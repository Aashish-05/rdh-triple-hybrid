"""Create an immutable, content-deduplicated versioned dataset split."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config


SPLIT_NAMES = ("train", "val", "holdout", "test")
PARTITION_PRIORITY = {"train": 0, "val": 1, "holdout": 2, "test": 3}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_versioned_split(splits, manifest, hash_function=sha256_file):
    """Keep duplicates in the highest-priority split; exclude redundant lower copies."""
    original_membership = {}
    for split_name in SPLIT_NAMES:
        for filename in splits[split_name]:
            if filename in original_membership:
                raise ValueError(f"Filename occurs in multiple original splits: {filename}")
            original_membership[filename] = split_name
            if filename not in manifest:
                raise ValueError(f"Missing manifest entry for {filename}")

    hashes = {}
    paths = {}
    by_hash = defaultdict(list)
    for filename, split_name in original_membership.items():
        path = Path(manifest[filename]["absolute_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file for {filename}: {path}")
        digest = hash_function(path)
        hashes[filename] = digest
        paths[filename] = path
        by_hash[digest].append(filename)

    new_splits = {name: [] for name in SPLIT_NAMES}
    excluded = []
    duplicate_groups = []
    assigned_hash_partition = {}
    for digest, filenames in by_hash.items():
        partitions = {original_membership[filename] for filename in filenames}
        keep_partition = min(partitions, key=PARTITION_PRIORITY.__getitem__)
        assigned_hash_partition[digest] = keep_partition
        for filename in filenames:
            original_split = original_membership[filename]
            if original_split == keep_partition:
                new_splits[keep_partition].append(filename)
            else:
                excluded.append({
                    "filename": filename,
                    "original_split": original_split,
                    "retained_split": keep_partition,
                    "retained_filenames": [
                        other for other in filenames if original_membership[other] == keep_partition
                    ],
                    "sha256": digest,
                    "reason": "identical content retained in higher-priority partition",
                })
        if len(filenames) > 1:
            duplicate_groups.append({
                "sha256": digest,
                "files": [
                    {"filename": filename, "original_split": original_membership[filename]}
                    for filename in filenames
                ],
                "retained_partition": keep_partition,
            })

    for name in SPLIT_NAMES:
        order = {filename: index for index, filename in enumerate(splits[name])}
        new_splits[name].sort(key=order.__getitem__)

    # Ensure the new active partitions are content-disjoint.
    active_hashes = {
        name: {hashes[filename] for filename in filenames}
        for name, filenames in new_splits.items()
    }
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1:]:
            if active_hashes[left] & active_hashes[right]:
                raise AssertionError(f"Content overlap remains: {left} and {right}")

    new_manifest = {}
    for filename, original_split in original_membership.items():
        entry = dict(manifest[filename])
        entry["original_split"] = original_split
        entry["sha256"] = hashes[filename]
        entry["split"] = (
            original_split if filename in new_splits[original_split] else "excluded_redundant"
        )
        new_manifest[filename] = entry

    metadata = {
        "schema_version": 2,
        "seed": 42,
        "source_splits": "data/processed/splits.json",
        "source_manifest": "data/processed/dataset_manifest.json",
        "duplicate_policy": (
            "Rehash every source in all four original splits. For content shared across "
            "partitions, retain copies in the highest-priority partition (train > val > "
            "holdout > test), exclude lower-priority copies from active splits, and record "
            "all exclusions. Keep duplicate copies when they remain in the same partition."
        ),
        "partition_priority": list(PARTITION_PRIORITY),
        "counts": {name: len(new_splits[name]) for name in SPLIT_NAMES},
        "excluded_redundant_count": len(excluded),
        "duplicate_content_groups": duplicate_groups,
    }
    return {
        "metadata": metadata,
        "splits": new_splits,
        "excluded_redundant": excluded,
    }, new_manifest


def main():
    cfg = get_config()
    with cfg.paths.splits_path.open("r", encoding="utf-8") as stream:
        source_splits = json.load(stream)
    with cfg.paths.manifest_path.open("r", encoding="utf-8") as stream:
        source_manifest = json.load(stream)

    # Refuse to overwrite any versioned split or manifest artifact.
    split_path = OUTPUT_DIR / "splits.json"
    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    if split_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Versioned split output already exists: {OUTPUT_DIR}")

    new_splits, new_manifest = build_versioned_split(source_splits, source_manifest)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_bytes = json.dumps(new_splits, indent=2).encode("utf-8")
    manifest_bytes = json.dumps(new_manifest, indent=2).encode("utf-8")
    with split_path.open("xb") as stream:
        stream.write(split_bytes)
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)

    print(f"Created versioned split: {split_path}")
    print(f"Created versioned manifest: {manifest_path}")
    print("Active split counts:", new_splits["metadata"]["counts"])
    print("Excluded redundant copies:", new_splits["metadata"]["excluded_redundant_count"])
    print("Exact-content duplicate groups:", len(new_splits["metadata"]["duplicate_content_groups"]))


if __name__ == "__main__":
    main()
