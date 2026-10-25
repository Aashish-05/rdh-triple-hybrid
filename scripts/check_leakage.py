"""
Recompute SHA-256 hashes for every source in all four dataset splits.

Manifest hashes are not trusted: the existing manifest intentionally omits
holdout hashes, and cross-split duplicates must be detected from disk.
"""
import hashlib
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config


SPLIT_NAMES = ("train", "val", "holdout", "test")


def compute_sha256(file_path: Path) -> str:
    """Hash a source file in bounded memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_content_duplicates(splits, manifest, hash_function=compute_sha256):
    """Hash all split sources and return duplicate groups and unavailable files."""
    hashes = defaultdict(lambda: defaultdict(list))
    missing = []
    for split_name in SPLIT_NAMES:
        for filename in splits[split_name]:
            entry = manifest.get(filename)
            if entry is None:
                missing.append((split_name, filename, "missing manifest entry"))
                continue
            file_path = Path(entry["absolute_path"])
            if not file_path.is_file():
                missing.append((split_name, filename, "missing source file"))
                continue
            hashes[hash_function(file_path)][split_name].append(filename)

    duplicates = []
    for digest, files_by_split in hashes.items():
        if sum(len(files) for files in files_by_split.values()) > 1:
            duplicates.append({
                "sha256": digest,
                "files_by_split": dict(files_by_split),
            })
    return duplicates, missing


def main():
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versioned",
        choices=("v2",),
        help="Audit a versioned split under data/processed/<version>/.",
    )
    args = parser.parse_args()
    split_path = cfg.paths.splits_path
    manifest_path = cfg.paths.manifest_path
    if args.versioned:
        version_dir = cfg.paths.processed_data_dir / args.versioned
        split_path = version_dir / "splits.json"
        manifest_path = version_dir / "dataset_manifest.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Splits file not found: {split_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with split_path.open("r", encoding="utf-8") as stream:
        split_document = json.load(stream)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    splits = split_document.get("splits", split_document)

    print("=" * 72)
    print("ALL-SPLIT DATA LEAKAGE & INTEGRITY AUDIT")
    print("=" * 72)
    for split_name in SPLIT_NAMES:
        print(f"  {split_name:8s}: {len(splits[split_name]):6d} files")

    filename_duplicates = []
    seen_filenames = {}
    for split_name in SPLIT_NAMES:
        for filename in splits[split_name]:
            if filename in seen_filenames:
                filename_duplicates.append((filename, seen_filenames[filename], split_name))
            else:
                seen_filenames[filename] = split_name

    print("\n[*] Recomputing SHA-256 for every split source (including holdout)...")
    duplicate_groups, missing = find_content_duplicates(splits, manifest)

    if filename_duplicates:
        print(f"[FAIL] {len(filename_duplicates)} filename overlap(s): {filename_duplicates}")
    else:
        print("[PASS] No filename overlaps.")

    cross_partition_duplicates = [
        group for group in duplicate_groups if len(group["files_by_split"]) > 1
    ]
    within_partition_duplicates = [
        group for group in duplicate_groups if len(group["files_by_split"]) == 1
    ]
    if cross_partition_duplicates:
        print(f"[FAIL] {len(cross_partition_duplicates)} cross-partition duplicate group(s):")
        for group in cross_partition_duplicates:
            print(f"  SHA-256 {group['sha256']}")
            for split_name, filenames in group["files_by_split"].items():
                print(f"    {split_name}: {', '.join(filenames)}")
    else:
        print("[PASS] No content is shared across partitions.")
    if within_partition_duplicates:
        print(f"[INFO] {len(within_partition_duplicates)} duplicate group(s) retained within one partition:")
        for group in within_partition_duplicates:
            split_name, filenames = next(iter(group["files_by_split"].items()))
            print(f"  {split_name}: {', '.join(filenames)}")

    if missing:
        print(f"[FAIL] {len(missing)} missing entries/files:")
        for split_name, filename, reason in missing:
            print(f"  {split_name}/{filename}: {reason}")

    dimension_errors = []
    for split_name in SPLIT_NAMES:
        for filename in splits[split_name]:
            entry = manifest.get(filename)
            if entry and (entry.get("width") != 512 or entry.get("height") != 512):
                dimension_errors.append((split_name, filename))
    if dimension_errors:
        print(f"[FAIL] {len(dimension_errors)} image(s) do not have 512x512 manifest dimensions.")
    else:
        print("[PASS] Manifest dimensions are 512x512 for all four splits.")

    passed = not filename_duplicates and not cross_partition_duplicates and not missing and not dimension_errors
    print("\n" + ("AUDIT PASSED" if passed else "AUDIT FAILED — DUPLICATES/INTEGRITY ISSUES FOUND"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
