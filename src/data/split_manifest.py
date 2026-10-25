"""Helpers for selecting and validating versioned dataset split manifests."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


SPLIT_NAMES = ("train", "val", "holdout", "test")


def resolve_split_paths(cfg, version: str) -> Tuple[Path, Path]:
    """Return split and manifest paths without changing the historical defaults."""
    if version == "historical":
        return cfg.paths.splits_path, cfg.paths.manifest_path
    if version == "v2":
        version_dir = cfg.paths.processed_data_dir / "v2"
        return version_dir / "splits.json", version_dir / "dataset_manifest.json"
    raise ValueError(f"Unsupported dataset split version: {version}")


def load_split_manifest(
    splits_path: Path,
    manifest_path: Path,
) -> Tuple[Dict[str, List[str]], dict, dict]:
    """Load either the historical split shape or the versioned nested shape."""
    with Path(splits_path).open("r", encoding="utf-8") as stream:
        split_document = json.load(stream)
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    splits = split_document.get("splits", split_document)
    missing_splits = [name for name in SPLIT_NAMES if name not in splits]
    if missing_splits:
        raise ValueError(f"Split file is missing required partitions: {missing_splits}")
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be a filename-to-entry object")

    for split_name in SPLIT_NAMES:
        filenames = splits[split_name]
        if not isinstance(filenames, list) or not all(
            isinstance(filename, str) for filename in filenames
        ):
            raise ValueError(f"Split '{split_name}' must be a list of filenames")
        if len(filenames) != len(set(filenames)):
            raise ValueError(f"Split '{split_name}' contains duplicate filenames")
        for filename in filenames:
            if filename not in manifest:
                raise KeyError(
                    f"Filename {filename!r} from split '{split_name}' is absent "
                    "from the dataset manifest"
                )
            entry = manifest[filename]
            if not isinstance(entry, dict) or not entry.get("absolute_path"):
                raise ValueError(
                    f"Manifest entry for {filename!r} must contain an absolute_path"
                )
            declared_split = entry.get("split")
            if declared_split is not None and declared_split != split_name:
                raise ValueError(
                    f"Manifest marks {filename!r} as {declared_split!r}, but it "
                    f"belongs to active split {split_name!r}"
                )

    return splits, manifest, split_document


def ordered_filename_sha256(filenames: List[str]) -> str:
    """Fingerprint an ordered list of filenames for experiment provenance."""
    return hashlib.sha256("\n".join(filenames).encode("utf-8")).hexdigest()


def validate_v2_thresholds(
    thresholds: dict,
    validation_filenames: List[str],
    cnn_sha256: str,
    transformer_sha256: str,
) -> None:
    """Require frozen router thresholds to match this v2 validation split and models."""
    expected = {
        "dataset_version": "v2",
        "selection_split": "val",
        "validation_count": len(validation_filenames),
        "validation_filenames_sha256": ordered_filename_sha256(validation_filenames),
    }
    mismatches = [
        key for key, value in expected.items() if thresholds.get(key) != value
    ]
    checkpoints = thresholds.get("checkpoints", {})
    if checkpoints.get("cnn_sha256") != cnn_sha256:
        mismatches.append("checkpoints.cnn_sha256")
    if checkpoints.get("transformer_sha256") != transformer_sha256:
        mismatches.append("checkpoints.transformer_sha256")
    if mismatches:
        raise ValueError(
            "Frozen thresholds do not match the current v2 validation split and "
            f"checkpoints: {', '.join(mismatches)}"
        )


def file_sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
