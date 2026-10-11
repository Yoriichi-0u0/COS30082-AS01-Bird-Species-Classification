"""Stage 1 candidate identity and portable paths, independent of TensorFlow.

An inspectable candidate is not an approved final lock. This module contains no
training, official-evaluation, export or account actions at import time.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

SOURCE_RUN = "iteration_05_20261009T042258_290143Z_full"
SOURCE_SHA256 = "942471b5c63c796ac8c4b066913ecc5c57fdc08a9955c545ddf14f5bbe467202"
SOURCE_BYTES = 79058025
CLASS_SHA256 = "ab490ae3ad95166b904c24b4173c2e018a5c17f9a1f8da87a1c86eb110ffdfa4"
SPLIT_SHA256 = "65c456049b19bc87173d0e9072cc7d8f3a20366a4d3e8ac0fd0ab7c3e4f358cd"
LOCK_PATH = "finalization/final_model_lock.json"
MODEL_PATH = "models/final_b3_380.keras"
REQUIRED_BOUND_FILES = {"src/__init__.py", "src/modeling.py", "src/inference.py", "src/final_bundle.py",
                        "models/model_metadata.json", "models/preprocessing.json"}
INPUT_CONTRACT = {"shape": [380, 380, 3], "dtype": "float32", "pixel_range": [0, 255],
                  "color": "RGB", "resize": "resize_with_pad", "method": "bilinear",
                  "padding_value": 0, "normalization": "inside saved backbone"}
INFERENCE_CONTRACT = {"protocol": "single", "training": False, "augmentation": False,
                      "tta": False, "compute_policy": "float32"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Strict JSON prevents NaN from becoming an ambiguous verification result."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def portable_path(root, relative):
    """Resolve a bundle member, rejecting absolute paths and symlink escapes."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Bundle path must be a portable project-relative path.")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError("Bundle path must remain inside the project.")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Bundle path leaves the project through a symlink.")
    return path


def validate_lock(lock):
    """Reject substitutions; Stage 1 cannot grant final approval to itself."""
    try:
        valid = (
            lock["schema_version"] == 1 and lock["status"] == "pending_external_review"
            and lock["model"]["architecture"] == "EfficientNetB3"
            and lock["model"]["seed"] == 42 and lock["model"]["classes"] == 200
            and lock["source"]["sha256"] == SOURCE_SHA256
            and lock["source"]["bytes"] == SOURCE_BYTES
            and lock["export"]["path"] == MODEL_PATH
            and lock["export"]["optimizer_state"] is False
            and lock["export"]["format"] == "keras_v3"
            and lock["class_mapping"]["path"] == "models/class_mapping.json"
            and lock["class_mapping"]["sha256"] == CLASS_SHA256
            and lock["class_mapping"]["classes"] == 200
            and lock["class_mapping"]["class_id_base"] == 0
            and lock["split"]["sha256"] == SPLIT_SHA256
            and lock["split"]["training_images"] == 3870
            and lock["split"]["validation_images"] == 959
            and lock["input"] == INPUT_CONTRACT
            and lock["inference"] == INFERENCE_CONTRACT
            and all(lock["inference"][k] is False for k in ("training", "augmentation", "tta"))
            and lock["compatibility"] == {"tensorflow": "2.19.1", "keras": "3.12.0"}
            and isinstance(lock["bound_files"], dict)
            and REQUIRED_BOUND_FILES.issubset(lock["bound_files"]))
        if not valid:
            raise ValueError("Final-model candidate identity or inference contract differs from approval.")
        for digest in [lock["export"]["sha256"], lock["weight_tensors_sha256"], *lock["bound_files"].values()]:
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid SHA-256 fingerprint in candidate.")
        if not isinstance(lock["export"]["bytes"], int) or lock["export"]["bytes"] <= 0:
            raise ValueError("Invalid export byte size.")
    except (KeyError, TypeError) as error:
        raise ValueError("Model-lock candidate is incomplete or malformed.") from error
    return lock


def verify_bundle(project_root, expected_model_path=None):
    """Verify portable export/mapping/backend identities before lazy loading."""
    root = Path(project_root).resolve()
    lock_path = portable_path(root, LOCK_PATH)
    try:
        lock = validate_lock(json.loads(lock_path.read_text(encoding="utf-8")))
        fingerprint = json.loads(portable_path(root, "finalization/lock_fingerprint.json").read_text(encoding="utf-8"))
        if fingerprint != {"path": LOCK_PATH, "sha256": sha256_file(lock_path)}:
            raise ValueError("Detached model-lock fingerprint differs from the candidate.")
        model_path = portable_path(root, lock["export"]["path"])
        if expected_model_path is not None and Path(expected_model_path).resolve() != model_path:
            raise ValueError("Checkpoint path disagrees with the locked export identity.")
        for record in [lock["export"], lock["class_mapping"]]:
            path = portable_path(root, record["path"])
            if (("bytes" in record and path.stat().st_size != record["bytes"])
                    or sha256_file(path) != record["sha256"]):
                raise ValueError(f"Bundle fingerprint differs: {record['path']}")
        for relative, digest in lock["bound_files"].items():
            if sha256_file(portable_path(root, relative)) != digest:
                raise ValueError(f"Bundle fingerprint differs: {relative}")
        names = json.loads(portable_path(root, lock["class_mapping"]["path"]).read_text(encoding="utf-8"))
        if not isinstance(names, list) or len(names) != 200 or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("Class mapping must contain 200 ordered species names.")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Final-model candidate is missing or unreadable: {error}") from error
    return {"project_root": root, "lock": lock, "lock_sha256": sha256_file(lock_path),
            "model_path": model_path, "class_names": names}


def protect_official_data(event, arguments):
    """Deny official paths before opening them, including synthetic test fixtures."""
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(arguments[0]))
    parts = [p.casefold() for p in path.parts]
    forbidden = path.name.casefold() in {"test.txt", "test.zip"} or any(p.startswith("official_test") for p in parts)
    if "test" in parts:
        # TensorFlow imports its own test API. Permit installed Python library
        # code only; this exception never allows images, labels or archives.
        code = path.suffix in {".py", ".pyc", ".so"} and path.resolve().is_relative_to(Path(sys.prefix).resolve() / "lib")
        forbidden = forbidden or not code
    if forbidden:
        raise PermissionError("Official-test access is prohibited during Stage 1.")
