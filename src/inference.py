"""Shared deterministic image preparation and saved-model inference.

Models contain their own rescaling and training-only augmentation. Inputs here
are RGB float32 pixels in [0, 255], padded to the saved input dimensions; never
normalise twice. Importing modeling registers the saved backbone wrappers.
"""
from pathlib import Path
import gc
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from .modeling import InferenceModeBackbone, TrainingPolicyBackbone, AdamWWithExclusions
from .final_bundle import LOCK_PATH, MODEL_PATH, verify_bundle

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 200
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def development_image_path(path):
    """Reject known official-test locations before opening a local image."""
    path = Path(path)
    parts = [part.casefold() for part in path.parts]
    if path.name.casefold() in {"test.zip", "test.txt"} or any(
            part == "test" or part.startswith("official_test") for part in parts):
        raise ValueError("Official-test images are disabled in the development validator.")
    return path


def decode_rgb(contents):
    """Decode JPEG/PNG bytes as RGB (first frame for an animated input)."""
    image = tf.io.decode_image(contents, channels=3, expand_animations=False)
    return tf.ensure_shape(image, (None, None, 3))


def resize_rgb(image, image_size=IMAGE_SIZE):
    """Preserve aspect ratio; fill the remaining input area with zero padding."""
    image_size = tuple(image_size)
    image = tf.image.resize_with_pad(image, *image_size, method="bilinear")
    return tf.ensure_shape(tf.cast(image, tf.float32), (*image_size, 3))


def preprocess_path(path, image_size=IMAGE_SIZE):
    """TensorFlow-compatible loader shared with the notebook's tf.data map."""
    if isinstance(path, (str, Path)):
        path = str(development_image_path(path))
    return resize_rgb(decode_rgb(tf.io.read_file(path)), image_size)


def preprocess_image(image, image_size=IMAGE_SIZE):
    """Prepare a local filename or uploaded bytes; return one unbatched image."""
    try:
        if isinstance(image, (bytes, bytearray, memoryview)):
            prepared = resize_rgb(decode_rgb(bytes(image)), image_size)
        else:
            path = development_image_path(image)
            if not path.is_file():
                raise ValueError("Select a readable image file first.")
            prepared = preprocess_path(str(path), image_size)
        if not bool(tf.reduce_all(tf.math.is_finite(prepared))):
            raise ValueError("Image contains non-finite pixel values.")
        return prepared
    except (tf.errors.OpError, TypeError, OSError) as error:
        raise ValueError("Image could not be decoded. Select a JPEG or PNG image.") from error


def load_class_mapping(path):
    """List position is the supplied numeric class ID, not alphabetic order."""
    try:
        names = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Class mapping is missing or unreadable.") from error
    if not isinstance(names, list) or len(names) != NUM_CLASSES:
        raise ValueError("Class mapping must contain exactly 200 ordered species names.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Class mapping contains an invalid species name.")
    return names


def model_image_size(model):
    """Read fixed RGB input dimensions from a checkpoint, including legacy 224 models."""
    shape = model.input_shape
    if not isinstance(shape, tuple) or len(shape) != 4 or shape[-1] != 3:
        raise ValueError("Model must accept one RGB image tensor.")
    if any(not isinstance(n, int) or n <= 0 for n in shape[1:3]):
        raise ValueError("Model must record fixed positive image dimensions.")
    return tuple(shape[1:3])


def load_model(path, device="/CPU:0"):
    """Load inference weights without optimizer state; CPU works without CUDA."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("Model checkpoint not available.")
    candidate = None
    if path.name == Path(MODEL_PATH).name:
        candidate = verify_bundle(path.parent.parent, expected_model_path=path)
        import importlib.metadata
        for package, version in candidate["lock"]["compatibility"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Final candidate requires verified {package} {version}.")
    try:
        with tf.device(device):
            # Functional's root dtype policy is not serialized in Keras 3.12.
            # Construct the final root in float32 without changing a notebook's
            # global policy permanently. Serialized child policies stay intact.
            previous_policy = keras.mixed_precision.global_policy()
            try:
                if candidate is not None:
                    keras.mixed_precision.set_global_policy("float32")
                model = keras.models.load_model(path, compile=False)
            finally:
                if candidate is not None:
                    keras.mixed_precision.set_global_policy(previous_policy)
        size = model_image_size(model)
        if model.output_shape[-1] != NUM_CLASSES:
            raise ValueError("Model must output 200 classes.")
        metadata_path = path.parent / "model_metadata.json"
        if metadata_path.is_file():
            contract = json.loads(metadata_path.read_text())
            # Stable final metadata describes its one export, not older 224
            # checkpoints that may share the models directory.
            describes_this_model = candidate is not None or contract.get("lock") != LOCK_PATH
            if describes_this_model and tuple(contract["image_size"]) != size:
                raise ValueError("Checkpoint input dimensions disagree with model metadata.")
        if candidate is not None:
            if size != (380, 380) or model.compute_dtype != "float32":
                raise ValueError("Final candidate input or compute precision is incompatible.")
            model.as01_final_class_names = tuple(candidate["class_names"])
            model.as01_inference_protocol = "single"
            model.as01_lock_sha256 = candidate["lock_sha256"]
        return model
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Model checkpoint is incompatible or unreadable: {error}") from error


def load_final_bundle(project_root=PROJECT_ROOT, device="/CPU:0"):
    """One reusable frontend entry point; pending review permits inspection only."""
    candidate = verify_bundle(project_root)
    return {"model": load_model(candidate["model_path"], device),
            "class_names": candidate["class_names"], "lock": candidate["lock"]}


def validate_probabilities(probabilities):
    values = np.asarray(probabilities, dtype=np.float32)
    if values.shape != (NUM_CLASSES,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Model did not return 200 finite, non-negative probabilities.")
    if not np.isclose(values.sum(), 1.0, atol=1e-4):
        raise ValueError("Model outputs are not normalised class probabilities.")
    return values


def top_k_predictions(probabilities, class_names, k=5):
    """Return exactly k ranked species; equal values use ascending class IDs."""
    values = validate_probabilities(probabilities)
    if len(class_names) != NUM_CLASSES or not 1 <= k <= NUM_CLASSES:
        raise ValueError("Provide 200 species names and k between 1 and 200.")
    ranked = np.argsort(-values, kind="stable")[:k]
    return [{"rank": rank, "class_id": int(label), "species": class_names[label],
             "probability": float(values[label])} for rank, label in enumerate(ranked, 1)]


def predict_probabilities(model, prepared, protocol="single"):
    """One shared deterministic batch API; TTA averages probabilities, not IDs."""
    if hasattr(model, "as01_final_class_names") and protocol != "single":
        raise ValueError("Final candidate inference protocol is single-view only.")
    if protocol not in ("single", "horizontal_flip"):
        raise ValueError("Unknown inference protocol.")
    prepared = tf.convert_to_tensor(prepared, dtype=tf.float32)
    probabilities = model(prepared, training=False).numpy()
    if protocol == "horizontal_flip":
        flipped = model(tf.image.flip_left_right(prepared), training=False).numpy()
        probabilities = (probabilities + flipped) / 2
    for row in probabilities:
        validate_probabilities(row)
    return probabilities


def predict_image(model, image, class_names, device="/CPU:0", protocol=None):
    """Image -> deterministic preprocessing -> probabilities -> ranked species."""
    if hasattr(model, "as01_final_class_names"):
        if tuple(class_names) != model.as01_final_class_names:
            raise ValueError("Class mapping ordering differs from the final candidate.")
        if protocol not in (None, "single"):
            raise ValueError("Final candidate inference protocol is single-view only.")
    with tf.device(device):
        prepared = preprocess_image(image, model_image_size(model))
        protocol = protocol or getattr(model, "as01_inference_protocol", "single")
        probabilities = predict_probabilities(model, prepared[None, ...], protocol)[0]
    probabilities = validate_probabilities(probabilities)
    ranked = top_k_predictions(probabilities, class_names)
    return {"top1": ranked[0], "top5": ranked, "probabilities": probabilities,
            "input_shape": tuple(prepared.shape), "input_dtype": prepared.dtype.name,
            "inference_protocol": protocol}


class ActiveModelCache:
    """Keep only the selected model; switching checkpoints releases its reference."""
    def __init__(self, device="/CPU:0"):
        self.device = device
        self.model = None
        self.key = None

    def get(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("Model checkpoint not available.")
        # Recheck candidate integrity even on a cache hit; a modified mapping or
        # manifest must not silently reuse an already-loaded final model.
        lock_sha = None
        if path.name == Path(MODEL_PATH).name:
            lock_sha = verify_bundle(path.parent.parent, expected_model_path=path)["lock_sha256"]
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size, lock_sha)
        if key != self.key:
            self.clear()
            self.model = load_model(path, device=self.device)
            self.key = key
        return self.model

    def clear(self):
        # Do not clear Keras's global session: a notebook may still own models.
        self.model = None
        self.key = None
        gc.collect()
