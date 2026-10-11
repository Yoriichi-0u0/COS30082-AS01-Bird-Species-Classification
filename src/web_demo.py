"""Small upload adapter; all neural-network inference stays in src.inference.

Pillow validates the container and builds a preview. The original uploaded bytes
go to the shared TensorFlow decoder, so preview handling cannot change inference.
"""
from io import BytesIO
import os
from pathlib import Path
import sys
from threading import Lock
import warnings

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
APPROVED_LOCK_SHA256 = "4826b177eb71b677d7b89e396dd2a2c9cb43e3a7a5014d540634481dde3c9b44"
APPROVED_EXPORT_SHA256 = "9fe9ea11937d67ca34d0f40f34511871d44a56a1610f229c8b8cfd410c80d441"


def deny_official_dataset_access(event, arguments):
    """App-only guard, including an isolated venv's inherited library code.

    TensorFlow has a Python package named 'test'. Only installed library code is
    exempt, never images/annotations. The fingerprint-bound backend is untouched.
    """
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(arguments[0]))
    parts = [part.casefold() for part in path.parts]
    forbidden = path.name.casefold() in {"test.zip", "test.txt"} or any(
        part.startswith("official_test") for part in parts)
    if "test" in parts:
        library_code = path.suffix in {".py", ".pyc", ".so"} and any(
            path.resolve().is_relative_to(Path(prefix).resolve() / "lib")
            for prefix in (sys.prefix, sys.base_prefix))
        forbidden = forbidden or not library_code
    if forbidden:
        raise PermissionError("Official-test dataset access is prohibited in the demonstration.")


def validate_upload(contents):
    """Accept bounded JPEG/PNG bytes, including grayscale and RGBA images."""
    if not isinstance(contents, (bytes, bytearray, memoryview)) or not contents:
        raise ValueError("Upload a JPEG or PNG image first.")
    contents = bytes(contents)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 10 MiB upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(contents)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported.")
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 20-megapixel limit.")
                image.verify()
            with Image.open(BytesIO(contents)) as image:
                image.load()
                preview = image.convert("RGB")
        return {"contents": contents, "preview": preview,
                "width": width, "height": height}
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Image is unreadable or incomplete. Select a valid JPEG or PNG.") from error


def species_display_name(name):
    """Improve display only; never reorder or alter the stored class mapping."""
    return name.split(".", 1)[-1].replace("_", " ")


class DemoPredictor:
    """One CPU model shared by sessions, with serialized calls to limit memory.

    The Streamlit resource cache owns this object. Predictions are not globally
    cached: uploaded image bytes and results stay in their own browser session.
    """
    def __init__(self, project_root=PROJECT_ROOT):
        # This affects only the demo process, never the training environment.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from .final_bundle import verify_bundle
        identity = verify_bundle(project_root)
        if (identity["lock_sha256"] != APPROVED_LOCK_SHA256
                or identity["lock"]["export"]["sha256"] != APPROVED_EXPORT_SHA256):
            raise ValueError("The approved final model identity differs; stop for review.")
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as error:
            if tf.config.get_visible_devices("GPU"):
                raise ValueError("The demo needs its own fresh CPU process.") from error
        from .inference import load_final_bundle, predict_image
        self.bundle = load_final_bundle(project_root, device="/CPU:0")
        self.model = self.bundle["model"]
        self.class_names = self.bundle["class_names"]
        self.predict_image = predict_image
        self.prediction_lock = Lock()

    def predict(self, contents):
        upload = validate_upload(contents)
        with self.prediction_lock:
            return self.predict_image(self.model, upload["contents"], self.class_names,
                                      device="/CPU:0", protocol="single")
