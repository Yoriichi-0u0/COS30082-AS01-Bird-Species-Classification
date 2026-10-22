"""Thin Gradio adapter for the frozen ConvNeXt export.

Training, evaluation, preprocessing and ranking stay in the accepted modules.
This file only validates uploads, owns one CPU model and formats its predictions.
Importing it does not import TensorFlow or load model weights.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
import time

from .web_demo import validate_upload, species_display_name, MAX_UPLOAD_BYTES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = '3c8d85465b3f0ae0732b4659e08f86caa7bae80dd1f39e1edfe3f8a0ef3ba0d5'
LOCK_SHA = '48c01dc05c98989dfb1ef7b840c1633a896b1b9e505dc16e264b6adbcc33f329'
MAPPING_SHA = 'ab490ae3ad95166b904c24b4173c2e018a5c17f9a1f8da87a1c86eb110ffdfa4'
_cache_lock = Lock()


def _permitted_path(path):
    """Check both spellings before reading, including symlinks to test folders."""
    path = Path(path)
    for candidate in (path, path.resolve()):
        parts = [part.casefold() for part in candidate.parts]
        if candidate.name.casefold() in {'test.zip', 'test.txt'} or any(
                part == 'test' or part.startswith('official_test') for part in parts):
            raise ValueError('Official-test inputs are not available in this application.')
    return path


def read_upload(value):
    """Pass original JPEG/PNG bytes to TensorFlow; preview conversion is separate.

    A File widget is used for uploads because an Image widget can re-encode a
    JPEG before prediction. Camera captures are newly generated PNG inputs.
    """
    if isinstance(value, (str, Path)):
        path = _permitted_path(value)
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            raise ValueError('Select a JPEG or PNG image.')
        try:
            if path.stat().st_size > MAX_UPLOAD_BYTES:
                raise ValueError('Image exceeds the 10 MiB upload limit.')
            value = path.read_bytes()
        except OSError as error:
            raise ValueError('Select a readable image; the selected file is missing.') from error
    return validate_upload(value)


class ConvNeXtPredictor:
    """One CPU model, with a lock preventing concurrent activation-memory spikes."""
    def __init__(self, project_root=PROJECT_ROOT):
        started = time.perf_counter()
        self.root = Path(project_root).resolve()
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Two TF compute threads approximate the CPU Basic resource allocation.
        os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
        os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
        os.environ.setdefault('OMP_NUM_THREADS', '2')
        manifest = self.root/'finalization/convnext_base_320/model_lock_candidate.json'
        mapping = self.root/'models/class_mapping.json'
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != LOCK_SHA:
            raise ValueError('Frozen ConvNeXt manifest identity differs.')
        if hashlib.sha256(mapping.read_bytes()).hexdigest() != MAPPING_SHA:
            raise ValueError('The ordered class mapping identity differs.')
        lock = json.loads(manifest.read_text())
        if lock['export']['sha256'] != MODEL_SHA or lock['export']['bytes'] != 353755501:
            raise ValueError('A different model cannot be substituted.')
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        from finalization.convnext_base_320.candidate import load_candidate
        from .inference import predict_image
        # The accepted loader verifies actual model bytes and every bound source,
        # restores the Functional root's mixed policy, and uses compile=False.
        self.model = load_candidate(self.root, device='/CPU:0')
        self.class_names = tuple(self.model.as01_final_class_names)
        if self.model.compute_dtype != 'float16' or self.model.output.dtype != 'float32':
            raise ValueError('Frozen model precision differs; no automatic conversion.')
        self.predict_image = predict_image
        self.prediction_lock = Lock()
        self.load_seconds = time.perf_counter() - started

    def predict(self, value):
        upload = read_upload(value)
        with self.prediction_lock:
            return self.predict_image(self.model, upload['contents'], self.class_names,
                                      device='/CPU:0', protocol='single')


@lru_cache(maxsize=1)
def _cached_predictor(project_root):
    return ConvNeXtPredictor(project_root)


def get_predictor(project_root=PROJECT_ROOT):
    # lru_cache alone can construct duplicates on simultaneous first requests.
    with _cache_lock:
        return _cached_predictor(Path(project_root).resolve())


def format_prediction(result):
    best = result['top1']
    headline = (f"### {species_display_name(best['species'])}\n\n"
                f"**Class ID {best['class_id']}** · **{best['probability']:.2%}** softmax confidence")
    # Preserve the accepted stable ranking, unrounded scores and original IDs.
    scores = {f"{species_display_name(row['species'])} · ID {row['class_id']}":
              row['probability'] for row in result['top5']}
    return headline, scores
