"""Inspect the pending ConvNeXt export without replacing the approved B3 bundle.

Importing this module reads no data, loads no model and grants no approval.
RGB preparation and predictions remain in the unchanged shared inference code.
"""
import importlib.metadata
import json
from pathlib import Path
import sys

from src.final_bundle import portable_path, protect_official_data, sha256_file

ROOT = Path(__file__).resolve().parents[2]
FOLDER = 'finalization/convnext_base_320'
MODEL_PATH = 'models/final_convnext_base_320.keras'
LOCK_PATH = FOLDER + '/model_lock_candidate.json'
SOURCE = Path('/home/user1/cos30082/as01/runs/iteration_06_20261014T141527_149836Z_full/experiments/base_full_320/best.keras')
SOURCE_SHA = '4217ab86d04f9b1fb84927b28192103d0cd8c3a89d7e91ab3fca1f526c730e9b'
WEIGHTS_SHA = '8c5c01b220163482e9d813a0f0692e7eb13b4e7cc6e3f6a1bc1fc6ade8b901a8'
SPLIT_SHA = '65c456049b19bc87173d0e9072cc7d8f3a20366a4d3e8ac0fd0ab7c3e4f358cd'
CLASS_SHA = 'ab490ae3ad95166b904c24b4173c2e018a5c17f9a1f8da87a1c86eb110ffdfa4'
FEATURES = 'native GAP before pretrained LayerNorm; pooling=avg'
INPUT = {'shape': [320, 320, 3], 'dtype': 'float32', 'pixel_range': [0, 255], 'color': 'RGB',
         'resize': 'resize_with_pad', 'method': 'bilinear', 'padding_value': 0, 'normalization': 'inside saved backbone'}
INFERENCE = {'protocol': 'single', 'training': False, 'augmentation': False, 'tta': False,
             'compute_policy': 'mixed_float16', 'output_dtype': 'float32', 'gpu_batch': 8, 'pad_last_batch': True}
BOUND_FILES = {'src/__init__.py', 'src/modeling.py', 'src/inference.py', 'src/final_bundle.py',
               'src/local_validator.py', FOLDER + '/candidate.py'}


def new_json(path, value):
    """Exclusive creation prevents a rerun from rewriting reviewed evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_lock(lock):
    try:
        valid = (lock['schema_version'] == 1 and lock['status'] == 'pending_external_review'
            and lock['model'] == {'architecture': 'ConvNeXtBase', 'seed': 42, 'classes': 200, 'parameters': 88193864}
            and lock['source']['sha256'] == SOURCE_SHA and lock['source']['bytes'] == 1765368530
            and lock['learned_weight_sha256'] == WEIGHTS_SHA
            and lock['export']['path'] == MODEL_PATH and lock['export']['optimizer_state'] is False
            and isinstance(lock['export']['bytes'], int) and lock['export']['bytes'] > 0
            and lock['class_mapping'] == {'path': 'models/class_mapping.json', 'sha256': CLASS_SHA}
            and lock['split'] == {'sha256': SPLIT_SHA, 'training_images': 3870, 'validation_images': 959}
            and lock['input'] == INPUT and lock['inference'] == INFERENCE
            and lock['feature_contract'] == FEATURES
            and lock['compatibility'] == {'tensorflow': '2.19.1', 'keras': '3.12.0'}
            and BOUND_FILES.issubset(lock['bound_files']))
        if not valid:
            raise ValueError('ConvNeXt identity or inference contract differs from the reviewed candidate.')
        for value in [lock['export']['sha256'], *lock['bound_files'].values()]:
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Malformed SHA-256 in candidate.')
    except (KeyError, TypeError) as error:
        raise ValueError('Candidate manifest is incomplete.') from error
    return lock


def verify_bundle(project_root=ROOT):
    """Verify identity for inspection; matching hashes are not authentication."""
    root = Path(project_root).resolve()
    try:
        lock_path = portable_path(root, LOCK_PATH)
        lock = validate_lock(json.loads(lock_path.read_text()))
        digest = sha256_file(lock_path)
        fingerprint = json.loads(portable_path(root, FOLDER + '/lock_fingerprint.json').read_text())
        if fingerprint != {'path': LOCK_PATH, 'sha256': digest}:
            raise ValueError('Detached manifest fingerprint differs.')
        approval = json.loads(portable_path(root, FOLDER + '/approval_status.json').read_text())
        if (approval.get('status') != 'pending_external_review' or approval.get('lock_sha256') != digest
                or approval.get('model_approved') is not False or approval.get('official_test_authorized') is not False
                or approval.get('approved_by') is not None):
            raise ValueError('This package permits pending inspection only, not model or test approval.')
        model_path = portable_path(root, lock['export']['path'])
        if model_path.stat().st_size != lock['export']['bytes'] or sha256_file(model_path) != lock['export']['sha256']:
            raise ValueError('Export file identity differs.')
        mapping = portable_path(root, lock['class_mapping']['path'])
        if sha256_file(mapping) != CLASS_SHA:
            raise ValueError('Class mapping identity differs.')
        names = json.loads(mapping.read_text())
        if not isinstance(names, list) or len(names) != 200 or len(set(names)) != 200:
            raise ValueError('Expected 200 unique ordered species names.')
        for relative, expected in lock['bound_files'].items():
            if sha256_file(portable_path(root, relative)) != expected:
                raise ValueError(f'Bound inference source changed: {relative}')
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f'Candidate bundle is missing or malformed: {error}') from error
    return {'lock': lock, 'lock_sha256': digest, 'model_path': model_path,
            'class_names': names, 'approval': approval}


def load_candidate(project_root=ROOT, device='/CPU:0'):
    """Retain the trained mixed policy; never change model layers for CPU use."""
    bundle = verify_bundle(project_root)
    for package, version in bundle['lock']['compatibility'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Candidate requires verified {package} {version}.')
    from tensorflow import keras
    from src.inference import load_model
    previous = keras.mixed_precision.global_policy()
    try:
        # Keras 3.12 does not serialize the Functional root policy. Child layer
        # policies are serialized; use the original root policy while loading.
        keras.mixed_precision.set_global_policy('mixed_float16')
        model = load_model(bundle['model_path'], device=device)
    finally:
        keras.mixed_precision.set_global_policy(previous)
    if model.input_shape != (None, 320, 320, 3) or model.output_shape != (None, 200):
        raise ValueError('Unexpected candidate model dimensions.')
    model.as01_final_class_names = tuple(bundle['class_names'])
    model.as01_inference_protocol = 'single'
    model.as01_lock_sha256 = bundle['lock_sha256']
    return model


def candidate_record(project_root=ROOT, local_root=None):
    bundle = verify_bundle(project_root)
    return {'label': 'ConvNeXt-Base 320 | seed 42 | pending external review',
            'stage': 'convnext_candidate', 'path': bundle['model_path'],
            'mapping': Path(project_root) / 'models/class_mapping.json', 'exported': True,
            'protocol': 'single', 'project_root': Path(project_root),
            'local_root': Path(local_root or Path.home() / 'cos30082/as01'),
            'lock_status': 'pending_external_review'}


def launch_candidate_validator(project_root=ROOT, show=True):
    """Add the pending export to the existing UI without changing its source."""
    from src.inference import ActiveModelCache
    from src.local_validator import LocalValidator
    root = Path(project_root).resolve()
    record = candidate_record(root)

    class CandidateCache(ActiveModelCache):
        def get(self, path):
            path = Path(path).resolve()
            if path != record['path']:
                return super().get(path)
            checked = verify_bundle(root)
            key = (str(path), path.stat().st_mtime_ns, path.stat().st_size, checked['lock_sha256'])
            if key != self.key:
                self.clear()
                self.model = load_candidate(root, device=self.device)
                self.key = key
            return self.model

    ui = LocalValidator(root, show=show)
    ui.models.insert(0, record)
    ui.records = {item['label']: item for item in ui.models}
    ui.model_selector.configure(values=list(ui.records))
    ui.cache = CandidateCache()
    ui.model_name.set(record['label'])
    ui.on_model_change()
    if show:
        ui.root.mainloop()
    return ui


def deny_official_data():
    """Install the existing audited development guard before ML/data imports."""
    sys.addaudithook(protect_official_data)
