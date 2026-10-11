"""Small shared builders used by the educational notebook and isolated VRAM probes.

Historical backbones use inference mode. Iteration 05 additionally tests explicit
outer-training propagation while keeping BatchNormalization frozen.
"""
import hashlib
import numpy as np
from tensorflow import keras

ARCHITECTURES = {
    "b0": {"constructor": "EfficientNetB0", "backbone": "efficientnetb0", "higher": 224},
    "b2": {"constructor": "EfficientNetB2", "backbone": "efficientnetb2", "higher": 260},
    "b3": {"constructor": "EfficientNetB3", "backbone": "efficientnetb3", "higher": 300},
    "convnext": {"constructor": "ConvNeXtTiny", "backbone": "convnext_tiny", "higher": 320},
}


@keras.utils.register_keras_serializable(package="AS01")
class InferenceModeBackbone(keras.layers.Wrapper):
    """Enforce inference-mode features even when the outer model is fitting.

    Keras 3.12 propagates an outer Functional training argument into nested
    models. A wrapper makes this policy explicit inside an actual Python call,
    where outer training mode cannot override it. Weight trainability is separate.
    The class is registered on import for safe CPU checkpoint loading.
    """
    def call(self, inputs, training=None):
        return self.layer(inputs, training=False)


@keras.utils.register_keras_serializable(package="AS01")
class TrainingPolicyBackbone(keras.layers.Wrapper):
    """Forward the actual fit/evaluate flag; never force stochastic inference."""
    def __init__(self, layer, policy="follow_training", **kwargs):
        if policy != "follow_training":
            raise ValueError("Use the historical wrapper for fixed inference mode.")
        super().__init__(layer, **kwargs)
        self.policy = policy

    def call(self, inputs, training=None):
        return self.layer(inputs, training=False if training is None else training)

    def get_config(self):
        return {**super().get_config(), "policy": self.policy}


@keras.utils.register_keras_serializable(package="AS01")
class AdamWWithExclusions(keras.optimizers.AdamW):
    """Keras 3.12 does not serialize decay exclusions; retain the small policy."""
    def __init__(self, exclusion_names=("bias", "gamma", "beta"), **kwargs):
        super().__init__(**kwargs)
        self.exclusion_names = list(exclusion_names)
        self.exclude_from_weight_decay(var_names=self.exclusion_names)

    def get_config(self):
        return {**super().get_config(), "exclusion_names": self.exclusion_names}


def augmentation(seed=42, name="training_augmentation"):
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(.04, fill_mode="reflect", seed=seed + 1),
        keras.layers.RandomZoom(.10, fill_mode="reflect", seed=seed + 2),
    ], name=name)


def build_classifier(architecture, resolution=224, weights="imagenet", seed=42, backbone_policy="fixed"):
    """RGB [0,255] -> original augmentation -> ImageNet backbone -> common head."""
    constructor = getattr(keras.applications, ARCHITECTURES[architecture]["constructor"])
    backbone = constructor(include_top=False, weights=weights,
                           input_shape=(resolution, resolution, 3))
    backbone.trainable = False
    inputs = keras.Input((resolution, resolution, 3), name="rgb_pixels")
    if backbone_policy == "fixed":
        features = InferenceModeBackbone(backbone, name="backbone_inference")(augmentation(seed)(inputs), training=False)
    elif backbone_policy == "follow_training":
        features = TrainingPolicyBackbone(backbone, name="backbone_inference")(augmentation(seed)(inputs))
    else:
        raise ValueError("Unknown backbone training policy.")
    # A spatial average keeps head capacity matched without a large Flatten layer.
    features = keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    features = keras.layers.Dropout(.3, seed=seed + 3, name="head_dropout")(features)
    outputs = keras.layers.Dense(200, activation="softmax", dtype="float32", name="species")(features)
    return keras.Model(inputs, outputs, name=f"{architecture}_birds")


def get_backbone(model):
    for layer in model.layers:
        if isinstance(layer, (InferenceModeBackbone, TrainingPolicyBackbone)):
            return layer.layer
    names = {record["backbone"] for record in ARCHITECTURES.values()}
    return next(layer for layer in model.layers if isinstance(layer, keras.Model) and layer.name in names)


def recursive_layers(model):
    """Include normalization inside ConvNeXt's nested downsampling models."""
    seen = set()
    def visit(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        yield layer
        for child in getattr(layer, "layers", []):
            yield from visit(child)
    for layer in model.layers:
        yield from visit(layer)


def set_fine_tuning(model, architecture, depth="deep"):
    if depth not in ("shallow", "deep"):
        raise ValueError("Fine-tuning depth must be shallow or deep.")
    backbone = get_backbone(model)
    backbone.trainable = True
    enabled = []
    for layer in backbone.layers:
        if architecture == "convnext":
            selected = (layer.name.startswith("convnext_tiny_stage_3_") or
                        layer.name == "convnext_tiny_downsampling_block_2" or
                        layer is backbone.layers[-1])
        else:
            selected = layer.name.startswith(("block6", "block7", "top_") if depth == "deep" else ("block7", "top_"))
        layer.trainable = selected and not isinstance(layer, keras.layers.BatchNormalization)
        if layer.trainable and layer.weights:
            enabled.append(layer.name)
    assert enabled
    assert all(not layer.trainable for layer in recursive_layers(backbone)
               if isinstance(layer, keras.layers.BatchNormalization))
    return enabled


def weight_fingerprint(model):
    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(str(weight.shape).encode())
        digest.update(str(weight.dtype).encode())
        digest.update(weight.tobytes())
    return digest.hexdigest()


def parameter_counts(model):
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return {"parameters": model.count_params(), "trainable_parameters": trainable,
            "non_trainable_parameters": model.count_params() - trainable}


def compile_classifier(model, learning_rate, optimizer="Adam", weight_decay=0.):
    if optimizer == "Adam":
        if weight_decay:
            raise ValueError("The Adam control has no weight decay.")
        chosen = keras.optimizers.Adam(learning_rate)
    elif optimizer == "AdamW":
        chosen = AdamWWithExclusions(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("Only the approved Adam/AdamW comparison is supported.")
    model.compile(optimizer=chosen,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=.05),
        metrics=[keras.metrics.CategoricalAccuracy(name="top1"),
                 keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")], jit_compile=False)


def rebuild_branch(parent, architecture, resolution, seed=42, depth="deep", backbone_policy="fixed"):
    """Only the fixed input size changes; all initial weight tensors are copied."""
    # EfficientNet's ImageNet constructor includes an extra fixed Rescaling
    # operation that weights=None omits. Reuse the cached ImageNet constructor
    # to preserve that graph, then replace ALL weights with the selected head.
    # This is not a new initialization: the parent tensors are authoritative.
    previous_policy = keras.mixed_precision.global_policy()
    try:
        # Functional root policy is not serialized by Keras 3.12. After loading
        # it can inherit the caller's global policy and disagree with saved layers.
        # The saved convolution policy is authoritative for the B3 experiment.
        saved_policy = (get_backbone(parent).get_layer("stem_conv").dtype_policy
                        if architecture == "b3" else parent.dtype_policy)
        keras.mixed_precision.set_global_policy(saved_policy)
        branch = build_classifier(architecture, resolution, weights="imagenet", seed=seed,
                                  backbone_policy=backbone_policy)
    finally:
        keras.mixed_precision.set_global_policy(previous_policy)
    branch.set_weights(parent.get_weights())
    # Keras Normalization caches broadcast mean/variance tensors. set_weights
    # updates their variables, but does not refresh those tensors in Keras 3.12.
    # Refresh them so training uses the same ImageNet normalization as reload.
    for layer in recursive_layers(get_backbone(branch)):
        if isinstance(layer, keras.layers.Normalization):
            layer.finalize_state()
    assert weight_fingerprint(branch) == weight_fingerprint(parent)
    if tuple(parent.input_shape[1:3]) == (resolution, resolution):
        pixels = np.full((1, resolution, resolution, 3), 100, np.float32)
        np.testing.assert_allclose(branch(pixels, training=False), parent(pixels, training=False),
                                   atol=1e-5, rtol=2e-3)
    set_fine_tuning(branch, architecture, depth)
    return branch
