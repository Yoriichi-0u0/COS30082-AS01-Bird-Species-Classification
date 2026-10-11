# Attribution and licensing notices

## EfficientNet pretrained components

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

EfficientNet source and upstream pretrained-model distribution are associated
with Apache License 2.0. The license text is included in
[third_party/APACHE-2.0.txt](third_party/APACHE-2.0.txt). Preserve these notices
when redistributing the upstream components or their derivatives; retain required
upstream notices, provide the license, and document modifications. No endorsement
by Google, TensorFlow, Keras or Hugging Face is implied.

The redistribution basis includes weight-specific evidence, not merely the
Keras source license:

- The [Google-published B3 model](https://huggingface.co/google/efficientnet-b3)
  declares Apache-2.0. Its model card was prepared by the Hugging Face team.
- The [official Transformers conversion source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/convert_efficientnet_to_pytorch.py)
  loads Keras `EfficientNetB3(weights="imagenet")` before converting weights.
- [Keras 3.12.0](https://github.com/keras-team/keras/blob/v3.12.0/keras/src/applications/efficientnet.py)
  and [Keras 2.11.0](https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/efficientnet.py)
  identify the same Google-hosted B3 top/no-top weight family. The no-top file
  used for this project matches MD5 `af6d107764bb5b1abb91932881670226`.
- Google's [original EfficientNet release](https://research.google/blog/efficientnet-improving-accuracy-and-efficiency-through-automl-and-model-scaling/)
  and [upstream repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
  publish the model family and pretrained checkpoints. Keras supplies an
  [upstream checkpoint conversion utility](https://github.com/keras-team/keras/blob/v2.12.0/keras/applications/efficientnet_weight_update_util.py).

This evidence supports treating the incorporated pretrained EfficientNet
components as Apache-2.0 redistribution with retained conditions. It is a
provenance-based interpretation, not a separate written license for the exact
HDF5 object or a legal guarantee covering every dataset-image right.

## Modifications in the included bird model

`models/final_b3_380.keras` is a fine-tuned derivative, not the original ImageNet
classifier: the 1,000-class head was replaced by a 200-species classifier,
selected block7/top weights were fine-tuned, and input size is 380 x 380. The
project used seed 42, a fixed development split, label smoothing, AdamW and
training-time stochastic regularization. The exported model excludes optimizer
state and uses deterministic single-view CPU float32 inference. These changes
were completed before publication; the frozen export has not been modified.

Original coursework code and new classifier contributions remain attributable
to the coursework author. No new blanket MIT or Apache license is assigned to
all original project material solely because the runtime uses Keras. Upstream
licensed components retain their existing rights and conditions.

Tan, M. and Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks*. ICML. [Paper](https://arxiv.org/abs/1905.11946).

## Dataset

Wah, C., Branson, S., Welinder, P., Perona, P. and Belongie, S. (2011).
*The Caltech-UCSD Birds-200-2011 Dataset*. CNS-TR-2011-001,
California Institute of Technology.

The [official Caltech page](https://www.vision.caltech.edu/datasets/cub_200_2011/)
restricts image use to non-commercial research and education and states that
Caltech does not own the image copyrights. This package contains no dataset
images or raw annotations. It does not grant rights to redistribute the training
photographs, and the image statement is not used as a license for model weights.
The intended application is a non-commercial university demonstration.

## Libraries

TensorFlow/Keras, Streamlit, NumPy, Pillow and other pinned runtime dependencies
are installed separately under their respective licenses. Their source code is
not vendored here. [Keras 3.12.0 license](https://github.com/keras-team/keras/blob/v3.12.0/LICENSE).
