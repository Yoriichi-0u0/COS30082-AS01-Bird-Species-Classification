# Attribution and redistribution notices

ConvNeXt: Liu et al. (2022), A ConvNet for the 2020s, CVPR.
Official project: https://github.com/facebookresearch/ConvNeXt
The official repository releases the project and links its pretrained checkpoints
under the MIT project license. The exact Meta copyright and MIT notice are retained
in third_party/CONVNEXT-MIT.txt. This is the supported upstream redistribution
basis; it is not a claim that a separate license was issued for this student's
exact fine-tuned binary. No endorsement by Meta or the dataset creators is implied.

Keras 3.12.0 ConvNeXt source identifies the converted pretrained weights and their
ImageNet-21k then ImageNet-1k provenance:
https://github.com/keras-team/keras/blob/v3.12.0/keras/src/applications/convnext.py
Keras and TensorFlow software are Apache-2.0; retain third_party/APACHE-2.0.txt.
These software licenses do not automatically assign a blanket license to original
student contributions or photographs.

The included export is a trained derivative: the original classifier was replaced
by a 200-species head, full fine-tuning was performed on the supplied training
partition, and the selected input resolution is 320. The immutable export excludes
optimizer state. Publication is authorized by the model owner for this educational
bird-classification demonstration. There is no new license grant for all coursework.

CUB-200-2011: Wah et al. (2011), California Institute of Technology.
https://www.vision.caltech.edu/datasets/cub_200_2011/
The dataset photographs have source-specific rights; this deployment includes no
dataset images, archives, raw annotations or per-image official test output.
Image use restrictions are not represented as a license to model weights.

The saved candidate provenance is retained unchanged for exact loading/source
checks. It records a generic historical Linux checkpoint path, not credentials.
The pending status in that historical inspection artifact is preserved; public
inference is authorized separately and grants no ability to evaluate a dataset.
