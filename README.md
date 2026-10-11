# COS30082 AS01 Multi-Class Classification

An educational Streamlit demonstration for classifying 200 bird species. It uses
one frozen EfficientNetB3 model with 380 x 380 RGB input and CPU float32 inference.

Upload a JPEG or PNG (up to 10 MiB and 20 megapixels), then select **Predict**.
The application previews the image, shows the Top-1 species and zero-based class
ID, and ranks the Top-5 species with softmax confidence bars.

## Local use

Use Python 3.12 in an isolated environment:

```bash
python -m venv .venv
# Activate the virtual environment using your operating system's normal command.
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The model and mapping are included. Training, WSL, CUDA and a GPU are unnecessary.
All runtime bundle paths are resolved relative to this repository. On Streamlit
Community Cloud, the entry point is `app.py` and Python 3.12 should be selected.
This repository does not itself deploy a cloud application.

## Verified results

These are saved aggregate results from the completed coursework evaluation;
running the app does not recompute accuracy or access the official dataset.

| Evaluation partition | Images | Top-1 | Average class accuracy | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Internal validation | 959 | 80.29% | 79.65% | 93.43% |
| Official assignment test | 1,204 | 76.41% | 75.85% | 93.69% |

The model is restricted to the 200 supplied species. It will still assign a class
to other birds or non-bird images. Softmax confidence is not a calibrated
probability or a guarantee of correctness; poor visibility, backgrounds and
unfamiliar image conditions can affect predictions. Upload only images you may
use. Hosted providers may handle network data under their own policies.

## Provenance and usage

This is the seed-42 B3-380 coursework model, fine-tuned from Keras ImageNet weights,
with single-view deterministic inference. Its ordered mapping and preprocessing
are verified before loading. No training/test images, per-image official results,
coursework notebooks or experiment checkpoints are included.

Read [ATTRIBUTION.md](ATTRIBUTION.md) for dataset and pretrained-model provenance,
upstream notices and the licensing evidence. This is an educational
demonstration. Upstream EfficientNet components retain their Apache-2.0
conditions; no blanket third-party image rights are claimed.

The provenance lock includes an historical local checkpoint path, published
with the author's explicit consent. It is provenance only; model loading uses
project-relative paths. Streamlit hosting and a Hugging Face landing page are
separate deployment steps.
