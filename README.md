# Bird species classifier — ConvNeXt-Base 320

A public educational Streamlit demonstration for 200 bird species. Upload a JPEG
or PNG (up to 10 MiB / 20 MP), preview it and obtain Top-1 plus Top-5 species,
class IDs and softmax confidence. Confidence is not a calibrated guarantee; the
closed-set classifier also assigns a bird label to unfamiliar or non-bird inputs.

## Exact frozen model

ConvNeXt-Base, seed 42, selected epoch 34, ImageNet-21k then ImageNet-1k pretraining.
RGB float32 [0,255] is bilinearly resized with aspect-preserving zero padding to
320 square. Normalization is internal; computation stays mixed_float16, outputs
float32, inference training=False, single view, no TTA. One CPU model is cached
per process and predictions are serialized. Uploaded bytes are passed unchanged
to the shared decoder. No dataset or evaluator is included.

Model SHA-256: `3c8d85465b3f0ae0732b4659e08f86caa7bae80dd1f39e1edfe3f8a0ef3ba0d5`.
Mapping SHA-256: `ab490ae3ad95166b904c24b4173c2e018a5c17f9a1f8da87a1c86eb110ffdfa4`.
The 353,755,501-byte optimizer-free export is tracked with Git LFS, not ordinary
Git. Clone with Git LFS and run `git lfs pull`; a pointer file cannot be loaded.
The loader verifies the actual file, mapping and bound source before inference.
Historical candidate metadata retains its original pending-review wording and
provenance bytes; that old status is not used to grant evaluation access.
Public inference is explicitly authorized by the model owner. There is no test
authorization or official dataset in this deployment.

## Run / host

Use Linux Python 3.12 with `pip install -r requirements.txt`, then
`streamlit run app.py`. TensorFlow 2.19.1 and Keras 3.12.0 are pinned. No GPU is
required. Community Cloud: repository branch `convnext-streamlit`, entrypoint
`app.py`, Python 3.12, free public hosting. An inactive app can hibernate and take
time to wake; the first prediction loads the model. No paid service is required.

## Existing results and limitations

Internal validation: 887/959 = 92.49218% Top-1, 92.34167% mean class accuracy.
Official test: 1,116/1,204 = 92.69103% Top-1, 92.56250% mean class accuracy,
99.08638% Top-5. These saved aggregates are not recomputed by the application.
The prior B3 official score was known before this replacement; repeated validation
development and one ConvNeXt seed limit the interpretation. Four identical
cross-split photographs have conflicting supplied labels and were all errors.
Comparisons with B3 change several factors, not just architecture.

See `ATTRIBUTION.md` and `third_party/` for source and upstream notices. No blanket
license is invented for third-party images or the student-derived model. Dataset
images, annotations, training records, evaluation claims and private credentials
are excluded. Earlier B3 deployment remains on branch `main`.
