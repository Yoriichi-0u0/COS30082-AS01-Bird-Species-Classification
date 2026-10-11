"""Stage 4A local Streamlit demonstration of the approved inference export."""
import hashlib
import json
import os
from pathlib import Path
import sys

# Set CPU visibility before the lazy TensorFlow import; no CUDA is required.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.dont_write_bytecode = True

import streamlit as st
from src.web_demo import (DemoPredictor, validate_upload, species_display_name,
                          deny_official_dataset_access)

if not getattr(sys, "as01_demo_guard_installed", False):
    sys.addaudithook(deny_official_dataset_access)
    sys.as01_demo_guard_installed = True

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource(max_entries=1, show_spinner=False)
def cached_predictor():
    """The single model survives reruns and is shared across browser sessions."""
    return DemoPredictor(PROJECT_ROOT)


def clear_upload():
    # A fresh widget key also clears the uploader's native selection.
    st.session_state["upload_version"] += 1
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_digest", None)


def show_saved_results():
    """Read a small aggregate snapshot; never open datasets or evaluator modules."""
    path = PROJECT_ROOT / "demo/verified_results.json"
    if not path.is_file():
        return
    with st.expander("Verified model results"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            for record in summary["results"]:
                rows.append({"Partition": record["partition"], "Images": record["images"],
                    "Top-1": f'{record["top1"]:.2%}',
                    "Average per class": f'{record["average_class_accuracy"]:.2%}',
                    "Top-5": f'{record["top5"]:.2%}',
                    "Common CE": f'{record["common_unsmoothed_crossentropy"]:.3f}'})
            st.table(rows)
            st.caption("Saved aggregate results from completed evaluations. Uploads do not rerun evaluation.")
        except (OSError, ValueError, KeyError, TypeError):
            st.info("Saved aggregate results are unavailable; image prediction is still available.")


def main():
    st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦", layout="wide")
    st.title("Bird species classifier")
    st.write("Upload a bird photograph to explore predictions across 200 species.")
    with st.sidebar:
        st.header("About this model")
        st.write("**EfficientNetB3 · seed 42**")
        st.write("380 × 380 RGB input · CPU inference")
        st.caption("RGB decoding → padded bilinear resize → CNN → 200 probabilities → Top-5")
        st.caption("Single-view inference. No training augmentation or test-time augmentation.")
        st.divider()
        st.caption("Softmax confidence is a model score, not a calibrated guarantee. The model chooses among its 200 bird species, even for unfamiliar or non-bird images.")

    st.session_state.setdefault("upload_version", 0)
    image_column, prediction_column = st.columns([1.15, 1], gap="large")
    upload = None
    contents = None
    with image_column:
        st.subheader("Your bird photograph")
        file = st.file_uploader("Choose a JPEG or PNG", type=["jpg", "jpeg", "png"],
                                key=f'upload_{st.session_state["upload_version"]}',
                                help="Maximum 10 MiB and 20 megapixels. Images are processed in memory.")
        if file is not None:
            try:
                contents = file.getvalue()
                upload = validate_upload(contents)
                st.image(upload["preview"], caption=f'{file.name} · {upload["width"]} × {upload["height"]}',
                         width="stretch")
            except ValueError as error:
                st.error(str(error))
        else:
            st.info("Choose an image to see a preview and enable prediction.")
        predict_pressed = st.button("Predict species", type="primary", disabled=upload is None,
                                    width="stretch")
        st.button("Clear image", on_click=clear_upload, width="stretch")

    digest = hashlib.sha256(contents).hexdigest() if upload is not None else None
    if digest != st.session_state.get("prediction_digest"):
        st.session_state.pop("prediction", None)
    if predict_pressed and upload is not None:
        try:
            with st.spinner("Loading the CPU model and predicting… The first prediction takes longer."):
                result = cached_predictor().predict(contents)
            st.session_state["prediction"] = result
            st.session_state["prediction_digest"] = digest
        except Exception as error:
            # Present a clear failure without leaving an old image's prediction.
            st.session_state.pop("prediction", None)
            st.error(f"Prediction could not complete: {error}")

    with prediction_column:
        st.subheader("Prediction")
        result = st.session_state.get("prediction")
        if result is None:
            st.write("Your prediction will appear here.")
        else:
            best = result["top1"]
            st.markdown(f'### {species_display_name(best["species"])}')
            st.write(f'Class ID: **{best["class_id"]}**')
            st.metric("Softmax confidence", f'{best["probability"]:.1%}')
            st.subheader("Top-5 species")
            for item in result["top5"]:
                label = (f'{item["rank"]}. {species_display_name(item["species"])}'
                         f' · ID {item["class_id"]} · {item["probability"]:.1%}')
                st.progress(item["probability"], text=label)
            st.caption("Class IDs use the original ordering, from 0 to 199.")
    st.divider()
    show_saved_results()
    st.caption("COS30082 Applied Machine Learning · local model demonstration")


if __name__ == "__main__":
    main()
