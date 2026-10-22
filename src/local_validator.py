"""A thin Tkinter model inspector; inference lives in src.inference.

No training cells or notebook variables are required. Exported submission
models are preferred; native training runs are optional development sources.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import hashlib
import json
import os
import queue
import random
import zipfile

from PIL import Image, ImageTk

from .inference import (PROJECT_ROOT, ActiveModelCache, decode_rgb,
                        load_class_mapping, predict_image, development_image_path)
from .final_bundle import LOCK_PATH, MODEL_PATH, verify_bundle

STAGE_NAMES = {"scratch": "Scratch CNN", "head": "EfficientNetB0 Frozen",
               "fine": "EfficientNetB0 Fine-tuned", "e3_a": "E3-A Augmentation",
               "e3_b": "E3-B Smoothing", "e3_c": "E3-C Combined",
               "ft_shallow": "E3-FT Shallow", "ft_deep": "E3-FT Deep"}
EXPORT_NAMES = {"scratch": "scratch.keras", "head": "efficientnet_frozen.keras",
                "fine": "efficientnet_finetuned.keras"}
SPLIT_SHA256 = "65c456049b19bc87173d0e9072cc7d8f3a20366a4d3e8ac0fd0ab7c3e4f358cd"


def checkpoint_available(path):
    """Cheap structural check; weight compatibility is checked at lazy loading."""
    try:
        with zipfile.ZipFile(path) as archive:
            if not {"config.json", "model.weights.h5"}.issubset(archive.namelist()):
                return False
            config = json.loads(archive.read("config.json"))
        layers = config["config"]["layers"]
        last = layers[-1]["config"]
        first = layers[0]["config"]
        shape = first.get("batch_shape", [])[1:]
        return (last.get("units") == 200 and last.get("activation") == "softmax"
                and len(shape) == 3 and shape[-1] == 3
                and all(isinstance(n, int) and n > 0 for n in shape[:2]))
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile):
        return False


def discover_models(project_root=PROJECT_ROOT, local_root=None):
    project_root = Path(project_root).resolve()
    local_root = Path(local_root or os.environ.get("AS01_LOCAL_ROOT", Path.home() / "cos30082/as01"))
    records = []
    candidates = []
    exported = project_root / "models"
    if (project_root / LOCK_PATH).exists() or (project_root / MODEL_PATH).exists():
        candidate = verify_bundle(project_root)
        if not checkpoint_available(candidate["model_path"]):
            raise ValueError("Final-model candidate checkpoint is incompatible.")
        candidates.append({"label": "EfficientNetB3 380 | seed 42 | pending external review",
                 "stage": "final_candidate", "path": candidate["model_path"],
                 "mapping": exported / "class_mapping.json", "exported": True,
                 "protocol": "single", "project_root": project_root,
                 "local_root": local_root, "lock_status": "pending_external_review"})
    sources = [(exported, "Export", exported / "class_mapping.json", True)]
    runs = local_root / "runs"
    if runs.is_dir():
        for run in sorted(runs.iterdir(), reverse=True):
            if run.is_dir():
                sources.append((run, run.name, run / "class_names.json", False))
    for directory, source, mapping, is_export in sources:
        try:
            load_class_mapping(mapping)
        except ValueError:
            continue
        manifest_path = directory / "models_manifest.json"
        if not is_export and manifest_path.is_file():
            for item in json.loads(manifest_path.read_text()):
                path = (directory / item["checkpoint"]).resolve()
                if not path.is_relative_to(directory.resolve()):
                    # A selected historical fallback is referenced, never copied.
                    # Permit it only inside native runs and when selection agrees.
                    selection_path = directory / "selection.json"
                    if not item.get("selected") or not path.is_relative_to(runs.resolve()) or not selection_path.is_file():
                        continue
                    selection = json.loads(selection_path.read_text())
                    if selection.get("final_test") is not False or Path(selection["checkpoint"]).resolve() != path:
                        continue
                if checkpoint_available(path):
                    records.append({"label": f'{item["architecture"]} {item["resolution"]} | {item["stage"]} | {source}',
                        "stage": item["stage"], "path": path, "mapping": mapping, "exported": False,
                        "protocol": item.get("inference_protocol", "single"),
                        "project_root": project_root, "local_root": local_root})
        # Stable exports still contain E2's three names; native runs may contain
        # additional E3 checkpoints. Export precedence stays unchanged.
        for stage in (EXPORT_NAMES if is_export else STAGE_NAMES):
            path = directory / (EXPORT_NAMES[stage] if is_export else f"{stage}.keras")
            if checkpoint_available(path):
                records.append({"label": f"{STAGE_NAMES[stage]} | {source}", "stage": stage,
                                "path": path, "mapping": mapping, "exported": is_export,
                                "project_root": project_root, "local_root": local_root})
    # A submission bundle is self-contained. Do not mix its models with unrelated
    # native run folders on a lecturer's machine merely because they also exist.
    exported_records = [record for record in records if record["exported"]]
    return candidates + (exported_records or records)


def validation_samples(record):
    """Expose internal validation only. Official test paths are never discovered."""
    if record["exported"]:
        directory = record["project_root"] / "models"
        manifest = directory / "validation_samples.csv"
    else:
        directory = record["local_root"] / "data/Train"
        manifest = record["local_root"] / "splits/train_validation_seed42.csv"
        if not manifest.is_file():
            return []
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != SPLIT_SHA256:
            raise ValueError("Validation split fingerprint differs from the frozen protocol.")
    if not manifest.is_file():
        return []
    samples = []
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("partition", "validation") != "validation":
                continue
            path = (directory / row.get("path", row["filename"])).resolve()
            # Exported sample paths must remain within the extracted bundle.
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Validation sample path leaves its data directory.")
            if path.is_file():
                label = int(row["label"])
                if not 0 <= label < 200:
                    raise ValueError("Invalid validation class ID.")
                samples.append({"filename": row["filename"], "label": label, "path": path})
    return samples


class LocalValidator:
    def __init__(self, project_root=PROJECT_ROOT, local_root=None, show=True):
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog
            self.root = tk.Tk()
        except (ImportError, RuntimeError) as error:
            raise RuntimeError("Local validator requires Python with Tcl/Tk and a desktop display.") from error
        except Exception as error:
            raise RuntimeError("Cannot open the desktop display. Use a local desktop or WSLg.") from error
        self.tk, self.ttk, self.filedialog = tk, ttk, filedialog
        self.root.title("AS01 · Local Model Validator")
        self.root.geometry("860x660")
        if not show:
            self.root.withdraw()
        self.discovery_error = None
        try:
            self.models = discover_models(project_root, local_root)
        except ValueError as error:
            # A broken final candidate is visible as an error, never replaced by
            # an unrelated historical model. The inspector itself stays usable.
            self.models = []
            self.discovery_error = str(error)
        self.records = {record["label"]: record for record in self.models}
        self.cache = ActiveModelCache()  # CPU by default; never reserves GPU memory.
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="as01-inference")
        self.messages = queue.Queue()
        self.generation = 0
        self.busy = False
        self.last_result = None
        self.image_path = None
        self.true_label = None
        self.samples = []
        self.photo = None
        self.status = tk.StringVar(value="Select a saved model and an image. Training is not required.")
        self.model_name = tk.StringVar(value=self.models[0]["label"] if self.models else "")
        self.source = tk.StringVar(value="External image")
        self.sample_name = tk.StringVar()
        self.truth = tk.StringVar(value="Ground truth: unknown (external image)")
        self.prediction = tk.StringVar(value="Prediction: —")

        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Local Model Validator", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Image → RGB + padded resize → CNN → 200 probabilities → Top-5").pack(anchor="w", pady=(4, 12))
        self.model_selector = ttk.Combobox(frame, textvariable=self.model_name,
                                           values=list(self.records), state="readonly", width=100)
        self.model_selector.pack(fill="x")
        self.model_selector.bind("<<ComboboxSelected>>", self.on_model_change)
        inputs = ttk.Frame(frame)
        inputs.pack(fill="x", pady=8)
        source = ttk.Combobox(inputs, textvariable=self.source,
                              values=["External image", "Validation sample"], state="readonly", width=19)
        source.pack(side="left")
        source.bind("<<ComboboxSelected>>", self.on_source_change)
        self.choose_button = ttk.Button(inputs, text="Select image…", command=self.choose_image)
        self.choose_button.pack(side="left", padx=6)
        self.random_button = ttk.Button(inputs, text="Random validation sample", command=self.random_sample)
        self.random_button.pack(side="left")
        self.sample_selector = ttk.Combobox(frame, textvariable=self.sample_name, state="readonly")
        self.sample_selector.pack(fill="x")
        self.sample_selector.bind("<<ComboboxSelected>>", self.choose_sample)
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, pady=10)
        self.preview = ttk.Label(body, text="Select an image for a preview", anchor="center", width=38)
        self.preview.pack(side="left", fill="both", expand=True, padx=(0, 14))
        results = ttk.Frame(body)
        results.pack(side="left", fill="both", expand=True)
        ttk.Label(results, textvariable=self.truth, wraplength=390).pack(anchor="w", pady=5)
        ttk.Label(results, textvariable=self.prediction, wraplength=390,
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=8)
        self.ranking = ttk.Treeview(results, columns=("id", "species", "probability"), show="headings", height=5)
        for column, title, width in [("id", "ID", 40), ("species", "Top-5 species", 235),
                                     ("probability", "Probability", 90)]:
            self.ranking.heading(column, text=title)
            self.ranking.column(column, width=width, stretch=column == "species")
        self.ranking.pack(fill="x")
        ttk.Label(results, text="Softmax confidence is a model score; it is not a calibrated guarantee.",
                  wraplength=390).pack(anchor="w", pady=10)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        self.predict_button = ttk.Button(buttons, text="Predict", command=self.on_predict)
        self.predict_button.pack(side="left")
        self.clear_button = ttk.Button(buttons, text="Clear", command=self.clear)
        self.clear_button.pack(side="left", padx=8)
        ttk.Label(frame, textvariable=self.status, wraplength=800).pack(anchor="w", pady=(12, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.on_model_change()
        self.poll_id = self.root.after(50, self.poll_results)

    def on_model_change(self, event=None):
        self.clear()
        self.samples = []
        record = self.records.get(self.model_name.get())
        if record is None:
            self.status.set(self.discovery_error or "Model checkpoint not available. Export models or configure AS01_LOCAL_ROOT.")
            return
        try:
            self.names = load_class_mapping(record["mapping"])
            self.samples = validation_samples(record)
            self.sample_options = {f'{sample["label"]}: {self.names[sample["label"]]} | {sample["filename"]}': sample
                                   for sample in self.samples}
            self.sample_selector.configure(values=list(self.sample_options))
            status = "Model selected. Weights load on CPU when Predict is pressed."
            if record.get("lock_status"):
                status += " Candidate pending external review; final approval has not been granted."
            self.status.set(status)
        except ValueError as error:
            self.status.set(str(error))

    def on_source_change(self, event=None):
        self.clear()
        if self.source.get() == "Validation sample" and not self.samples:
            self.status.set("No internal validation samples are available. Select an external image.")

    def choose_image(self):
        path = self.filedialog.askopenfilename(title="Select bird image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*")])
        if path:
            self.source.set("External image")
            self.set_image(path)

    def choose_sample(self, event=None):
        sample = getattr(self, "sample_options", {}).get(self.sample_name.get())
        if sample:
            self.source.set("Validation sample")
            self.set_image(sample["path"], sample["label"])

    def random_sample(self):
        if not self.samples:
            self.status.set("No internal validation samples available for this model.")
            return
        sample = random.choice(self.samples)
        self.source.set("Validation sample")
        self.sample_name.set(next(name for name, value in self.sample_options.items() if value is sample))
        self.set_image(sample["path"], sample["label"])

    def set_image(self, path, true_label=None):
        self.clear(clear_sample=False)
        try:
            path = development_image_path(path)
            pixels = decode_rgb(path.read_bytes()).numpy()
            preview = Image.fromarray(pixels)
            preview.thumbnail((360, 330))
            self.photo = ImageTk.PhotoImage(preview, master=self.root)
            self.preview.configure(image=self.photo, text="")
            self.image_path = Path(path)
            self.true_label = true_label
            self.truth.set("Ground truth: unknown (external image)" if true_label is None
                           else f"Ground truth: {self.names[true_label]} (ID {true_label})")
            self.status.set(f"Ready: {self.image_path.name}. Deterministic inference; no random augmentation.")
        except Exception as error:
            self.status.set(f"Image could not be decoded: {error}")

    def on_predict(self):
        if self.busy:
            return
        record = self.records.get(self.model_name.get())
        if record is None or not record["path"].is_file():
            self.status.set("Model checkpoint not available.")
            return
        if self.image_path is None or not self.image_path.is_file():
            self.status.set("Select a readable image file first.")
            return
        self.busy = True
        self.predict_button.configure(state="disabled")
        self.status.set("Loading/predicting on CPU…")
        generation = self.generation
        path, names = self.image_path, list(self.names)
        self.worker.submit(self.predict_job, generation, record["path"], path, names, record.get("protocol", "single"))

    def predict_job(self, generation, model_path, image_path, names, protocol="single"):
        # No Tk calls here: all GUI updates remain on the desktop event thread.
        try:
            model = self.cache.get(model_path)
            result = predict_image(model, image_path, names, protocol=protocol)
            self.messages.put((generation, result, None))
        except Exception as error:
            self.messages.put((generation, None, str(error)))

    def poll_results(self):
        try:
            while True:
                generation, result, error = self.messages.get_nowait()
                if generation != self.generation:
                    continue
                self.busy = False
                self.predict_button.configure(state="normal")
                if error:
                    self.status.set(error)
                    continue
                self.last_result = result
                top1 = result["top1"]
                correctness = "" if self.true_label is None else (
                    " · Correct" if top1["class_id"] == self.true_label else " · Incorrect")
                self.prediction.set(f'{top1["species"]} (ID {top1["class_id"]})\n'
                                    f'Confidence: {top1["probability"]:.1%}{correctness}')
                self.ranking.delete(*self.ranking.get_children())
                for row in result["top5"]:
                    self.ranking.insert("", "end", values=(row["class_id"], row["species"],
                                                           f'{row["probability"]:.2%}'))
                height, width, _ = result["input_shape"]
                self.status.set(f"Prediction complete: {height} × {width} RGB → 200 probabilities → Top-5. "
                                f"Protocol: {result.get('inference_protocol', 'single')}.")
        except queue.Empty:
            pass
        self.poll_id = self.root.after(50, self.poll_results)

    def clear(self, clear_sample=True):
        self.generation += 1  # Discard a stale prediction if the input changes.
        self.busy = False
        self.last_result = None
        self.image_path = None
        self.true_label = None
        self.photo = None
        self.preview.configure(image="", text="Select an image for a preview")
        self.truth.set("Ground truth: unknown (external image)")
        self.prediction.set("Prediction: —")
        self.ranking.delete(*self.ranking.get_children())
        self.predict_button.configure(state="normal")
        if clear_sample:
            self.sample_name.set("")
        self.status.set("Select an image and press Predict.")

    def close(self):
        # Cancel the event before destroying its Tcl command; otherwise a later
        # Tk event loop (e.g. another inspector) can fire a stale timer callback.
        self.root.after_cancel(self.poll_id)
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch_validator(project_root=PROJECT_ROOT, local_root=None, show=True):
    """Launch the local inspector without training; return its UI for testing."""
    ui = LocalValidator(project_root, local_root, show=show)
    if show:
        try:
            from IPython import get_ipython
            shell = get_ipython()
        except ImportError:
            shell = None
        if shell is not None:
            shell.enable_gui("tk")
        else:
            ui.root.mainloop()
    return ui
