"""FORCORA Desktop GUI — Inference, Training, Evaluation, and Results Viewer."""

import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model"
EVAL_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "test_evaluation" / "test_evaluation_summary.txt"
TEST_METRICS_PATH = PROJECT_ROOT / "outputs" / "test_evaluation" / "test_metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "outputs" / "test_evaluation" / "test_confusion_matrix.png"
HISTORY_PATH = PROJECT_ROOT / "outputs" / "logs" / "training_history.json"
DATA_DIR = PROJECT_ROOT / "DATA" / "RAW" / "privacy_reviews"

CLASS_NAMES = {
    0: "Feature Request",
    1: "Bug Report",
    2: "Privacy-Irrelevant",
}


class ForcoraGUI(tk.Tk):
    """Tkinter GUI for FORCORA privacy review classification."""

    def __init__(self):
        super().__init__()
        self.title("FORCORA — Privacy Review Classifier")
        self.geometry("920x680")
        self.minsize(780, 560)

        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded_checkpoint = None
        self._active_proc = None
        self._cm_image = None

        self._build_ui()
        self._refresh_results()

    def _build_ui(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        header = ttk.Frame(self, padding=(14, 10))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="FORCORA — DistilBERT Privacy Review Classifier",
            font=("Segoe UI", 14, "bold"),
        ).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self.status_var, foreground="#555555").pack(
            side=tk.RIGHT
        )

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.tab_infer = ttk.Frame(notebook, padding=12)
        self.tab_train = ttk.Frame(notebook, padding=12)
        self.tab_results = ttk.Frame(notebook, padding=12)
        self.tab_viz = ttk.Frame(notebook, padding=12)

        notebook.add(self.tab_infer, text="Classify Review")
        notebook.add(self.tab_train, text="Train & Evaluate")
        notebook.add(self.tab_results, text="Evaluation Results")
        notebook.add(self.tab_viz, text="Visualizer")

        self._build_infer_tab()
        self._build_train_tab()
        self._build_results_tab()
        self._build_viz_tab()

    # ------------------------------------------------------------------
    # Tab 1: Inference
    # ------------------------------------------------------------------
    def _build_infer_tab(self):
        ckpt_frame = ttk.LabelFrame(self.tab_infer, text="Model Checkpoint", padding=8)
        ckpt_frame.pack(fill=tk.X, pady=(0, 10))

        self.ckpt_var = tk.StringVar(value=str(DEFAULT_CHECKPOINT))
        ttk.Entry(ckpt_frame, textvariable=self.ckpt_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        ttk.Button(ckpt_frame, text="Browse...", command=self._browse_checkpoint).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(ckpt_frame, text="Load Model", command=self._load_model_async).pack(
            side=tk.LEFT
        )

        input_frame = ttk.LabelFrame(self.tab_infer, text="Review Text", padding=8)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.review_text = scrolledtext.ScrolledText(
            input_frame, wrap=tk.WORD, height=7, font=("Segoe UI", 10)
        )
        self.review_text.pack(fill=tk.BOTH, expand=True)
        self.review_text.insert(
            "1.0",
            "Please add a toggle to opt out of third-party analytics tracking in settings.",
        )

        btn_row = ttk.Frame(self.tab_infer)
        btn_row.pack(fill=tk.X, pady=(0, 10))
        self.classify_btn = ttk.Button(
            btn_row, text="Classify Review", command=self._classify_async
        )
        self.classify_btn.pack(side=tk.LEFT)
        ttk.Button(
            btn_row,
            text="Clear",
            command=lambda: self.review_text.delete("1.0", tk.END),
        ).pack(side=tk.LEFT, padx=(8, 0))

        out_frame = ttk.LabelFrame(self.tab_infer, text="Prediction", padding=10)
        out_frame.pack(fill=tk.X)

        self.pred_label_var = tk.StringVar(value="Predicted Class: —")
        ttk.Label(
            out_frame, textvariable=self.pred_label_var, font=("Segoe UI", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 8))

        self.prob_bars = {}
        self.prob_labels = {}
        for idx, name in CLASS_NAMES.items():
            row = ttk.Frame(out_frame)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=f"{name:<20}", width=20).pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, maximum=100.0, mode="determinate")
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            pct_var = tk.StringVar(value="0.0%")
            ttk.Label(row, textvariable=pct_var, width=8, anchor=tk.E).pack(side=tk.RIGHT)
            self.prob_bars[idx] = bar
            self.prob_labels[idx] = pct_var

    def _browse_checkpoint(self):
        chosen = filedialog.askdirectory(initialdir=str(PROJECT_ROOT / "outputs"))
        if chosen:
            self.ckpt_var.set(chosen)

    def _load_model_async(self):
        ckpt_path = Path(self.ckpt_var.get().strip())
        if not ckpt_path.exists():
            messagebox.showerror(
                "Checkpoint Not Found",
                f"Checkpoint directory does not exist:\n{ckpt_path}\n\n"
                "Run training first (Train & Evaluate tab) to generate best_model.",
            )
            return

        self.status_var.set("Loading model...")
        threading.Thread(target=self._load_model_worker, args=(ckpt_path,), daemon=True).start()

    def _load_model_worker(self, ckpt_path: Path):
        try:
            import torch
            from transformers import AutoTokenizer, DistilBertForSequenceClassification

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
            model = DistilBertForSequenceClassification.from_pretrained(
                ckpt_path,
                num_labels=len(CLASS_NAMES),
                id2label=CLASS_NAMES,
                label2id={v: k for k, v in CLASS_NAMES.items()},
            )
            model.to(device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._loaded_checkpoint = ckpt_path
            self.after(
                0,
                lambda: self.status_var.set(f"Model loaded ({device})"),
            )
        except Exception as exc:
            self.after(0, lambda: self.status_var.set("Model load failed"))
            self.after(0, lambda e=exc: messagebox.showerror("Model Load Error", str(e)))

    def _classify_async(self):
        text = self.review_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty Input", "Please enter a review to classify.")
            return

        ckpt_path = Path(self.ckpt_var.get().strip())
        if self._model is None or self._loaded_checkpoint != ckpt_path:
            if not ckpt_path.exists():
                messagebox.showerror(
                    "Checkpoint Not Found",
                    f"Checkpoint not found at:\n{ckpt_path}\n\n"
                    "Train the model first in the 'Train & Evaluate' tab.",
                )
                return

        self.classify_btn.config(state=tk.DISABLED)
        self.status_var.set("Classifying...")
        threading.Thread(
            target=self._classify_worker, args=(text, ckpt_path), daemon=True
        ).start()

    def _classify_worker(self, text: str, ckpt_path: Path):
        try:
            import torch

            if self._model is None or self._loaded_checkpoint != ckpt_path:
                self._load_model_worker(ckpt_path)
                if self._model is None:
                    self.after(0, lambda: self.classify_btn.config(state=tk.NORMAL))
                    return

            inputs = self._tokenizer(
                text,
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

            pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
            pred_name = CLASS_NAMES[pred_id]

            def _update():
                self.pred_label_var.set(
                    f"Predicted Class: {pred_name} ({probs[pred_id] * 100:.1f}%)"
                )
                for idx, prob in enumerate(probs):
                    pct = prob * 100.0
                    self.prob_bars[idx]["value"] = pct
                    self.prob_labels[idx].set(f"{pct:.1f}%")
                self.status_var.set(f"Done ({self._device})")
                self.classify_btn.config(state=tk.NORMAL)

            self.after(0, _update)
        except Exception as exc:
            self.after(0, lambda: self.classify_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.status_var.set("Classification failed"))
            self.after(0, lambda e=exc: messagebox.showerror("Inference Error", str(e)))

    # ------------------------------------------------------------------
    # Tab 2: Train & Evaluate
    # ------------------------------------------------------------------
    def _build_train_tab(self):
        cfg_frame = ttk.LabelFrame(self.tab_train, text="Training Parameters", padding=8)
        cfg_frame.pack(fill=tk.X, pady=(0, 10))

        self.epochs_var = tk.StringVar(value="3")
        self.batch_var = tk.StringVar(value="8")
        self.lr_var = tk.StringVar(value="2e-5")
        self.no_fp16_var = tk.BooleanVar(value=False)

        ttk.Label(cfg_frame, text="Epochs:").grid(row=0, column=0, padx=6, pady=4, sticky=tk.W)
        ttk.Entry(cfg_frame, textvariable=self.epochs_var, width=8).grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(cfg_frame, text="Batch Size:").grid(row=0, column=2, padx=6, pady=4, sticky=tk.W)
        ttk.Entry(cfg_frame, textvariable=self.batch_var, width=8).grid(row=0, column=3, padx=6, pady=4)

        ttk.Label(cfg_frame, text="Learning Rate:").grid(row=0, column=4, padx=6, pady=4, sticky=tk.W)
        ttk.Entry(cfg_frame, textvariable=self.lr_var, width=10).grid(row=0, column=5, padx=6, pady=4)

        ttk.Checkbutton(
            cfg_frame, text="Disable FP16", variable=self.no_fp16_var
        ).grid(row=0, column=6, padx=10, pady=4)

        actions = ttk.Frame(self.tab_train)
        actions.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            actions,
            text="1. Verify Datasets",
            command=lambda: self._run_cli([sys.executable, str(PROJECT_ROOT / "test.py")]),
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            actions,
            text="2. Dry Run",
            command=lambda: self._run_cli(self._build_train_cmd(dry_run=True)),
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            actions,
            text="3. Train Model",
            command=lambda: self._run_cli(self._build_train_cmd(dry_run=False)),
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            actions,
            text="4. Evaluate Test Set",
            command=lambda: self._run_cli([sys.executable, str(PROJECT_ROOT / "eval_test.py")]),
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(actions, text="Stop Process", command=self._stop_cli).pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(self.tab_train, text="Console Output", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.console = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED
        )
        self.console.pack(fill=tk.BOTH, expand=True)

    def _build_train_cmd(self, dry_run: bool):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_train.py"),
            "--epochs",
            self.epochs_var.get().strip(),
            "--batch_size",
            self.batch_var.get().strip(),
            "--lr",
            self.lr_var.get().strip(),
        ]
        if self.no_fp16_var.get():
            cmd.append("--no_fp16")
        if dry_run:
            cmd.append("--dry_run")
        return cmd

    def _append_log(self, text: str):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def _run_cli(self, cmd):
        if self._active_proc is not None and self._active_proc.poll() is None:
            messagebox.showwarning("Process Running", "Wait for the current process to finish or stop it first.")
            return

        self.console.config(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.config(state=tk.DISABLED)
        self._append_log(f"$ {' '.join(cmd)}\n\n")
        self.status_var.set("Running command...")

        def _worker():
            try:
                self._active_proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in self._active_proc.stdout:
                    self.after(0, lambda l=line: self._append_log(l))
                rc = self._active_proc.wait()
                self.after(0, lambda: self._append_log(f"\n[Process exited with code {rc}]\n"))
                self.after(0, lambda: self.status_var.set(f"Command finished (exit {rc})"))
                self.after(0, self._refresh_results)
            except Exception as exc:
                self.after(0, lambda e=exc: self._append_log(f"\n[Error] {e}\n"))
                self.after(0, lambda: self.status_var.set("Command error"))

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_cli(self):
        if self._active_proc is not None and self._active_proc.poll() is None:
            self._active_proc.terminate()
            self._append_log("\n[Terminated by user]\n")
            self.status_var.set("Stopped")

    # ------------------------------------------------------------------
    # Tab 3: Evaluation Results
    # ------------------------------------------------------------------
    def _build_results_tab(self):
        top = ttk.Frame(self.tab_results)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(top, text="Refresh Results", command=self._refresh_results).pack(side=tk.LEFT)

        paned = ttk.PanedWindow(self.tab_results, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(paned, text="Test Summary & Training History", padding=8)
        right = ttk.LabelFrame(paned, text="Confusion Matrix", padding=8)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        self.results_text = scrolledtext.ScrolledText(
            left, wrap=tk.NONE, font=("Consolas", 9), state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self.cm_label = ttk.Label(right, text="No confusion matrix image found.", anchor=tk.CENTER)
        self.cm_label.pack(fill=tk.BOTH, expand=True)

    def _refresh_results(self):
        chunks = []
        if EVAL_SUMMARY_PATH.exists():
            chunks.append(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        else:
            chunks.append("No test evaluation summary found yet. Run 'Evaluate Test Set' first.")

        if HISTORY_PATH.exists():
            try:
                history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
                chunks.append("\n\nTRAINING HISTORY:\n" + json.dumps(history, indent=2))
            except Exception:
                pass

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", "\n".join(chunks))
        self.results_text.config(state=tk.DISABLED)

        if CONFUSION_MATRIX_PATH.exists():
            try:
                img = tk.PhotoImage(file=str(CONFUSION_MATRIX_PATH))
                # Downsample large 300-DPI PNG to fit comfortably in panel
                factor = max(1, img.width() // 360)
                if factor > 1:
                    img = img.subsample(factor, factor)
                self._cm_image = img
                self.cm_label.config(image=self._cm_image, text="")
            except Exception as exc:
                self.cm_label.config(image="", text=f"Could not render PNG: {exc}")

    # ------------------------------------------------------------------
    # Tab 4: Visualizer
    # ------------------------------------------------------------------
    def _build_viz_tab(self):
        controls = ttk.Frame(self.tab_viz)
        controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="Chart:").pack(side=tk.LEFT, padx=(0, 6))
        self.viz_choice = tk.StringVar(value="Training Curves")
        combo = ttk.Combobox(
            controls,
            textvariable=self.viz_choice,
            values=["Training Curves", "Per-Class Test Metrics", "Dataset Distribution"],
            state="readonly",
            width=26,
        )
        combo.pack(side=tk.LEFT, padx=(0, 8))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._render_viz())

        ttk.Button(controls, text="Refresh", command=self._render_viz).pack(side=tk.LEFT)

        self.viz_container = ttk.Frame(self.tab_viz)
        self.viz_container.pack(fill=tk.BOTH, expand=True)
        self._viz_canvas = None
        self._render_viz()

    def _render_viz(self):
        for child in self.viz_container.winfo_children():
            child.destroy()

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError as exc:
            ttk.Label(
                self.viz_container,
                text=f"Matplotlib is required for the Visualizer tab: {exc}",
            ).pack(expand=True)
            return

        choice = self.viz_choice.get()
        fig = Figure(figsize=(8.2, 4.6), dpi=100)

        if choice == "Training Curves":
            self._plot_training_curves(fig)
        elif choice == "Per-Class Test Metrics":
            self._plot_test_metrics(fig)
        else:
            self._plot_dataset_distribution(fig)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._viz_canvas = canvas

    def _plot_training_curves(self, fig):
        if not HISTORY_PATH.exists():
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No training_history.json found.", ha="center", va="center")
            ax.axis("off")
            return

        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        val_acc = [h["val_accuracy"] * 100 for h in history]
        val_f1 = [h["val_macro_f1"] * 100 for h in history]

        ax1 = fig.add_subplot(121)
        ax1.plot(epochs, train_loss, marker="o", label="Train Loss")
        ax1.plot(epochs, val_loss, marker="s", label="Val Loss")
        ax1.set_title("Loss per Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_xticks(epochs)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.plot(epochs, val_acc, marker="o", label="Val Accuracy (%)")
        ax2.plot(epochs, val_f1, marker="s", label="Val Macro-F1 (%)")
        ax2.set_title("Validation Metrics per Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score (%)")
        ax2.set_xticks(epochs)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()

    def _plot_test_metrics(self, fig):
        ax = fig.add_subplot(111)
        if not TEST_METRICS_PATH.exists():
            ax.text(0.5, 0.5, "No test_metrics.json found.", ha="center", va="center")
            ax.axis("off")
            return

        data = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))
        per_class = data.get("per_class", {})
        labels = [per_class[k]["name"] for k in sorted(per_class.keys(), key=int)]
        precision = [per_class[k]["precision"] * 100 for k in sorted(per_class.keys(), key=int)]
        recall = [per_class[k]["recall"] * 100 for k in sorted(per_class.keys(), key=int)]
        f1 = [per_class[k]["f1"] * 100 for k in sorted(per_class.keys(), key=int)]

        x = list(range(len(labels)))
        w = 0.25
        ax.bar([i - w for i in x], precision, width=w, label="Precision")
        ax.bar(x, recall, width=w, label="Recall")
        ax.bar([i + w for i in x], f1, width=w, label="F1-Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (%)")
        ax.set_title(
            f"Test Set Performance (Accuracy: {data.get('accuracy', 0)*100:.2f}%, "
            f"Macro-F1: {data.get('macro_f1', 0)*100:.2f}%)"
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend()

    def _plot_dataset_distribution(self, fig):
        ax = fig.add_subplot(111)
        try:
            import pandas as pd
        except ImportError:
            ax.text(0.5, 0.5, "pandas is required to plot dataset distribution.", ha="center", va="center")
            ax.axis("off")
            return

        splits = [
            ("Train", DATA_DIR / "original_12756 (1).csv"),
            ("Validation", DATA_DIR / "val_1595.csv"),
            ("Test", DATA_DIR / "test_1594.csv"),
        ]
        counts_by_split = {}
        for split_name, path in splits:
            if path.exists():
                df = pd.read_csv(path, usecols=["Label"])
                vc = df["Label"].value_counts().to_dict()
                counts_by_split[split_name] = [int(vc.get(i, 0)) for i in sorted(CLASS_NAMES)]

        if not counts_by_split:
            ax.text(0.5, 0.5, "No dataset CSVs found in DATA/RAW/privacy_reviews.", ha="center", va="center")
            ax.axis("off")
            return

        class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
        x = list(range(len(class_labels)))
        n = len(counts_by_split)
        w = 0.75 / max(1, n)
        for idx, (split_name, vals) in enumerate(counts_by_split.items()):
            offset = (idx - (n - 1) / 2) * w
            ax.bar([i + offset for i in x], vals, width=w, label=split_name)

        ax.set_xticks(x)
        ax.set_xticklabels(class_labels)
        ax.set_ylabel("Sample Count")
        ax.set_title("Privacy Review Dataset Class Distribution")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend()


def main():
    app = ForcoraGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
