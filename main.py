from pathlib import Path
import traceback

import customtkinter as ctk
from tkinter import filedialog

from components.clean import build_corpus, OUT_PATH as CORPUS_PATH
from components.ngram_model import SmartNGramModel, load_model
from components.train_ngrams import main as train_all_ngrams
from components.story_ollama import call_ollama, build_prompt

MODEL_FILES = {
    "2": Path("models/ngram_2.pkl"),
    "3": Path("models/ngram_3.pkl"),
    "4": Path("models/ngram_4.pkl"),
    "5": Path("models/ngram_5.pkl"),
}


class StoryApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Story Generator – n-gram + LLM")
        self.geometry("1000x700")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.ngram_models: dict[str, SmartNGramModel] = {}

        self._build_ui()

    def log(self, widget: ctk.CTkTextbox, text: str):
        widget.configure(state="normal")
        widget.insert("end", text + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _build_ui(self):
        tabview = ctk.CTkTabview(self)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)

        tab_corpus = tabview.add("1. Corpus")
        tab_ngrams = tabview.add("2. N-gram models")
        tab_llm = tabview.add("3. N-gram + LLM")

        self._build_tab_corpus(tab_corpus)
        self._build_tab_ngrams(tab_ngrams)
        self._build_tab_llm(tab_llm)

    def _build_tab_corpus(self, parent: ctk.CTkFrame):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1)

        label = ctk.CTkLabel(
            parent,
            text="Step 1: load raw text and clean it into stories.txt",
            anchor="w",
        )
        label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        btn_load = ctk.CTkButton(
            parent,
            text="Select raw .txt and clean",
            command=self.on_select_raw_corpus,
        )
        btn_load.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.corpus_log = ctk.CTkTextbox(parent, wrap="word")
        self.corpus_log.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.corpus_log.configure(state="disabled")

    def on_select_raw_corpus(self):
        path = filedialog.askopenfilename(
            title="Select raw text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            raw_path = Path(path)
            self.log(self.corpus_log, f"[INFO] Selected file: {raw_path}")

            build_corpus(raw_path=raw_path, out_path=CORPUS_PATH)
            self.log(
                self.corpus_log,
                f"[OK] Corpus cleaned and saved to: {CORPUS_PATH.resolve()}",
            )
        except Exception as e:
            self.log(self.corpus_log, "[ERROR] Error while cleaning corpus:")
            self.log(self.corpus_log, str(e))
            self.log(self.corpus_log, traceback.format_exc())

    def _build_tab_ngrams(self, parent: ctk.CTkFrame):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(3, weight=1)

        frame_top = ctk.CTkFrame(parent)
        frame_top.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        frame_top.grid_columnconfigure((0, 1), weight=0)
        frame_top.grid_columnconfigure(2, weight=1)

        btn_train = ctk.CTkButton(
            frame_top,
            text="Train n-gram models (2,3,4,5)",
            command=self.on_train_ngrams,
        )
        btn_train.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        lbl_input = ctk.CTkLabel(frame_top, text="Input text for n-gram comparison:")
        lbl_input.grid(row=1, column=0, padx=5, pady=(10, 5), sticky="w")

        self.entry_ngram_input = ctk.CTkEntry(
            frame_top,
            placeholder_text="Once upon a time in a small town...",
            width=600,
        )
        self.entry_ngram_input.grid(
            row=1, column=1, columnspan=2, padx=5, pady=(10, 5), sticky="ew"
        )

        btn_compare = ctk.CTkButton(
            frame_top,
            text="Compare n-grams (2 / 3 / 4 / 5)",
            command=self.on_compare_ngrams,
        )
        btn_compare.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.ngram_log = ctk.CTkTextbox(parent, wrap="word")
        self.ngram_log.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.ngram_log.configure(state="disabled")

    def on_train_ngrams(self):
        try:
            self.log(self.ngram_log, "[INFO] Training n-gram models (2,3,4,5)...")
            train_all_ngrams()
            self.log(self.ngram_log, "[OK] Training finished. Models saved to /models.")
            self.ngram_models.clear()
        except Exception as e:
            self.log(self.ngram_log, "[ERROR] Error while training n-grams:")
            self.log(self.ngram_log, str(e))
            self.log(self.ngram_log, traceback.format_exc())

    def _load_ngram_model_cached(self, n_str: str) -> SmartNGramModel | None:
        if n_str in self.ngram_models:
            return self.ngram_models[n_str]

        path = MODEL_FILES.get(n_str)
        if path is None or not path.exists():
            self.log(
                self.ngram_log,
                f"[WARN] Model for n={n_str} not found ({path}). Train n-grams first.",
            )
            return None

        try:
            model = load_model(path)
            self.ngram_models[n_str] = model
            self.log(self.ngram_log, f"[INFO] Loaded model n={n_str} from {path}")
            return model
        except Exception as e:
            self.log(self.ngram_log, f"[ERROR] Failed to load model n={n_str}: {e}")
            return None

    def on_compare_ngrams(self):
        text = self.entry_ngram_input.get().strip()
        if not text:
            self.log(self.ngram_log, "[WARN] Enter text to compare n-grams.")
            return

        self.log(self.ngram_log, "\n=== N-gram comparison ===")
        self.log(self.ngram_log, f"[INPUT] {text}")

        for n_str in ["2", "3", "4", "5"]:
            model = self._load_ngram_model_cached(n_str)
            if model is None:
                continue
            try:
                draft = model.generate_multi(text, num_sentences=3, max_tokens=80)
                self.log(self.ngram_log, f"\n[n={n_str} draft]:")
                self.log(self.ngram_log, draft)
            except Exception as e:
                self.log(self.ngram_log, f"[ERROR] Generation error for n={n_str}: {e}")

    def _build_tab_llm(self, parent: ctk.CTkFrame):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(4, weight=1)

        frame_top = ctk.CTkFrame(parent)
        frame_top.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        frame_top.grid_columnconfigure(1, weight=1)

        lbl_ng = ctk.CTkLabel(frame_top, text="Select n-gram model for draft:")
        lbl_ng.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.option_ng = ctk.CTkOptionMenu(
            frame_top,
            values=["2", "3", "4", "5"],
        )
        self.option_ng.set("4")
        self.option_ng.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        lbl_input = ctk.CTkLabel(frame_top, text="Story beginning:")
        lbl_input.grid(row=1, column=0, padx=5, pady=(10, 5), sticky="w")

        self.entry_llm_input = ctk.CTkEntry(
            frame_top,
            placeholder_text="Once upon a time in a small town...",
            width=600,
        )
        self.entry_llm_input.grid(
            row=1, column=1, padx=5, pady=(10, 5), sticky="ew"
        )

        lbl_genre = ctk.CTkLabel(frame_top, text="Genre (optional):")
        lbl_genre.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.entry_genre = ctk.CTkEntry(
            frame_top,
            placeholder_text="horror / fantasy / comedy ...",
            width=200,
        )
        self.entry_genre.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        btn_generate = ctk.CTkButton(
            frame_top,
            text="Generate (n-gram + LLM)",
            command=self.on_generate_llm_pipeline,
        )
        btn_generate.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        self.llm_output = ctk.CTkTextbox(parent, wrap="word")
        self.llm_output.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
        self.llm_output.configure(state="disabled")

    def on_generate_llm_pipeline(self):
        text = self.entry_llm_input.get().strip()
        if not text:
            self.log(self.llm_output, "[WARN] Enter a story beginning.")
            return

        genre = self.entry_genre.get().strip() or None
        n_str = self.option_ng.get()

        model = self._load_ngram_model_cached(n_str)
        if model is None:
            self.log(self.llm_output, f"[WARN] No model for n={n_str}. Train n-grams first.")
            return

        try:
            draft = model.generate_multi(text, num_sentences=3, max_tokens=80)
        except Exception as e:
            self.log(self.llm_output, f"[ERROR] Draft generation error: {e}")
            return

        prompt = build_prompt(user_input=text, draft=draft, genre=genre)

        self.log(self.llm_output, "\n=== N-gram + LLM ===")
        self.log(self.llm_output, f"[n={n_str} draft]:")
        self.log(self.llm_output, draft)
        self.log(self.llm_output, "\n[LLM continuation]:")

        try:
            llm_output = call_ollama(prompt)
        except Exception as e:
            self.log(self.llm_output, f"[ERROR] Ollama call error: {e}")
            self.log(self.llm_output, traceback.format_exc())
            return

        self.log(self.llm_output, llm_output)


if __name__ == "__main__":
    app = StoryApp()
    app.mainloop()
