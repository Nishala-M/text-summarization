# fix_tokenizer.py
# Re-downloads and saves correct tokenizers to replace corrupted/missing ones.
# Run this once from the project root:  python fix_tokenizer.py

import os
from transformers import BartTokenizer, T5Tokenizer, AutoModelForSeq2SeqLM

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
BART_MODEL_PATH = os.path.join(BASE_DIR, "my_bart_model")
T5_MODEL_PATH   = os.path.join(BASE_DIR, "my_t5_model")

print(f"Project dir : {BASE_DIR}")
print()

# ── BART ───────────────────────────────────────────────────────────────────────
print("Downloading BART tokenizer + model (facebook/bart-large-cnn) …")
bart_tok   = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
bart_tok.save_pretrained(BART_MODEL_PATH)
bart_model.save_pretrained(BART_MODEL_PATH)
print(f"✅  BART saved → {BART_MODEL_PATH}\n")

# ── T5 ────────────────────────────────────────────────────────────────────────
print("Downloading T5 tokenizer + model (t5-small) …")
t5_tok   = T5Tokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
t5_tok.save_pretrained(T5_MODEL_PATH)
t5_model.save_pretrained(T5_MODEL_PATH)
print(f"✅  T5 saved → {T5_MODEL_PATH}\n")

print("All done!  Now launch the app:")
print("    streamlit run app.py")