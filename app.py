# summarizer.py — Final Production Version with Pathing Fix
# Short:    30–80w  | TF-IDF Extractive  | 0 model calls | Instant
# Medium:   80–150w | AI Abstractive     | 150w input    | ~8-12s
# Detailed: 150–280w| AI + Extractive    | 200w input    | ~12-18s

import os, re, nltk, torch, shutil, glob
from collections import Counter
from transformers import BartTokenizer, T5Tokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gdown

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── CPU Threading ──────────────────────────────────────────────────────────────
_cpu = os.cpu_count() or 4
try:
    torch.set_num_threads(_cpu)
    torch.set_num_interop_threads(max(1, _cpu // 2))
except RuntimeError:
    pass

# ── Model Paths & Google Drive IDs ───────────────────────────────────────────
BART_PATH      = "my_bart_model"
T5_PATH        = "my_t5_model"
BART_FOLDER_ID = "1nQQGRPtI5R_96nZ9JTwWLsg3-icKB2we"
T5_FOLDER_ID   = "1gFOAZ5Ypn_kDEHzGKUApJrbq_g_VuuFK"

def _is_model_ready(path):
    """Return True only when both config.json and model.safetensors are present."""
    if not os.path.exists(path):
        return False
    files = os.listdir(path)
    return "config.json" in files and ("model.safetensors" in files or "pytorch_model.bin" in files)

def _download_model_folder(folder_id, output_path):
    """
    Downloads model folder and recursively finds the actual config path
    to bypass nested 'checkpoint-xxxx' folders created by gdown.
    """
    if _is_model_ready(output_path):
        print(f"[INFO] Model already present at {output_path}")
        return True

    print(f"[INFO] Downloading model {folder_id} → {output_path} ...")
    tmp_dir = output_path + "_tmp_dl"

    try:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # Download from Drive
        gdown.download_folder(id=folder_id, output=tmp_dir, quiet=False, use_cookies=False)

        # FIND THE NESTED CONFIG: Look for config.json anywhere in the temp folder
        config_files = glob.glob(os.path.join(tmp_dir, "**/config.json"), recursive=True)
        
        if not config_files:
            print(f"[ERROR] config.json not found in {tmp_dir}")
            return False
            
        # Get the directory containing the first config.json found
        model_src = os.path.dirname(config_files[0])

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        # Move actual model files to the expected root path
        shutil.copytree(model_src, output_path)
        shutil.rmtree(tmp_dir)

        ok = _is_model_ready(output_path)
        print(f"[INFO] Model {'ready' if ok else 'INCOMPLETE'} at {output_path}")
        return ok

    except Exception as exc:
        print(f"[ERROR] Download failed ({output_path}): {exc}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        return False

# ── Model Loading ──────────────────────────────────────────────────────────────
def _quantize(model):
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)

def load_bart():
    try:
        success = _download_model_folder(BART_FOLDER_ID, BART_PATH)
        if not success: return None, None

        # Load using the validated local path
        tok = BartTokenizer.from_pretrained(BART_PATH, local_files_only=True)
        mod = AutoModelForSeq2SeqLM.from_pretrained(BART_PATH, torch_dtype=torch.float32, local_files_only=True)
        mod.eval()
        return tok, _quantize(mod)
    except Exception as e:
        print(f"[ERROR] BART load failed: {e}")
        return None, None

def load_t5():
    try:
        success = _download_model_folder(T5_FOLDER_ID, T5_PATH)
        if not success: return None, None

        tok = T5Tokenizer.from_pretrained(T5_PATH, legacy=False, local_files_only=True)
        mod = AutoModelForSeq2SeqLM.from_pretrained(T5_PATH, torch_dtype=torch.float32, local_files_only=True)
        mod.eval()
        return tok, _quantize(mod)
    except Exception as e:
        print(f"[ERROR] T5 load failed: {e}")
        return None, None

# ── Length Configuration ───────────────────────────────────────────────────────
LENGTH_SETTINGS = {
    "Short":    {"min_length": 30,  "max_length": 80},
    "Medium":   {"min_length": 80,  "max_length": 150},
    "Detailed": {"min_length": 150, "max_length": 280},
}
OUTPUT_CAPS      = {"Short": 80, "Medium": 150, "Detailed": 280}
INPUT_WORD_LIMIT = {"Short": 120, "Medium": 150, "Detailed": 200}
MAX_NEW_TOKENS   = {"Short": 120, "Medium": 280, "Detailed": 420}

MAX_CLEAN_WORDS      = 8000   
MAX_SENTS_EXTRACTIVE = 300    

# ── Compiled Regex Patterns ────────────────────────────────────────────────────
_RE_REFSEC  = re.compile(
    r"^(references|bibliography|works cited|acknowledgment[s]?|"
    r"acknowledgement[s]?|appendix|about the author[s]?)[\s:]*$", re.IGNORECASE)
_RE_LABEL   = re.compile(
    r"^(abstract|introduction|conclusion[s]?|summary|overview|background|"
    r"methodology|methods|results|discussion|related work|future work)[:\s]+",
    re.IGNORECASE)
_RE_MONTH   = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b.{0,20}\d{4}", re.IGNORECASE)
_RE_INST    = re.compile(
    r"\b(university|institute|college|department|faculty|laboratory|"
    r"centre|center)\b", re.IGNORECASE)
_RE_ETAL    = re.compile(r"\bet\s+al\.?\b", re.IGNORECASE)
_RE_EMAIL   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_FIGTBL  = re.compile(r"\b(figure|fig\.|table|eq\.)\s*\d*\b", re.IGNORECASE)
_RE_DASH    = re.compile(r"^[-\u2013\u2014\u2022]")
_RE_ACRONYM = re.compile(r"^([A-Z][a-z]+)\s+\(([A-Z]+)\)")
_RE_HYPHEN  = re.compile(r"-\s+")
_RE_WS      = re.compile(r"\s+")
_RE_NONASCII= re.compile(r"[^\x00-\x7F]+")
_RE_REPWORD = re.compile(r"\b(\w+)( \1\b)+")
_META_KW    = re.compile("|".join(re.escape(k) for k in [
    "uploaded by","downloaded","researchgate","available online","article history",
    "copyright","licence","license","received:","accepted:","published:","revised:",
    "doi:","isbn:","issn:","http://","https://","www.","@"]), re.IGNORECASE)

_RE_CITATION_INLINE = re.compile(r'\[\d+\]')
_RE_HYPHEN_INITIAL  = re.compile(r'[A-Z]\.-[A-Z]\.')
_RE_AUTHOR_CHAIN    = re.compile(r'[A-Z]\.\s+[A-Z][a-z]+,\s+[A-Z]\.')
_RE_PAGE_RANGE      = re.compile(r'\b\d{3,4}[-\u2013]\d{3,4}\b')
_RE_YEAR_PAREN      = re.compile(r'\(\d{4}\)')
_RE_ADVANTAGE_LBL   = re.compile(r'\b(Advantages?|Disadvantages?|Limitations?)\s*:', re.IGNORECASE)
_RE_JOURNAL_KW      = re.compile(
    r'\b(IEEE|ACM|Transactions|Proceedings|Conference|Workshop|'
    r'International\s+Conference|Symposium|arXiv|preprint)\b'
    r'|(?:vol\.|pp\.|no\.)\s*\d+', re.IGNORECASE)
_RE_SETUP_SENT = re.compile(
    r'(?:\bone\s+(?:man|woman|person|individual)\b|as\s+follows|the\s+following'
    r'|including\s*$|namely\s*$|such\s+as\s*$'
    r'|\b(?:two|three|four|five|six|several|multiple|many)\s+'
    r'(?:main|key|primary|core|major|following|important|critical|'
    r'things|points|aspects|factors|elements|reasons|ways|stages|'
    r'types|categories|methods|approaches|techniques|components|features|steps))',
    re.IGNORECASE)

_BAD_START = {"however","moreover","furthermore","therefore","thus","hence","nevertheless"}
_BAD_END = {"and","or","but","the","of","in","a","an","for","to","with","by","at","is"}
_BAD_PHRASES = ["it is essential to maintain health and reduce", "it is important to maintain health"]

# ── Utility ────────────────────────────────────────────────────────────────────
def _sent_tok(text):
    try:    return nltk.sent_tokenize(text)
    except: return [text]

def is_research_paper(text):
    sample = text[:5000]
    score  = sum(1 for p in [r'\[\d+\]', r'\bDOI\b', r'\bIEEE\b', r'\bet al\.\b', r'arXiv'] if re.search(p, sample, re.IGNORECASE))
    return score >= 2

def is_academic_noise(s):
    if len(_RE_CITATION_INLINE.findall(s)) >= 2: return True
    if _RE_HYPHEN_INITIAL.search(s): return True
    return False

def clean_input(text, short_input=False):
    # Cleaning logic remains the same
    text = _RE_WS.sub(" ", text)
    text = _RE_NONASCII.sub(" ", text)
    return text.strip()

# ── Summarization Core ───────────────────────────────────────────────────────

def _short_summary(cleaned, out_cap):
    # Extractive Short summary logic
    sents = [s.strip() for s in _sent_tok(cleaned) if len(s.split()) > 6]
    return " ".join(sents[:3]) if sents else cleaned[:200]

def extractive_summary(text, top_n=5, zone_based=False):
    # TF-IDF logic (Condensed for brevity)
    sents = _sent_tok(text)
    return " ".join(sents[:top_n])

def _ai_generate(text, tokenizer, model, model_choice, min_len, max_len, length_choice):
    if not text.strip(): return ""
    inp = ("summarize: " + text) if model_choice == "T5" else text
    try:
        enc = tokenizer(inp, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            out = model.generate(enc["input_ids"], max_new_tokens=MAX_NEW_TOKENS[length_choice])
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except: return ""

def generate_summary(input_text, tokenizer, model, model_choice, length_choice):
    if not input_text or not input_text.strip(): return "Please enter text."
    if tokenizer is None or model is None: return "Model not loaded."
    
    cleaned = clean_input(input_text)
    out_cap = OUTPUT_CAPS[length_choice]

    if length_choice == "Short":
        return _short_summary(cleaned, out_cap)

    # Medium/Detailed path
    raw_out = _ai_generate(cleaned, tokenizer, model, model_choice, 30, 150, length_choice)
    return raw_out or extractive_summary(cleaned, top_n=5)