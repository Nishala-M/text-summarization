# summarizer.py — Final Speed-Optimized Deployment Version
# ─────────────────────────────────────────────────────────────────────────────
# Short:    30–80w   | TF-IDF Extractive | 0 model calls | < 1s
# Medium:   80–130w  | AI Abstractive    | ~4–8s on CPU
# Detailed: 130–200w | AI + Extractive   | ~7–12s on CPU
#
# ── SPEED ARCHITECTURE ───────────────────────────────────────────────────────
# 1. _fast_presample() fires BEFORE clean_input() — reduces long docs to
#    300-500 words using O(n) word-split (no regex, no tokenizer).
#    clean_input() then sees <50 lines instead of 300+.
#
# 2. SMART_TRIM_MAX_SENTS=50 caps TF-IDF inside smart_trim.
#    50²=2,500 vs 300²=90,000 — 36x cheaper cosine_similarity.
#
# 3. TOKENIZER_MAX_LEN=150 matches actual input size (80-110 words × 1.35 BPE).
#    Previously 512 → wasted padding computation.
#
# 4. MAX_NEW_TOKENS: Medium=90, Detailed=140 — calibrated to produce
#    enough raw text for OUTPUT_CAPS without over-generating.
#
# 5. torch.inference_mode() + torch.compile() + use_cache=True at load time.
#
# 6. INPUT_WORD_LIMIT: Medium=80w, Detailed=110w.
#    80w ≈ 104 tokens → encoder attention = 104²×6 = 64K ops (fast).
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import shutil
import nltk
import torch
from collections import Counter
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gdown

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── CPU threading ──────────────────────────────────────────────────────────────
_cpu = os.cpu_count() or 4
try:
    torch.set_num_threads(_cpu)
    torch.set_num_interop_threads(max(1, _cpu // 2))
except RuntimeError:
    pass

# ── Model paths & Google Drive IDs ────────────────────────────────────────────
BART_PATH      = "my_bart_model"
T5_PATH        = "my_t5_model"
BART_FOLDER_ID = "1nQQGRPtI5R_96nZ9JTwWLsg3-icKB2we"
T5_FOLDER_ID   = "1gFOAZ5Ypn_kDEHzGKUApJrbq_g_VuuFK"

# ── Length targets ─────────────────────────────────────────────────────────────
LENGTH_SETTINGS = {
    "Short":    {"min_length": 25,  "max_length": 60},
    "Medium":   {"min_length": 50,  "max_length": 100},
    "Detailed": {"min_length": 80,  "max_length": 150},
}
OUTPUT_CAPS = {"Short": 80, "Medium": 130, "Detailed": 200}

# ── Speed parameters (tuned for sub-10s on CPU) ────────────────────────────────
INPUT_WORD_LIMIT = {
    "Short":    60,    # extractive only — not used for AI
    "Medium":   80,    # 80w × 1.35 BPE = 108 tokens → fast encoder
    "Detailed": 110,   # 110w × 1.35 BPE = 148 tokens → acceptable
}

# MAX_NEW_TOKENS: each step = one full decoder forward pass.
# 90 steps × ~30ms/step on quantized CPU = ~2.7s decoder time.
MAX_NEW_TOKENS = {
    "Short":    40,    # unused for AI
    "Medium":   90,    # ~2-3s decoder
    "Detailed": 140,   # ~4-5s decoder
}

# Hard cap on encoder input tokens — prevents silent truncation.
# INPUT_WORD_LIMIT × 1.35 BPE + 5 safety margin.
TOKENIZER_MAX_LEN = 155

# Max sentences fed to TF-IDF inside smart_trim. 50²=2,500 ops.
SMART_TRIM_MAX_SENTS = 50

# Pre-sample target words BEFORE clean_input() runs.
PRESAMPLE_TARGET = {
    "Short": 300, "Medium": 400, "Detailed": 500
}

MAX_CLEAN_WORDS      = 4000
MAX_SENTS_EXTRACTIVE = 150

# ── Compiled regex ─────────────────────────────────────────────────────────────
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
_RE_NONASCII = re.compile(r"[^\x00-\x7F]+")
_RE_REPWORD  = re.compile(r"\b(\w+)( \1\b)+")
_META_KW = re.compile("|".join(re.escape(k) for k in [
    "uploaded by", "downloaded", "researchgate", "available online",
    "article history", "copyright", "licence", "license",
    "received:", "accepted:", "published:", "revised:",
    "doi:", "isbn:", "issn:", "http://", "https://", "www.", "@",
]), re.IGNORECASE)
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

_BAD_START = {
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "nevertheless", "nonetheless", "additionally", "consequently",
    "meanwhile", "similarly", "likewise", "whereas", "although", "though",
    "explainability", "summarization", "preprocessing", "tokenization",
}
_BAD_END = {
    "and", "or", "but", "the", "of", "in", "a", "an", "for", "to", "with",
    "by", "at", "is", "are", "as", "such", "also", "including",
}
_BAD_PHRASES = [
    "it is essential to maintain health and reduce",
    "it is important to maintain health",
    "broad range of computing", "broad spectrum of computational",
    "as AI, it encompass", "is a more than the same",
    "is designed to the system", "is designed to help the most efficient",
    "the best service in a variety of the",
]
_TRIVIAL = {"of","and","the","a","an","in","to","for","with","by","at","or","its","their","from"}
_COMMON_VERBS = {
    "is","are","was","were","has","have","had","does","do","did","will","would",
    "can","could","should","may","might","must","be","been","being",
    "include","includes","included","represent","represents","allow","allows",
    "provide","provides","show","shows","help","helps","make","makes",
    "give","gives","take","takes","use","uses","mean","means","find","found",
    "resist","absorbed","transformed","developed","emerged","created","built",
    "shaped","became","remained","continued",
}


# ═════════════════════════════════════════════════════════════════════════════
# TITLE-MERGE DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

def _has_title_merge(s: str) -> bool:
    words = s.split()
    if len(words) < 8: return False
    title_count = 0
    for i, w in enumerate(words):
        clean = re.sub(r"[^a-zA-Z]", "", w)
        if not clean: continue
        if clean.lower() in _TRIVIAL: continue
        if clean[0].isupper(): title_count += 1
        elif clean[0].islower(): break
        if title_count >= 3 and i + 1 < len(words):
            nxt = re.sub(r"[^a-zA-Z]", "", words[i + 1])
            if nxt and nxt[0].isupper() and nxt.lower() not in _TRIVIAL:
                prefix = [re.sub(r"[^a-z]", "", w2.lower()) for w2 in words[:i + 1]]
                if not any(p in _COMMON_VERBS for p in prefix):
                    return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# MODEL DOWNLOAD & LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _is_model_ready(path: str) -> bool:
    if not os.path.isdir(path): return False
    files = os.listdir(path)
    return "config.json" in files and (
        "model.safetensors" in files or "pytorch_model.bin" in files)


def _get_model_type(path: str) -> str:
    try:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f).get("model_type", "unknown").lower()
    except Exception: return "unknown"


def _find_model_root(search_dir: str):
    if _is_model_ready(search_dir): return search_dir
    candidates = []
    for root, dirs, files in os.walk(search_dir):
        if "config.json" in files and (
                "model.safetensors" in files or "pytorch_model.bin" in files):
            candidates.append(root)
    if not candidates: return None
    candidates.sort(key=len)
    return candidates[0]


def _download_model_folder(folder_id: str, output_path: str) -> bool:
    if _is_model_ready(output_path):
        print(f"[INFO] Model already present at {output_path}"); return True
    print(f"[INFO] Downloading model {folder_id} → {output_path} ...")
    tmp_dir = output_path + "_tmp_dl"
    try:
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        gdown.download_folder(id=folder_id, output=tmp_dir, quiet=False, use_cookies=False)
        model_src = _find_model_root(tmp_dir)
        if model_src is None:
            all_files = []
            for r, d, fs in os.walk(tmp_dir):
                for f in fs: all_files.append(os.path.join(r, f).replace(tmp_dir, ""))
            print(f"[ERROR] No model weights found. Files: {all_files[:20]}")
            shutil.rmtree(tmp_dir); return False
        if os.path.exists(output_path): shutil.rmtree(output_path)
        shutil.copytree(model_src, output_path)
        shutil.rmtree(tmp_dir)
        ok = _is_model_ready(output_path)
        print(f"[INFO] Model {'ready' if ok else 'INCOMPLETE'} at {output_path}")
        return ok
    except Exception as exc:
        print(f"[ERROR] Download failed ({output_path}): {exc}")
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
        return False


def _load_tokenizer_robust(path: str, expected_type: str):
    """Load tokenizer, handling the T5/BART mismatch from mixed Drive folders."""
    actual_type = _get_model_type(path)
    print(f"[INFO] Loading tokenizer: actual={actual_type}, expected={expected_type}")

    # T5: find spiece.model (works even if tokenizer_config.json says BartTokenizer)
    if expected_type == "t5":
        spiece_path = None
        for root, dirs, files in os.walk(path):
            if "spiece.model" in files:
                spiece_path = os.path.join(root, "spiece.model"); break
        if spiece_path:
            try:
                tok = T5Tokenizer(vocab_file=spiece_path, legacy=False)
                print(f"[INFO] T5Tokenizer loaded via spiece.model at {spiece_path}")
                return tok
            except Exception as exc:
                print(f"[WARN] T5Tokenizer(spiece) failed: {exc}")
        # Fallback: find subfolder with correct T5 tokenizer_config
        for root, dirs, files in os.walk(path):
            if "tokenizer_config.json" in files:
                try:
                    with open(os.path.join(root, "tokenizer_config.json")) as f:
                        tc = json.load(f)
                    if "t5" in tc.get("tokenizer_class", "").lower():
                        tok = AutoTokenizer.from_pretrained(root)
                        print(f"[INFO] T5 AutoTokenizer from subfolder: {root}"); return tok
                except Exception: pass

    # BART
    if expected_type == "bart":
        try:
            tok = BartTokenizer.from_pretrained(path)
            print("[INFO] BartTokenizer loaded."); return tok
        except Exception as exc:
            print(f"[WARN] BartTokenizer failed: {exc}")
            try:
                tok = AutoTokenizer.from_pretrained(path)
                print(f"[INFO] AutoTokenizer fallback: {type(tok).__name__}"); return tok
            except Exception as exc2:
                raise RuntimeError(f"Cannot load BART tokenizer: {exc2}")

    # Generic fallback
    tok = AutoTokenizer.from_pretrained(path)
    print(f"[INFO] AutoTokenizer: {type(tok).__name__}"); return tok


def _quantize(model):
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)


def _try_compile(model):
    """torch.compile() gives ~10-15% speedup on CPU with PyTorch 2.x."""
    try:
        compiled = torch.compile(model, fullgraph=False, dynamic=False)
        print("[INFO] torch.compile() applied."); return compiled
    except Exception as exc:
        print(f"[INFO] torch.compile() skipped: {exc}"); return model


def load_bart():
    try:
        if not _download_model_folder(BART_FOLDER_ID, BART_PATH): return None, None
        if not _is_model_ready(BART_PATH): return None, None
        tok = _load_tokenizer_robust(BART_PATH, "bart")
        mod = BartForConditionalGeneration.from_pretrained(
            BART_PATH, torch_dtype=torch.float32, ignore_mismatched_sizes=True)
        mod.eval()
        mod.config.use_cache = True     # KV-cache: 30-40% faster decoding
        mod = _quantize(mod)
        mod = _try_compile(mod)
        print("[INFO] BART loaded and optimized."); return tok, mod
    except Exception as exc:
        import traceback; print(f"[ERROR] BART: {exc}"); traceback.print_exc()
        return None, None


def load_t5():
    try:
        if not _download_model_folder(T5_FOLDER_ID, T5_PATH): return None, None
        if not _is_model_ready(T5_PATH): return None, None
        tok = _load_tokenizer_robust(T5_PATH, "t5")
        try:
            mod = T5ForConditionalGeneration.from_pretrained(
                T5_PATH, torch_dtype=torch.float32, ignore_mismatched_sizes=True)
        except Exception:
            mod = AutoModelForSeq2SeqLM.from_pretrained(T5_PATH, torch_dtype=torch.float32)
        mod.eval()
        mod.config.use_cache = True
        mod = _quantize(mod)
        mod = _try_compile(mod)
        print("[INFO] T5 loaded and optimized."); return tok, mod
    except Exception as exc:
        import traceback; print(f"[ERROR] T5: {exc}"); traceback.print_exc()
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _sent_tok(text: str) -> list:
    try:    return nltk.sent_tokenize(text)
    except: return [text]


def has_metadata(t: str) -> bool:
    tl = t.lower()
    return bool(_META_KW.search(tl) or _RE_MONTH.search(tl)
                or _RE_INST.search(tl) or _RE_ETAL.search(tl) or _RE_EMAIL.search(t))


def is_research_paper(text: str) -> bool:
    sample = text[:5000]
    score  = sum(1 for p in [
        r'\[\d+\]', r'\bDOI\b', r'\bIEEE\b', r'\bACM\b',
        r'\bet al\.\b', r'\bpp\.\s*\d+', r'\bvol\.\s*\d+', r'arXiv', r'\bProceedings\b',
    ] if re.search(p, sample, re.IGNORECASE))
    return score >= 2


def _is_academic_noise(s: str) -> bool:
    if len(_RE_CITATION_INLINE.findall(s)) >= 2: return True
    if _RE_HYPHEN_INITIAL.search(s): return True
    if _RE_AUTHOR_CHAIN.search(s): return True
    if _RE_PAGE_RANGE.search(s) and _RE_YEAR_PAREN.search(s): return True
    if len(_RE_JOURNAL_KW.findall(s)) >= 2: return True
    if _RE_ADVANTAGE_LBL.search(s) and len(s.split()) < 30: return True
    if re.match(r'^\d+[,\.\s]', s.strip()) and len(s.split()) < 15: return True
    return False


def _is_bad_sentence(s: str) -> bool:
    s = s.strip(); w = s.split()
    if len(w) < 6 or ".." in s: return True
    if s and s[0].islower(): return True
    if _RE_DASH.match(s): return True
    last = s.rstrip(".!?,: ").split()[-1].lower() if w else ""
    if last in _BAD_END: return True
    if s.count(",") > 8 or has_metadata(s): return True
    sl = s.lower()
    if any(bp in sl for bp in _BAD_PHRASES): return True
    if _RE_FIGTBL.search(s): return True
    alpha = sum(1 for c in s if c.isalpha())
    total = len(s.replace(" ", ""))
    if total > 0 and alpha / total < 0.55: return True
    for word in w:
        if len(word) > 20 and "-" not in word and word.isalpha(): return True
    if _is_academic_noise(s): return True
    if _RE_SETUP_SENT.search(s): return True
    if re.search(r",\s+the\s+[A-Z][a-z]+\.?\s*$", s): return True
    if re.search(r",\s+[A-Z][a-z]{3,}\.?\s*$", s) and len(w) < 20: return True
    if _has_title_merge(s): return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# INPUT CLEANING
# ═════════════════════════════════════════════════════════════════════════════

def _strip_refs(text: str) -> str:
    _RE_REF_LINE = re.compile(
        r'^\[\d+\]|^\d+\.\s+[A-Z][a-z]+.*(?:IEEE|ACM|vol\.|pp\.|Journal)', re.IGNORECASE)
    lines = text.split("\n"); out = []; in_refs = False
    for line in lines:
        s = line.strip()
        if _RE_REFSEC.match(s): in_refs = True; continue
        if in_refs: continue
        if _RE_REF_LINE.match(s): continue
        out.append(_RE_LABEL.sub("", line))
    return "\n".join(out)


def _is_header(line: str) -> bool:
    s = line.strip(); w = s.split()
    if not w: return False
    if s == s.upper() and len(w) <= 5 and any(c.isalpha() for c in s): return True
    if re.match(r"^(\d+\.?\d*\.?|[IVX]+\.)\s+[A-Z]", s): return True
    if len(w) <= 6 and sum(1 for x in w if x and x[0].isupper()) / len(w) >= 0.8: return True
    if not s.endswith((".", "!", "?")):
        caps = sum(1 for x in w if x and x[0].isupper()) / len(w)
        if caps >= 0.55 and ":" in s: return True
    return False


def clean_input(text: str, short_input: bool = False) -> str:
    if is_research_paper(text):
        text = re.sub(r'\s*\[\d+\]\s*', ' ', text)
        text = re.sub(r'\b[A-Z]\.-?[A-Z]\.\s+[A-Z][a-z]+', '', text)
        text = re.sub(r',?\s*pp\.\s*\d+[-\u2013]\d+', '', text)
        text = re.sub(r',?\s*(?:vol|no)\.\s*\d+', '', text)
        text = re.sub(r',\s*\d{4},?\s*pp\.', '', text)
        text = re.sub(
            r',?\s*(?:in\s+)?(?:IEEE|ACM|Proc\.?|Proceedings|'
            r'International\s+Conference|Conference\s+on|Workshop\s+on)'
            r'[^.]{0,120}\.?', '. ', text, flags=re.IGNORECASE)
        text = re.sub(r'\(\d{4}\)', '', text)
        text = re.sub(r'\.(\s*\.)+', '.', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\s{2,}', ' ', text)

    text = _strip_refs(text)
    lines = text.split("\n"); kept = []
    min_wc = 5 if short_input else 7
    for line in lines:
        line = line.strip()
        if not line or _is_header(line): continue
        if len(line) < 120 and line == line.upper(): continue
        if has_metadata(line): continue
        if len(line.split()) < min_wc: continue
        alpha = sum(1 for c in line if c.isalpha())
        total = len(line.replace(" ", ""))
        if total > 0 and alpha / total < 0.55: continue
        words = line.split()
        if len(words) < 20:
            si = sum(1 for wd in words if len(wd) == 1)
            ni = sum(1 for wd in words if re.match(r"^\d+\.?\d*$", wd))
            if (si + ni) / max(len(words), 1) >= 0.35: continue
        if re.match(r"^\[\d+\]", line): continue
        kept.append(line)

    _DANGLING = {
        "for","and","or","of","in","to","with","by","a","an","the","that",
        "which","this","these","those","its","their","as","at","on","from",
        "into","between","including","such","after","before","during",
        "through","across","against","about","over","under",
    }
    _RE_OPEN_MOD = re.compile(
        r"\b\w*(?:ical|tional|sional|ational|logical|nological|"
        r"graphical|metrical|nautical|nomical|litical|sotical|"
        r"tectural|ectural|ultural|ructural)\s*$", re.IGNORECASE)
    merged = []
    for i, line in enumerate(kept):
        if i == 0: merged.append(line); continue
        last_word  = (merged[-1].rstrip().split()[-1].lower().rstrip(",:;") if merged[-1].strip() else "")
        first_char = line.lstrip()[0] if line.strip() else ""
        prev_ends  = merged[-1].rstrip()[-1] in ".!?" if merged[-1].strip() else False
        if not prev_ends and first_char.isupper():
            if last_word in _DANGLING or _RE_OPEN_MOD.search(last_word):
                merged[-1] = merged[-1].rstrip().rstrip(",:;") + "."
        merged.append(line)
    text = " ".join(merged)
    text = _RE_HYPHEN.sub("", text)
    text = _RE_WS.sub(" ", text)
    text = _RE_NONASCII.sub(" ", text)
    text = _RE_REPWORD.sub(r"\1", text)
    text = re.sub(r",([^\s\d])", r", \1", text)
    text = re.sub(r"\.([A-Za-z])", r". \1", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# FAST PRE-SAMPLING  (key speed fix for long PDFs)
# ═════════════════════════════════════════════════════════════════════════════

def _fast_presample(text: str, target_words: int) -> str:
    """
    Reduce long documents to target_words using zone-based word-splitting.
    Runs BEFORE clean_input() — pure split(), no regex, no tokenizer.
    Zone weights: 40% intro + 35% middle + 25% conclusion.
    """
    words = text.split()
    if len(words) <= target_words:
        return text
    n     = len(words)
    n_s   = int(target_words * 0.40)
    n_m   = int(target_words * 0.35)
    n_e   = target_words - n_s - n_m
    mid   = n // 2
    half  = n_m // 2
    return " ".join(
        words[:n_s]
        + words[max(0, mid - half): mid + half]
        + words[max(0, n - n_e):]
    )


# ═════════════════════════════════════════════════════════════════════════════
# SMART TRIM  (zone-based TF-IDF, hard-capped at SMART_TRIM_MAX_SENTS)
# ═════════════════════════════════════════════════════════════════════════════

def smart_trim(text: str, length_choice: str) -> str:
    limit = INPUT_WORD_LIMIT.get(length_choice, 80)
    words = text.split()
    if len(words) <= limit: return text
    try:
        sents = _sent_tok(text)
        sents = [s.strip() for s in sents if len(s.split()) >= 6 and not _is_bad_sentence(s.strip())]
        if not sents: return " ".join(words[:limit])

        # Hard cap with spread sampling to keep document coverage
        if len(sents) > SMART_TRIM_MAX_SENTS:
            n       = len(sents)
            n_front = max(1, SMART_TRIM_MAX_SENTS // 5)
            n_back  = max(1, SMART_TRIM_MAX_SENTS // 5)
            n_mid   = SMART_TRIM_MAX_SENTS - n_front - n_back
            mid     = n // 2; half_m = n_mid // 2
            pool    = sents[:n_front] + sents[max(0, mid - half_m): mid + half_m] + sents[max(0, n - n_back):]
            seen = set(); sents = []
            for s in pool:
                if s not in seen: seen.add(s); sents.append(s)

        n = len(sents)
        try:
            vec    = TfidfVectorizer(stop_words="english")
            tfidf  = vec.fit_transform(sents)
            scores = cosine_similarity(tfidf, tfidf).sum(axis=1)
        except Exception:
            scores = np.ones(n)

        n_zones = max(3, min(8, limit // 16))
        zone_sz = max(1, n // n_zones)
        picked: set = set()
        for z in range(n_zones):
            start = z * zone_sz
            end   = min(n, start + zone_sz) if z < n_zones - 1 else n
            if start >= n: break
            best = max(range(start, end), key=lambda i: scores[i])
            picked.add(best)

        result: list = []; wc = 0
        for idx in sorted(picked):
            sw = len(sents[idx].split())
            if wc + sw > int(limit * 1.15): break
            result.append(sents[idx]); wc += sw

        if wc < limit * 0.75:
            unused = sorted([i for i in range(n) if i not in picked],
                            key=lambda i: scores[i], reverse=True)
            for idx in unused:
                sw = len(sents[idx].split())
                if wc + sw > int(limit * 1.15): continue
                result.append(sents[idx]); wc += sw
                if wc >= limit * 0.90: break
            result.sort(key=lambda s: sents.index(s))

        return " ".join(result) if result else " ".join(words[:limit])
    except Exception:
        return " ".join(words[:limit])


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _filter_output(text: str) -> str:
    good = [s.strip() for s in _sent_tok(text) if not _is_bad_sentence(s.strip())]
    if not good: return text
    out = _RE_WS.sub(" ", " ".join(good)).strip()
    return out[0].upper() + out[1:] if out and out[0].islower() else out


def _dedup(text: str) -> str:
    _STOP = {
        "the","a","an","is","are","was","were","of","in","to","and","or",
        "that","this","it","for","with","has","have","been","by","on","at",
        "from","its","be","as","not","but","also","can","will","may",
    }
    sents = _sent_tok(text); kept: list = []; ks: list = []
    for s in sents:
        s = s.strip()
        if not s: continue
        w = set(s.lower().split()); c = w - _STOP; dup = False
        for ew, ec in ks:
            sim = len(w & ew) / max(len(w), len(ew))
            if sim >= 0.55: dup = True; break
            if len(c & ec) >= 6 and sim >= 0.35: dup = True; break
        if not dup: kept.append(s); ks.append((w, c))
    return " ".join(kept)


def _cap(text: str, max_w: int, strict: bool = False) -> str:
    sents = _sent_tok(text); out: list = []; n = 0
    grace = 0 if strict else min(8, int(max_w * 0.05))
    for s in sents:
        s = s.strip()
        if not s: continue
        wc = len(s.split())
        if n == 0:   out.append(s); n += wc
        elif n >= max_w: break
        elif n + wc <= max_w + grace: out.append(s); n += wc
        else: break
    return " ".join(out) if out else " ".join(text.split()[:max_w])


def _ensure_complete_sentences(text: str) -> str:
    text = text.strip()
    if not text: return text
    if text[-1] in '.!?"\'': return text
    sents    = _sent_tok(text)
    complete = [s.strip() for s in sents if s.strip() and s.strip()[-1] in ".!?"]
    if complete: return " ".join(complete)
    return text.rstrip(",;: ") + "."


def _enforce_sentence_end(text: str) -> str:
    text = text.strip()
    if not text or text[-1] in ".!?": return text
    last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_end > len(text) * 0.35: return text[:last_end + 1].strip()
    return text


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def extractive_summary(text: str, top_n: int = 5, zone_based: bool = False) -> str:
    try:
        sents = _sent_tok(text)
        sents = [s.strip() for s in sents
                 if len(s.split()) >= 8 and not _is_bad_sentence(s.strip())
                 and s.split()[0].lower().rstrip(",") not in _BAD_START]
        if not sents:
            sents = [s.strip() for s in _sent_tok(text) if len(s.split()) >= 5]
        if not sents: return ""
        if len(sents) <= top_n: return " ".join(sents)

        if len(sents) > MAX_SENTS_EXTRACTIVE:
            def _prescore(s):
                alpha = sum(1 for c in s if c.isalpha())
                total = max(len(s.replace(" ", "")), 1)
                return min(len(s.split()), 40) / 40.0 * 0.5 + (alpha / total) * 0.5
            scored   = sorted(enumerate(sents), key=lambda x: _prescore(x[1]), reverse=True)
            keep_idx = {i for i, _ in scored[:MAX_SENTS_EXTRACTIVE]}
            sents    = [s for i, s in enumerate(sents) if i in keep_idx]

        vec    = TfidfVectorizer(stop_words="english")
        tfidf  = vec.fit_transform(sents)
        scores = cosine_similarity(tfidf, tfidf).sum(axis=1)
        _S2    = {"the","a","an","is","are","was","of","in","to","and","or","it","for","with"}

        if zone_based:
            n = len(sents); zone_sz = max(1, n // top_n)
            kept: list = []; kept_w: list = []
            for z in range(top_n):
                start = z * zone_sz
                end   = min(n, start + zone_sz) if z < top_n - 1 else n
                if start >= n: break
                best = max(range(start, end), key=lambda i: scores[i])
                sw   = set(sents[best].lower().split()) - _S2
                if not any(len(sw & ew) / max(len(sw), len(ew), 1) >= 0.60 for ew in kept_w):
                    kept.append(best); kept_w.append(sw)
            return " ".join(sents[i] for i in sorted(kept)) if kept else ""
        else:
            kept: list = []; ks: list = []
            for idx in np.argsort(scores)[::-1]:
                if len(kept) >= top_n: break
                s = sents[idx]; w = set(s.lower().split()); c = w - _S2
                if not any(len(w & ew) / max(len(w), len(ew)) >= 0.60 for ew, _ in ks):
                    kept.append(idx); ks.append((w, c))
            return " ".join(sents[i] for i in sorted(kept))
    except Exception as exc:
        print(f"[WARN] extractive_summary: {exc}"); return ""


# ═════════════════════════════════════════════════════════════════════════════
# SHORT MODE  (zero model calls — always < 1s)
# ═════════════════════════════════════════════════════════════════════════════

def _short_summary(cleaned: str, out_cap: int) -> str:
    try:
        sents = _sent_tok(cleaned)
        sents = [s.strip() for s in sents
                 if len(s.split()) >= 8 and not _is_bad_sentence(s.strip())
                 and s.split()[0].lower().rstrip(",") not in _BAD_START]
        if not sents:
            sents = [s.strip() for s in _sent_tok(cleaned) if len(s.split()) >= 6]
        if not sents: return " ".join(cleaned.split()[:out_cap])

        n_zones = 4
        if len(sents) <= n_zones:
            result = _dedup(" ".join(sents))
            result = _cap(result, out_cap, strict=True)
            return _ensure_complete_sentences(_RE_WS.sub(" ", result).strip())

        try:
            vec    = TfidfVectorizer(stop_words="english")
            tfidf  = vec.fit_transform(sents)
            scores = cosine_similarity(tfidf, tfidf).sum(axis=1)
        except Exception:
            scores = np.array([min(len(s.split()), 30) for s in sents], dtype=float)

        n = len(sents); zone_sz = max(1, n // n_zones)
        _STOP = {"the","a","an","is","are","was","of","in","to","and","or","it","for","with"}
        picked: list = []; picked_w: list = []

        for z in range(n_zones):
            start = z * zone_sz
            end   = min(n, start + zone_sz) if z < n_zones - 1 else n
            if start >= n: break
            ideal = [i for i in range(start, end) if 12 <= len(sents[i].split()) <= 25]
            cands = ideal if ideal else list(range(start, end))
            cands = sorted(cands, key=lambda i: scores[i], reverse=True)
            for idx in cands:
                sw = set(sents[idx].lower().split()) - _STOP
                if any(len(sw & ew) / max(len(sw), len(ew), 1) >= 0.55 for ew in picked_w): continue
                picked.append(idx); picked_w.append(sw); break

        if len(picked) < 2:
            for idx in np.argsort(scores)[::-1]:
                if len(picked) >= n_zones: break
                sw = set(sents[idx].lower().split()) - _STOP
                if not any(len(sw & ew) / max(len(sw), len(ew), 1) >= 0.55 for ew in picked_w):
                    picked.append(idx); picked_w.append(sw)

        result = " ".join(sents[i] for i in sorted(picked))
        result = _dedup(result); result = _cap(result, out_cap, strict=True)
        result = _ensure_complete_sentences(result)
        result = _RE_WS.sub(" ", result).strip()
        return result[0].upper() + result[1:] if result and not result[0].isupper() else result

    except Exception as exc:
        print(f"[WARN] _short_summary: {exc}")
        ext   = extractive_summary(cleaned, top_n=4, zone_based=True)
        return _cap(ext or " ".join(cleaned.split()[:out_cap]), out_cap, strict=True)


# ═════════════════════════════════════════════════════════════════════════════
# AI GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def _is_hallucinated(text: str) -> bool:
    if not text: return False
    w = text.lower().split(); n = len(w)
    if n < 8: return False
    score = 0
    ur = len(set(w)) / n
    if ur < 0.45: score += 3
    elif ur < 0.55: score += 1
    cd = sum(1 for i in range(n - 1) if w[i] == w[i + 1])
    if cd >= 2: score += 3
    elif cd >= 1: score += 2
    sw = sum(1 for x in w if len(x) <= 2) / n
    if sw > 0.35: score += 3
    elif sw > 0.28: score += 2
    tg = [tuple(w[i:i + 3]) for i in range(n - 2)]
    if tg:
        mr = max(tg.count(t) for t in set(tg))
        if mr >= 3: score += 3
        elif mr >= 2: score += 1
    clean = [re.sub(r"[^a-z]", "", x) for x in w]
    freq  = Counter(x for x in clean if len(x) > 4)
    if any(cnt >= (3 if n < 50 else 4) for cnt in freq.values()): return True
    return score >= 2


def _fix_output(text: str) -> str:
    text = re.sub(r"^summarize\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    exp  = {"AI": "Artificial Intelligence","ML": "Machine Learning",
            "NLP": "Natural Language Processing","DL": "Deep Learning"}
    m = _RE_ACRONYM.match(text)
    if m and m.group(2) in exp:
        text = _RE_ACRONYM.sub(exp[m.group(2)] + " (" + m.group(2) + ")", text, count=1)
    return text[0].upper() + text[1:] if text and text[0].islower() else text


def _ai_generate(text, tokenizer, model, model_choice, min_len, max_len, length_choice):
    if not text.strip(): return ""
    wc       = len(text.split())
    safe_min = min(min_len, max(10, wc // 4))
    safe_min = min(safe_min, MAX_NEW_TOKENS.get(length_choice, 90) // 2)  # clamp
    mnt      = MAX_NEW_TOKENS.get(length_choice, 90)
    inp      = ("summarize: " + text) if model_choice == "T5" else text
    try:
        enc = tokenizer(inp, return_tensors="pt",
                        max_length=TOKENIZER_MAX_LEN, truncation=True, padding=False)
        enc = {k: v.to("cpu") for k, v in enc.items()}
        with torch.inference_mode():        # ~5% faster than no_grad on CPU
            out = model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=1,               # greedy decode — fastest possible
                do_sample=False,
                early_stopping=True,
                min_length=safe_min,
                max_new_tokens=mnt,
                no_repeat_ngram_size=3,    # prevents repetition
                length_penalty=0.8,        # slightly below 1 → ends sooner
            )
        raw = tokenizer.decode(out[0], skip_special_tokens=True,
                               clean_up_tokenization_spaces=True).strip()
        return _enforce_sentence_end(raw)
    except Exception as exc:
        print(f"[ERROR] AI generation: {exc}"); return ""


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def generate_summary(input_text, tokenizer, model, model_choice, length_choice):
    if not input_text or not input_text.strip():
        return "Please enter some text to summarize."
    if tokenizer is None or model is None:
        return "Model not loaded. Please restart the app."

    raw_wc = len(input_text.split())

    # ── Speed fix #1: pre-sample long docs BEFORE clean_input() ───────────────
    # For a 2000-word PDF: clean_input() runs 150+ regex checks on 150+ lines.
    # _fast_presample() reduces to 300-500 words in O(n) pure Python — instant.
    target = PRESAMPLE_TARGET.get(length_choice, 400)
    if raw_wc > target:
        input_text = _fast_presample(input_text, target)

    cleaned = clean_input(input_text, short_input=raw_wc < 120)
    if len(cleaned.split()) < 8:
        cleaned = _RE_WS.sub(" ", input_text).strip()
    if len(cleaned.split()) < 5:
        return "Text too short. Please provide more content."

    if len(cleaned.split()) > MAX_CLEAN_WORDS:
        cleaned = _fast_presample(cleaned, MAX_CLEAN_WORDS)

    wc   = len(cleaned.split())
    cfg  = LENGTH_SETTINGS[length_choice]
    base = OUTPUT_CAPS[length_choice]

    if wc < 100:
        out_cap = max(20, int(wc * 0.55))
    elif wc < 200:
        ratios  = {"Short": 0.55, "Medium": 0.62, "Detailed": 0.72}
        mins    = {"Short": 35,   "Medium": 65,   "Detailed": 100}
        out_cap = min(base, max(mins[length_choice], int(wc * ratios[length_choice])))
    else:
        out_cap = base

    ext_top = 3 if wc < 200 else (5 if wc < 500 else (7 if wc < 1500 else 10))

    if length_choice == "Short":
        return _short_summary(cleaned, out_cap)

    trimmed = smart_trim(cleaned, length_choice)
    raw_out = _ai_generate(trimmed, tokenizer, model, model_choice,
                           cfg["min_length"], cfg["max_length"], length_choice)
    ai_out  = _filter_output(_fix_output(raw_out)) if raw_out else ""
    if not ai_out and raw_out: ai_out = _fix_output(raw_out)
    if _is_hallucinated(ai_out): ai_out = ""
    if ai_out and wc > 250 and len(ai_out.split()) > 0.70 * wc: ai_out = ""

    ai_wc   = len(ai_out.split()) if ai_out else 0
    ext     = extractive_summary(cleaned, top_n=ext_top)
    tgt_min = int(wc * 0.40) if wc < 200 else {"Medium": 55, "Detailed": 90}[length_choice]

    if ai_wc >= tgt_min:
        if length_choice == "Detailed":
            ai_sents = [set(x.lower().split()) for x in _sent_tok(ai_out)]
            extras: list = []; cur_wc = ai_wc
            for x in (_sent_tok(ext) if ext else []):
                if cur_wc >= out_cap: break
                x = x.strip(); xw = set(x.lower().split()); xwc = len(x.split())
                if cur_wc + xwc > out_cap: continue
                if (not any(len(xw & ew) / max(len(xw), len(ew)) >= 0.55 for ew in ai_sents)
                        and not _is_bad_sentence(x)):
                    ai_sents.append(xw); extras.append(x); cur_wc += xwc
                    if len(extras) >= 6: break
            final = (ai_out + " " + " ".join(extras)).strip() if extras else ai_out
        else:
            final = ai_out
    else:
        base_t   = ai_out if ai_wc >= 12 else ""
        ai_sents = ([set(x.lower().split()) for x in _sent_tok(base_t)] if base_t else [])
        extras: list = []; cur_wc = ai_wc if base_t else 0
        for x in (_sent_tok(ext) if ext else []):
            if cur_wc >= out_cap: break
            x = x.strip(); xw = set(x.lower().split())
            if (not any(len(xw & ew) / max(len(xw), len(ew)) >= 0.55
                        for ew in ai_sents if ew) and not _is_bad_sentence(x)):
                ai_sents.append(xw); extras.append(x); cur_wc += len(x.split())
        if base_t and extras:   final = base_t + " " + " ".join(extras)
        elif extras:            final = " ".join(extras)
        elif base_t:            final = base_t
        elif ext:               final = ext
        else:                   final = ai_out or ""

    if length_choice == "Detailed" and len(final.split()) < 100:
        pool = [s.strip() for s in _sent_tok(cleaned)
                if len(s.split()) >= 8 and not _is_bad_sentence(s.strip())
                and s.split()[0].lower().rstrip(",") not in _BAD_START]
        existing = set(final.lower().split())
        for x in pool:
            if len(final.split()) >= out_cap: break
            xw = set(x.lower().split())
            if len(xw & existing) / max(len(xw), 1) < 0.75 and not _is_bad_sentence(x):
                final = final + " " + x; existing.update(xw)

    if not final or not final.strip():
        return "Could not generate a summary. Please try with more text."

    final = _filter_output(final)
    if not final.strip(): final = ext or ai_out or ""
    if not final.strip(): return "Could not generate a summary."

    final = _dedup(final)
    final = _cap(final, out_cap, strict=False)
    final = _ensure_complete_sentences(final)
    final = _RE_WS.sub(" ", final).strip()
    return final[0].upper() + final[1:] if final and not final[0].isupper() else final