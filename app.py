# app.py — Final Deployment Version
# Professional AI Text Summarizer — Multilingual Support
# Fixes: PDF UI improved, translation pipeline shows intermediate results,
#        speed optimizations passed through, clean deployable code.

import streamlit as st
import pdfplumber
import time
import io
from summarizer import load_bart, load_t5, generate_summary
from explainability import get_important_sentences
from translator import (
    is_available as translation_available,
    detect_language,
    get_language_name,
    translate_to_english,
    translate_from_english,
    SUPPORTED_LANGUAGES,
)

st.set_page_config(
    page_title="SummarAI — Intelligent Text Summarizer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:       #eef0f9;
    --bg2:      #ffffff;
    --bg3:      #e4e7f5;
    --bg4:      #dde0f2;
    --border:   #c8cde8;
    --accent:   #4f46e5;
    --accent2:  #e8458a;
    --accent3:  #059669;
    --text:     #1a1d2e;
    --muted:    #5a6080;
    --shadow:   0 2px 14px rgba(79,70,229,0.10);
    --shadow2:  0 4px 24px rgba(79,70,229,0.16);
}

html, body,
.stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.main, .block-container,
section.main, [class*="css"] {
    background-color: #eef0f9 !important;
    color: #1a1d2e !important;
    color-scheme: light !important;
}
[data-theme="dark"], [data-theme="dark"] body, [data-theme="dark"] .stApp {
    --background-color: #eef0f9 !important;
    --secondary-background-color: #e4e7f5 !important;
    --text-color: #1a1d2e !important;
    background-color: #eef0f9 !important;
    color: #1a1d2e !important;
    color-scheme: light !important;
}
@media (prefers-color-scheme: dark) {
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    .main, .block-container {
        background-color: #eef0f9 !important;
        color: #1a1d2e !important;
        color-scheme: light !important;
    }
}
p, span, label, div, h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"],
[data-testid="stText"],
.stSelectbox label, .stToggle label,
.stSlider label, .stFileUploader label {
    color: #1a1d2e !important;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #eef0f9 !important;
    color: #1a1d2e !important;
}
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Hero ──────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.8rem 1rem 1.8rem;
    background: linear-gradient(135deg, rgba(79,70,229,.07) 0%, rgba(232,69,138,.05) 100%);
    border-radius: 18px;
    border: 1px solid rgba(79,70,229,.12);
    margin-bottom: 1.8rem;
}
.hero-badge {
    display:inline-block;
    background:linear-gradient(135deg,rgba(79,70,229,.15),rgba(232,69,138,.10));
    border:1px solid rgba(79,70,229,.3); border-radius:50px;
    padding:5px 16px; font-size:.72rem; font-family:'Syne',sans-serif;
    letter-spacing:.12em; text-transform:uppercase; color:var(--accent);
    margin-bottom:1rem;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; line-height:1.15;
    background:linear-gradient(135deg,#1a1d2e 20%,var(--accent) 60%,var(--accent2) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin-bottom:.8rem;
}
.hero-sub {
    font-size:1rem; color:var(--muted); max-width:500px;
    margin:0 auto 1.8rem; line-height:1.7;
}
.hero-divider {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(79,70,229,.25),transparent);
    margin:0 auto 1.8rem; max-width:500px;
}
.stats-row { display:flex; justify-content:center; gap:.8rem; flex-wrap:wrap; margin-bottom:.5rem; }
.stat-pill {
    background: rgba(255,255,255,0.7);
    border:1px solid rgba(79,70,229,.18);
    border-radius:50px; padding:6px 16px; font-size:.78rem;
    color:var(--muted); display:flex; align-items:center; gap:5px;
    box-shadow: 0 2px 8px rgba(79,70,229,.08);
}
.stat-pill span { color:var(--accent); font-weight:600; }

/* ── Card title ─────────────────────────────────────────────── */
.card-title {
    font-family:'Syne',sans-serif; font-size:.68rem; font-weight:700;
    letter-spacing:.14em; text-transform:uppercase; color:var(--muted);
    margin-bottom:.8rem; display:flex; align-items:center; gap:7px;
}
.card-title::before {
    content:''; display:inline-block; width:3px; height:11px;
    background:var(--accent); border-radius:2px;
}

/* ── Input wrapper ──────────────────────────────────────────── */
.input-wrapper {
    background: #ffffff;
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem 1.5rem 1.1rem;
    box-shadow: var(--shadow);
    margin-bottom: .8rem;
}

/* ── Tabs ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg3) !important;
    border-radius: 9px !important;
    gap: 4px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    width: 100% !important;
    margin-bottom: .2rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: .85rem !important;
    font-weight: 600 !important;
    letter-spacing: .04em !important;
    flex: 1 !important;
    text-align: center !important;
    justify-content: center !important;
    padding: .45rem 1rem !important;
}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] div { color: var(--muted) !important; }
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(79,70,229,.3) !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div { color: white !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ── Textarea ───────────────────────────────────────────────── */
.stTextArea textarea {
    background: #fafbff !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .93rem !important;
    line-height: 1.78 !important;
    padding: 1rem 1.1rem !important;
    box-shadow: none !important;
    transition: border-color .22s !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.1) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #b0b7c5 !important; }
[data-baseweb="textarea"],
[data-baseweb="base-input"] { border: none !important; box-shadow: none !important; }

/* ── Buttons ────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #6d64f5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .65rem 1.8rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: .88rem !important;
    letter-spacing: .04em !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
    display: block !important;
    margin: 0 auto !important;
    transition: all .22s !important;
    box-shadow: 0 4px 14px rgba(79,70,229,.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(79,70,229,.45) !important;
    color: #ffffff !important;
}
.stButton > button p, .stButton > button span, .stButton > button div { color: #ffffff !important; }

.btn-clear .stButton > button {
    background: #fff0f0 !important;
    color: #ef4444 !important;
    border: 1.5px solid #fca5a5 !important;
    box-shadow: none !important;
    margin-top: .5rem !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
.btn-clear .stButton > button:hover {
    background: #ffe4e4 !important;
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    transform: none !important;
    box-shadow: none !important;
}
.btn-clear .stButton > button p,
.btn-clear .stButton > button span,
.btn-clear .stButton > button div { color: #ef4444 !important; }

div[data-testid="stColumn"] div[data-testid="stButton"],
div[data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
}

/* ── Selectbox ──────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    box-shadow: var(--shadow) !important;
}

/* ── PDF Uploader ───────────────────────────────────────────── */
[data-testid="stFileUploader"] > section {
    background: #fafbff !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    transition: border-color .22s, background .22s !important;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: var(--accent) !important;
    background: rgba(79,70,229,.03) !important;
}
[data-testid="stFileUploader"] > section > div { gap: .5rem !important; }
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: .85rem !important;
    color: var(--muted) !important;
}
/* Hide the default "Browse files" button and replace with styled one */
[data-testid="stFileUploader"] button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .4rem 1.2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: .8rem !important;
    margin-top: .4rem !important;
    box-shadow: 0 2px 8px rgba(79,70,229,.25) !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #3730d3 !important;
}
.pdf-upload-hint {
    font-size: .75rem;
    color: var(--muted);
    text-align: center;
    margin-top: .5rem;
}

/* ── Result boxes ───────────────────────────────────────────── */
.result-box {
    background: linear-gradient(135deg, rgba(79,70,229,.06), rgba(5,150,105,.04));
    border: 1.5px solid rgba(79,70,229,.2);
    border-radius: 16px; padding: 1.6rem;
    margin: .8rem 0; position: relative; overflow: hidden;
    box-shadow: var(--shadow2);
}
.result-box::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
}
.result-text { font-size: .97rem; line-height: 1.85; color: var(--text); }

/* ── Translation boxes ──────────────────────────────────────── */
.trans-input-box {
    background: linear-gradient(135deg, rgba(232,69,138,.05), rgba(79,70,229,.03));
    border: 1.5px solid rgba(232,69,138,.25);
    border-radius: 14px; padding: 1.4rem;
    margin: .6rem 0; position: relative; overflow: hidden;
}
.trans-input-box::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.trans-label {
    font-family: 'Syne', sans-serif; font-size: .65rem; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
    margin-bottom: .6rem; display: flex; align-items: center; gap: 6px;
}
.trans-label-in  { color: var(--accent2); }
.trans-label-en  { color: var(--accent3); }
.trans-label-out { color: var(--accent); }
.trans-text { font-size: .9rem; line-height: 1.8; color: var(--text); }
.trans-pipeline-bar {
    display: flex; align-items: center; justify-content: center;
    gap: .6rem; flex-wrap: wrap;
    background: rgba(79,70,229,.05);
    border: 1px solid rgba(79,70,229,.15);
    border-radius: 10px; padding: .6rem 1rem;
    margin-bottom: 1rem; font-size: .78rem; color: var(--muted);
}
.trans-pipeline-bar .pipe-step {
    background: #fff; border: 1px solid var(--border);
    border-radius: 6px; padding: 3px 10px;
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: .72rem;
}
.trans-pipeline-bar .pipe-arrow { color: var(--accent); font-size: .9rem; }

/* ── Stats ──────────────────────────────────────────────────── */
.stats-bar { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: .8rem; }
.stat-chip {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 8px; padding: 5px 13px; font-size: .76rem; color: var(--muted);
    box-shadow: 0 1px 4px rgba(79,70,229,.06);
}
.stat-chip b { color: var(--accent3); }

/* ── Explain boxes ──────────────────────────────────────────── */
.explain-box {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.3rem; margin-bottom: .5rem;
    border-left: 3px solid var(--accent); font-size: .88rem;
    line-height: 1.7; color: var(--text); transition: all .2s;
    box-shadow: 0 1px 6px rgba(79,70,229,.07);
}
.explain-box:hover {
    border-left-color: var(--accent2);
    background: linear-gradient(135deg, rgba(79,70,229,.03), rgba(232,69,138,.02));
}
.explain-num {
    font-family: 'Syne', sans-serif; font-size: .65rem; font-weight: 700;
    color: var(--accent); letter-spacing: .1em; margin-bottom: .25rem;
    text-transform: uppercase;
}

/* ── Model badge ────────────────────────────────────────────── */
.model-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 6px;
    font-size: .72rem; font-family: 'Syne', sans-serif;
    font-weight: 700; letter-spacing: .07em;
}
.badge-bart { background:rgba(79,70,229,.12); color:var(--accent); border:1px solid rgba(79,70,229,.3); }
.badge-t5   { background:rgba(5,150,105,.12); color:var(--accent3); border:1px solid rgba(5,150,105,.3); }
.badge-ok   { background:rgba(5,150,105,.12); color:var(--accent3); border:1px solid rgba(5,150,105,.3); }
.badge-err  { background:rgba(232,69,138,.12); color:var(--accent2); border:1px solid rgba(232,69,138,.3); }

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8eaf6 0%, #eceef8 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 16px rgba(79,70,229,.08) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* ── Progress bar ───────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 50px !important;
}
.stProgress > div { background: var(--bg3) !important; border-radius: 50px !important; }

/* ── Expander ───────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #ffffff !important; border-radius: 10px !important;
    border: 1px solid var(--border) !important; font-size: .85rem !important;
    color: var(--text) !important;
}
.streamlit-expanderContent {
    background: #fafbff !important; border: 1px solid var(--border) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}

/* ── Download button ────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #4f46e5, #6d64f5) !important;
    color: #ffffff !important;
    border: 2px solid rgba(255,255,255,0.7) !important;
    border-radius: 10px !important;
    padding: .65rem 1.8rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: .95rem !important;
    letter-spacing: .06em !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
    width: 320px !important;
    min-width: 320px !important;
    max-width: 320px !important;
    display: block !important;
    margin: 0 auto !important;
    box-shadow: 0 4px 14px rgba(79,70,229,.4) !important;
    transition: all .22s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(79,70,229,.55) !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,1.0) !important;
}
[data-testid="stDownloadButton"] > button p,
[data-testid="stDownloadButton"] > button span,
[data-testid="stDownloadButton"] > button div,
[data-testid="stDownloadButton"] > button * {
    color: #ffffff !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
}
div[data-testid="stDownloadButton"] {
    display: flex !important;
    justify-content: center !important;
}

/* ── Misc ───────────────────────────────────────────────────── */
.sec-div {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}
.summary-section {
    background: linear-gradient(135deg, rgba(79,70,229,.04), rgba(232,69,138,.02));
    border: 1px solid rgba(79,70,229,.1);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.keysent-section {
    background: linear-gradient(135deg, rgba(5,150,105,.04), rgba(79,70,229,.02));
    border: 1px solid rgba(5,150,105,.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    border: 1.5px dashed var(--border);
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(79,70,229,.03), rgba(232,69,138,.02));
}
.empty-icon  { font-size: 1.6rem; color: #a5b0d0; margin-bottom: .6rem; }
.empty-title {
    font-family: 'Syne', sans-serif; font-size: .8rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
    color: #9ca8c8; margin-bottom: .4rem;
}
.empty-sub   { font-size: .8rem; line-height: 1.7; color: #9ca8c8; }
.empty-sub b { color: var(--accent); opacity: .7; }

.info-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: var(--shadow);
    height: 100%;
}
.info-card-header {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: .95rem;
    color: var(--accent); margin-bottom: .5rem; display: flex; align-items: center; gap: 8px;
}
.info-card p { font-size: .83rem; color: var(--muted); line-height: 1.7; margin: 0; }
.info-card ul { font-size: .83rem; color: var(--muted); line-height: 1.9;
                padding-left: 1.1rem; margin: .4rem 0 0; }
.info-card .tag {
    display: inline-block; background: rgba(79,70,229,.09);
    color: var(--accent); border-radius: 5px; padding: 1px 8px;
    font-size: .72rem; font-weight: 600; margin-right: 4px; margin-top: 6px;
}
.info-card .tag-green { background: rgba(5,150,105,.09); color: var(--accent3); }
.step-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: .65rem .9rem;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: .5rem;
    box-shadow: 0 1px 5px rgba(79,70,229,.05);
}
.step-num {
    min-width: 28px; height: 28px; border-radius: 50%;
    background: var(--accent); color: #fff;
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: .82rem;
    display: flex; align-items: center; justify-content: center; margin-top: 1px;
}
.step-text { font-size: .85rem; color: var(--text); line-height: 1.6; }
.step-text b { color: var(--accent); }

/* ── Translation banner (simple) ────────────────────────────── */
.trans-banner {
    background: linear-gradient(135deg,rgba(79,70,229,.07),rgba(232,69,138,.04));
    border: 1px solid rgba(79,70,229,.2);
    border-radius: 10px; padding: .7rem 1.1rem;
    font-size: .82rem; color: var(--muted); margin-bottom: .8rem;
    display: flex; align-items: center; gap: 8px;
}
.trans-banner b { color: var(--accent); }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg3); }
::-webkit-scrollbar-thumb { background: #b0b8d8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
for k, v in [
    ("history", []),
    ("last_summary", ""),
    ("last_summary_en", ""),       # always English version of summary
    ("last_input", ""),            # English (translated if needed)
    ("last_input_original", ""),   # raw input as typed
    ("last_translated_en", ""),    # translated-to-english text (for display)
    ("total_runs", 0),
    ("total_reduced", 0),
    ("last_detected_lang", "en"),
    ("last_output_lang", "en"),
    ("did_translate_in", False),
    ("did_translate_out", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return load_bart(), load_t5()

with st.spinner("Loading AI models..."):
    (bart_tok, bart_mod), (t5_tok, t5_mod) = load_models()


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:.5rem 0 1.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.5rem; font-weight:800;
                    background:linear-gradient(135deg,#4f46e5,#e8458a);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            ✦ SummarAI
        </div>
        <div style='font-size:.7rem; color:#6b7280; letter-spacing:.12em;
                    text-transform:uppercase; margin-top:3px;'>
            Intelligent Summarizer
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title">Model Status</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ok = bart_mod is not None
        st.markdown(
            f'<div class="model-badge {"badge-ok" if ok else "badge-err"}">{"✓" if ok else "✗"} BART</div>',
            unsafe_allow_html=True)
    with c2:
        ok = t5_mod is not None
        st.markdown(
            f'<div class="model-badge {"badge-ok" if ok else "badge-err"}">{"✓" if ok else "✗"} T5</div>',
            unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#c8cde8; margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚙ Settings</div>', unsafe_allow_html=True)
    model_choice  = st.selectbox("AI Model", ["BART", "T5"],
                                  help="BART: Better for news & articles. T5: General text.")
    length_choice = st.select_slider("Summary Length",
                                      options=["Short", "Medium", "Detailed"],
                                      value="Medium")
    show_explain  = st.toggle("Show Key Sentences", value=True)
    show_history  = st.toggle("Show History Panel", value=True)

    st.markdown("<hr style='border-color:#c8cde8; margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌐 Translation</div>', unsafe_allow_html=True)

    if translation_available():
        auto_detect = st.toggle("Auto-detect Input Language", value=True)
        lang_names  = list(SUPPORTED_LANGUAGES.keys())
        output_lang = st.selectbox(
            "Output Summary Language",
            lang_names,
            index=0,
            help="The language your summary will be returned in."
        )
        st.markdown(
            '<div style="font-size:.74rem;color:#9ca3af;margin-top:.3rem;">'
            'Detects non-English input, translates to English for summarization, '
            'then delivers result in your chosen language.</div>',
            unsafe_allow_html=True
        )
    else:
        auto_detect = False
        output_lang = "English"
        st.markdown(
            '<div style="font-size:.78rem;color:#e8458a;">⚠ Translation unavailable.<br>'
            'Install: <code>deep-translator langdetect</code></div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:#c8cde8; margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Session Stats</div>', unsafe_allow_html=True)
    avg = round(st.session_state.total_reduced / st.session_state.total_runs) \
          if st.session_state.total_runs > 0 else 0
    st.markdown(f"""
    <div style='display:flex; flex-direction:column; gap:7px;'>
        <div class='stat-chip'>Summaries: <b>{st.session_state.total_runs}</b></div>
        <div class='stat-chip'>Avg reduction: <b>{avg}%</b></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#c8cde8; margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">📖 Length Guide</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.78rem; color:#5a6080; line-height:2;'>
        <b style='color:#1a1d2e'>Short</b> — 40–80 words<br>
        Extractive · Key sentences only<br><br>
        <b style='color:#1a1d2e'>Medium</b> — 70–130 words<br>
        Abstractive · AI rewrites content<br><br>
        <b style='color:#1a1d2e'>Detailed</b> — 130–220 words<br>
        Abstractive + Extractive hybrid
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI-Powered · Multilingual · Explainable · Multi-Model</div>
    <div class="hero-title">Summarize Any Text,<br>Instantly</div>
    <div class="hero-sub">
        Powered by BART &amp; T5 transformers. Paste text or upload a PDF —
        get clean, accurate summaries in seconds. Supports 8 languages.
    </div>
    <div class="hero-divider"></div>
    <div class="stats-row">
        <div class="stat-pill">⚡ <span>Fast</span> Single-pass</div>
        <div class="stat-pill">🧠 <span>2</span> AI Models</div>
        <div class="stat-pill">🌐 <span>8</span> Languages</div>
        <div class="stat-pill">📄 <span>Any</span> Document type</div>
        <div class="stat-pill">🔍 <span>Explainable</span> Output</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ABOUT THE MODELS (expander)
# ══════════════════════════════════════════════════════════════
with st.expander("🧠  About the AI Models & How to Use", expanded=False):
    st.markdown("""
    <div style='padding:.4rem 0 .2rem;'>
        <div class='card-title'>Understanding the Models</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-header'>🟣 BART — Best for Articles &amp; Reports</div>
            <p><b>BART</b> (Bidirectional and Auto-Regressive Transformer) by Meta AI.
            Reads the full text in both directions to deeply understand context,
            then rewrites it into a clean, fluent summary.</p>
            <ul>
                <li>Best for: news articles, research papers, reports</li>
                <li>Produces natural, well-structured sentences</li>
                <li>Fine-tuned on CNN/DailyMail dataset</li>
            </ul>
            <div style='margin-top:.6rem;'>
                <span class='tag'>facebook/bart-base</span>
                <span class='tag'>140M parameters</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-header' style='color:#059669;'>🟢 T5 — Best for General Text</div>
            <p><b>T5</b> (Text-to-Text Transfer Transformer) by Google. Treats every NLP
            task as a text-to-text problem. Converts input into a shorter output text.</p>
            <ul>
                <li>Best for: general text, emails, blog posts</li>
                <li>Flexible and fast on CPU</li>
                <li>Fine-tuned on CNN/DailyMail dataset</li>
            </ul>
            <div style='margin-top:.6rem;'>
                <span class='tag tag-green'>google/t5-small</span>
                <span class='tag tag-green'>60M parameters</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>How to Use SummarAI</div>", unsafe_allow_html=True)
    steps = [
        ("1", "Choose your <b>AI Model</b> (BART or T5) from the sidebar."),
        ("2", "Select your <b>Summary Length</b> — Short, Medium, or Detailed."),
        ("3", "Toggle <b>Auto-detect Input Language</b> ON for non-English text."),
        ("4", "Select your desired <b>Output Language</b> from the dropdown."),
        ("5", "Paste your text in <b>Paste Text</b> tab, or switch to <b>Upload PDF</b>."),
        ("6", "Click <b>✦ Generate Summary</b>. Translation pipeline runs automatically."),
        ("7", "If non-English input: view <b>Translated English Text</b> + both summaries."),
        ("8", "Click <b>⬇ Download Summary</b> to save the result as a .txt file."),
    ]
    for num, text in steps:
        st.markdown(f"""
        <div class='step-row'>
            <div class='step-num'>{num}</div>
            <div class='step-text'>{text}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(79,70,229,.05); border:1px solid rgba(79,70,229,.15);
                border-radius:12px; padding:1rem 1.3rem; margin-top:1rem;'>
        <div style='font-family:Syne,sans-serif; font-size:.75rem; font-weight:700;
                    color:var(--accent); letter-spacing:.1em; margin-bottom:.6rem;'>
            💡 TIPS FOR BEST RESULTS
        </div>
        <div style='font-size:.82rem; color:var(--muted); line-height:1.9;'>
            ✦ &nbsp;For <b>research papers</b>, use BART + Detailed mode<br>
            ✦ &nbsp;For <b>news articles</b>, BART + Medium gives the cleanest output<br>
            ✦ &nbsp;For <b>emails or blog posts</b>, T5 + Short is fastest<br>
            ✦ &nbsp;For <b>Tamil / Hindi text</b>, enable Auto-detect + set Output Language<br>
            ✦ &nbsp;Toggle <b>Key Sentences OFF</b> in sidebar for a cleaner view
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ① INPUT
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
st.markdown('<div class="card-title">📥 Input</div>', unsafe_allow_html=True)

input_text = ""

st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
tab_text, tab_pdf = st.tabs(["✏  Paste Text", "📄  Upload PDF"])

with tab_text:
    txt = st.text_area(
        "Input text", height=300,
        placeholder="Paste your article, report, research paper or any text here... (any language supported)",
        label_visibility="hidden"
    )
    if txt.strip():
        input_text = txt
        wc = len(txt.split())
        st.markdown(
            f'<div style="font-size:.74rem;color:#9ca3af;text-align:right;margin-top:.3rem">'
            f'{wc:,} words · {len(txt):,} characters</div>',
            unsafe_allow_html=True
        )

with tab_pdf:
    # Improved PDF upload area
    st.markdown("""
    <div style='text-align:center; padding:.4rem 0 .2rem;'>
        <div style='font-size:2rem; margin-bottom:.3rem;'>📄</div>
        <div style='font-family:Syne,sans-serif; font-size:.78rem; font-weight:700;
                    color:var(--muted); letter-spacing:.08em; margin-bottom:.1rem;'>
            DROP YOUR PDF HERE
        </div>
        <div style='font-size:.72rem; color:#9ca3af;'>or click Browse to select a file</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF document",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload any PDF document — articles, reports, research papers, books."
    )

    st.markdown("""
    <div class='pdf-upload-hint'>
        ✦ Supports text-based PDFs up to any size &nbsp;·&nbsp;
        Multi-page documents supported &nbsp;·&nbsp;
        Max recommended: 50 pages for best speed
    </div>
    """, unsafe_allow_html=True)

    if uploaded:
        try:
            file_bytes = uploaded.read()
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages    = [p.extract_text() for p in pdf.pages if p.extract_text()]
                pdf_text = "\n".join(pages)

            if not pdf_text.strip():
                st.error("⚠ This PDF appears to be scanned/image-based. Only text-based PDFs are supported.")
            else:
                input_text = pdf_text
                word_count = len(pdf_text.split())
                char_count = len(pdf_text)

                # Success banner
                st.markdown(f"""
                <div style='background:rgba(5,150,105,.08);border:1.5px solid rgba(5,150,105,.3);
                            border-radius:12px;padding:1rem 1.2rem;margin-top:.8rem;'>
                    <div style='display:flex; align-items:center; gap:10px; flex-wrap:wrap;'>
                        <div style='font-size:1.4rem;'>✅</div>
                        <div>
                            <div style='font-family:Syne,sans-serif; font-weight:700;
                                        font-size:.85rem; color:#059669;'>
                                {uploaded.name}
                            </div>
                            <div style='font-size:.75rem; color:#6b7280; margin-top:2px;'>
                                {len(pages)} page{"s" if len(pages) != 1 else ""} &nbsp;·&nbsp;
                                {word_count:,} words &nbsp;·&nbsp;
                                {char_count:,} characters &nbsp;·&nbsp;
                                {round(len(file_bytes)/1024, 1)} KB
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Preview
                preview_text = pdf_text[:600].replace("\n", " ")
                with st.expander("👁  Preview extracted text (first 600 chars)", expanded=False):
                    st.markdown(
                        f'<div style="font-size:.82rem;color:#5a6080;line-height:1.7;'
                        f'background:#fafbff;border-radius:8px;padding:1rem;">'
                        f'{preview_text}...'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"⚠ Could not read PDF: {e}")
            st.markdown(
                '<div style="font-size:.78rem;color:#9ca3af;margin-top:.4rem;">'
                'Tip: Make sure the file is a valid, non-password-protected PDF.</div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)

# ── Buttons ───────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stButton"] { display: flex !important; justify-content: center !important; }
</style>
""", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_btn = st.button("✦  Generate Summary", use_container_width=True)
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    clr_btn = st.button("✕  Clear", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if clr_btn:
    st.session_state.last_summary       = ""
    st.session_state.last_summary_en    = ""
    st.session_state.last_input         = ""
    st.session_state.last_input_original= ""
    st.session_state.last_translated_en = ""
    st.session_state.last_detected_lang = "en"
    st.session_state.last_output_lang   = "en"
    st.session_state.did_translate_in   = False
    st.session_state.did_translate_out  = False
    st.rerun()


# ══════════════════════════════════════════════════════════════
#  RUN SUMMARIZATION  (full translation pipeline)
# ══════════════════════════════════════════════════════════════
if run_btn:
    if not input_text or not input_text.strip():
        st.warning("Please enter or upload some text first.")
    else:
        tok = bart_tok if model_choice == "BART" else t5_tok
        mod = bart_mod if model_choice == "BART" else t5_mod
        if mod is None:
            st.error(f"{model_choice} model failed to load. Please restart.")
        else:
            prog = st.progress(0)

            # ── Step 1: Language detection & translate input → English ──
            detected_lang    = "en"
            translated_in    = input_text
            did_translate_in = False

            if translation_available() and auto_detect:
                with st.spinner("🔍 Detecting language..."):
                    detected_lang = detect_language(input_text)
                    lang_name     = get_language_name(detected_lang)

                if detected_lang != "en":
                    with st.spinner(f"🌐 Translating {lang_name} → English..."):
                        translated_in, did_translate_in = translate_to_english(
                            input_text, detected_lang
                        )
            prog.progress(20)

            # ── Step 2: Summarise in English ───────────────────────────
            with st.spinner(f"🧠 Summarizing with {model_choice}..."):
                for i in range(20, 70):
                    time.sleep(0.003)
                    prog.progress(i)
                summary_en = generate_summary(
                    translated_in, tok, mod, model_choice, length_choice
                )
            prog.progress(70)

            # ── Step 3: Translate output → chosen language ─────────────
            out_lang_code     = SUPPORTED_LANGUAGES.get(output_lang, "en")
            final_summary     = summary_en
            did_translate_out = False

            if translation_available() and out_lang_code != "en":
                with st.spinner(f"🌐 Translating summary → {output_lang}..."):
                    final_summary     = translate_from_english(summary_en, out_lang_code)
                    did_translate_out = True

            for i in range(70, 100):
                time.sleep(0.002)
                prog.progress(i)
            prog.empty()

            # ── Update state ───────────────────────────────────────────
            in_wc  = len(input_text.split())
            out_wc = len(final_summary.split())
            pct    = round((1 - out_wc / max(in_wc, 1)) * 100)

            st.session_state.last_summary        = final_summary
            st.session_state.last_summary_en     = summary_en
            st.session_state.last_input          = translated_in
            st.session_state.last_input_original = input_text
            st.session_state.last_translated_en  = translated_in if did_translate_in else ""
            st.session_state.last_detected_lang  = detected_lang
            st.session_state.last_output_lang    = out_lang_code
            st.session_state.did_translate_in    = did_translate_in
            st.session_state.did_translate_out   = did_translate_out
            st.session_state.total_runs         += 1
            st.session_state.total_reduced      += pct
            st.session_state.history.insert(0, {
                "model":    model_choice,
                "length":   length_choice,
                "in_wc":    in_wc,
                "out_wc":   out_wc,
                "pct":      pct,
                "full":     final_summary,
                "full_en":  summary_en,
                "lang_in":  get_language_name(detected_lang),
                "lang_out": output_lang,
            })
            if len(st.session_state.history) > 6:
                st.session_state.history.pop()
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  ② SUMMARY OUTPUT
# ══════════════════════════════════════════════════════════════
if st.session_state.last_summary:

    summary          = st.session_state.last_summary
    summary_en       = st.session_state.last_summary_en
    detected_lang    = st.session_state.last_detected_lang
    out_lang_code    = st.session_state.last_output_lang
    did_translate_in = st.session_state.did_translate_in
    did_translate_out= st.session_state.did_translate_out
    translated_en_text = st.session_state.last_translated_en

    in_wc  = len(st.session_state.last_input_original.split())
    out_wc = len(summary.split())
    pct    = round((1 - out_wc / max(in_wc, 1)) * 100)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TRANSLATION PIPELINE (shown only when non-English involved)
    # ──────────────────────────────────────────────────────────
    is_multilingual = did_translate_in or did_translate_out

    if is_multilingual:
        lang_in_name  = get_language_name(detected_lang)
        lang_out_name = get_language_name(out_lang_code)

        # Pipeline flow indicator
        steps_html = f'<span class="pipe-step">📥 {lang_in_name} Input</span>'
        if did_translate_in:
            steps_html += '<span class="pipe-arrow">→</span><span class="pipe-step">🇬🇧 Translate to English</span>'
        steps_html += '<span class="pipe-arrow">→</span><span class="pipe-step">🧠 AI Summarize</span>'
        if did_translate_out:
            steps_html += f'<span class="pipe-arrow">→</span><span class="pipe-step">🌐 Translate to {lang_out_name}</span>'

        st.markdown(f'<div class="trans-pipeline-bar">{steps_html}</div>',
                    unsafe_allow_html=True)

        # ── Box A: Original Input Language (collapsed preview) ──
        if did_translate_in:
            orig_preview = st.session_state.last_input_original[:500]
            with st.expander(f"📥 Original Input ({lang_in_name}) — click to expand", expanded=False):
                st.markdown(
                    f'<div class="trans-text" style="padding:.3rem 0">{orig_preview}'
                    f'{"..." if len(st.session_state.last_input_original) > 500 else ""}</div>',
                    unsafe_allow_html=True
                )

            # ── Box B: Translated English Text ──────────────────
            st.markdown("""
            <div class="trans-input-box" style="
                background:linear-gradient(135deg,rgba(5,150,105,.06),rgba(79,70,229,.03));
                border-color:rgba(5,150,105,.3);">
                <div class="trans-label trans-label-en">
                    🇬🇧 Translated to English (input sent to AI)
                </div>
            """, unsafe_allow_html=True)
            # Show first 400 words of translated English
            en_words = translated_en_text.split()
            en_preview = " ".join(en_words[:400])
            if len(en_words) > 400:
                en_preview += "..."
            st.markdown(
                f'<div class="trans-text">{en_preview}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin-bottom:.5rem'></div>", unsafe_allow_html=True)

        # ── Box C: English Summary (always shown in multilingual mode) ──
        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-title">📤 Summary in English</div>',
            unsafe_allow_html=True
        )
        badge = "badge-bart" if model_choice == "BART" else "badge-t5"
        st.markdown(
            f'<div class="model-badge {badge}" style="margin-bottom:.7rem">'
            f'✦ {model_choice} · {length_choice} · English</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="result-box"><div class="result-text">{summary_en}</div></div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Box D: Final Output in selected language ─────────────
        if did_translate_out and out_lang_code != "en":
            st.markdown('<div class="summary-section" style="border-color:rgba(232,69,138,.2);">',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="card-title" style="color:var(--accent2);">'
                f'🌐 Summary in {lang_out_name}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="model-badge badge-bart" style="margin-bottom:.7rem;'
                f'background:rgba(232,69,138,.12);color:var(--accent2);'
                f'border-color:rgba(232,69,138,.3);">'
                f'✦ {model_choice} · {length_choice} · {lang_out_name}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="result-box" style="background:linear-gradient(135deg,'
                f'rgba(232,69,138,.07),rgba(79,70,229,.04));'
                f'border-color:rgba(232,69,138,.25);">'
                f'<div class="result-text">{summary}</div></div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="stats-bar">
                <div class="stat-chip">📥 Input <b>{in_wc:,}</b> words ({lang_in_name})</div>
                <div class="stat-chip">📤 Output <b>{out_wc}</b> words ({lang_out_name})</div>
                <div class="stat-chip">📉 Reduced by <b>{pct}%</b></div>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Download: offer both languages
        st.markdown("<div style='margin-top:.8rem'>", unsafe_allow_html=True)
        dl_col1, dl_col2 = st.columns(2, gap="medium")
        with dl_col1:
            st.download_button(
                "⬇  Download English Summary",
                data=summary_en,
                file_name="summary_english.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_en"
            )
        with dl_col2:
            if did_translate_out and out_lang_code != "en":
                st.download_button(
                    f"⬇  Download {lang_out_name} Summary",
                    data=summary,
                    file_name=f"summary_{out_lang_code}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_out"
                )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ── Standard (English-only) output ──────────────────────
        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📤 Summary</div>', unsafe_allow_html=True)

        badge = "badge-bart" if model_choice == "BART" else "badge-t5"
        st.markdown(
            f'<div class="model-badge {badge}" style="margin-bottom:.7rem">'
            f'✦ {model_choice} · {length_choice}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="result-box"><div class="result-text">{summary}</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-chip">📥 Input <b>{in_wc:,}</b> words</div>
            <div class="stat-chip">📤 Output <b>{out_wc}</b> words</div>
            <div class="stat-chip">📉 Reduced by <b>{pct}%</b></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:.8rem'>", unsafe_allow_html=True)
        _, dl_col, _ = st.columns([1, 2, 1])
        with dl_col:
            st.download_button(
                "⬇  Download Summary (.txt)",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Key Sentences (always from English source) ────────────
    if show_explain:
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="keysent-section">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Key Source Sentences</div>',
                    unsafe_allow_html=True)
        try:
            # Use English input for key sentence extraction
            english_input = st.session_state.last_input
            key_sents = get_important_sentences(english_input, summary_en, top_n=3)
            if key_sents:
                for i, s in enumerate(key_sents, 1):
                    st.markdown(
                        f'<div class="explain-box">'
                        f'<div class="explain-num">Key Sentence {i}</div>{s}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No key sentences found.")
        except Exception:
            pass
        st.markdown('</div>', unsafe_allow_html=True)

    # ── History ───────────────────────────────────────────────
    if show_history and st.session_state.history:
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🕘 Recent Summaries</div>',
                    unsafe_allow_html=True)
        for i, h in enumerate(st.session_state.history):
            lang_tag = ""
            if h.get("lang_in", "English") != "English" or h.get("lang_out", "English") != "English":
                lang_tag = f" · {h.get('lang_in','EN')} → {h.get('lang_out','EN')}"
            with st.expander(
                f"#{i+1}  {h['model']} · {h['length']} · "
                f"{h['in_wc']:,} → {h['out_wc']} words  ({h['pct']}% reduced){lang_tag}"
            ):
                if lang_tag and h.get("full_en") and h["full_en"] != h["full"]:
                    st.markdown(
                        f'<div style="font-size:.78rem;color:#9ca3af;margin-bottom:.3rem">'
                        f'English:</div>'
                        f'<div style="font-size:.88rem;line-height:1.75;color:#1a1d2e;'
                        f'padding:.2rem 0 .8rem">{h["full_en"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="font-size:.78rem;color:#9ca3af;margin-bottom:.3rem">'
                        f'{h.get("lang_out","Output")}:</div>',
                        unsafe_allow_html=True
                    )
                st.markdown(
                    f'<div style="font-size:.88rem;line-height:1.75;color:#1a1d2e;padding:.2rem 0">'
                    f'{h["full"]}</div>', unsafe_allow_html=True
                )
                dl_c1, dl_c2 = st.columns(2)
                with dl_c1:
                    st.download_button(
                        "⬇ Download", data=h["full"],
                        file_name=f"summary_{i+1}.txt", key=f"dl_{i}"
                    )
                if lang_tag and h.get("full_en") and h["full_en"] != h["full"]:
                    with dl_c2:
                        st.download_button(
                            "⬇ English", data=h["full_en"],
                            file_name=f"summary_{i+1}_en.txt", key=f"dl_en_{i}"
                        )

else:
    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">✦</div>
        <div class="empty-title">Ready to Summarize</div>
        <div class="empty-sub">
            Paste your text or upload a PDF above,<br>
            then click <b>Generate Summary</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-top:4rem; padding:2rem 0 1rem;
            border-top:1px solid #c8cde8;'>
    <div style='font-family:Syne,sans-serif; font-size:.68rem; font-weight:700;
                letter-spacing:.15em; text-transform:uppercase; color:#a0a8c8;'>
        SummarAI · BART + T5 Transformers · Multilingual · Built with Streamlit
    </div>
</div>
""", unsafe_allow_html=True)