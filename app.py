# app.py — SummarAI Final Production Version
# ─────────────────────────────────────────────────────────────────────────────
# Fixes:
#   1. Sidebar collapse issue → ALL controls moved to main area sticky top bar
#   2. Sidebar now only shows branding + status (cosmetic, not critical)
#   3. PDF text cleaning pipeline (_clean_pdf_text) included
#   4. Speed fixes from summarizer.py flow through correctly
#   5. Translation pipeline: shows English translation → English summary →
#      native-language summary when non-English is selected
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pdfplumber
import time
import io
import re
from summarizer import load_bart, load_t5, generate_summary, clean_input
from explainability import get_important_sentences
from translator import (
    SUPPORTED_LANGUAGES,
    translate_to_english,
    translate_from_english,
    is_available,
)

st.set_page_config(
    page_title="SummarAI — AI Text Summarizer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",   # ← collapsed by default; controls are in main area
)


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def _clean_pdf_text(text: str) -> str:
    """7-step pipeline to fix common PDF extraction artifacts."""
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)        # 1. lone page numbers
    text = re.sub(r'\f', '\n', text)                     # 1. form feeds
    text = re.sub(r'-(\s*\n\s*)', '', text)              # 2. hyphenated line breaks

    lines = text.split('\n'); joined = []; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: joined.append(''); i += 1; continue
        while i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if not nxt: break
            if nxt == nxt.upper() and len(nxt.split()) <= 6: break
            ends_mid   = line and line[-1] not in '.!?:' and len(line) > 20
            starts_low = nxt and nxt[0].islower()
            if ends_mid or starts_low: line = line + ' ' + nxt; i += 1
            else: break
        joined.append(line); i += 1
    text = '\n'.join(joined)

    text = re.sub(r',([^\s\d])', r', \1', text)
    text = re.sub(r'\.([A-Za-z])', r'. \1', text)
    text = re.sub(r',\s+([A-Z][a-z])', lambda m: '. ' + m.group(1), text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:       #eef0f9;
    --bg2:      #ffffff;
    --bg3:      #e4e7f5;
    --border:   #c8cde8;
    --accent:   #4f46e5;
    --accent2:  #e8458a;
    --accent3:  #059669;
    --text:     #1a1d2e;
    --muted:    #5a6080;
    --shadow:   0 2px 14px rgba(79,70,229,0.10);
    --shadow2:  0 4px 24px rgba(79,70,229,0.16);
}

/* ── Force light mode everywhere ───────────────────────────────────────── */
html,body,.stApp,.stApp>div,
[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],[data-testid="stHorizontalBlock"],
.main,.block-container,section.main,[class*="css"]{
    background-color:#eef0f9!important;color:#1a1d2e!important;color-scheme:light!important;}
[data-theme="dark"],[data-theme="dark"] body,[data-theme="dark"] .stApp{
    background-color:#eef0f9!important;color:#1a1d2e!important;color-scheme:light!important;}
@media(prefers-color-scheme:dark){
    html,body,.stApp,[data-testid="stAppViewContainer"],.main,.block-container{
        background-color:#eef0f9!important;color:#1a1d2e!important;color-scheme:light!important;}}
p,span,label,div,h1,h2,h3,h4,h5,h6,
[data-testid="stMarkdownContainer"],[data-testid="stText"],
.stSelectbox label,.stToggle label,.stSlider label{color:#1a1d2e!important;}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.main .block-container{padding:1.5rem 2rem 4rem;max-width:1180px;}
#MainMenu,footer,header{visibility:hidden;}.stDeployButton{display:none;}

/* ── Hero ───────────────────────────────────────────────────────────────── */
.hero{text-align:center;padding:2.2rem 1rem 1.4rem;
    background:linear-gradient(135deg,rgba(79,70,229,.07),rgba(232,69,138,.05));
    border-radius:16px;border:1px solid rgba(79,70,229,.12);margin-bottom:1.4rem;}
.hero-badge{display:inline-block;
    background:linear-gradient(135deg,rgba(79,70,229,.15),rgba(232,69,138,.10));
    border:1px solid rgba(79,70,229,.3);border-radius:50px;padding:4px 14px;
    font-size:.68rem;font-family:'Syne',sans-serif;letter-spacing:.12em;
    text-transform:uppercase;color:var(--accent);margin-bottom:.8rem;}
.hero-title{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;line-height:1.15;
    background:linear-gradient(135deg,#1a1d2e 20%,var(--accent) 60%,var(--accent2) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin-bottom:.6rem;}
.hero-sub{font-size:.92rem;color:var(--muted);max-width:460px;margin:0 auto 1.2rem;line-height:1.7;}
.hero-divider{height:1px;
    background:linear-gradient(90deg,transparent,rgba(79,70,229,.25),transparent);
    margin:0 auto 1.2rem;max-width:400px;}
.stats-row{display:flex;justify-content:center;gap:.6rem;flex-wrap:wrap;}
.stat-pill{background:rgba(255,255,255,0.7);border:1px solid rgba(79,70,229,.18);
    border-radius:50px;padding:5px 14px;font-size:.74rem;color:var(--muted);
    display:flex;align-items:center;gap:4px;box-shadow:0 2px 6px rgba(79,70,229,.07);}
.stat-pill span{color:var(--accent);font-weight:600;}

/* ── Settings bar (replaces sidebar) ────────────────────────────────────── */
.settings-bar{
    background:#ffffff;
    border:1.5px solid var(--border);
    border-radius:14px;
    padding:1rem 1.4rem;
    box-shadow:var(--shadow);
    margin-bottom:1.2rem;
    display:flex;
    flex-wrap:wrap;
    gap:1rem;
    align-items:flex-end;
}
.settings-bar-title{
    font-family:'Syne',sans-serif;font-size:.62rem;font-weight:700;
    letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
    margin-bottom:.35rem;
}
.settings-group{flex:1;min-width:130px;}

/* Model status pills inside settings bar */
.status-row{display:flex;gap:.5rem;margin-top:.2rem;}
.status-pill{display:inline-flex;align-items:center;gap:4px;
    padding:3px 9px;border-radius:6px;font-size:.70rem;
    font-family:'Syne',sans-serif;font-weight:700;letter-spacing:.05em;}
.sp-ok{background:rgba(5,150,105,.12);color:#059669;border:1px solid rgba(5,150,105,.3);}
.sp-err{background:rgba(232,69,138,.12);color:#e8458a;border:1px solid rgba(232,69,138,.3);}

/* ── Section title ───────────────────────────────────────────────────────── */
.card-title{font-family:'Syne',sans-serif;font-size:.65rem;font-weight:700;
    letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
    margin-bottom:.7rem;display:flex;align-items:center;gap:6px;}
.card-title::before{content:'';display:inline-block;width:3px;height:11px;
    background:var(--accent);border-radius:2px;}

/* ── Input wrapper ───────────────────────────────────────────────────────── */
.input-wrapper{background:#ffffff;border:1.5px solid var(--border);
    border-radius:14px;padding:1.1rem 1.3rem 1rem;
    box-shadow:var(--shadow);margin-bottom:.8rem;}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{background:var(--bg3)!important;
    border-radius:9px!important;gap:4px!important;padding:4px!important;
    border:1px solid var(--border)!important;width:100%!important;margin-bottom:.2rem!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:7px!important;
    color:var(--muted)!important;font-family:'Syne',sans-serif!important;font-size:.82rem!important;
    font-weight:600!important;letter-spacing:.04em!important;flex:1!important;
    text-align:center!important;justify-content:center!important;padding:.4rem .8rem!important;}
.stTabs [data-baseweb="tab"] p,.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] div{color:var(--muted)!important;}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:white!important;
    box-shadow:0 2px 8px rgba(79,70,229,.3)!important;}
.stTabs [aria-selected="true"] p,.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div{color:white!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:.8rem!important;}

/* ── Textarea ────────────────────────────────────────────────────────────── */
.stTextArea textarea{background:#fafbff!important;border:1.5px solid var(--border)!important;
    border-radius:10px!important;color:var(--text)!important;
    font-family:'Inter',sans-serif!important;font-size:.9rem!important;
    line-height:1.75!important;padding:.9rem 1rem!important;
    box-shadow:none!important;transition:border-color .22s!important;resize:none!important;}
.stTextArea textarea:focus{border-color:var(--accent)!important;
    box-shadow:0 0 0 3px rgba(79,70,229,.1)!important;outline:none!important;}
.stTextArea textarea::placeholder{color:#b0b7c5!important;}
[data-baseweb="textarea"],[data-baseweb="base-input"]{border:none!important;box-shadow:none!important;}

/* ── Selectbox ───────────────────────────────────────────────────────────── */
.stSelectbox>div>div{background:#ffffff!important;border:1.5px solid var(--border)!important;
    border-radius:10px!important;color:var(--text)!important;box-shadow:var(--shadow)!important;}

/* ── Toggle ──────────────────────────────────────────────────────────────── */
.stToggle{margin-top:.2rem;}

/* ── Main buttons ────────────────────────────────────────────────────────── */
.stButton>button{
    background:linear-gradient(135deg,var(--accent),#6d64f5)!important;
    color:#ffffff!important;border:none!important;border-radius:10px!important;
    padding:.6rem 1.6rem!important;font-family:'Syne',sans-serif!important;
    font-weight:700!important;font-size:.85rem!important;letter-spacing:.04em!important;
    width:260px!important;min-width:260px!important;max-width:260px!important;
    display:block!important;margin:0 auto!important;transition:all .22s!important;
    box-shadow:0 4px 14px rgba(79,70,229,.3)!important;}
.stButton>button:hover{transform:translateY(-2px)!important;
    box-shadow:0 6px 22px rgba(79,70,229,.45)!important;color:#ffffff!important;}
.stButton>button p,.stButton>button span,.stButton>button div{color:#ffffff!important;}

.btn-clear .stButton>button{
    background:#fff0f0!important;color:#ef4444!important;
    border:1.5px solid #fca5a5!important;box-shadow:none!important;margin-top:.5rem!important;
    width:260px!important;min-width:260px!important;max-width:260px!important;}
.btn-clear .stButton>button:hover{background:#ffe4e4!important;
    border-color:#ef4444!important;color:#ef4444!important;
    transform:none!important;box-shadow:none!important;}
.btn-clear .stButton>button p,.btn-clear .stButton>button span,
.btn-clear .stButton>button div{color:#ef4444!important;}
div[data-testid="stColumn"] div[data-testid="stButton"],
div[data-testid="stButton"]{display:flex!important;justify-content:center!important;}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"]>section{
    background:#fafbff!important;border:2px dashed var(--border)!important;
    border-radius:12px!important;padding:1.2rem!important;
    text-align:center!important;transition:border-color .22s,background .22s!important;}
[data-testid="stFileUploader"]>section:hover{
    border-color:var(--accent)!important;background:rgba(79,70,229,.03)!important;}
[data-testid="stFileUploader"] button{
    background:var(--accent)!important;color:#fff!important;
    border:none!important;border-radius:8px!important;
    padding:.38rem 1.1rem!important;font-family:'Syne',sans-serif!important;
    font-weight:700!important;font-size:.78rem!important;margin-top:.3rem!important;
    box-shadow:0 2px 8px rgba(79,70,229,.25)!important;}
[data-testid="stFileUploader"] button:hover{background:#3730d3!important;}
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button div{color:#fff!important;}

/* ── Result & translation boxes ─────────────────────────────────────────── */
.result-box{background:linear-gradient(135deg,rgba(79,70,229,.06),rgba(5,150,105,.04));
    border:1.5px solid rgba(79,70,229,.2);border-radius:14px;padding:1.4rem;
    margin:.6rem 0;position:relative;overflow:hidden;box-shadow:var(--shadow2);}
.result-box::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent3));}
.result-text{font-size:.94rem;line-height:1.85;color:var(--text);}

.trans-en-box{background:linear-gradient(135deg,rgba(5,150,105,.05),rgba(79,70,229,.03));
    border:1.5px solid rgba(5,150,105,.25);border-radius:14px;padding:1.2rem 1.4rem;
    margin:.6rem 0;position:relative;overflow:hidden;}
.trans-en-box::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,var(--accent3),var(--accent));}
.trans-native-box{background:linear-gradient(135deg,rgba(232,69,138,.06),rgba(79,70,229,.03));
    border:1.5px solid rgba(232,69,138,.25);border-radius:14px;padding:1.2rem 1.4rem;
    margin:.6rem 0;position:relative;overflow:hidden;}
.trans-native-box::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,var(--accent2),var(--accent));}
.trans-label{font-family:'Syne',sans-serif;font-size:.62rem;font-weight:700;
    letter-spacing:.12em;text-transform:uppercase;margin-bottom:.5rem;}
.trans-text{font-size:.88rem;line-height:1.8;color:var(--text);}

/* Pipeline bar */
.pipeline-bar{display:flex;align-items:center;justify-content:center;
    gap:.5rem;flex-wrap:wrap;
    background:rgba(79,70,229,.04);border:1px solid rgba(79,70,229,.15);
    border-radius:10px;padding:.55rem 1rem;margin-bottom:.9rem;
    font-size:.75rem;color:var(--muted);}
.pipe-step{background:#fff;border:1px solid var(--border);border-radius:6px;
    padding:2px 9px;font-family:'Syne',sans-serif;font-weight:700;font-size:.68rem;}
.pipe-arrow{color:var(--accent);font-size:.85rem;}

/* ── Stats bar ───────────────────────────────────────────────────────────── */
.stats-bar{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:.7rem;}
.stat-chip{background:#ffffff;border:1px solid var(--border);border-radius:8px;
    padding:4px 11px;font-size:.73rem;color:var(--muted);
    box-shadow:0 1px 4px rgba(79,70,229,.06);}
.stat-chip b{color:var(--accent3);}

/* ── Key sentences ───────────────────────────────────────────────────────── */
.explain-box{background:#ffffff;border:1px solid var(--border);border-radius:10px;
    padding:.9rem 1.2rem;margin-bottom:.5rem;border-left:3px solid var(--accent);
    font-size:.86rem;line-height:1.7;color:var(--text);transition:all .2s;
    box-shadow:0 1px 5px rgba(79,70,229,.07);}
.explain-box:hover{border-left-color:var(--accent2);
    background:linear-gradient(135deg,rgba(79,70,229,.03),rgba(232,69,138,.02));}
.explain-num{font-family:'Syne',sans-serif;font-size:.62rem;font-weight:700;
    color:var(--accent);letter-spacing:.1em;margin-bottom:.2rem;text-transform:uppercase;}

/* ── Progress ────────────────────────────────────────────────────────────── */
.stProgress>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important;
    border-radius:50px!important;}
.stProgress>div{background:var(--bg3)!important;border-radius:50px!important;}

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader{background:#ffffff!important;border-radius:10px!important;
    border:1px solid var(--border)!important;font-size:.83rem!important;color:var(--text)!important;}
.streamlit-expanderContent{background:#fafbff!important;border:1px solid var(--border)!important;
    border-top:none!important;border-radius:0 0 10px 10px!important;}

/* ── Download button ─────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"]>button{
    background:linear-gradient(135deg,#4f46e5,#6d64f5)!important;color:#ffffff!important;
    border:2px solid rgba(255,255,255,0.7)!important;border-radius:10px!important;
    padding:.6rem 1.6rem!important;font-family:'Syne',sans-serif!important;
    font-weight:800!important;font-size:.9rem!important;letter-spacing:.05em!important;
    text-shadow:0 1px 4px rgba(0,0,0,0.3)!important;
    width:300px!important;min-width:300px!important;max-width:300px!important;
    display:block!important;margin:0 auto!important;
    box-shadow:0 4px 14px rgba(79,70,229,.4)!important;transition:all .22s!important;}
[data-testid="stDownloadButton"]>button:hover{transform:translateY(-2px)!important;
    box-shadow:0 6px 22px rgba(79,70,229,.55)!important;color:#ffffff!important;}
[data-testid="stDownloadButton"]>button p,[data-testid="stDownloadButton"]>button span,
[data-testid="stDownloadButton"]>button div,[data-testid="stDownloadButton"]>button *{
    color:#ffffff!important;text-shadow:0 1px 4px rgba(0,0,0,0.3)!important;}
div[data-testid="stDownloadButton"]{display:flex!important;justify-content:center!important;}

/* ── Layout helpers ──────────────────────────────────────────────────────── */
.sec-div{height:1px;
    background:linear-gradient(90deg,transparent,var(--border),transparent);margin:1.6rem 0;}
.summary-section{background:linear-gradient(135deg,rgba(79,70,229,.04),rgba(232,69,138,.02));
    border:1px solid rgba(79,70,229,.1);border-radius:14px;
    padding:1.2rem 1.4rem;margin-bottom:.9rem;}
.keysent-section{background:linear-gradient(135deg,rgba(5,150,105,.04),rgba(79,70,229,.02));
    border:1px solid rgba(5,150,105,.12);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:.9rem;}
.empty-state{text-align:center;padding:3rem 2rem;border:1.5px dashed var(--border);
    border-radius:14px;background:linear-gradient(135deg,rgba(79,70,229,.03),rgba(232,69,138,.02));}
.empty-icon{font-size:1.5rem;color:#a5b0d0;margin-bottom:.5rem;}
.empty-title{font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;
    letter-spacing:.1em;text-transform:uppercase;color:#9ca8c8;margin-bottom:.3rem;}
.empty-sub{font-size:.78rem;line-height:1.7;color:#9ca8c8;}
.empty-sub b{color:var(--accent);opacity:.7;}

/* Info cards */
.info-card{background:#ffffff;border:1px solid var(--border);border-radius:14px;
    padding:1.1rem 1.3rem;box-shadow:var(--shadow);height:100%;}
.info-card-header{font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;
    color:var(--accent);margin-bottom:.4rem;display:flex;align-items:center;gap:7px;}
.info-card p{font-size:.8rem;color:var(--muted);line-height:1.7;margin:0;}
.info-card ul{font-size:.8rem;color:var(--muted);line-height:1.9;
    padding-left:1.1rem;margin:.3rem 0 0;}
.info-card .tag{display:inline-block;background:rgba(79,70,229,.09);color:var(--accent);
    border-radius:5px;padding:1px 8px;font-size:.68rem;font-weight:600;
    margin-right:4px;margin-top:5px;}
.info-card .tag-green{background:rgba(5,150,105,.09);color:var(--accent3);}
.step-row{display:flex;align-items:flex-start;gap:10px;padding:.55rem .8rem;
    background:#ffffff;border:1px solid var(--border);border-radius:9px;
    margin-bottom:.4rem;box-shadow:0 1px 4px rgba(79,70,229,.05);}
.step-num{min-width:26px;height:26px;border-radius:50%;background:var(--accent);color:#fff;
    font-family:'Syne',sans-serif;font-weight:700;font-size:.78rem;
    display:flex;align-items:center;justify-content:center;margin-top:1px;}
.step-text{font-size:.82rem;color:var(--text);line-height:1.6;}
.step-text b{color:var(--accent);}

/* Sidebar (cosmetic only — no controls) */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#e8eaf6 0%,#eceef8 100%)!important;
    border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] .block-container{padding:1.2rem .9rem!important;}

::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--bg3);}
::-webkit-scrollbar-thumb{background:#b0b8d8;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent);}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "history": [], "last_summary": "", "last_input": "",
    "last_input_clean": "", "total_runs": 0, "total_reduced": 0,
    "summary_native": "", "summary_english": "",
    "lang_choice_run": "English", "translated_en_text": "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return load_bart(), load_t5()

with st.spinner("⚙ Loading AI models — this runs once, then stays cached..."):
    (bart_tok, bart_mod), (t5_tok, t5_mod) = load_models()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  (branding only — no controls)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:.8rem 0 1.2rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                    background:linear-gradient(135deg,#4f46e5,#e8458a);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            ✦ SummarAI
        </div>
        <div style='font-size:.65rem;color:#6b7280;letter-spacing:.12em;
                    text-transform:uppercase;margin-top:3px;'>
            Intelligent Summarizer
        </div>
    </div>""", unsafe_allow_html=True)

    ok_b = bart_mod is not None
    ok_t = t5_mod  is not None
    st.markdown(f"""
    <div style='font-family:Syne,sans-serif;font-size:.6rem;font-weight:700;
                letter-spacing:.14em;text-transform:uppercase;color:#5a6080;margin-bottom:.5rem;'>
        Model Status
    </div>
    <div style='display:flex;gap:.5rem;'>
        <div class='status-pill {"sp-ok" if ok_b else "sp-err"}'>{"✓" if ok_b else "✗"} BART</div>
        <div class='status-pill {"sp-ok" if ok_t else "sp-err"}'>{"✓" if ok_t else "✗"} T5</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#c8cde8;margin:1rem 0'>", unsafe_allow_html=True)
    avg = (round(st.session_state.total_reduced / st.session_state.total_runs)
           if st.session_state.total_runs > 0 else 0)
    st.markdown(f"""
    <div style='font-family:Syne,sans-serif;font-size:.6rem;font-weight:700;
                letter-spacing:.14em;text-transform:uppercase;color:#5a6080;margin-bottom:.5rem;'>
        Session Stats
    </div>
    <div style='font-size:.78rem;color:#5a6080;line-height:2;'>
        Summaries: <b style='color:#1a1d2e'>{st.session_state.total_runs}</b><br>
        Avg reduction: <b style='color:#1a1d2e'>{avg}%</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#c8cde8;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.72rem;color:#9ca3af;line-height:1.7;'>
        ✦ All controls are in the <b style='color:#4f46e5'>Settings Bar</b> on the main page.<br>
        No need to open this sidebar.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI-Powered · Multilingual · Explainable · Multi-Model</div>
    <div class="hero-title">Summarize Any Text, Instantly</div>
    <div class="hero-sub">
        BART &amp; T5 transformers. Paste text or upload a PDF.
        Supports 8 languages. Results in seconds.
    </div>
    <div class="hero-divider"></div>
    <div class="stats-row">
        <div class="stat-pill">⚡ <span>Fast</span> CPU</div>
        <div class="stat-pill">🧠 <span>2</span> AI Models</div>
        <div class="stat-pill">🌐 <span>8</span> Languages</div>
        <div class="stat-pill">📄 <span>Any</span> PDF</div>
        <div class="stat-pill">🔍 <span>Explainable</span></div>
    </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ① SETTINGS BAR  (always visible — replaces sidebar controls)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">⚙ Settings</div>', unsafe_allow_html=True)
st.markdown('<div class="settings-bar">', unsafe_allow_html=True)

cfg_col1, cfg_col2, cfg_col3, cfg_col4, cfg_col5 = st.columns([1.4, 1.6, 1.6, 1.2, 1.2])

with cfg_col1:
    st.markdown('<div class="settings-bar-title">AI Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("AI Model", ["BART", "T5"], label_visibility="collapsed",
                                 help="BART: articles & reports. T5: general text.")

with cfg_col2:
    st.markdown('<div class="settings-bar-title">Summary Length</div>', unsafe_allow_html=True)
    length_choice = st.select_slider("Length", options=["Short", "Medium", "Detailed"],
                                      value="Medium", label_visibility="collapsed")

with cfg_col3:
    st.markdown('<div class="settings-bar-title">Language</div>', unsafe_allow_html=True)
    lang_list   = list(SUPPORTED_LANGUAGES.keys())
    lang_choice = st.selectbox("Language", lang_list, index=0,
                                label_visibility="collapsed",
                                help="Language of your input/output. Auto-translates.")

with cfg_col4:
    st.markdown('<div class="settings-bar-title">Options</div>', unsafe_allow_html=True)
    show_explain = st.toggle("Key Sentences", value=True)
    show_history = st.toggle("History", value=True)

with cfg_col5:
    st.markdown('<div class="settings-bar-title">Model Status</div>', unsafe_allow_html=True)
    ok_b = bart_mod is not None
    ok_t = t5_mod  is not None
    st.markdown(f"""
    <div class='status-row'>
        <div class='status-pill {"sp-ok" if ok_b else "sp-err"}'>{"✓" if ok_b else "✗"} BART</div>
        <div class='status-pill {"sp-ok" if ok_t else "sp-err"}'>{"✓" if ok_t else "✗"} T5</div>
    </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close settings-bar

# Language indicator
if lang_choice != "English":
    lang_code = SUPPORTED_LANGUAGES.get(lang_choice, "en")
    st.markdown(f"""
    <div style='background:rgba(5,150,105,.07);border:1px solid rgba(5,150,105,.22);
                border-radius:9px;padding:.55rem 1rem;font-size:.78rem;color:#059669;
                margin-bottom:.8rem;display:flex;align-items:center;gap:7px;'>
        🌐 <b>Auto-translate mode:</b> {lang_choice} input → English for AI → {lang_choice} output
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ABOUT EXPANDER
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🧠  About the AI Models & How to Use", expanded=False):
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-header'>🟣 BART — Best for Articles &amp; Reports</div>
            <p><b>BART</b> (Meta AI) reads text bidirectionally then rewrites it into a
            clean, fluent summary. Best for news articles, research papers, reports.</p>
            <ul>
                <li>Natural, well-structured sentences</li>
                <li>Fine-tuned on CNN/DailyMail dataset</li>
            </ul>
            <div style='margin-top:.5rem;'>
                <span class='tag'>facebook/bart-base</span>
                <span class='tag'>140M params</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-header' style='color:#059669;'>🟢 T5 — Best for General Text</div>
            <p><b>T5</b> (Google) treats summarization as text-to-text. Fast and flexible
            for emails, blog posts, general documents.</p>
            <ul>
                <li>Slightly shorter outputs, fast on CPU</li>
                <li>Fine-tuned on CNN/DailyMail dataset</li>
            </ul>
            <div style='margin-top:.5rem;'>
                <span class='tag tag-green'>google/t5-small</span>
                <span class='tag tag-green'>60M params</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>How to Use</div>", unsafe_allow_html=True)
    for num, text in [
        ("1", "Set <b>AI Model</b>, <b>Summary Length</b>, and <b>Language</b> in the Settings Bar above."),
        ("2", "Paste text in the <b>Paste Text</b> tab, or upload a PDF in <b>Upload PDF</b>."),
        ("3", "Click <b>✦ Generate Summary</b>."),
        ("4", "If non-English: see <b>translated English text</b>, then <b>English summary</b>, then <b>native summary</b>."),
        ("5", "Check <b>Key Source Sentences</b> to see what the AI used."),
        ("6", "Click <b>⬇ Download</b> to save as .txt."),
    ]:
        st.markdown(f"""
        <div class='step-row'>
            <div class='step-num'>{num}</div>
            <div class='step-text'>{text}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(79,70,229,.05);border:1px solid rgba(79,70,229,.15);
                border-radius:10px;padding:.9rem 1.2rem;margin-top:.9rem;'>
        <div style='font-family:Syne,sans-serif;font-size:.7rem;font-weight:700;
                    color:var(--accent);letter-spacing:.1em;margin-bottom:.5rem;'>
            💡 SPEED GUIDE
        </div>
        <div style='font-size:.79rem;color:var(--muted);line-height:1.9;'>
            ✦ &nbsp;<b>Short</b>: instant — no AI model call, pure extractive<br>
            ✦ &nbsp;<b>Medium</b>: ~6–10s — AI generates 90-word context window<br>
            ✦ &nbsp;<b>Detailed</b>: ~10–15s — AI generates 120-word context + extractive padding<br>
            ✦ &nbsp;Long PDFs are auto-sampled (intro + middle + end) for speed<br>
            ✦ &nbsp;For research papers: BART + Detailed; for emails: T5 + Short
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ② INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
st.markdown('<div class="card-title">📥 Input</div>', unsafe_allow_html=True)

input_text = ""
st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
tab_text, tab_pdf = st.tabs(["✏  Paste Text", "📄  Upload PDF"])

with tab_text:
    txt = st.text_area("Paste text here", height=280,
                       placeholder="Paste your article, report, research paper or any text here — any language...",
                       label_visibility="collapsed")
    if txt.strip():
        input_text = txt
        wc = len(txt.split())
        st.markdown(
            f'<div style="font-size:.72rem;color:#9ca3af;text-align:right;margin-top:.3rem">'
            f'{wc:,} words · {len(txt):,} characters</div>', unsafe_allow_html=True)

with tab_pdf:
    st.markdown("""
    <div style='text-align:center;padding:.3rem 0 .5rem;'>
        <div style='font-size:1.8rem;margin-bottom:.2rem;'>📄</div>
        <div style='font-family:Syne,sans-serif;font-size:.75rem;font-weight:700;
                    color:var(--muted);letter-spacing:.08em;margin-bottom:.2rem;'>
            DROP YOUR PDF HERE
        </div>
        <div style='font-size:.69rem;color:#9ca3af;'>or click Browse to select · text-based PDFs only</div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        pdf_bytes = uploaded.read()
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = []
                for page in pdf.pages:
                    pg = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3) or page.extract_text()
                    if pg: pages.append(pg)
                raw_pdf  = "\n".join(pages)
                pdf_text = _clean_pdf_text(raw_pdf)

            if not pdf_text.strip():
                st.error("⚠ This PDF appears to be scanned/image-based. Only text-based PDFs are supported.")
            else:
                input_text = pdf_text
                wc_pdf = len(pdf_text.split())
                st.markdown(f"""
                <div style='background:rgba(5,150,105,.08);border:1.5px solid rgba(5,150,105,.3);
                            border-radius:11px;padding:.85rem 1.1rem;margin-top:.7rem;'>
                    <div style='display:flex;align-items:center;gap:9px;flex-wrap:wrap;'>
                        <span style='font-size:1.2rem;'>✅</span>
                        <div>
                            <div style='font-family:Syne,sans-serif;font-weight:700;
                                        font-size:.82rem;color:#059669;'>{uploaded.name}</div>
                            <div style='font-size:.72rem;color:#6b7280;margin-top:1px;'>
                                {len(pages)} page{"s" if len(pages)!=1 else ""} &nbsp;·&nbsp;
                                {wc_pdf:,} words &nbsp;·&nbsp;
                                {round(len(pdf_bytes)/1024,1)} KB
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                with st.expander("👁  Preview extracted text (first 500 chars)", expanded=False):
                    preview = pdf_text[:500].replace("\n", " ")
                    st.markdown(
                        f'<div style="font-size:.8rem;color:#5a6080;line-height:1.65;'
                        f'background:#fafbff;border-radius:8px;padding:.9rem;">'
                        f'{preview}...</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠ Could not read PDF: {e}")
            st.markdown('<div style="font-size:.75rem;color:#9ca3af;margin-top:.3rem;">'
                        'Tip: ensure the file is a valid, non-password-protected PDF.</div>',
                        unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Generate / Clear buttons ──────────────────────────────────────────────────
st.markdown("""
<style>div[data-testid="stButton"]{display:flex!important;justify-content:center!important;}</style>
""", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_btn = st.button("✦  Generate Summary", use_container_width=True)
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    clr_btn = st.button("✕  Clear", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if clr_btn:
    for k in ["last_summary", "last_input", "last_input_clean",
              "summary_native", "summary_english", "translated_en_text"]:
        st.session_state[k] = ""
    st.session_state["lang_choice_run"] = "English"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ③ SUMMARIZATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
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
            try:
                lang_code = SUPPORTED_LANGUAGES.get(lang_choice, "en")

                # Step 1: translate input to English if needed
                prog.progress(8)
                with st.spinner("🔍 Processing input..."):
                    en_text, translated = translate_to_english(input_text, lang_code)
                prog.progress(20)

                # Step 2: generate summary in English
                with st.spinner(f"🧠 Summarizing with {model_choice} ({length_choice})..."):
                    for i in range(20, 75):
                        time.sleep(0.005)
                        prog.progress(i)
                    summary_en = generate_summary(en_text, tok, mod, model_choice, length_choice)
                prog.progress(78)

                # Step 3: translate output to native language if needed
                summary_native = ""
                if lang_choice != "English" and translated:
                    with st.spinner(f"🌐 Translating summary → {lang_choice}..."):
                        summary_native = translate_from_english(summary_en, lang_code)

                prog.progress(100)

            finally:
                prog.empty()

            # Determine final summary for display
            summary = summary_en  # always show English base

            in_wc  = len(input_text.split())
            out_wc = len(summary_en.split())
            pct    = round((1 - out_wc / max(in_wc, 1)) * 100)

            st.session_state.last_summary     = summary
            st.session_state.last_input       = input_text
            st.session_state.last_input_clean = clean_input(en_text)
            st.session_state.summary_english  = summary_en
            st.session_state.summary_native   = summary_native
            st.session_state.lang_choice_run  = lang_choice
            st.session_state.translated_en_text = en_text if translated and lang_choice != "English" else ""
            st.session_state.total_runs      += 1
            st.session_state.total_reduced   += pct
            st.session_state.history.insert(0, {
                "model": model_choice, "length": length_choice,
                "in_wc": in_wc, "out_wc": out_wc, "pct": pct,
                "full": summary_en, "native": summary_native, "lang": lang_choice,
            })
            if len(st.session_state.history) > 6:
                st.session_state.history.pop()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ④ OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.last_summary:
    summary_en     = st.session_state.summary_english or st.session_state.last_summary
    summary_native = st.session_state.summary_native
    lang_run       = st.session_state.lang_choice_run
    en_translation = st.session_state.translated_en_text
    in_wc          = len(st.session_state.last_input.split())
    out_wc         = len(summary_en.split())
    pct            = round((1 - out_wc / max(in_wc, 1)) * 100)
    is_multilingual = lang_run != "English" and bool(summary_native)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── Multilingual output ──────────────────────────────────────────────────
    if is_multilingual:
        # Pipeline indicator
        st.markdown(f"""
        <div class='pipeline-bar'>
            <span class='pipe-step'>📥 {lang_run} Input</span>
            <span class='pipe-arrow'>→</span>
            <span class='pipe-step'>🇬🇧 Translate → English</span>
            <span class='pipe-arrow'>→</span>
            <span class='pipe-step'>🧠 AI Summarize</span>
            <span class='pipe-arrow'>→</span>
            <span class='pipe-step'>🌐 Translate → {lang_run}</span>
        </div>""", unsafe_allow_html=True)

        # Box A: Translated English input (what was actually sent to AI)
        if en_translation:
            en_words   = en_translation.split()
            en_preview = " ".join(en_words[:300]) + ("..." if len(en_words) > 300 else "")
            with st.expander(f"📥 Translated English Input (sent to AI) — {len(en_words):,} words", expanded=False):
                st.markdown(
                    f'<div style="font-size:.84rem;color:#1a1d2e;line-height:1.75;">{en_preview}</div>',
                    unsafe_allow_html=True)

        # Box B: English summary
        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🇬🇧 Summary in English</div>', unsafe_allow_html=True)
        badge = "badge-bart" if model_choice == "BART" else "badge-t5"
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;'
            f'border-radius:6px;font-size:.68rem;font-family:Syne,sans-serif;font-weight:700;'
            f'letter-spacing:.07em;background:rgba(79,70,229,.12);color:#4f46e5;'
            f'border:1px solid rgba(79,70,229,.3);margin-bottom:.7rem;">'
            f'✦ {model_choice} · {length_choice}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box"><div class="result-text">{summary_en}</div></div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-chip">📥 Input <b>{in_wc:,}</b> words ({lang_run})</div>
            <div class="stat-chip">📤 Output <b>{out_wc}</b> words</div>
            <div class="stat-chip">📉 Reduced <b>{pct}%</b></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Box C: Native language summary
        st.markdown('<div class="summary-section" style="border-color:rgba(232,69,138,.2);">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title" style="color:var(--accent2);">🌐 Summary in {lang_run}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;'
            f'border-radius:6px;font-size:.68rem;font-family:Syne,sans-serif;font-weight:700;'
            f'letter-spacing:.07em;background:rgba(232,69,138,.12);color:#e8458a;'
            f'border:1px solid rgba(232,69,138,.3);margin-bottom:.7rem;">'
            f'✦ {model_choice} · {length_choice} · {lang_run}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-box" style="border-color:rgba(232,69,138,.25);'
            f'background:linear-gradient(135deg,rgba(232,69,138,.06),rgba(79,70,229,.03));">'
            f'<div class="result-text">{summary_native}</div></div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Downloads
        st.markdown("<div style='margin-top:.8rem'>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2, gap="medium")
        with dl1:
            st.download_button("⬇  Download English Summary",
                               data=summary_en, file_name="summary_english.txt",
                               mime="text/plain", use_container_width=True, key="dl_en")
        with dl2:
            combined = (f"=== ENGLISH SUMMARY ===\n{summary_en}"
                        f"\n\n=== {lang_run.upper()} SUMMARY ===\n{summary_native}")
            st.download_button(f"⬇  Download {lang_run} Summary",
                               data=combined, file_name=f"summary_{lang_run.lower()}.txt",
                               mime="text/plain", use_container_width=True, key="dl_native")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ── Standard English output ──────────────────────────────────────────
        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📤 Summary</div>', unsafe_allow_html=True)
        badge_cls = "badge-bart" if model_choice == "BART" else "badge-t5"
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;'
            f'border-radius:6px;font-size:.68rem;font-family:Syne,sans-serif;font-weight:700;'
            f'letter-spacing:.07em;background:rgba(79,70,229,.12);color:#4f46e5;'
            f'border:1px solid rgba(79,70,229,.3);margin-bottom:.7rem;">'
            f'✦ {model_choice} · {length_choice}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box"><div class="result-text">{summary_en}</div></div>',
                    unsafe_allow_html=True)
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
            st.download_button("⬇  Download Summary (.txt)",
                               data=summary_en, file_name="summary.txt",
                               mime="text/plain", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Key sentences ─────────────────────────────────────────────────────────
    if show_explain:
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="keysent-section">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Key Source Sentences</div>', unsafe_allow_html=True)
        try:
            source = st.session_state.last_input_clean or st.session_state.last_input
            key_sents = get_important_sentences(source, summary_en, top_n=3)
            if key_sents:
                for i, s in enumerate(key_sents, 1):
                    st.markdown(
                        f'<div class="explain-box">'
                        f'<div class="explain-num">Key Sentence {i}</div>{s}'
                        f'</div>', unsafe_allow_html=True)
            else:
                st.info("No key sentences found.")
        except Exception:
            pass
        st.markdown('</div>', unsafe_allow_html=True)

    # ── History ───────────────────────────────────────────────────────────────
    if show_history and st.session_state.history:
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🕘 Recent Summaries</div>', unsafe_allow_html=True)
        for i, h in enumerate(st.session_state.history):
            lang_tag = f" · {h.get('lang','EN')}" if h.get("lang", "English") != "English" else ""
            with st.expander(
                f"#{i+1}  {h['model']} · {h['length']} · "
                f"{h['in_wc']:,} → {h['out_wc']} words ({h['pct']}% reduced){lang_tag}"):
                st.markdown(
                    f'<div style="font-size:.85rem;line-height:1.75;color:#1a1d2e;padding:.2rem 0">'
                    f'{h["full"]}</div>', unsafe_allow_html=True)
                if h.get("native"):
                    st.markdown(
                        f'<div style="font-size:.8rem;color:#9ca3af;margin:.4rem 0 .2rem;">'
                        f'{h.get("lang","")} translation:</div>'
                        f'<div style="font-size:.85rem;line-height:1.75;color:#1a1d2e;">'
                        f'{h["native"]}</div>', unsafe_allow_html=True)
                dl_c1, dl_c2 = st.columns(2)
                with dl_c1:
                    st.download_button("⬇ English", data=h["full"],
                                       file_name=f"summary_{i+1}_en.txt", key=f"dl_{i}_en")
                if h.get("native"):
                    with dl_c2:
                        st.download_button(f"⬇ {h.get('lang','')}", data=h["native"],
                                           file_name=f"summary_{i+1}_{h.get('lang','')}.txt",
                                           key=f"dl_{i}_nat")

else:
    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">✦</div>
        <div class="empty-title">Ready to Summarize</div>
        <div class="empty-sub">
            Paste text or upload a PDF above, then click <b>Generate Summary</b>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-top:3.5rem;padding:1.5rem 0 1rem;
            border-top:1px solid #c8cde8;'>
    <div style='font-family:Syne,sans-serif;font-size:.64rem;font-weight:700;
                letter-spacing:.15em;text-transform:uppercase;color:#a0a8c8;'>
        SummarAI · BART + T5 Transformers · Multilingual · Built with Streamlit
    </div>
</div>""", unsafe_allow_html=True)