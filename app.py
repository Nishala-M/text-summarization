# app.py
# Professional AI Text Summarizer — Single Column Layout

import streamlit as st
import pdfplumber
import time
import io
from summarizer import load_bart, load_t5, generate_summary
from explainability import get_important_sentences

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

/* ══ FORCE LIGHT MODE ══ */
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

/* ── Hero ── */
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

/* ── Section titles ── */
.card-title {
    font-family:'Syne',sans-serif; font-size:.68rem; font-weight:700;
    letter-spacing:.14em; text-transform:uppercase; color:var(--muted);
    margin-bottom:.8rem; display:flex; align-items:center; gap:7px;
}
.card-title::before {
    content:''; display:inline-block; width:3px; height:11px;
    background:var(--accent); border-radius:2px;
}

/* ── INPUT WRAPPER ── */
.input-wrapper {
    background: #ffffff;
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem 1.5rem 1.1rem;
    box-shadow: var(--shadow);
    margin-bottom: .8rem;
}

/* ── Tabs ── */
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

/* ── Textarea ── */
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

/* ── Generate button ── */
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

/* ── Clear button ── */
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

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    box-shadow: var(--shadow) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #fafbff !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Result Box ── */
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

/* ── Stats chips ── */
.stats-bar { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: .8rem; }
.stat-chip {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 8px; padding: 5px 13px; font-size: .76rem; color: var(--muted);
    box-shadow: 0 1px 4px rgba(79,70,229,.06);
}
.stat-chip b { color: var(--accent3); }

/* ── Key sentences ── */
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

/* ── Model badges ── */
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

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8eaf6 0%, #eceef8 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 16px rgba(79,70,229,.08) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 50px !important;
}
.stProgress > div { background: var(--bg3) !important; border-radius: 50px !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #ffffff !important; border-radius: 10px !important;
    border: 1px solid var(--border) !important; font-size: .85rem !important;
    color: var(--text) !important;
}
.streamlit-expanderContent {
    background: #fafbff !important; border: 1px solid var(--border) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}

/* ── Download button — same style as Generate, wider ── */
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

/* ── Section divider ── */
.sec-div {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ── Summary section wrapper ── */
.summary-section {
    background: linear-gradient(135deg, rgba(79,70,229,.04), rgba(232,69,138,.02));
    border: 1px solid rgba(79,70,229,.1);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Key sentences section wrapper ── */
.keysent-section {
    background: linear-gradient(135deg, rgba(5,150,105,.04), rgba(79,70,229,.02));
    border: 1px solid rgba(5,150,105,.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Empty state ── */
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

/* ── Info cards ── */
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
.info-card .tag-green {
    background: rgba(5,150,105,.09); color: var(--accent3);
}

/* ── How-to steps ── */
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

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg3); }
::-webkit-scrollbar-thumb { background: #b0b8d8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
for k, v in [("history", []), ("last_summary", ""), ("last_input", ""),
             ("total_runs", 0), ("total_reduced", 0)]:
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
        st.markdown(f'<div class="model-badge {"badge-ok" if ok else "badge-err"}">{"✓" if ok else "✗"} BART</div>', unsafe_allow_html=True)
    with c2:
        ok = t5_mod is not None
        st.markdown(f'<div class="model-badge {"badge-ok" if ok else "badge-err"}">{"✓" if ok else "✗"} T5</div>', unsafe_allow_html=True)

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
        <b style='color:#1a1d2e'>Detailed</b> — 130–160 words<br>
        Abstractive + Extractive hybrid
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI-Powered · Explainable · Multi-Model</div>
    <div class="hero-title">Summarize Any Text,<br>Instantly</div>
    <div class="hero-sub">
        Powered by BART &amp; T5 transformers. Paste text or upload a PDF —
        get clean, accurate summaries in seconds.
    </div>
    <div class="hero-divider"></div>
    <div class="stats-row">
        <div class="stat-pill">⚡ <span>Fast</span> Single-pass</div>
        <div class="stat-pill">🧠 <span>2</span> AI Models</div>
        <div class="stat-pill">📄 <span>Any</span> Document type</div>
        <div class="stat-pill">🔍 <span>Explainable</span> Output</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ★ NEW — ABOUT THE MODELS SECTION
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
            <div class='info-card-header'>🟣 BART — Best for Articles & Reports</div>
            <p><b>BART</b> (Bidirectional and Auto-Regressive Transformer) is developed by
            Meta AI. It reads the full text in both directions to deeply understand context,
            then rewrites it into a clean, fluent summary.</p>
            <ul>
                <li>Best for: news articles, research papers, reports</li>
                <li>Produces natural, well-structured sentences</li>
                <li>Fine-tuned on CNN/DailyMail news dataset</li>
                <li>Handles long documents well</li>
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
            <p><b>T5</b> (Text-to-Text Transfer Transformer) is developed by Google. It
            treats every NLP task — including summarization — as a text-to-text problem.
            It converts input text into a shorter output text.</p>
            <ul>
                <li>Best for: general text, emails, blog posts</li>
                <li>Flexible and fast on CPU</li>
                <li>Fine-tuned on CNN/DailyMail news dataset</li>
                <li>Slightly shorter outputs than BART</li>
            </ul>
            <div style='margin-top:.6rem;'>
                <span class='tag tag-green'>google/t5-small</span>
                <span class='tag tag-green'>60M parameters</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Summary Length Modes</div>", unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3, gap="medium")
    with col3:
        st.markdown("""
        <div class='info-card' style='border-top:3px solid #4f46e5;'>
            <div class='info-card-header'>📌 Short Mode</div>
            <p><b>40–80 words.</b> Uses TF-IDF extractive method — picks the 3 most
            important sentences directly from your text. No rewriting. Fast and precise.</p>
            <p style='margin-top:.5rem;'>Use when you need a <b>quick overview</b> of the main point.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='info-card' style='border-top:3px solid #e8458a;'>
            <div class='info-card-header'>📝 Medium Mode</div>
            <p><b>70–130 words.</b> Uses BART/T5 to abstractively rewrite the text.
            The AI reads and rephrases — not just copying sentences.</p>
            <p style='margin-top:.5rem;'>Use when you need a <b>balanced summary</b> covering main ideas.</p>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class='info-card' style='border-top:3px solid #059669;'>
            <div class='info-card-header'>📄 Detailed Mode</div>
            <p><b>130–160 words.</b> Hybrid approach — first extracts key sentences,
            then AI rewrites them into a comprehensive summary.</p>
            <p style='margin-top:.5rem;'>Use when you need <b>full coverage</b> of all key topics.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>How to Use SummarAI</div>", unsafe_allow_html=True)

    steps = [
        ("1", "Choose your <b>AI Model</b> (BART or T5) from the sidebar on the left."),
        ("2", "Select your <b>Summary Length</b> — Short, Medium, or Detailed."),
        ("3", "Paste your text in the <b>Paste Text</b> tab, or switch to <b>Upload PDF</b> to upload a file."),
        ("4", "Click <b>✦ Generate Summary</b> to create your summary."),
        ("5", "Read the <b>Summary</b> output and check the <b>Key Source Sentences</b> to see which parts of your text were most important."),
        ("6", "Click <b>⬇ Download Summary</b> to save the result as a .txt file."),
    ]
    for num, text in steps:
        st.markdown(f"""
        <div class='step-row'>
            <div class='step-num'>{num}</div>
            <div class='step-text'>{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Tips box
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
            ✦ &nbsp;Toggle <b>Key Sentences OFF</b> in sidebar for a cleaner view<br>
            ✦ &nbsp;Use <b>History panel</b> to compare BART vs T5 outputs side by side
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
        placeholder="Paste your article, report, research paper or any text here...",
        label_visibility="hidden"
    )
    if txt.strip():
        input_text = txt
        wc = len(txt.split())
        st.markdown(
            f'<div style="font-size:.74rem;color:#9ca3af;text-align:right;margin-top:.3rem">'
            f'{wc:,} words</div>',
            unsafe_allow_html=True
        )

with tab_pdf:
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="hidden")
    if uploaded:
        try:
            with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
                pages    = [p.extract_text() for p in pdf.pages if p.extract_text()]
                pdf_text = "\n".join(pages)
            input_text = pdf_text
            st.markdown(f"""
            <div style='background:rgba(5,150,105,.08);border:1px solid rgba(5,150,105,.22);
                        border-radius:10px;padding:.75rem 1rem;font-size:.82rem;
                        color:#059669;margin-top:.5rem;'>
                ✓ <b>{uploaded.name}</b> · {len(pages)} pages · {len(pdf_text.split()):,} words
            </div>""", unsafe_allow_html=True)
            with st.expander("Preview first 400 characters"):
                st.markdown(
                    f'<div style="font-size:.82rem;color:#5a6080;line-height:1.6">'
                    f'{pdf_text[:400]}...</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ── Buttons ───────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stButton"] { display: flex !important; justify-content: center !important; }
.btn-group { display:flex; flex-direction:column; align-items:center; gap:0; margin:0.5rem 0; }
.btn-group div[data-testid="stButton"] > button { width: 260px !important; }
</style>
<div class="btn-group">
""", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_btn = st.button("✦  Generate Summary", use_container_width=True)
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    clr_btn = st.button("✕  Clear", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

if clr_btn:
    st.session_state.last_summary = ""
    st.session_state.last_input   = ""
    st.rerun()


# ══════════════════════════════════════════════════════════════
#  RUN SUMMARIZATION
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
            with st.spinner(f"Summarizing with {model_choice}..."):
                for i in range(50):
                    time.sleep(0.008)
                    prog.progress(i + 1)
                summary = generate_summary(
                    input_text, tok, mod, model_choice, length_choice
                )
                for i in range(50, 100):
                    time.sleep(0.004)
                    prog.progress(i + 1)
            prog.empty()

            in_wc  = len(input_text.split())
            out_wc = len(summary.split())
            pct    = round((1 - out_wc / max(in_wc, 1)) * 100)

            st.session_state.last_summary  = summary
            st.session_state.last_input    = input_text
            st.session_state.total_runs   += 1
            st.session_state.total_reduced += pct
            st.session_state.history.insert(0, {
                "model": model_choice, "length": length_choice,
                "in_wc": in_wc, "out_wc": out_wc, "pct": pct, "full": summary
            })
            if len(st.session_state.history) > 6:
                st.session_state.history.pop()
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  ② SUMMARY
# ══════════════════════════════════════════════════════════════
if st.session_state.last_summary:

    summary = st.session_state.last_summary
    in_wc   = len(st.session_state.last_input.split())
    out_wc  = len(summary.split())
    pct     = round((1 - out_wc / max(in_wc, 1)) * 100)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

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
            data=summary, file_name="summary.txt",
            mime="text/plain", use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── KEY SENTENCES ─────────────────────────────────────────
    if show_explain:
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="keysent-section">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Key Source Sentences</div>',
                    unsafe_allow_html=True)
        try:
            key_sents = get_important_sentences(
                st.session_state.last_input, summary, top_n=3
            )
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

    # ── HISTORY ───────────────────────────────────────────────
    if show_history and st.session_state.history:
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🕘 Recent Summaries</div>',
                    unsafe_allow_html=True)
        for i, h in enumerate(st.session_state.history):
            with st.expander(
                f"#{i+1}  {h['model']} · {h['length']} · "
                f"{h['in_wc']:,} → {h['out_wc']} words  ({h['pct']}% reduced)"
            ):
                st.markdown(
                    f'<div style="font-size:.88rem;line-height:1.75;color:#1a1d2e;padding:.2rem 0">'
                    f'{h["full"]}</div>', unsafe_allow_html=True
                )
                st.download_button(
                    "⬇ Download", data=h["full"],
                    file_name=f"summary_{i+1}.txt", key=f"dl_{i}"
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
        SummarAI · BART + T5 Transformers · Built with Streamlit
    </div>
</div>
""", unsafe_allow_html=True)