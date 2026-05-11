import os
# ── ALL env vars MUST be set before any framework import ──────────────────────
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"   
os.environ["OMP_NUM_THREADS"] = "1"              
os.environ["TF_NUM_INTEROP_THREADS"] = "1"       
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"       

import torch
torch.classes.__path__ = []
torch.set_num_threads(1)

import streamlit as st
import numpy as np
import pickle, json, re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import nltk

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ — DMU NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.main { background-color: #0a0e1a; }
.block-container { background-color: #0a0e1a !important; padding-top: 2rem !important; }
[data-testid="stAppViewContainer"] { background-color: #0a0e1a !important; }
[data-testid="stHeader"] { background-color: #0a0e1a !important; }
p, label, div { color: #e2e8f0; }
.stTextArea textarea {
    background-color: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 12px !important;
}
.stButton > button {
    background-color: #1e2d4a !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3f5a !important;
    border-radius: 8px !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #4f9eff, #a78bfa) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
}
.stCheckbox label { color: #94a3b8 !important; }
hr { border-color: #1e2d4a !important; }
section[data-testid="stSidebar"] { background: #0f1629 !important; border-right: 1px solid #1e2d4a; }
.hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #4f9eff 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 0.2rem;
}
.hero-sub { color: #6b7fa3; font-size: 1rem; font-weight: 300; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-card {
    background: linear-gradient(135deg, #111827, #1a2540);
    border: 1px solid #1e2d4a; border-radius: 16px;
    padding: 1.4rem 1.6rem; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #e2e8f0; }
.metric-lbl { font-size: 0.78rem; color: #6b7fa3; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
.badge { display: inline-block; padding: 0.4rem 1.2rem; border-radius: 999px; font-weight: 700; font-size: 1.05rem; }
.badge-pos { background: #05422a; color: #34d399; border: 1px solid #34d399; }
.badge-neg { background: #3b0f0f; color: #f87171; border: 1px solid #f87171; }
.badge-neu { background: #1e2d4a; color: #93c5fd; border: 1px solid #93c5fd; }
.section-header {
    font-size: 0.75rem; font-weight: 600; color: #4f9eff; text-transform: uppercase;
    letter-spacing: 0.12em; margin: 1.4rem 0 0.6rem 0;
    border-bottom: 1px solid #1e2d4a; padding-bottom: 0.4rem;
}
.result-box {
    background: linear-gradient(135deg, #111827, #131c2e);
    border: 1px solid #1e2d4a; border-radius: 16px; padding: 1.8rem; margin-top: 1rem;
}
.conf-label { font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_DIR   = "saved_models"
LABEL_MAP   = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLOR = {"Negative": "#f87171", "Neutral": "#93c5fd", "Positive": "#34d399"}
BADGE_CLASS = {"Negative": "badge-neg", "Neutral": "badge-neu", "Positive": "badge-pos"}

# ── Preprocessing ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\$\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ── Model Loaders ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_classical():
    models = {}
    try:
        with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
            models["tfidf"] = pickle.load(f)
        with open(f"{MODEL_DIR}/logistic_regression.pkl", "rb") as f:
            models["lr"] = pickle.load(f)
        with open(f"{MODEL_DIR}/best_lr.pkl", "rb") as f:
            models["best_lr"] = pickle.load(f)
        with open(f"{MODEL_DIR}/naive_bayes.pkl", "rb") as f:
            models["nb"] = pickle.load(f)
        with open(f"{MODEL_DIR}/svm.pkl", "rb") as f:
            models["svm"] = pickle.load(f)
    except Exception as e:
        st.warning(f"Classical model load error: {e}")
    return models

@st.cache_resource
def load_bilstm():
    try:
        # TF thread limits applied via env vars at top of file (before any TF import)
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        with open(f"{MODEL_DIR}/keras_tokenizer.json") as f:
            tok = tokenizer_from_json(f.read())
        with open(f"{MODEL_DIR}/lstm_config.json") as f:
            cfg = json.load(f)

        mdl = Sequential([
            Embedding(10000, 128, input_length=30),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(3, activation='softmax')
        ])
        mdl.build(input_shape=(None, 30))
        mdl.load_weights(f"{MODEL_DIR}/bilstm_weights.weights.h5")
        return mdl, tok, cfg
    except Exception as e:
        st.warning(f"BiLSTM load error: {e}")
        return None, None, None

@st.cache_resource
def load_finbert():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = f"{MODEL_DIR}/finbert_model"
        tok = AutoTokenizer.from_pretrained(path)
        # Mac fix: load on CPU explicitly, avoid MPS issues
        mdl = AutoModelForSequenceClassification.from_pretrained(path)
        mdl = mdl.to("cpu")
        mdl.eval()
        return mdl, tok
    except Exception as e:
        st.warning(f"FinBERT load error: {e}")
        return None, None

# ── Load All Models ────────────────────────────────────────────────────────────
classical       = load_classical()
bilstm_bundle   = load_bilstm()
finbert_bundle  = load_finbert()

# ── Predict Functions ──────────────────────────────────────────────────────────
def predict_sklearn(text_clean, tfidf, model):
    vec  = tfidf.transform([text_clean])
    pred = int(model.predict(vec)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0].tolist()
    else:
        proba = [0.0, 0.0, 0.0]
        proba[pred] = 1.0
    return {"label": LABEL_MAP[pred], "proba": proba, "type": "Classical ML"}

def _bilstm_inner(text_clean, bundle):
    mdl, tok, cfg = bundle
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = tok.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=cfg["max_len"], padding="post", truncating="post")
    proba  = mdl.predict(padded, verbose=0)[0].tolist()
    pred   = int(np.argmax(proba))
    return {"label": LABEL_MAP[pred], "proba": proba, "type": "Deep Learning"}

def predict_bilstm(text_clean, bundle):
    if bundle[0] is None:
        return None
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_bilstm_inner, text_clean, bundle)
            return fut.result(timeout=30)
    except FuturesTimeout:
        st.warning("BiLSTM timed out — skipping.")
        return None
    except Exception as e:
        st.warning(f"BiLSTM prediction error: {e}")
        return None

def _finbert_inner(text_clean, bundle):
    mdl, tok = bundle
    inputs = tok(text_clean, return_tensors="pt", truncation=True, padding=True, max_length=96)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
    proba = torch.softmax(logits, dim=1)[0].tolist()
    pred  = int(np.argmax(proba))
    return {"label": LABEL_MAP[pred], "proba": proba, "type": "Transformer"}

def predict_finbert(text_clean, bundle):
    if bundle[0] is None:
        return None
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_finbert_inner, text_clean, bundle)
            return fut.result(timeout=30)
    except FuturesTimeout:
        st.warning("FinBERT timed out — skipping.")
        return None
    except Exception as e:
        st.warning(f"FinBERT prediction error: {e}")
        return None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem 0'>
        <div style='font-size:1.6rem;font-weight:700;color:#e2e8f0;'>🧠 SentimentIQ</div>
        <div style='font-size:0.72rem;color:#4f9eff;letter-spacing:0.1em;text-transform:uppercase;'>DMU NLP Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Select Models</div>", unsafe_allow_html=True)
    use_lr      = st.checkbox("Logistic Regression",   value=True)
    use_best_lr = st.checkbox("Best LR (GridSearch)",  value=True)
    use_nb      = st.checkbox("Naive Bayes",           value=True)
    use_svm     = st.checkbox("SVM",                   value=True)
    use_bilstm  = st.checkbox("BiLSTM (Deep)",         value=True)
    use_finbert = st.checkbox("FinBERT (Fine-tuned)",  value=True)

    st.markdown("<div class='section-header'>Model Status</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.82rem;line-height:2;'>
    LR / Best LR / NB / SVM: {'✅ Loaded' if classical.get('tfidf') else '❌ Failed'}<br>
    BiLSTM: {'✅ Loaded' if bilstm_bundle[0] else '❌ Failed'}<br>
    FinBERT: {'✅ Loaded' if finbert_bundle[0] else '❌ Failed'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Accuracies</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Model":    ["LR", "Best LR", "NB", "SVM", "BiLSTM", "FinBERT"],
        "Accuracy": ["80.27%", "82.08%", "78.39%", "82.08%", "75.92%", "85.76%"]
    }), hide_index=True, use_container_width=True)

    st.markdown("<div class='section-header'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem;color:#6b7fa3;line-height:1.6;'>
    Financial sentiment analysis comparing classical ML,
    deep learning, and transformer architectures.<br><br>
    Dataset: FinancialPhraseBank<br>
    Labels: Negative · Neutral · Positive
    </div>
    """, unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>Sentiment Intelligence Dashboard</div>
<div class='hero-sub'>Financial text · Multi-model NLP analysis</div>
<hr style='border-color:#1e2d4a;margin:1.2rem 0;'>
""", unsafe_allow_html=True)

col_input, col_examples = st.columns([3, 1])

with col_input:
    text_input = st.text_area(
        "Enter financial text",
        height=130,
        placeholder="e.g. 'The company reported strong earnings growth driven by higher margins...'",
        label_visibility="collapsed"
    )

with col_examples:
    st.markdown("<div style='font-size:0.8rem;color:#6b7fa3;margin-bottom:0.5rem;'>Quick examples</div>", unsafe_allow_html=True)
    ex_pos = st.button("📈 Positive", use_container_width=True)
    ex_neg = st.button("📉 Negative", use_container_width=True)
    ex_neu = st.button("📊 Neutral",  use_container_width=True)

EXAMPLES = {
    "pos": "The company reported record revenues and exceeded analyst expectations, boosting investor confidence.",
    "neg": "Losses widened significantly as demand fell and supply chain disruptions hammered margins.",
    "neu": "The firm released its quarterly report today, showing figures broadly in line with forecasts."
}
if ex_pos: text_input = EXAMPLES["pos"]
if ex_neg: text_input = EXAMPLES["neg"]
if ex_neu: text_input = EXAMPLES["neu"]

analyze_btn = st.button("🔍  Analyze Sentiment", type="primary", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze_btn and text_input.strip():
    cleaned     = clean_text(text_input)
    all_results = {}

    with st.spinner("Running models..."):
        tfidf = classical.get("tfidf")

        if use_lr and classical.get("lr") and tfidf:
            all_results["Logistic Regression"] = predict_sklearn(cleaned, tfidf, classical["lr"])

        if use_best_lr and classical.get("best_lr") and tfidf:
            all_results["Best LR"] = predict_sklearn(cleaned, tfidf, classical["best_lr"])

        if use_nb and classical.get("nb") and tfidf:
            all_results["Naive Bayes"] = predict_sklearn(cleaned, tfidf, classical["nb"])

        if use_svm and classical.get("svm") and tfidf:
            all_results["SVM"] = predict_sklearn(cleaned, tfidf, classical["svm"])

        if use_bilstm and bilstm_bundle[0] is not None:
            r = predict_bilstm(cleaned, bilstm_bundle)
            if r: all_results["BiLSTM"] = r

        if use_finbert and finbert_bundle[0] is not None:
            r = predict_finbert(cleaned, finbert_bundle)
            if r: all_results["FinBERT"] = r

    if not all_results:
        st.error("No models loaded. Check saved_models/ folder.")
        st.stop()

    # Majority vote
    votes          = [v["label"] for v in all_results.values()]
    vote_counter   = Counter(votes)
    majority_label = vote_counter.most_common(1)[0][0]

    # Metrics
    st.markdown("<div class='section-header'>Ensemble Result</div>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Majority Vote</div>
            <div style='margin-top:0.6rem'><span class='badge {BADGE_CLASS[majority_label]}'>{majority_label}</span></div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        agree_pct = int(vote_counter[majority_label] / len(votes) * 100)
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val'>{agree_pct}%</div>
            <div class='metric-lbl'>Model Agreement</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val'>{len(all_results)}</div>
            <div class='metric-lbl'>Models Run</div>
        </div>""", unsafe_allow_html=True)
    with mc4:
        avg_conf = np.mean([max(v["proba"]) for v in all_results.values()])
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val'>{avg_conf:.0%}</div>
            <div class='metric-lbl'>Avg Confidence</div>
        </div>""", unsafe_allow_html=True)

    # Per-model cards
    st.markdown("<div class='section-header'>Per-Model Predictions</div>", unsafe_allow_html=True)
    cols = st.columns(len(all_results))
    for i, (mname, res) in enumerate(all_results.items()):
        with cols[i]:
            conf_pct = int(max(res["proba"]) * 100)
            color    = LABEL_COLOR[res["label"]]
            st.markdown(f"""<div class='result-box'>
                <div style='font-size:0.7rem;color:#4f9eff;text-transform:uppercase;letter-spacing:0.1em;'>{res['type']}</div>
                <div style='font-weight:700;color:#e2e8f0;font-size:1rem;margin:0.3rem 0;'>{mname}</div>
                <span class='badge {BADGE_CLASS[res["label"]]}'>{res['label']}</span>
                <div style='margin-top:1rem;'>
                    <div class='conf-label'>Confidence: {conf_pct}%</div>
                    <div style='background:#1e2d4a;border-radius:999px;height:6px;margin-top:4px;'>
                        <div style='background:{color};width:{conf_pct}%;height:6px;border-radius:999px;'></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Confidence distribution bar chart
    st.markdown("<div class='section-header'>Confidence Distribution</div>", unsafe_allow_html=True)
    chart_data = []
    for mname, res in all_results.items():
        for idx, lbl in LABEL_MAP.items():
            chart_data.append({"Model": mname, "Sentiment": lbl, "Probability": round(res["proba"][idx], 4)})

    fig = px.bar(pd.DataFrame(chart_data), x="Model", y="Probability",
                 color="Sentiment", barmode="group",
                 color_discrete_map=LABEL_COLOR, template="plotly_dark", height=380)
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font_family="Space Grotesk", font_color="#94a3b8", legend_title_text="",
        xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a", tickformat=".0%"),
        margin=dict(t=20, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("<div class='section-header'>Model Agreement Radar</div>", unsafe_allow_html=True)
    model_names = list(all_results.keys())
    pos_probs   = [r["proba"][2] for r in all_results.values()]
    neg_probs   = [r["proba"][0] for r in all_results.values()]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=pos_probs + [pos_probs[0]], theta=model_names + [model_names[0]],
        fill="toself", name="Positive", line_color="#34d399", fillcolor="rgba(52,211,153,0.15)"
    ))
    fig2.add_trace(go.Scatterpolar(
        r=neg_probs + [neg_probs[0]], theta=model_names + [model_names[0]],
        fill="toself", name="Negative", line_color="#f87171", fillcolor="rgba(248,113,113,0.15)"
    ))
    fig2.update_layout(
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e2d4a", color="#6b7fa3"),
            angularaxis=dict(gridcolor="#1e2d4a", color="#94a3b8")
        ),
        paper_bgcolor="#0a0e1a", font_family="Space Grotesk", font_color="#94a3b8",
        height=400, legend=dict(bgcolor="#111827", bordercolor="#1e2d4a"),
        margin=dict(t=20, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🔎 View Preprocessed Text"):
        st.code(cleaned, language="text")

elif analyze_btn:
    st.warning("Please enter some text to analyze.")

else:
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem;color:#6b7fa3;'>
        <div style='font-size:3rem;margin-bottom:1rem;'>📊</div>
        <div style='font-size:1.1rem;font-weight:600;color:#94a3b8;'>Enter financial text above and click Analyze</div>
        <div style='font-size:0.85rem;margin-top:0.5rem;'>Runs all 6 models simultaneously — LR · Best LR · NB · SVM · BiLSTM · FinBERT</div>
    </div>
    """, unsafe_allow_html=True)
