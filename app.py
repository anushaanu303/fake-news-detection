"""Fake News Detection – news-intelligence dashboard (Streamlit).   Run: streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, MAX_BULK_ROWS, PREPROCESS_CONFIG, PROCESSED_DIR, SAMPLE_CSV
from src.data_loader import DataValidationError, read_csv, validate_dataset
from src.predict import FakeNewsPredictor, ModelNotAvailableError
from src.preprocessing import preprocess_series
from src import visualization as viz

st.set_page_config(page_title="News Intelligence | Fake News Detection", page_icon="📰",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1F3A 0%, #12325C 100%); }
[data-testid="stSidebar"] * { color: #E6EEF8 !important; }
.brand { font-size: 1.25rem; font-weight: 800; letter-spacing: .3px; margin-bottom: .2rem; }
.brand-sub { font-size: .78rem; opacity: .75; margin-bottom: 1.2rem; }
.hero { background: linear-gradient(120deg, #0B1F3A 0%, #1D4ED8 100%); color: #fff; padding: 1.8rem 2rem;
        border-radius: 18px; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(11,31,58,.25); }
.hero h1 { margin: 0 0 .3rem 0; font-size: 2rem; font-weight: 800; color: #fff; }
.hero p { margin: 0; opacity: .88; font-size: 1rem; }
.card { background: #fff; border: 1px solid #E5EAF2; border-radius: 14px; padding: 1.05rem 1.2rem;
        box-shadow: 0 2px 10px rgba(15,23,42,.06); height: 100%; }
.card .k { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: #64748B; font-weight: 600; }
.card .v { font-size: 1.65rem; font-weight: 800; color: #0B1F3A; margin-top: .15rem; }
.card .s { font-size: .8rem; color: #64748B; }
.disclaimer { background: #FFF7E6; border-left: 5px solid #F5A524; padding: .85rem 1rem; border-radius: 8px;
              color: #5B4100; font-size: .9rem; margin: 1rem 0; }
.demo { background: #FDECEC; border-left: 5px solid #E5484D; padding: .85rem 1rem; border-radius: 8px;
        color: #7A1F22; font-size: .9rem; margin: 0 0 1rem 0; }
.result { border-radius: 16px; padding: 1.3rem 1.6rem; color: #fff; margin: .6rem 0 1rem 0; }
.result.fake { background: linear-gradient(120deg, #B42318, #E5484D); }
.result.real { background: linear-gradient(120deg, #0B7A6D, #12A594); }
.result .lab { font-size: 2.1rem; font-weight: 800; }
.result .sub { opacity: .92; font-size: .92rem; }
.stButton > button { border-radius: 10px; font-weight: 600; padding: .55rem 1.4rem; }
@media (max-width: 768px) { .hero h1 { font-size: 1.4rem; } .card .v { font-size: 1.25rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def card(label, value, sub=""):
    return f'<div class="card"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def disclaimer():
    st.markdown(f'<div class="disclaimer"><b>⚠️ Important:</b> {DISCLAIMER}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained model…")
def load_predictor():
    try:
        return FakeNewsPredictor.load(), None
    except ModelNotAvailableError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    clean, report = validate_dataset(read_csv(path))
    clean["processed_text"] = preprocess_series(clean["text"], PREPROCESS_CONFIG)
    return clean, report


predictor, load_error = load_predictor()
meta = predictor.metadata if predictor else {}

st.sidebar.markdown('<div class="brand">📰 NEWS INTELLIGENCE</div><div class="brand-sub">Fake News Detection using NLP</div>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["🏠 Home Dashboard", "📝 Single News Prediction", "📂 Bulk CSV Prediction",
                                     "📈 Model Performance", "🔎 Dataset Explorer", "ℹ️ About Project"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")
if predictor:
    st.sidebar.caption(f"Active model: **{predictor.model_name}**")
else:
    st.sidebar.warning("Model not trained yet.")


def require_model():
    if predictor is None:
        st.error(f"🚫 No trained model available. {load_error}")
        st.code("python train_model.py --data data/sample_news.csv", language="bash")
        st.stop()


def demo_banner():
    if meta.get("dataset", {}).get("is_demo_dataset"):
        st.markdown('<div class="demo"><b>Demo model:</b> trained on a small <b>synthetic</b> sample dataset. '
                    'Scores below describe that toy data only, not real-world accuracy. Retrain on a real dataset '
                    'for meaningful results.</div>', unsafe_allow_html=True)


def best_result():
    return next((m for m in meta.get("models", []) if m["name"] == meta.get("best_model")), None)


# ------------------------------------------------------------------ Home
if page.startswith("🏠"):
    hero("News Intelligence Dashboard", "Machine-learning pattern analysis of news text, trained on labelled examples.")
    require_model()
    demo_banner()
    b = best_result()
    c = st.columns(4)
    c[0].markdown(card("Selected model", meta["best_model"], "by cross-validated macro-F1"), unsafe_allow_html=True)
    c[1].markdown(card("Test accuracy", f"{b['accuracy']:.1%}", "held-out test split"), unsafe_allow_html=True)
    c[2].markdown(card("Test F1 (REAL)", f"{b['f1']:.3f}", "harmonic mean of precision & recall"), unsafe_allow_html=True)
    c[3].markdown(card("Training articles", f"{meta['split']['train_rows']:,}",
                       f"{meta['split']['test_rows']:,} held out for testing"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    counts = meta["dataset"]["class_counts"]
    left.plotly_chart(viz.label_distribution_fig(pd.DataFrame({"label": ["FAKE"] * counts.get("FAKE", 0) + ["REAL"] * counts.get("REAL", 0)})),
                      width="stretch")
    right.plotly_chart(viz.model_comparison_fig(meta["models"]), width="stretch")
    disclaimer()
    st.caption(f"Trained {meta.get('trained_at', 'n/a')} on {meta['dataset']['file']}.")

# ------------------------------------------------------------------ Single
elif page.startswith("📝"):
    hero("Single News Prediction", "Paste an article and see which class its wording resembles in the training data.")
    require_model()
    demo_banner()
    text = st.text_area("News article text", height=280, placeholder="Paste the full news article or headline here…")
    go = st.button("🔍 Predict", type="primary")
    if go:
        try:
            res = predictor.predict_one(text)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            cls = res["label"].lower()
            conf = (f"Model confidence: {res['confidence']:.1%}" if res["confidence"] is not None
                    else "Confidence score is not available for this model type.")
            st.markdown(f'<div class="result {cls}"><div class="sub">Model prediction</div>'
                        f'<div class="lab">{"🚩" if cls == "fake" else "✅"} {res["label"]}</div>'
                        f'<div class="sub">{conf}</div></div>', unsafe_allow_html=True)
            if res["probabilities"]:
                st.progress(res["probabilities"].get("REAL", 0.0), text=f"P(REAL) = {res['probabilities'].get('REAL', 0):.1%}")
            terms = predictor.explain(text)
            if terms:
                st.markdown(f"**Why?** The wording of this text resembles training articles labelled **{res['label']}**. "
                            "The terms below contributed most to that prediction.")
                st.plotly_chart(viz.contribution_fig(terms, viz.COLORS[res["label"]]), width="stretch")
            else:
                st.info("No term-level explanation is available for this text/model.")
            disclaimer()
    else:
        disclaimer()

# ------------------------------------------------------------------ Bulk
elif page.startswith("📂"):
    hero("Bulk CSV Prediction", "Upload a CSV with a `text` column to classify many articles at once.")
    require_model()
    demo_banner()
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df, report = validate_dataset(read_csv(up), require_label=False)
        except DataValidationError as exc:
            st.error(str(exc))
            st.stop()
        if len(df) > MAX_BULK_ROWS:
            st.warning(f"Only the first {MAX_BULK_ROWS:,} rows will be processed.")
            df = df.head(MAX_BULK_ROWS)
        out = pd.concat([df.reset_index(drop=True), predictor.predict_many(df["text"])], axis=1)
        k = st.columns(3)
        k[0].markdown(card("Rows processed", f"{len(out):,}", f"{report['missing_text']} blank rows skipped"), unsafe_allow_html=True)
        k[1].markdown(card("Predicted FAKE", f"{(out['predicted_label'] == 'FAKE').sum():,}"), unsafe_allow_html=True)
        k[2].markdown(card("Predicted REAL", f"{(out['predicted_label'] == 'REAL').sum():,}"), unsafe_allow_html=True)
        st.write("")
        st.dataframe(out, width="stretch", height=380)
        st.download_button("⬇️ Download CSV report", out.to_csv(index=False).encode("utf-8"),
                           file_name="fake_news_predictions.csv", mime="text/csv")
        disclaimer()
    else:
        st.info("Required column: `text`. Any other columns are kept in the report.")
        disclaimer()

# ------------------------------------------------------------------ Performance
elif page.startswith("📈"):
    hero("Model Performance", "Metrics on the held-out test split. Nothing here is estimated or invented.")
    require_model()
    demo_banner()
    table = pd.DataFrame([{"Model": m["name"], "Accuracy": m["accuracy"], "Precision": m["precision"],
                           "Recall": m["recall"], "F1": m["f1"], "CV macro-F1 (train)": m["cv_f1_macro_mean"],
                           "Confidence available": "Yes" if m["supports_confidence"] else "No"}
                          for m in meta["models"]])
    st.subheader("Performance summary")
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns[1:6]}), width="stretch", hide_index=True)
    st.success(f"Selected model: **{meta['best_model']}** — chosen by {meta['selection_criterion']}.")
    a, b = st.columns(2)
    a.plotly_chart(viz.model_comparison_fig(meta["models"]), width="stretch")
    pick = b.selectbox("Confusion matrix for", [m["name"] for m in meta["models"]],
                       index=[m["name"] for m in meta["models"]].index(meta["best_model"]))
    sel = next(m for m in meta["models"] if m["name"] == pick)
    b.plotly_chart(viz.confusion_matrix_fig(sel["confusion_matrix"], title=pick), width="stretch")
    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown("""
- **Accuracy** – out of all test articles, the share the model labelled correctly.
- **Precision (REAL)** – of the articles the model called REAL, how many really were labelled REAL. High precision = few false alarms.
- **Recall (REAL)** – of all articles labelled REAL, how many the model found. High recall = few missed.
- **F1-score** – a single number balancing precision and recall (low if either is low).
- **Confusion matrix** – a 2×2 count table: correct predictions on the diagonal, mistakes off it.
- **Cross-validated macro-F1** – average F1 over both classes across training folds; used to choose the model.
""")
    with st.expander("Classification report"):
        st.dataframe(pd.DataFrame(sel["classification_report"]).T.round(3), width="stretch")
    st.subheader("Representative incorrect predictions")
    wrong = meta.get("misclassified_examples", [])
    if wrong:
        st.dataframe(pd.DataFrame(wrong), width="stretch", hide_index=True)
        st.markdown("**Possible reasons:** the article mixes sensational and neutral wording; it contains few "
                    "words the model learned from; or its style differs from the training data. The model "
                    "learns wording patterns, not facts, so well-written falsehoods or sensational true stories can be misjudged.")
    else:
        st.info("The selected model made no mistakes on this test split (common with very small or easy datasets, "
                "and not evidence of real-world accuracy).")

# ------------------------------------------------------------------ Explorer
elif page.startswith("🔎"):
    hero("Dataset Explorer", "Inspect data quality, class balance, text length and vocabulary.")
    src = st.radio("Dataset", ["Bundled sample (synthetic)", "Upload my own CSV"], horizontal=True)
    path, report = None, None
    if src.startswith("Upload"):
        up = st.file_uploader("CSV with `text` and `label` columns", type=["csv"], key="explorer")
        if up is None:
            st.stop()
        try:
            raw = read_csv(up)
            data, report = validate_dataset(raw)
            data["processed_text"] = preprocess_series(data["text"], PREPROCESS_CONFIG)
        except DataValidationError as exc:
            st.error(str(exc))
            st.stop()
    else:
        raw = read_csv(SAMPLE_CSV)
        data, report = load_dataset(str(SAMPLE_CSV))
    k = st.columns(4)
    k[0].markdown(card("Raw rows", f"{report['rows_raw']:,}"), unsafe_allow_html=True)
    k[1].markdown(card("Usable rows", f"{report['rows_clean']:,}"), unsafe_allow_html=True)
    k[2].markdown(card("Missing text", report["missing_text"]), unsafe_allow_html=True)
    k[3].markdown(card("Duplicates", report["duplicates"], f"{report['invalid_labels']} invalid labels"), unsafe_allow_html=True)
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Distribution & quality", "Text length", "Vocabulary", "Before / after preprocessing"])
    with t1:
        a, b = st.columns(2)
        a.plotly_chart(viz.label_distribution_fig(data), width="stretch")
        b.plotly_chart(viz.data_quality_fig(report), width="stretch")
        st.dataframe(raw.head(20), width="stretch")
    with t2:
        st.plotly_chart(viz.length_hist_fig(data), width="stretch")
    with t3:
        a, b = st.columns(2)
        a.plotly_chart(viz.top_words_fig(data.loc[data.label == "FAKE", "processed_text"], "Top words – FAKE", viz.COLORS["FAKE"]), width="stretch")
        b.plotly_chart(viz.top_words_fig(data.loc[data.label == "REAL", "processed_text"], "Top words – REAL", viz.COLORS["REAL"]), width="stretch")
        st.plotly_chart(viz.word_comparison_fig(data), width="stretch")
        st.caption("Frequency bar charts are used instead of word clouds: they are easier to read and compare.")
    with t4:
        ex = data.sample(min(5, len(data)), random_state=1)[["text", "processed_text"]]
        ex.columns = ["Raw text", "Processed text"]
        st.dataframe(ex, width="stretch", hide_index=True)

# ------------------------------------------------------------------ About
else:
    hero("About This Project", "Fake News Detection Using Natural Language Processing")
    disclaimer()
    st.markdown("""
### What it does
Text is cleaned (lowercase, HTML/URL/punctuation removal, tokenisation, stopword removal), converted to
**TF-IDF** features, and classified by **Logistic Regression**, **Multinomial Naive Bayes** and a
**Passive Aggressive** classifier. The model with the best cross-validated macro-F1 on the training data is kept;
the test split is used only to report results.

### Limits you should know
- The model learns **writing-style patterns** in its training data. It does **not** check facts or sources.
- Performance depends entirely on the dataset; it can fail on new topics, languages or writing styles.
- Never use a prediction as the sole basis for decisions about a person, publisher or article.

### Dataset
The bundled `data/sample_news.csv` is **synthetic**, template-generated for demonstration and tests
(`scripts/generate_sample_data.py`), created for this project with no third-party license. For real use, train on a
published labelled dataset (for example the ISOT Fake News dataset or a Kaggle fake/real news dataset) and **check
its license and citation requirements first**.
""")
    if meta:
        st.json({k: meta[k] for k in ("trained_at", "best_model", "tfidf", "preprocessing", "versions") if k in meta})
