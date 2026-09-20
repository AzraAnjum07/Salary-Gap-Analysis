from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_earners.csv"
METRICS_PATH = BASE_DIR / "outputs" / "gam_metrics.json"
EFFECTS_PATH = BASE_DIR / "outputs" / "gam_effects.csv"

st.set_page_config(page_title="Salary Gap Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .main { background: #f4f7fb; }
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        h1 { color: #0b2545; }
        h2 { color: #123b66; margin-top: 1.2rem; }
        .metric-card { background: linear-gradient(135deg, #0b2545, #123b66); color: white; padding: 1rem; border-radius: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gender-Based Salary Gap Analysis")
st.caption("PLFS dataset • exploratory analysis + generalized additive model (GAM)")

if not DATA_PATH.exists():
    st.error("Processed data not found. Run the data preparation notebook first.")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["gender_label"] = df["gender"].map({1: "Male", 2: "Female", 3: "Other"}).fillna("Unknown")
    return df

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_effects():
    if not EFFECTS_PATH.exists():
        return None
    return pd.read_csv(EFFECTS_PATH)


df = load_data()
metrics = load_metrics()
effects = load_effects()

st.sidebar.header("Filters")
all_genders = sorted(df["gender_label"].dropna().unique())
selected_genders = st.sidebar.multiselect("Gender", all_genders, default=all_genders)
filtered = df[df["gender_label"].isin(selected_genders)].copy()

with st.sidebar:
    st.markdown("---")
    st.caption("The dashboard distinguishes descriptive findings from adjusted GAM associations.")

overview_col1, overview_col2, overview_col3 = st.columns(3)
overview_col1.markdown(
    "<div class='metric-card'><h3>Workers</h3><h2>{:,}</h2></div>".format(len(filtered)),
    unsafe_allow_html=True,
)
overview_col2.markdown(
    "<div class='metric-card'><h3>Median earnings</h3><h2>₹{:,.0f}</h2></div>".format(
        filtered["earnings"].median()
    ),
    unsafe_allow_html=True,
)
overview_col3.markdown(
    "<div class='metric-card'><h3>Mean earnings</h3><h2>₹{:,.0f}</h2></div>".format(
        filtered["earnings"].mean()
    ),
    unsafe_allow_html=True,
)

overview_text = (
    "This dashboard combines exploratory data analysis with a generalized additive model to explain "
    "how earnings vary across age, education, gender, employment status, occupation, and industry."
)
st.markdown(f"<p style='font-size:1.02rem; color:#213a5b; margin-top:1rem;'>{overview_text}</p>", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "EDA", "GAM"])

with tabs[0]:
    st.subheader("Project summary")
    st.markdown(
        "- EDA explores the raw patterns in the dataset and helps identify where salary gaps are strongest.\n"
        "- GAM provides a flexible statistical model to estimate adjusted relationships between earnings and several explanatory variables.\n"
        "- The dashboard is designed for interpretation, not for proving causality."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        gender_summary = filtered.groupby("gender_label")["earnings"].median().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(gender_summary.index, gender_summary.values, color=["#4c78a8", "#f58518", "#54a24b"][:len(gender_summary)])
        ax.set_title("Median earnings by gender")
        ax.set_ylabel("Median earnings (₹)")
        for i, value in enumerate(gender_summary.values):
            ax.text(i, value + max(gender_summary.values) * 0.02, f"₹{value:,.0f}", ha="center")
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        education_summary = filtered.groupby("education_level")["earnings"].median().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(education_summary.index, education_summary.values, marker="o", color="#2ca25f", linewidth=2)
        ax.set_title("Median earnings by education level")
        ax.set_xlabel("Education level")
        ax.set_ylabel("Median earnings (₹)")
        st.pyplot(fig)
        plt.close(fig)

with tabs[1]:
    st.subheader("Exploratory findings")

    col1, col2 = st.columns(2)
    with col1:
        age_bins = [0, 20, 30, 40, 50, 60, 70, 200]
        age_labels = ["11–20", "21–30", "31–40", "41–50", "51–60", "61–70", "71+"]
        filtered_copy = filtered.copy()
        filtered_copy["age_group"] = pd.cut(filtered_copy["age"], bins=age_bins, labels=age_labels, right=False)
        age_summary = filtered_copy.groupby("age_group", observed=False)["earnings"].median()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(age_summary.index.astype(str), age_summary.values, color="#7a6ccf")
        ax.set_title("Median earnings by age group")
        ax.set_ylabel("Median earnings (₹)")
        ax.set_xlabel("Age group")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        employment_summary = filtered.groupby("employment_status")["earnings"].median().sort_values().tail(8)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([str(v) for v in employment_summary.index], employment_summary.values, color="#e76f51")
        ax.set_title("Top employment-status earnings")
        ax.set_xlabel("Median earnings (₹)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Occupation distribution")
    occupation_counts = filtered["occupation"].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([str(v) for v in occupation_counts.index], occupation_counts.values, color="#3a86ff")
    ax.set_title("Top 10 occupations by worker count")
    ax.set_xlabel("Number of workers")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)

with tabs[2]:
    st.subheader("GAM model evaluation")
    if metrics is None:
        st.info("Run `python src/gam_model.py` to generate `outputs/gam_metrics.json`.")
    else:
        metric_cols = st.columns(3)
        metric_cols[0].metric("R²", f"{metrics['r2']:.4f}")
        metric_cols[1].metric("RMSE", f"₹{metrics['rmse']:,.0f}")
        metric_cols[2].metric("MAE", f"₹{metrics['mae']:,.0f}")
        st.caption(f"Training rows: {metrics['train_size']:,} • Test rows: {metrics['test_size']:,}")

    if effects is None:
        st.info("`outputs/gam_effects.csv` is not available yet. Run the GAM script to produce the effect curves.")
    else:
        st.subheader("GAM effect curves")
        feature_order = ["Age", "Formal education years", "Gender", "Employment status", "Industry", "Occupation"]
        for feature in feature_order:
            feature_df = effects[effects["feature"] == feature].sort_values("value")
            if feature_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 4.5))
            if feature in ["Age", "Formal education years"]:
                ax.plot(feature_df["value"], feature_df["effect"], color="#0d6efd", linewidth=2)
                ax.fill_between(feature_df["value"], feature_df["lower"], feature_df["upper"], alpha=0.2)
                ax.set_title(f"{feature} effect")
                ax.set_xlabel(feature)
                ax.set_ylabel("Partial effect")
            else:
                ax.bar(feature_df["value"].astype(str), feature_df["effect"], color="#ff7f0e")
                ax.axhline(0, color="black", linewidth=1)
                ax.set_title(f"{feature} effect")
                ax.set_xlabel(feature)
                ax.set_ylabel("Partial effect")
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    st.markdown(
        "The GAM is used to estimate adjusted, potentially nonlinear associations between earnings and the selected variables. "
        "This supports interpretation beyond a raw descriptive comparison, but it remains an observational study and not proof of causation."
    )

st.markdown("---")
st.caption("Built for the salary-gap analysis project: EDA + GAM + dashboard presentation.")
