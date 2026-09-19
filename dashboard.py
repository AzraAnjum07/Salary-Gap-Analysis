import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_earners.csv"
METRICS_PATH = BASE_DIR / "outputs" / "gam_metrics.json"

st.set_page_config(page_title="Gender-Based Salary Gap Analysis", layout="wide")
st.title("Gender-Based Salary Gap Analysis")
st.caption("PLFS 2023–24 | Exploratory Data Analysis and GAM model evaluation")

if not DATA_PATH.exists():
    st.error("Processed data not found. Run notebooks/01_data_preparation.ipynb first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df["gender_label"] = df["gender"].map({1: "Male", 2: "Female", 3: "Other"}).fillna("Unknown")

st.sidebar.header("Filters")
genders = sorted(df["gender_label"].unique())
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
filtered = df[df["gender_label"].isin(selected_genders)].copy()

st.header("Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Workers", f"{len(filtered):,}")
c2.metric("Median earnings", f"₹{filtered['earnings'].median():,.0f}")
c3.metric("Mean earnings", f"₹{filtered['earnings'].mean():,.0f}")

st.header("EDA Findings")

st.subheader("Gender-wise earnings")
gender_summary = filtered.groupby("gender_label")["earnings"].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(gender_summary.index, gender_summary.values, color=["#377eb8", "#e41a1c", "#4daf4a"][:len(gender_summary)])
ax.set_ylabel("Median earnings (₹)")
ax.set_xlabel("Gender")
for i, value in enumerate(gender_summary.values):
    ax.text(i, value, f"₹{value:,.0f}", ha="center", va="bottom")
st.pyplot(fig)
plt.close(fig)

st.subheader("Education and earnings")
education_summary = filtered.groupby("education_level")["earnings"].median().sort_index()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(education_summary.index, education_summary.values, marker="o", color="#2ca25f")
ax.set_xlabel("Education level")
ax.set_ylabel("Median earnings (₹)")
ax.set_title("Median earnings by education level")
st.pyplot(fig)
plt.close(fig)

st.subheader("Age and earnings")
age_bins = [0, 20, 30, 40, 50, 60, 70, 200]
age_labels = ["11–20", "21–30", "31–40", "41–50", "51–60", "61–70", "71+"]
filtered["age_group"] = pd.cut(filtered["age"], bins=age_bins, labels=age_labels, right=False)
age_summary = filtered.groupby("age_group", observed=False)["earnings"].median()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(age_summary.index.astype(str), age_summary.values, color="#756bb1")
ax.set_xlabel("Age group")
ax.set_ylabel("Median earnings (₹)")
ax.set_title("Median earnings by age group")
st.pyplot(fig)
plt.close(fig)

st.subheader("Employment status and earnings")
employment_summary = filtered.groupby("employment_status")["earnings"].median().sort_values(ascending=True).tail(10)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(employment_summary.index.astype(str), employment_summary.values, color="#e6550d")
ax.set_xlabel("Median earnings (₹)")
ax.set_ylabel("Employment status")
ax.set_title("Median earnings by employment-status code")
st.pyplot(fig)
plt.close(fig)

st.subheader("Occupation distribution")
occupation_counts = filtered["occupation"].value_counts().head(10).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(occupation_counts.index.astype(str), occupation_counts.values, color="#3182bd")
ax.set_xlabel("Number of workers")
ax.set_ylabel("Occupation")
ax.set_title("Top occupations")
st.pyplot(fig)
plt.close(fig)

st.header("GAM Model Evaluation")
if METRICS_PATH.exists():
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['r2']:.4f}")
    m2.metric("RMSE", f"₹{metrics['rmse']:,.0f}")
    m3.metric("MAE", f"₹{metrics['mae']:,.0f}")
    st.write(f"Training rows: {metrics['train_size']:,} | Test rows: {metrics['test_size']:,}")
else:
    st.info("GAM metrics are not available yet. Run `python src/gam_model.py` first.")

st.header("Interpretation")
st.write(
    "EDA describes differences in earnings and worker distributions. "
    "The GAM adds a flexible model for examining how earnings are associated with age, "
    "education, gender, employment status, industry, and occupation. "
    "These are observational associations and should not be interpreted as proof of causation."
)
