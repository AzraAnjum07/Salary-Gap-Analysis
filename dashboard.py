from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_earners.csv"
MAPPINGS_PATH = BASE_DIR / "data" / "processed" / "mappings.json"
METRICS_PATH = BASE_DIR / "outputs" / "gam_metrics.json"
EFFECTS_PATH = BASE_DIR / "outputs" / "gam_effects.csv"
GENDER_PREDICTIONS_PATH = BASE_DIR / "outputs" / "gam_gender_predictions.csv"

INDUSTRY_BROAD_LABELS = {
    1: "Agriculture, forestry and fishing",
    2: "Mining and quarrying",
    3: "Manufacturing",
    4: "Electricity, gas and water",
    5: "Construction",
    6: "Trade and repair",
    7: "Transport and storage",
    8: "Accommodation and food services",
    9: "Information and communication",
}

OCCUPATION_BROAD_LABELS = {
    0: "Armed forces",
    1: "Managers",
    2: "Professionals",
    3: "Technicians and associate professionals",
    4: "Clerical support workers",
    5: "Service and sales workers",
    6: "Skilled agricultural workers",
    7: "Craft and related trades workers",
    8: "Plant and machine operators",
    9: "Elementary occupations",
}

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
def load_mappings():
    if MAPPINGS_PATH.exists():
        with MAPPINGS_PATH.open("r", encoding="utf-8") as mappings_file:
            return json.load(mappings_file)
    return {}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["gender"].isin([1, 2])].copy()
    mappings = load_mappings()

    for column, mapping_name in {
        "gender": "gender",
        "education_level": "education_level",
        "employment_status": "employment_status",
        "industry": "industry",
        "occupation": "occupation",
    }.items():
        mapping = mappings.get(mapping_name, {})
        df[f"{column}_label"] = df[column].map(
            lambda value: mapping.get(
                str(int(value)) if pd.notna(value) and float(value).is_integer() else str(value),
                f"{column.replace('_', ' ').title()} group",
            )
        )
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


@st.cache_data
def load_gender_predictions():
    if not GENDER_PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(GENDER_PREDICTIONS_PATH)


df = load_data()
metrics = load_metrics()
effects = load_effects()
gender_predictions = load_gender_predictions()
mappings = load_mappings()

st.sidebar.header("Filters")
all_genders = [gender for gender in ["Male", "Female"] if gender in df["gender_label"].unique()]
selected_genders = st.sidebar.multiselect("Gender", all_genders, default=all_genders)
filtered = df[df["gender_label"].isin(selected_genders)].copy()


def finish_axes(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return ax


def show_chart(fig, explanation):
    st.pyplot(fig)
    st.caption(explanation)
    plt.close(fig)


def readable_effect_label(feature, value, label_maps, index):
    code = int(value)
    exact_label = label_maps.get(str(code))
    if exact_label:
        return exact_label
    if feature == "Industry":
        broad_label = INDUSTRY_BROAD_LABELS.get(int(str(code)[0]), "Other industry sector")
    else:
        broad_label = OCCUPATION_BROAD_LABELS.get(code // 100, "Other occupation")
    return f"{broad_label} subgroup {index + 1}"


def broad_effect_label(feature, value):
    code = int(value)
    if feature == "Industry":
        return INDUSTRY_BROAD_LABELS.get(int(str(code)[0]), "Other industry sector")
    return OCCUPATION_BROAD_LABELS.get(code // 100, "Other occupation")

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
        show_chart(fig, "X-axis: gender. Y-axis: median observed earnings in rupees. This compares the typical earnings of Male and Female workers.")

    with col_right:
        education_summary = filtered.groupby("education_level_label")["earnings"].median().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(education_summary.index, education_summary.values, marker="o", color="#2ca25f", linewidth=2)
        ax.set_title("Median earnings by education level")
        ax.set_xlabel("Education level")
        ax.set_ylabel("Median earnings (₹)")
        ax.tick_params(axis="x", rotation=45)
        finish_axes(ax)
        show_chart(fig, "X-axis: education level. Y-axis: median observed earnings in rupees. Higher points indicate education groups with higher typical earnings.")

with tabs[1]:
    st.subheader("Exploratory findings")
    st.caption("These 15 charts reproduce the EDA notebook. Earnings charts are descriptive observed comparisons, not causal estimates.")
    chart_colors = ["#4c78a8", "#f58518"]
    mapped_occupation_labels = set(mappings.get("occupation", {}).values())
    mapped_occupation_data = filtered[filtered["occupation_label"].isin(mapped_occupation_labels)]
    top_occupation_labels = mapped_occupation_data["occupation_label"].value_counts().head(15).index
    top_occupation_data = mapped_occupation_data[mapped_occupation_data["occupation_label"].isin(top_occupation_labels)]

    occupation_counts = top_occupation_data["occupation_label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(occupation_counts.index, occupation_counts.values, color="#3a86ff")
    ax.set_title("Worker distribution across top 15 occupations")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Occupation")
    finish_axes(ax)
    show_chart(fig, "X-axis: number of workers. Y-axis: occupation. Longer bars show occupations containing more workers in the filtered dataset.")

    occupation_median = top_occupation_data.groupby("occupation_label")["earnings"].median().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(occupation_median.index, occupation_median.values, color="#2a9d8f")
    ax.set_title("Median earnings by top occupation")
    ax.set_xlabel("Median earnings (₹)")
    ax.set_ylabel("Occupation")
    finish_axes(ax)
    show_chart(fig, "X-axis: median observed earnings in rupees. Y-axis: occupation. Each bar shows the typical earnings for that occupation.")

    top_occupation_gender = top_occupation_data.groupby(["occupation_label", "gender_label"])["earnings"].median().unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    top_occupation_gender.plot.bar(ax=ax, color=chart_colors)
    ax.set_title("Gender-wise median earnings by occupation")
    ax.set_xlabel("Occupation")
    ax.set_ylabel("Median earnings (₹)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Gender")
    finish_axes(ax)
    show_chart(fig, "X-axis: occupation. Y-axis: median earnings in rupees. The two bars compare typical Male and Female earnings within each occupation.")

    education_counts = filtered["education_level_label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(education_counts.index, education_counts.values, color="#6a994e")
    ax.set_title("Education-level distribution")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Education level")
    finish_axes(ax)
    show_chart(fig, "X-axis: number of workers. Y-axis: education level. This shows how workers are distributed across education groups.")

    education_median = filtered.groupby("education_level_label")["earnings"].median().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(education_median.index, education_median.values, color="#80b918")
    ax.set_title("Median earnings by education level")
    ax.set_xlabel("Median earnings (₹)")
    ax.set_ylabel("Education level")
    finish_axes(ax)
    show_chart(fig, "X-axis: median observed earnings in rupees. Y-axis: education level. This compares typical earnings across education groups.")

    education_gender = filtered.groupby(["education_level_label", "gender_label"])["earnings"].median().unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    education_gender.plot.bar(ax=ax, color=chart_colors)
    ax.set_title("Gender-wise median earnings by education")
    ax.set_xlabel("Education level")
    ax.set_ylabel("Median earnings (₹)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Gender")
    finish_axes(ax)
    show_chart(fig, "X-axis: education level. Y-axis: median earnings in rupees. Each pair compares Male and Female typical earnings at that education level.")

    age_bins = [0, 20, 30, 40, 50, 60, 70, 200]
    age_labels = ["11–20", "21–30", "31–40", "41–50", "51–60", "61–70", "71+"]
    age_groups = pd.cut(filtered["age"], bins=age_bins, labels=age_labels, right=False)
    age_counts = age_groups.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(age_counts.index.astype(str), age_counts.values, color="#7a6ccf")
    ax.set_title("Age-group distribution")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Number of workers")
    finish_axes(ax)
    show_chart(fig, "X-axis: age group in years. Y-axis: number of workers. This shows the sample size in each age band.")

    age_earnings = filtered.assign(age_group=age_groups).groupby("age_group", observed=False)["earnings"].agg(["median", "mean"]).dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    age_earnings.plot.bar(ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_title("Mean and median earnings by age group")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Earnings (₹)")
    ax.legend(["Median", "Mean"])
    finish_axes(ax)
    show_chart(fig, "X-axis: age group. Y-axis: earnings in rupees. Median shows the typical worker; mean is more affected by unusually high earnings.")

    age_gender = filtered.assign(age_group=age_groups).groupby(["age_group", "gender_label"], observed=False)["earnings"].median().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    age_gender.plot.bar(ax=ax, color=chart_colors)
    ax.set_title("Gender-wise median earnings by age group")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Median earnings (₹)")
    ax.legend(title="Gender")
    finish_axes(ax)
    show_chart(fig, "X-axis: age group. Y-axis: median earnings in rupees. Each pair compares Male and Female typical earnings within an age band.")

    employment_counts = filtered["employment_status_label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(employment_counts.index, employment_counts.values, color="#e76f51")
    ax.set_title("Employment-status distribution")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Employment status")
    finish_axes(ax)
    show_chart(fig, "X-axis: number of workers. Y-axis: employment status. This shows the composition of workers by work arrangement.")

    employment_median = filtered.groupby("employment_status_label")["earnings"].median().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(employment_median.index, employment_median.values, color="#55a868")
    ax.set_title("Median earnings by employment status")
    ax.set_xlabel("Median earnings (₹)")
    ax.set_ylabel("Employment status")
    finish_axes(ax)
    show_chart(fig, "X-axis: median observed earnings in rupees. Y-axis: employment status. This compares typical earnings across work arrangements.")

    top_employment = filtered["employment_status_label"].value_counts().head(10).index
    employment_gender = filtered[filtered["employment_status_label"].isin(top_employment)].groupby(["employment_status_label", "gender_label"])["earnings"].median().unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    employment_gender.plot.bar(ax=ax, color=chart_colors)
    ax.set_title("Gender-wise median earnings by employment status")
    ax.set_xlabel("Employment status")
    ax.set_ylabel("Median earnings (₹)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Gender")
    finish_axes(ax)
    show_chart(fig, "X-axis: employment status. Y-axis: median earnings in rupees. The pairs compare Male and Female typical earnings for the ten largest status groups.")

    numeric_columns = [column for column in ["age", "formal_education_years", "earnings", "regular_earnings", "self_employed_earnings"] if column in filtered.columns]
    correlation = filtered[numeric_columns].corr()
    correlation_labels = {"age": "Age", "formal_education_years": "Formal education years", "earnings": "Total earnings", "regular_earnings": "Regular earnings", "self_employed_earnings": "Self-employed earnings"}
    labeled_correlation = correlation.rename(index=correlation_labels, columns=correlation_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(labeled_correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labeled_correlation.columns)), labeled_correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(labeled_correlation.index)), labeled_correlation.index)
    for row in range(len(labeled_correlation.index)):
        for column in range(len(labeled_correlation.columns)):
            ax.text(column, row, f"{labeled_correlation.iloc[row, column]:.3f}", ha="center", va="center")
    ax.set_title("Correlation matrix of numerical variables")
    fig.colorbar(image, ax=ax, label="Pearson correlation coefficient")
    show_chart(fig, "X-axis and Y-axis: numerical variables. Cell color and value show Pearson correlation, from -1 (opposite movement) to +1 (same movement).")

    fig, ax = plt.subplots(figsize=(12, 7))
    for gender, color in [("Male", "#4c78a8"), ("Female", "#f58518")]:
        gender_data = filtered[filtered["gender_label"] == gender]
        ax.scatter(gender_data["age"], gender_data["earnings"], alpha=0.25, s=16, color=color, label=gender)
        if len(gender_data) > 2:
            sorted_age = np.sort(gender_data["age"].to_numpy())
            coefficients = np.polyfit(gender_data["age"], gender_data["earnings"], 2)
            ax.plot(sorted_age, np.polyval(coefficients, sorted_age), color=color, linewidth=2, label=f"{gender} trend")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Earnings (₹)")
    ax.set_title("Age versus earnings by gender")
    ax.legend()
    finish_axes(ax)
    show_chart(fig, "X-axis: age in years. Y-axis: observed earnings in rupees. Points show workers; curved lines summarize the broad age trend for each gender.")

    fig, ax = plt.subplots(figsize=(12, 7))
    for gender, color in [("Male", "#4c78a8"), ("Female", "#f58518")]:
        gender_data = filtered[filtered["gender_label"] == gender]
        ax.scatter(gender_data["formal_education_years"], gender_data["earnings"], alpha=0.25, s=16, color=color, label=gender)
    ax.set_xlabel("Years of formal education")
    ax.set_ylabel("Earnings (₹)")
    ax.set_title("Formal education years versus earnings by gender")
    ax.legend()
    finish_axes(ax)
    show_chart(fig, "X-axis: completed years of formal education. Y-axis: observed earnings in rupees. Each point represents a worker, colored by gender.")

with tabs[2]:
    st.subheader("GAM model evaluation")
    st.info(
        "**What is a GAM?** A Generalized Additive Model estimates earnings as the sum of separate contributions from age, education, gender, employment status, industry, and occupation. "
        "It is used here because age and education may have curved rather than straight-line relationships with earnings. "
        "The GAM graphs show one variable's contribution while the other variables are held at typical values. "
        "The horizontal zero line is the model reference: bars above it raise the predicted earnings estimate, while bars below it lower that estimate. "
        "These are adjusted associations, not proof that a variable causes the difference."
    )
    if metrics is None:
        st.warning("GAM metrics are not available. Run `python src/gam_model.py` and refresh this page.")
    else:
        has_cv_metrics = "cv_r2_mean" in metrics
        if not has_cv_metrics:
            st.warning("The metrics file uses the previous evaluation format. Rerun `python src/gam_model.py` to generate 5-fold CV results.")
        st.markdown("**5-fold cross-validation on the training partition**")
        cv_cols = st.columns(3)
        cv_cols[0].metric(
            "CV R²",
            "Unavailable" if not has_cv_metrics else f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}",
        )
        cv_cols[1].metric(
            "CV RMSE",
            "Unavailable" if not has_cv_metrics else f"₹{metrics['cv_rmse_mean']:,.0f} ± ₹{metrics['cv_rmse_std']:,.0f}",
        )
        cv_cols[2].metric(
            "CV MAE",
            "Unavailable" if not has_cv_metrics else f"₹{metrics['cv_mae_mean']:,.0f} ± ₹{metrics['cv_mae_std']:,.0f}",
        )
        st.markdown("**Final evaluation on untouched test data**")
        metric_cols = st.columns(3)
        test_r2 = metrics.get("test_r2", metrics.get("r2"))
        test_rmse = metrics.get("test_rmse", metrics.get("rmse"))
        test_mae = metrics.get("test_mae", metrics.get("mae"))
        metric_cols[0].metric("Test R²", "Unavailable" if test_r2 is None else f"{test_r2:.4f}")
        metric_cols[1].metric("Test RMSE", "Unavailable" if test_rmse is None else f"₹{test_rmse:,.0f}")
        metric_cols[2].metric("Test MAE", "Unavailable" if test_mae is None else f"₹{test_mae:,.0f}")
        st.caption(f"Training rows: {metrics['train_size']:,} • Test rows: {metrics['test_size']:,}")
        if has_cv_metrics:
            st.success("Validation uses only the training partition; the final test partition is evaluated once at the end.")
            if metrics.get("cv_used_representative_subset", False):
                st.info(
                    f"CV used a reproducible representative subset of {metrics['cv_training_rows']:,} rows from the training partition to control memory. "
                    "The final GAM still used all training rows."
                )
        if "baseline_rmse" in metrics:
            improvement = 1 - test_rmse / metrics["baseline_rmse"]
            st.info(
                f"Held-out test performance: the GAM improves RMSE over a mean-earnings baseline by {improvement:.1%}. "
                "R² is the share of test-set earnings variation explained by the model; it is not a causal estimate."
            )

    if effects is None or effects.empty:
        st.warning("GAM effect data is not available. Run `python src/gam_model.py` and refresh this page.")
    else:
        st.subheader("GAM effect curves")
        st.markdown(
            "These curves show each variable's **partial effect on predicted earnings**, in rupees, while the other variables are held at typical values. "
            "The zero line is the model's reference level: a negative effect means lower predicted earnings than that reference, not negative salary. "
            "The shaded band is a 95% confidence interval. Category codes are converted to readable labels where mappings are available."
        )
        feature_order = ["Age", "Formal education years", "Gender", "Employment status", "Industry", "Occupation"]
        available_features = [feature for feature in feature_order if feature in effects["feature"].unique()]
        st.caption(f"Loaded effect curves: {', '.join(available_features)}")
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
                if feature == "Gender":
                    if gender_predictions is None:
                        gender_prediction_data = feature_df[["value", "effect"]].copy()
                        gender_prediction_data["gender_label"] = gender_prediction_data["value"].map(
                            {1.0: "Male", 2.0: "Female"}
                        )
                        gender_prediction_data["predicted_earnings"] = (
                            filtered["earnings"].mean() + gender_prediction_data["effect"]
                        )
                    else:
                        gender_prediction_data = gender_predictions
                    ax.bar(
                        gender_prediction_data["gender_label"],
                        gender_prediction_data["predicted_earnings"],
                        color=["#4c78a8", "#f58518"],
                    )
                    ax.set_title("Adjusted predicted earnings by gender")
                    ax.set_xlabel("Gender")
                    ax.set_ylabel("Predicted earnings (₹)")
                    for index, value in enumerate(gender_prediction_data["predicted_earnings"]):
                        ax.text(index, value, f"₹{value:,.0f}", ha="center", va="bottom")
                    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
                    finish_axes(ax)
                    show_chart(
                        fig,
                        "X-axis: gender. Y-axis: predicted earnings in rupees for a typical worker with other variables held at their reference values. This is an adjusted comparison, not a raw group average.",
                    )
                    continue
                if feature in ["Industry", "Occupation"]:
                    feature_df = feature_df.copy()
                    feature_df["category_label"] = feature_df["value"].map(
                        lambda value: broad_effect_label(feature, value)
                    )
                    feature_df = (
                        feature_df.groupby("category_label", as_index=False)[
                            ["effect", "lower", "upper"]
                        ]
                        .mean()
                        .sort_values("effect")
                    )
                effect_mapping_names = {
                    "Gender": "gender",
                    "Employment status": "employment_status",
                    "Industry": "industry",
                    "Occupation": "occupation",
                }
                label_maps = mappings.get(effect_mapping_names.get(feature, ""), {})
                if feature in ["Industry", "Occupation"]:
                    category_labels = feature_df["category_label"].tolist()
                else:
                    category_labels = [
                        readable_effect_label(feature, value, label_maps, index)
                        for index, value in enumerate(feature_df["value"])
                    ]
                if feature in ["Industry", "Occupation"]:
                    ax.barh(category_labels, feature_df["effect"], color="#ff7f0e")
                    ax.set_xlabel("Partial effect")
                    ax.set_ylabel(feature)
                    ax.grid(axis="x", alpha=0.25)
                else:
                    ax.bar(category_labels, feature_df["effect"], color="#ff7f0e")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Partial effect")
                    ax.tick_params(axis="x", rotation=45)
                ax.set_title(f"{feature} effect")
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
            finish_axes(ax)
            if feature == "Gender":
                explanation = (
                    "X-axis: gender. Y-axis: adjusted predicted earnings in rupees for a typical worker. "
                    "This compares model predictions after holding the other variables at their reference values."
                )
            else:
                explanation = (
                    f"X-axis: {feature.lower()}. Y-axis: partial effect on predicted earnings in rupees. "
                    "Positive values indicate a higher model estimate than the reference level; negative values indicate a lower estimate. "
                    "Industry and occupation show the 15 strongest effects for readability."
                )
            show_chart(
                fig,
                explanation,
            )

    st.markdown("---")
    st.markdown(
        "The GAM is used to estimate adjusted, potentially nonlinear associations between earnings and the selected variables. "
        "This supports interpretation beyond a raw descriptive comparison, but it remains an observational study and not proof of causation."
    )

st.markdown("---")
st.caption("Built for the salary-gap analysis project: EDA + GAM + dashboard presentation.")
