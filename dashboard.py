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

# Four-digit values are NIC-2008 industry classes. Their first two digits
# identify the division, which maps to an official NIC-2008 section.
# Source: MoSPI, NIC 2008 Codes & Descriptions.
INDUSTRY_SECTION_RANGES = [
    (1, 3, "Agriculture, forestry and fishing"),
    (5, 9, "Mining and quarrying"),
    (10, 33, "Manufacturing"),
    (35, 35, "Electricity, gas, steam and air conditioning"),
    (36, 39, "Water supply, sewerage and waste management"),
    (41, 43, "Construction"),
    (45, 47, "Wholesale and retail trade; vehicle repair"),
    (49, 53, "Transportation and storage"),
    (55, 56, "Accommodation and food services"),
    (58, 63, "Information and communication"),
    (64, 66, "Financial and insurance activities"),
    (68, 68, "Real estate activities"),
    (69, 75, "Professional, scientific and technical activities"),
    (77, 82, "Administrative and support services"),
    (84, 84, "Public administration and defence"),
    (85, 85, "Education"),
    (86, 88, "Human health and social work"),
    (90, 93, "Arts, entertainment and recreation"),
    (94, 96, "Other service activities"),
    (97, 98, "Household employment and own-use production"),
    (99, 99, "Activities of extraterritorial organizations"),
]

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


@st.cache_data
def load_training_baseline_and_category_counts():
    """Recreate the GAM training split to get its exact baseline and weights."""
    from sklearn.model_selection import train_test_split

    features = [
        "age", "formal_education_years", "gender", "employment_status",
        "industry", "occupation", "earnings",
    ]
    model_data = df[features].copy()
    for column in features:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
    model_data = model_data[model_data["gender"].isin([1, 2])].dropna()
    X_train, _, _, _ = train_test_split(
        model_data[features[:-1]], model_data["earnings"],
        test_size=0.2, random_state=42,
    )

    baseline = {}
    for feature in features[:-1]:
        if feature in {"gender", "employment_status", "industry", "occupation"}:
            baseline[feature] = float(X_train[feature].mode().iloc[0])
        else:
            baseline[feature] = float(X_train[feature].median())
    counts = {
        feature: X_train[feature].value_counts().to_dict()
        for feature in ["employment_status", "industry", "occupation"]
    }
    return baseline, counts


df = load_data()
metrics = load_metrics()
effects = load_effects()
gender_predictions = load_gender_predictions()
mappings = load_mappings()
training_baseline, training_category_counts = load_training_baseline_and_category_counts()

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
    if isinstance(explanation, (list, tuple)):
        st.markdown("\n".join(f"- {item}" for item in explanation))
    else:
        st.caption(explanation)
    plt.close(fig)


def readable_effect_label(feature, value, label_maps, index):
    code = int(value)
    exact_label = label_maps.get(str(code))
    if exact_label:
        return exact_label
    if feature == "Occupation":
        broad_label = OCCUPATION_BROAD_LABELS.get(code // 100, "Other occupation")
        return broad_label
    return "Other industry category (label unavailable)"


def broad_effect_label(feature, value):
    code = int(value)
    return OCCUPATION_BROAD_LABELS.get(code // 100, "Other occupation")


def industry_section_label(value):
    # The clean CSV stores NIC codes as numbers, so a leading zero can be lost
    # (for example, 01111 becomes 1111). Restore the five-digit format first.
    division = int(str(int(value)).zfill(5)[:2])
    for first_division, last_division, label in INDUSTRY_SECTION_RANGES:
        if first_division <= division <= last_division:
            return label
    return "Other / unclassified industry code"


def summarize_smooth_effect(feature_df, feature, gender_label):
    feature_df = feature_df.sort_values("value").reset_index(drop=True)
    x = feature_df["value"].to_numpy(dtype=float)
    estimate = feature_df["effect"].to_numpy(dtype=float)
    lower = feature_df["lower"].to_numpy(dtype=float)
    upper = feature_df["upper"].to_numpy(dtype=float)
    smoothed = pd.Series(estimate).rolling(
        window=nine_point_window(len(estimate)), center=True, min_periods=1
    ).mean().to_numpy()
    slopes = np.gradient(smoothed, x)
    slope_scale = np.nanpercentile(np.abs(slopes), 75)
    tolerance = max(slope_scale * 0.12, 1e-8)
    directions = np.where(slopes > tolerance, 1, np.where(slopes < -tolerance, -1, 0))

    runs = []
    run_start = 0
    for index in range(1, len(directions) + 1):
        if index == len(directions) or directions[index] != directions[run_start]:
            runs.append((int(directions[run_start]), run_start, index - 1))
            run_start = index

    min_run = max(5, int(len(x) * 0.06))
    substantial_runs = [run for run in runs if run[2] - run[1] + 1 >= min_run]
    merged_runs = []
    for direction, start, end in substantial_runs:
        if merged_runs and merged_runs[-1][0] == direction:
            merged_runs[-1] = (direction, merged_runs[-1][1], end)
        else:
            merged_runs.append((direction, start, end))
    substantial_runs = merged_runs
    trend_names = {1: "increases", -1: "decreases", 0: "is relatively flat"}
    if substantial_runs:
        trend = "; then ".join(
            f"{trend_names[direction]} from about {x[start]:.1f} to {x[end]:.1f}"
            for direction, start, end in substantial_runs[:4]
        )
    else:
        net_change = smoothed[-1] - smoothed[0]
        trend = "is broadly flat" if abs(net_change) <= slope_scale * (x[-1] - x[0]) * 0.1 else (
            "generally increases" if net_change > 0 else "generally decreases"
        )

    turning_points = []
    for first, second in zip(substantial_runs, substantial_runs[1:]):
        if first[0] * second[0] < 0:
            turn_index = first[2]
            turning_points.append(f"a change in direction near {x[turn_index]:.1f}")
    turning_text = (
        " The estimated curve changes direction, with " + " and ".join(turning_points[:2]) + "."
        if turning_points
        else " No clear turning point is identified by the smoothed curve."
    )

    median_band_width = float(np.median(upper - lower))
    widest_index = int(np.argmax(upper - lower))
    band_text = (
        f" The shaded region is pyGAM's pointwise 95% confidence interval for the smooth term contribution; "
        f"its median width is ₹{median_band_width:,.0f}, and it is widest near {x[widest_index]:.1f}. "
        "A wider interval indicates greater uncertainty in that part of the curve."
    )
    x_description = "age in years" if feature == "Age" else "completed years of formal education"
    return (
        f"{gender_label}: {trend}; contribution ranges ₹{estimate.min():,.0f} to ₹{estimate.max():,.0f}.",
        f"{gender_label} 95% band: median width ₹{median_band_width:,.0f}; widest near {x[widest_index]:.1f}.",
        turning_text.strip(),
    )


def summarize_gender_smooths(feature_df, feature):
    curves = {
        gender: feature_df[feature_df["gender_label"] == gender].sort_values("value")
        for gender in ["Male", "Female"]
    }
    if any(curve.empty for curve in curves.values()):
        return "Gender-specific curves are missing from the saved GAM effects. Rerun the GAM script to generate the interaction-model output."

    male_curve = curves["Male"].set_index("value")
    female_curve = curves["Female"].set_index("value")
    shared_x = male_curve.index.intersection(female_curve.index)
    male_minus_female = (
        male_curve.loc[shared_x, "effect"] - female_curve.loc[shared_x, "effect"]
    )
    average_difference = float(male_minus_female.mean())
    direction = "higher" if average_difference >= 0 else "lower"
    x_description = "age in years" if feature == "Age" else "years of formal education"
    male_summary, male_uncertainty, male_turning = summarize_smooth_effect(
        curves["Male"], feature, "Male"
    )
    female_summary, female_uncertainty, female_turning = summarize_smooth_effect(
        curves["Female"], feature, "Female"
    )
    return [
        f"Axes: X = {x_description} (increases left to right); Y = gender-specific partial contribution to modeled earnings (₹).",
        "Zero line: no contribution from this smooth term; above zero adds to, below zero subtracts from the modeled estimate.",
        f"{male_summary} {male_turning} {male_uncertainty}",
        f"{female_summary} {female_turning} {female_uncertainty}",
        f"With other variables held at baseline, the Male smooth contribution averages ₹{abs(average_difference):,.0f} {direction} than Female. This is the smooth-term difference, not the full earnings gap or a causal effect.",
    ]


def nine_point_window(length):
    """Return an odd smoothing window no larger than the available curve."""
    return max(1, min(9, length if length % 2 else length - 1))


def gender_category_profiles(feature_df, feature, gender_predictions):
    """Create baseline-adjusted category profiles from additive GAM terms."""
    feature_column = {
        "Employment status": "employment_status",
        "Industry": "industry",
        "Occupation": "occupation",
    }[feature]
    detail = feature_df.copy()
    mapping = mappings.get(feature_column, {})
    if feature == "Industry":
        detail["display_label"] = detail["value"].map(industry_section_label)
        grouped = True
    elif feature == "Occupation":
        detail["display_label"] = detail["value"].map(lambda value: broad_effect_label(feature, value))
        grouped = True
    else:
        detail["display_label"] = [
            readable_effect_label(feature, value, mapping, index)
            for index, value in enumerate(detail["value"])
        ]
        grouped = False

    if grouped:
        counts = training_category_counts[feature_column]
        detail["training_count"] = detail["value"].map(
            lambda value: counts.get(value, counts.get(int(value), 0))
        )
        display_rows = []
        for label, group in detail.groupby("display_label", sort=False):
            weights = group["training_count"].to_numpy(dtype=float)
            display_rows.append({
                "display_label": label,
                "effect": float(np.average(group["effect"], weights=weights)),
                "weight": float(weights.sum()),
            })
        display = pd.DataFrame(display_rows)
        if feature == "Industry":
            order = [label for _, _, label in INDUSTRY_SECTION_RANGES]
        else:
            order = [OCCUPATION_BROAD_LABELS[index] for index in sorted(OCCUPATION_BROAD_LABELS)]
        display["display_label"] = pd.Categorical(display["display_label"], categories=order, ordered=True)
        display = display.sort_values("display_label")
    else:
        display = detail[["display_label", "value", "effect"]].copy()
        display["weight"] = 1.0

    baseline_value = training_baseline[feature_column]
    baseline_effect = float(detail.loc[np.isclose(detail["value"], baseline_value), "effect"].iloc[0])
    prediction_by_gender = gender_predictions.set_index("gender_label")["predicted_earnings"]
    rows = []
    for _, item in display.iterrows():
        contribution_delta = float(item["effect"] - baseline_effect)
        for gender in ["Male", "Female"]:
            rows.append({
                "display_label": str(item["display_label"]),
                "gender_label": gender,
                "predicted_earnings": float(prediction_by_gender[gender] + contribution_delta),
            })
    return pd.DataFrame(rows), grouped


def summarize_gender_category_profiles(profiles, feature, grouped):
    male = profiles[profiles["gender_label"] == "Male"]
    female = profiles[profiles["gender_label"] == "Female"]
    male_high = male.loc[male["predicted_earnings"].idxmax()]
    female_high = female.loc[female["predicted_earnings"].idxmax()]
    differences = male.set_index("display_label")["predicted_earnings"] - female.set_index("display_label")["predicted_earnings"]
    average_gap = float(differences.mean())
    if feature == "Industry":
        grouping_text = "Four- and five-digit NIC codes are restored to their five-digit form and grouped into official NIC-2008 sections; group estimates are weighted by training-category counts."
    elif feature == "Occupation":
        grouping_text = "Occupation codes are grouped into NCO major groups; group estimates are weighted by training-category counts."
    else:
        grouping_text = "Employment statuses are shown in code order with readable labels."
    category_axis = {
        "Industry": "NIC-2008 industry section",
        "Occupation": "NCO major group",
        "Employment status": "employment-status category",
    }[feature]
    return [
        f"Axes: X = adjusted GAM earnings estimate (₹); Y = {category_axis}. The legend identifies Male and Female.",
        f"Highest displayed estimates: Male, {male_high['display_label']} (₹{male_high['predicted_earnings']:,.0f}); Female, {female_high['display_label']} (₹{female_high['predicted_earnings']:,.0f}).",
        f"Mean Male-minus-Female difference across displayed categories: ₹{abs(average_gap):,.0f} at this constructed baseline. {grouping_text}",
        "These are complete adjusted profile estimates, not partial effects or observed group averages. No confidence interval is shown for these derived profiles.",
        "The model has a shared category term and no gender-by-category interaction. Paired points show each gender's fitted level; they do not imply that the category association itself differs by gender. This is not causal.",
    ]

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
    artifacts_have_schema = bool(
        metrics
        and metrics.get("artifact_schema_version") == 3
        and metrics.get("response_scale") == "raw_earnings_in_rupees"
        and metrics.get("model_terms") == (
            "s(age, by=male_indicator) + s(age, by=female_indicator) + "
            "s(formal_education_years, by=male_indicator) + "
            "s(formal_education_years, by=female_indicator) + f(gender) + "
            "f(employment_status) + f(industry) + f(occupation)"
        )
        and effects is not None
        and not effects.empty
        and "artifact_schema_version" in effects.columns
        and effects["artifact_schema_version"].eq(3).all()
        and gender_predictions is not None
        and "artifact_schema_version" in gender_predictions.columns
        and gender_predictions["artifact_schema_version"].eq(3).all()
    )
    run_id_fields = [
        bool(metrics and "run_id" in metrics),
        bool(effects is not None and "run_id" in effects.columns),
        bool(gender_predictions is not None and "run_id" in gender_predictions.columns),
    ]
    artifacts_have_run_ids = all(run_id_fields)
    artifacts_verified = artifacts_have_schema and (
        not any(run_id_fields)
        or (
            artifacts_have_run_ids
            and effects["run_id"].eq(metrics["run_id"]).all()
            and gender_predictions["run_id"].eq(metrics["run_id"]).all()
        )
    )
    st.markdown(
        "**GAM at a glance**\n"
        "- Shows adjusted earnings associations; it does not predict individual salaries.\n"
        "- Age and education each have separate Male and Female curves.\n"
        "- Effects use raw earnings in rupees; zero means no contribution from that term.\n"
        "- Curves are associations, not proof of cause."
    )
    if not artifacts_verified:
        st.warning(
            "The saved GAM outputs predate the current artifact metadata and cannot be verified as products of the current validation workflow. "
            "They are hidden here. Run `python src/gam_model.py` to complete the full training fit and generate verified metrics and effects."
        )
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

    if not artifacts_verified or effects is None or effects.empty:
        if artifacts_verified:
            st.warning("GAM effect data is not available. Run `python src/gam_model.py` and refresh this page.")
    else:
        st.subheader("GAM effect curves")
        feature_order = ["Age", "Formal education years", "Gender", "Employment status", "Industry", "Occupation"]
        available_features = [feature for feature in feature_order if feature in effects["feature"].unique()]
        st.caption(f"Loaded effect curves: {', '.join(available_features)}")
        for feature in feature_order:
            feature_df = effects[effects["feature"] == feature].sort_values("value")
            if feature_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 4.5))
            if feature in ["Age", "Formal education years"]:
                if "gender_label" not in feature_df.columns:
                    st.warning("Saved smooth curves predate gender-specific fitting. Rerun `python src/gam_model.py` to generate separate Male and Female curves.")
                    plt.close(fig)
                    continue
                colors = {"Male": "#2563eb", "Female": "#e76f51"}
                line_styles = {"Male": "-", "Female": "--"}
                for gender_label in ["Male", "Female"]:
                    gender_curve = feature_df[feature_df["gender_label"] == gender_label].sort_values("value")
                    if gender_curve.empty:
                        continue
                    ax.plot(
                        gender_curve["value"],
                        gender_curve["effect"],
                        color=colors[gender_label],
                        linestyle=line_styles[gender_label],
                        linewidth=2.2,
                        label=gender_label,
                    )
                    ax.fill_between(
                        gender_curve["value"],
                        gender_curve["lower"],
                        gender_curve["upper"],
                        color=colors[gender_label],
                        alpha=0.16,
                    )
                x_label = "Age (years)" if feature == "Age" else "Years of Formal Education"
                ax.set_title(f"Adjusted GAM {feature} Smooth by Gender")
                ax.set_xlabel(x_label)
                ax.set_ylabel("Partial Effect on Modeled Earnings (₹)")
                ax.legend(title="Gender")
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
                finish_axes(ax)
                show_chart(fig, summarize_gender_smooths(feature_df, feature))
                continue
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
                    ax.set_ylabel("Adjusted Earnings Estimate (₹)")
                    for index, value in enumerate(gender_prediction_data["predicted_earnings"]):
                        ax.text(index, value, f"₹{value:,.0f}", ha="center", va="bottom")
                    finish_axes(ax)
                    show_chart(
                        fig,
                        [
                            "X-axis: gender. Y-axis: GAM-fitted earnings estimate in rupees for a typical baseline profile.",
                            f"Male: ₹{gender_prediction_data.iloc[0]['predicted_earnings']:,.0f}; Female: ₹{gender_prediction_data.iloc[1]['predicted_earnings']:,.0f}.",
                            f"Difference at this baseline: ₹{abs(gender_prediction_data.iloc[0]['predicted_earnings'] - gender_prediction_data.iloc[1]['predicted_earnings']):,.0f}. This is adjusted, not a raw group average.",
                            "Other variables are held at training baseline values. This comparison is not causal; no confidence interval is shown.",
                        ],
                    )
                    continue
                profiles, grouped_categories = gender_category_profiles(
                    feature_df, feature, gender_predictions
                )
                category_labels = profiles["display_label"].drop_duplicates().tolist()
                fig.set_size_inches(10, max(4.8, 0.42 * len(category_labels) + 1.7))
                positions = {label: index for index, label in enumerate(category_labels)}
                colors = {"Male": "#2563eb", "Female": "#e76f51"}
                offsets = {"Male": -0.14, "Female": 0.14}
                male_estimates = profiles[profiles["gender_label"] == "Male"].set_index("display_label")
                female_estimates = profiles[profiles["gender_label"] == "Female"].set_index("display_label")
                for label in category_labels:
                    ax.plot(
                        [male_estimates.loc[label, "predicted_earnings"], female_estimates.loc[label, "predicted_earnings"]],
                        [positions[label], positions[label]],
                        color="#9aa3ad", linewidth=1, alpha=0.65, zorder=1,
                    )
                for gender in ["Male", "Female"]:
                    gender_profiles = profiles[profiles["gender_label"] == gender]
                    ax.scatter(
                        gender_profiles["predicted_earnings"],
                        [positions[label] + offsets[gender] for label in gender_profiles["display_label"]],
                        color=colors[gender], s=58, label=gender, zorder=2,
                    )
                ax.set_yticks(list(positions.values()), labels=category_labels)
                ax.set_xlabel("Adjusted Modeled Earnings (₹)")
                ax.set_ylabel("NIC-2008 industry section" if feature == "Industry" else "NCO major group" if feature == "Occupation" else "Employment status")
                title_feature = "NIC-2008 Industry Section" if feature == "Industry" else "Occupation Major Group" if feature == "Occupation" else feature
                ax.set_title(f"Adjusted Earnings Profiles by {title_feature} and Gender")
                ax.legend(title="Gender")
                ax.grid(axis="x", alpha=0.25)
                finish_axes(ax)
                show_chart(fig, summarize_gender_category_profiles(profiles, feature, grouped_categories))
                continue

    st.markdown("---")
    st.markdown(
        "The GAM is used to estimate adjusted, potentially nonlinear associations between earnings and the selected variables. "
        "This supports interpretation beyond a raw descriptive comparison, but it remains an observational study and not proof of causation."
    )

st.markdown("---")
st.caption("Built for the salary-gap analysis project: EDA + GAM + dashboard presentation.")
