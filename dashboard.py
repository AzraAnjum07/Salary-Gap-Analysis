from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
        .stApp { background: var(--background-color); color: var(--text-color); }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
        h1, h2, h3, p, label { color: var(--text-color); }
        .metric-card { background: #172B4D; color: #FFFFFF; padding: 1rem; border-radius: 0.5rem; }
        .metric-card h3, .metric-card h2 { color: #FFFFFF; }
        div[data-testid="stExpander"] { border-color: var(--border-color, rgba(128, 128, 128, 0.35)); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gender-Based Earnings Gap Analysis")
dataset_mode = st.radio(
    "Dataset",
    ["Original PLFS Dataset", "Upload Custom Dataset"],
    horizontal=True,
    key="dataset_mode",
)
previous_dataset_mode = st.session_state.get("_previous_dataset_mode")
if previous_dataset_mode is not None and previous_dataset_mode != dataset_mode:
    st.session_state.pop("user_dataset_report", None)
    st.session_state.pop("selected_gam_graph", None)
    st.session_state.pop("selected_upload_gam_graph", None)
st.session_state["_previous_dataset_mode"] = dataset_mode
if dataset_mode == "Upload Custom Dataset":
    if st.button("Reset to Original Dataset", type="secondary"):
        st.session_state.pop("user_dataset_report", None)
        st.session_state["dataset_mode"] = "Original PLFS Dataset"
        st.rerun()
    from src.upload_dashboard import render_upload_dashboard

    render_upload_dashboard()
    st.stop()
else:
    st.session_state.pop("user_dataset_report", None)

st.caption("Active dataset: Original PLFS Dataset • exploratory analysis + GAM")

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
def load_metrics(file_stamp=None):
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_effects(file_stamp=None):
    if not EFFECTS_PATH.exists():
        return None
    return pd.read_csv(EFFECTS_PATH)


@st.cache_data
def load_gender_predictions(file_stamp=None):
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
metrics = load_metrics(METRICS_PATH.stat().st_mtime_ns if METRICS_PATH.exists() else None)
effects = load_effects(EFFECTS_PATH.stat().st_mtime_ns if EFFECTS_PATH.exists() else None)
gender_predictions = load_gender_predictions(
    GENDER_PREDICTIONS_PATH.stat().st_mtime_ns if GENDER_PREDICTIONS_PATH.exists() else None
)
mappings = load_mappings()
training_baseline, training_category_counts = load_training_baseline_and_category_counts()

st.markdown("#### Filters")
all_genders = [gender for gender in ["Male", "Female"] if gender in df["gender_label"].unique()]
selected_genders = st.multiselect("Gender", all_genders, default=all_genders, key="original_gender_filter")
filtered = df[df["gender_label"].isin(selected_genders)].copy()
if filtered.empty:
    st.warning("Choose at least one gender to display the descriptive analysis.")
    st.stop()
active_section = st.radio("Analysis section", ["EDA", "GAM Analysis", "Conclusions"], horizontal=True, key="original_analysis_section")


def finish_axes(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return ax


def show_chart(fig, explanation):
    st.pyplot(fig)
    with st.expander("How to read this chart", expanded=False):
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


def _gam_plot_layout(fig, title, x_title, y_title):
    fig.update_layout(
        title=title,
        font={"size": 13},
        margin={"l": 55, "r": 25, "t": 60, "b": 55},
        legend={"orientation": "h", "y": -0.22},
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def _smooth_explanation(feature_df, feature):
    x_label = "age in years" if feature == "Age" else "completed years of formal education"
    plotted_values = feature_df["value"].to_numpy(dtype=float)
    parts = [
        f"This graph asks how the model's age-related pattern changes across {x_label}, and whether its fitted pattern differs for Male and Female workers.",
        f"X is {x_label}; the curve runs from about {plotted_values.min():.1f} to {plotted_values.max():.1f}. The model uses the training data's 1st–99th percentile range to avoid emphasizing extreme tails. Y is the smooth term's contribution in rupees, not a worker's full earnings. The model adds this term to its other terms and intercept to form a fitted earnings estimate.",
    ]
    trends = []
    for gender in ["Male", "Female"]:
        curve = feature_df[feature_df["gender_label"] == gender].sort_values("value")
        if curve.empty:
            continue
        values = curve["value"].to_numpy(dtype=float)
        effects_value = curve["effect"].to_numpy(dtype=float)
        quarter = max(1, len(curve) // 4)
        start_mean = float(np.mean(effects_value[:quarter]))
        end_mean = float(np.mean(effects_value[-quarter:]))
        direction = "higher" if end_mean > start_mean else "lower" if end_mean < start_mean else "about the same"
        trends.append(
            f"The {gender} curve is {direction} at the high end than at the low end (about ₹{start_mean:,.0f} versus ₹{end_mean:,.0f} when averaged over the first and last quarters of the plotted range)."
        )
    band_note = "The shaded ribbons are pointwise 95% confidence intervals from pyGAM; wider areas mean the fitted curve is less certain there, not that 95% of workers fall inside it."
    parts.append(
        "Each solid/dashed line is the model's learned curve for one gender, while the colored shading shows uncertainty. Moving right means older age or more completed schooling; a rising/falling line means that term contributes more/less across that range. "
        + band_note
    )
    parts.append(
        "pyGAM calculated these values from the fitted smooth term while the other input columns were held at the training-data reference profile. Zero is the smooth term's reference point in the additive model, so positive/negative means above/below that term reference—not added/subtracted rupees from a person's observed salary."
    )
    parts.append(
        " ".join(trends) + " The curves show adjusted associations in this dataset; they do not show that aging or schooling causes earnings to change, and their separation is not the full gender earnings gap."
    )
    return parts


def _categorical_explanation(profiles, feature, grouped):
    axis_name = {
        "Employment status": "employment-status group",
        "Industry": "NIC-2008 industry section",
        "Occupation": "NCO occupation group",
    }[feature]
    male = profiles[profiles["gender_label"] == "Male"].set_index("display_label")
    female = profiles[profiles["gender_label"] == "Female"].set_index("display_label")
    common = male.index.intersection(female.index)
    if len(common):
        gaps = male.loc[common, "predicted_earnings"] - female.loc[common, "predicted_earnings"]
        gap_sentence = f"Across these displayed categories, the average Male-minus-Female difference is ₹{gaps.mean():,.0f} in this constructed profile comparison."
        high_m = male.loc[common, "predicted_earnings"].idxmax()
        high_f = female.loc[common, "predicted_earnings"].idxmax()
        high_sentence = f"The highest displayed fitted profile is {high_m} for Male (₹{male.loc[high_m, 'predicted_earnings']:,.0f}) and {high_f} for Female (₹{female.loc[high_f, 'predicted_earnings']:,.0f})."
    else:
        gap_sentence = "A paired gender comparison is unavailable because the graph has no shared categories."
        high_sentence = ""
    grouping = (
        "Industry codes are grouped into NIC-2008 sections, and occupation codes into broad NCO groups; group values are weighted by their training counts."
        if grouped else "Each employment-status category is shown separately."
    )
    return [
        f"This graph asks how fitted earnings profiles vary across {axis_name} while keeping the remaining characteristics at the model's reference profile.",
        f"X is the model's fitted earnings estimate in rupees. Y lists the {axis_name}. Each point is a constructed profile estimate, not the observed average salary for that group.",
        "Blue points are Male and coral points are Female. The short connector makes the within-category distance easier to compare. Read the legend as well as the colors. " + grouping,
        "The points start from the GAM's gender-specific fitted estimate at its reference profile, then replace that category's reference contribution with the displayed category contribution. No confidence interval is available for these constructed profile estimates; do not read visual distance as statistical certainty.",
        f"{high_sentence} {gap_sentence} This is a model-adjusted association in the observed data, not proof that category or gender causes an earnings difference."
    ]


def _gender_explanation(predictions):
    values = predictions.set_index("gender_label")["predicted_earnings"]
    male_value, female_value = float(values["Male"]), float(values["Female"])
    gap = male_value - female_value
    direction = "higher" if gap >= 0 else "lower"
    return [
        "This graph asks what earnings the fitted GAM gives to Male and Female workers when it uses the same typical values for age, education, employment, industry, and occupation.",
        "X lists gender. Y is a fitted earnings estimate in rupees for one reference profile—not raw earnings and not a group average.",
        "The two bars are the model's estimates for that same profile. Compare their heights; the legend/labels identify the group.",
        "The estimates come directly from the fitted GAM's predictions after setting numeric inputs to training medians and categories to training modes, changing only gender. They are full fitted values, not partial contributions. This chart has no confidence intervals.",
        f"For this reference profile, the model estimates ₹{male_value:,.0f} for Male and ₹{female_value:,.0f} for Female; Male is ₹{abs(gap):,.0f} {direction}. It is one adjusted comparison and cannot show that gender caused a difference or describe every worker."
    ]


def _clear_original_graph_dialog():
    st.session_state.pop("selected_gam_graph", None)


@st.dialog("GAM graph details", width="large", on_dismiss=_clear_original_graph_dialog)
def _show_gam_dialog(graph):
    st.markdown(f"### {graph['title']}")
    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(graph["figure"], use_container_width=True, key=f"detail_{graph['key']}")
    with right:
        section_names = ["A. What question does this answer?", "B. Understanding the axes", "C. How to read it", "D. Where the numbers come from", "E. What we learned"]
        for heading, paragraph in zip(section_names, graph["explanation"]):
            st.markdown(f"**{heading}**")
            st.write(paragraph)
    if st.button("Close", key=f"close_{graph['key']}"):
        st.session_state.pop("selected_gam_graph", None)
        st.rerun()


def render_gam_gallery(effects, gender_predictions):
    graphs = []
    colors = {"Male": "#2563EB", "Female": "#D97757"}
    for feature in ["Age", "Formal education years"]:
        feature_df = effects[effects["feature"] == feature].copy()
        if feature_df.empty:
            continue
        fig = go.Figure()
        for gender, dash in [("Male", "solid"), ("Female", "dash")]:
            curve = feature_df[feature_df["gender_label"] == gender].sort_values("value")
            if curve.empty:
                continue
            fig.add_trace(go.Scatter(
                x=curve["value"], y=curve["upper"], mode="lines", line={"width": 0},
                showlegend=False, hoverinfo="skip", name=f"{gender} upper interval",
            ))
            fig.add_trace(go.Scatter(
                x=curve["value"], y=curve["lower"], mode="lines", line={"width": 0},
                fill="tonexty", fillcolor=("rgba(37,99,235,.14)" if gender == "Male" else "rgba(217,119,87,.14)"),
                showlegend=False, hoverinfo="skip", name=f"{gender} 95% interval",
            ))
            fig.add_trace(go.Scatter(
                x=curve["value"], y=curve["effect"], mode="lines", name=gender,
                line={"color": colors[gender], "width": 2.5, "dash": dash},
                customdata=np.column_stack([curve["lower"], curve["upper"]]),
                hovertemplate=f"{gender}<br>Value: %{{x:.2f}}<br>Contribution: ₹%{{y:,.0f}}<br>95% interval: ₹%{{customdata[0]:,.0f}} to ₹%{{customdata[1]:,.0f}}<extra></extra>",
            ))
        fig.add_hline(y=0, line_width=1, line_dash="dot")
        _gam_plot_layout(fig, f"Adjusted {feature.lower()} pattern by gender", feature, "Smooth-term contribution (₹)")
        graphs.append({"key": feature.lower().replace(" ", "_"), "title": f"{feature} pattern", "subtitle": "Does the modelled contribution change across this value?", "figure": fig, "explanation": _smooth_explanation(feature_df, feature)})

    gender_rows = gender_predictions.set_index("gender_label").reindex(["Male", "Female"]).reset_index()
    fig = go.Figure(go.Bar(
        x=gender_rows["gender_label"], y=gender_rows["predicted_earnings"],
        marker_color=[colors[gender] for gender in gender_rows["gender_label"]],
        text=[f"₹{value:,.0f}" for value in gender_rows["predicted_earnings"]], textposition="outside",
        hovertemplate="%{x}<br>Fitted earnings: ₹%{y:,.0f}<extra></extra>",
    ))
    _gam_plot_layout(fig, "Adjusted fitted earnings by gender", "Gender", "Fitted earnings estimate (₹)")
    graphs.append({"key": "gender", "title": "Gender comparison", "subtitle": "What does the model estimate for the same typical profile?", "figure": fig, "explanation": _gender_explanation(gender_predictions)})

    category_features = ["Employment status", "Industry", "Occupation"]
    for feature in category_features:
        feature_df = effects[effects["feature"] == feature].copy()
        if feature_df.empty:
            continue
        profiles, grouped = gender_category_profiles(feature_df, feature, gender_predictions)
        labels = profiles["display_label"].drop_duplicates().tolist()
        y_pos = {label: index for index, label in enumerate(labels)}
        fig = go.Figure()
        male = profiles[profiles["gender_label"] == "Male"].set_index("display_label")
        female = profiles[profiles["gender_label"] == "Female"].set_index("display_label")
        for label in labels:
            if label in male.index and label in female.index:
                fig.add_trace(go.Scatter(
                    x=[male.loc[label, "predicted_earnings"], female.loc[label, "predicted_earnings"]],
                    y=[y_pos[label], y_pos[label]], mode="lines", line={"color": "#A9B7B4", "width": 1.5},
                    showlegend=False, hoverinfo="skip",
                ))
        for group in ["Male", "Female"]:
            group_data = profiles[profiles["gender_label"] == group]
            fig.add_trace(go.Scatter(
                x=group_data["predicted_earnings"], y=[y_pos[label] for label in group_data["display_label"]],
                mode="markers", name=group, marker={"color": colors[group], "size": 10},
                customdata=np.column_stack([group_data["display_label"], group_data["gender_label"]]),
                hovertemplate="%{customdata[1]}<br>Category: %{customdata[0]}<br>Fitted estimate: ₹%{x:,.0f}<extra></extra>",
            ))
        fig.update_yaxes(tickmode="array", tickvals=list(y_pos.values()), ticktext=labels, autorange="reversed")
        _gam_plot_layout(fig, f"Adjusted earnings profiles: {feature}", "Fitted earnings estimate (₹)", feature)
        graphs.append({"key": feature.lower().replace(" ", "_"), "title": f"{feature} profiles", "subtitle": f"How do fitted profiles vary across {feature.lower()}?", "figure": fig, "explanation": _categorical_explanation(profiles, feature, grouped)})

    st.markdown("### GAM graph gallery")
    st.caption("Select **Explore graph** for the larger view and graph-specific explanation. Hover or tap a line or point to inspect its plotted values.")
    for start in range(0, len(graphs), 2):
        columns = st.columns(2)
        for column, graph in zip(columns, graphs[start:start + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### {graph['title']}")
                    st.caption(graph["subtitle"])
                    preview = go.Figure(graph["figure"])
                    preview.update_layout(height=300, margin={"l": 45, "r": 15, "t": 30, "b": 55}, font={"size": 11})
                    st.plotly_chart(preview, use_container_width=True, key=f"preview_{graph['key']}")
                    if st.button("Explore graph", key=f"open_{graph['key']}", type="primary"):
                        st.session_state["selected_gam_graph"] = graph["key"]
    selected_key = st.session_state.get("selected_gam_graph")
    selected_graph = next((graph for graph in graphs if graph["key"] == selected_key), None)
    if selected_graph is not None:
        _show_gam_dialog(selected_graph)

if active_section == "EDA":
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
    st.markdown(f"<p style='font-size:1.02rem; color:var(--text-color); margin-top:1rem;'>{overview_text}</p>", unsafe_allow_html=True)
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
        ax.bar(gender_summary.index, gender_summary.values, color=[{"Male": "#2563EB", "Female": "#D97757"}.get(label, "#287D78") for label in gender_summary.index])
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

if active_section == "EDA":
    st.subheader("Exploratory findings")
    st.caption("These 15 charts reproduce the EDA notebook. Earnings charts are descriptive observed comparisons, not causal estimates.")
    gender_colors = {"Male": "#2563EB", "Female": "#D97757"}
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
    top_occupation_gender.plot.bar(ax=ax, color=[gender_colors.get(label, "#287D78") for label in top_occupation_gender.columns])
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
    education_gender.plot.bar(ax=ax, color=[gender_colors.get(label, "#287D78") for label in education_gender.columns])
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
    age_earnings.plot.bar(ax=ax, color=["#287D78", "#172B4D"])
    ax.set_title("Mean and median earnings by age group")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Earnings (₹)")
    ax.legend(["Median", "Mean"])
    finish_axes(ax)
    show_chart(fig, "X-axis: age group. Y-axis: earnings in rupees. Median shows the typical worker; mean is more affected by unusually high earnings.")

    age_gender = filtered.assign(age_group=age_groups).groupby(["age_group", "gender_label"], observed=False)["earnings"].median().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    age_gender.plot.bar(ax=ax, color=[gender_colors.get(label, "#287D78") for label in age_gender.columns])
    ax.set_title("Gender-wise median earnings by age group")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Median earnings (₹)")
    ax.legend(title="Gender")
    finish_axes(ax)
    show_chart(fig, "X-axis: age group. Y-axis: median earnings in rupees. Each pair compares Male and Female typical earnings within an age band.")

    employment_counts = filtered["employment_status_label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(employment_counts.index, employment_counts.values, color="#287D78")
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
    employment_gender.plot.bar(ax=ax, color=[gender_colors.get(label, "#287D78") for label in employment_gender.columns])
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
    for gender, color in gender_colors.items():
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
    for gender, color in gender_colors.items():
        gender_data = filtered[filtered["gender_label"] == gender]
        ax.scatter(gender_data["formal_education_years"], gender_data["earnings"], alpha=0.25, s=16, color=color, label=gender)
    ax.set_xlabel("Years of formal education")
    ax.set_ylabel("Earnings (₹)")
    ax.set_title("Formal education years versus earnings by gender")
    ax.legend()
    finish_axes(ax)
    show_chart(fig, "X-axis: completed years of formal education. Y-axis: observed earnings in rupees. Each point represents a worker, colored by gender.")

artifacts_have_schema = bool(
    metrics
    and metrics.get("artifact_schema_version") == 4
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
    and effects["artifact_schema_version"].eq(4).all()
    and gender_predictions is not None
    and "artifact_schema_version" in gender_predictions.columns
    and gender_predictions["artifact_schema_version"].eq(4).all()
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

if active_section == "GAM Analysis":
    st.subheader("GAM model evaluation")
    st.markdown(
        "**What the GAM does:** It builds an earnings estimate by adding together the model's learned parts for age, education, gender, and work categories.\n\n"
        "**What a contribution means:** A smooth curve shows just one part of that sum, not anyone's full salary. In this model, zero is the smooth term's reference point: a positive value is above it and a negative value is below it.\n\n"
        "**Important:** These are adjusted patterns after accounting for the other model inputs. They describe associations in this dataset; they do not prove that a factor caused an earnings change."
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
        render_gam_gallery(effects, gender_predictions)
    st.markdown("---")
    st.markdown(
        "The GAM is used to estimate adjusted, potentially nonlinear associations between earnings and the selected variables. "
        "This supports interpretation beyond a raw descriptive comparison, but it remains an observational study and not proof of causation."
    )

if active_section == "Conclusions":
    st.subheader("Conclusions")
    st.markdown("### A. What does this dataset contain?")
    st.caption(
        "This summary always uses the full original PLFS analysis sample. The Gender filter changes the EDA charts only; it does not change the GAM or these conclusions."
    )
    st.write(
        f"The analyzed file contains {len(df):,} workers and records earnings, gender, age, formal education, employment status, industry, and occupation."
    )

    raw_gender_stats = df.groupby("gender_label")["earnings"].agg(["count", "median", "mean"])
    st.markdown("### B. What did EDA reveal?")
    if {"Male", "Female"}.issubset(raw_gender_stats.index):
        male_median = float(raw_gender_stats.loc["Male", "median"])
        female_median = float(raw_gender_stats.loc["Female", "median"])
        raw_median_gap = male_median - female_median
        female_median_difference_pct = (
            100 * raw_median_gap / male_median if male_median else float("nan")
        )
        raw_cols = st.columns(3)
        raw_cols[0].metric("Male workers: middle earnings", f"₹{male_median:,.0f}")
        raw_cols[1].metric("Female workers: middle earnings", f"₹{female_median:,.0f}")
        raw_cols[2].metric("Difference between middle values", f"₹{raw_median_gap:,.0f}")
        st.caption(
            f"Workers in this comparison: Male {int(raw_gender_stats.loc['Male', 'count']):,}; "
            f"Female {int(raw_gender_stats.loc['Female', 'count']):,}."
        )
        st.markdown(
            f"The middle value is ₹{male_median:,.0f} for Male workers and ₹{female_median:,.0f} for Female workers. "
            "Half of each group earned more than its middle value, and half earned less. "
            f"The Female group's middle value is {female_median_difference_pct:.1f}% lower. "
            "This first comparison uses actual earnings and does not account for differences in age, education, or type of work."
        )
    else:
        st.info("The cleaned analysis sample does not contain both Male and Female groups for this comparison.")

    st.markdown("### C. What did the GAM analysis reveal?")
    st.write(
        "The GAM compares model-estimated earnings after accounting for the other included characteristics. Its smooth curves show how the age and education parts change across their ranges; category profiles show fitted estimates for typical workers in different work groups. These are adjusted patterns, not proof of causes."
    )
    if artifacts_verified and gender_predictions is not None:
        adjusted = gender_predictions.set_index("gender_label")["predicted_earnings"]
        if {"Male", "Female"}.issubset(adjusted.index):
            male_adjusted = float(adjusted.loc["Male"])
            female_adjusted = float(adjusted.loc["Female"])
            adjusted_gap = male_adjusted - female_adjusted
            adjusted_cols = st.columns(3)
            adjusted_cols[0].metric("Estimated earnings: Male", f"₹{male_adjusted:,.0f}")
            adjusted_cols[1].metric("Estimated earnings: Female", f"₹{female_adjusted:,.0f}")
            adjusted_cols[2].metric("Estimated difference", f"₹{adjusted_gap:,.0f}")
            st.markdown(
                "The model compares Male and Female workers using the same typical age, schooling, and work characteristics. "
                "These are estimated earnings for that one example profile, not the average earnings of all workers."
            )
            st.markdown("### Main takeaway")
            if {"Male", "Female"}.issubset(raw_gender_stats.index):
                st.info(
                    f"In this sample, Male workers' middle earnings are higher (₹{male_median:,.0f}) than Female workers' "
                    f"(₹{female_median:,.0f}). When the model compares workers with the same typical characteristics, "
                    f"it still estimates higher earnings for Male (₹{male_adjusted:,.0f}) than Female (₹{female_adjusted:,.0f}). "
                    "This shows a pattern in this dataset; it cannot tell us that gender alone caused the difference."
                )
            else:
                st.info(
                    f"For workers with the same typical characteristics, the model estimates ₹{male_adjusted:,.0f} for Male "
                    f"and ₹{female_adjusted:,.0f} for Female. This is a pattern in the data, not proof of what caused it."
                )
    else:
        st.info("Verified GAM estimates are not available. Run the GAM model to generate current outputs.")

    if artifacts_verified:
        for feature in ["Age", "Formal education years"]:
            smooth_data = effects[(effects["feature"] == feature) & effects["gender_label"].isin(["Male", "Female"])]
            if smooth_data.empty:
                continue
            trend_parts = []
            for group in ["Male", "Female"]:
                curve = smooth_data[smooth_data["gender_label"] == group].sort_values("value")
                if curve.empty:
                    continue
                quarter = max(1, len(curve) // 4)
                start_mean = float(curve["effect"].iloc[:quarter].mean())
                end_mean = float(curve["effect"].iloc[-quarter:].mean())
                move = "rises" if end_mean - start_mean > 0 else "falls" if end_mean - start_mean < 0 else "stays similar"
                trend_parts.append(f"{group} curve {move} from roughly ₹{start_mean:,.0f} to ₹{end_mean:,.0f}")
            st.markdown(f"**{feature}:** " + "; ".join(trend_parts) + ". These are changes in that model component, not full salaries.")
        for feature in ["Employment status", "Industry", "Occupation"]:
            category_data = effects[effects["feature"] == feature].copy()
            if category_data.empty:
                continue
            profiles, _ = gender_category_profiles(category_data, feature, gender_predictions)
            profile_summaries = []
            for group in ["Male", "Female"]:
                group_profiles = profiles[profiles["gender_label"] == group]
                if group_profiles.empty:
                    continue
                highest = group_profiles.loc[group_profiles["predicted_earnings"].idxmax()]
                profile_summaries.append(
                    f"{group}'s highest displayed {feature.lower()} profile is {highest['display_label']} (₹{highest['predicted_earnings']:,.0f})"
                )
            if profile_summaries:
                st.markdown(
                    f"**{feature}:** " + "; ".join(profile_summaries)
                    + ". These are fitted comparisons for the model's typical profile, not actual group averages."
                )
        held_out_r2 = metrics.get("test_r2", metrics.get("r2"))
        if held_out_r2 is not None:
            st.markdown("### How well did the model match other workers?")
            st.markdown(
                f"When checked on workers it was not trained on, the model captured about **{held_out_r2:.1%} of the differences in earnings**. "
                "The rest was not explained by this model. This score does not tell us why the gender difference exists."
            )

    with st.expander("What these results cannot tell us"):
        st.markdown(
            "- The model can only account for the information included in it; other things may also affect earnings.\n"
            "- This dataset shows patterns, but cannot prove what caused them.\n"
            "- The model does not calculate a separate Male/Female difference for every job, industry, or employment type.\n"
            "- The first comparison uses actual group earnings; the second is an estimate for one example profile. They are different kinds of numbers."
        )

st.markdown("---")
st.caption("Built for the salary-gap analysis project: EDA + GAM + dashboard presentation.")
