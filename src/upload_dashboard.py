"""In-session EDA and GAM workflow for user-supplied CSV files."""

from __future__ import annotations

import hashlib
import io
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pygam import LinearGAM, f, s
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


MAX_MODEL_TRAIN_ROWS = 15_000
MAX_TEST_SCORING_ROWS = 20_000
MAX_CATEGORY_LEVELS = 20


def _suggest_column(columns, keywords):
    lowered = {str(column).lower().replace(" ", "_"): column for column in columns}
    for keyword in keywords:
        for normalized, original in lowered.items():
            if keyword in normalized:
                return original
    return columns[0] if columns else None


def _column_index(columns, guess):
    return columns.index(guess) if guess in columns else 0


def _read_csv(uploaded_file):
    return pd.read_csv(io.BytesIO(uploaded_file.getvalue()), encoding="utf-8-sig")


def _prepare_frame(raw, outcome_col, gender_col, male_value, female_value, numeric_cols, category_cols):
    if male_value == female_value:
        raise ValueError("Choose two different values for the Male and Female groups.")
    if len(numeric_cols) > 4 or len(category_cols) > 4:
        raise ValueError("Choose no more than four numeric and four category columns for the model.")

    selected = list(dict.fromkeys([outcome_col, gender_col, *numeric_cols, *category_cols]))
    frame = raw[selected].copy()
    frame["earnings"] = pd.to_numeric(frame[outcome_col], errors="coerce")
    frame["__gender_value"] = frame[gender_col].astype("string")
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    chosen = frame["__gender_value"].isin([male_value, female_value])
    frame = frame.loc[chosen].copy()
    frame["gender_label"] = np.where(frame["__gender_value"] == male_value, "Male", "Female")
    for column in category_cols:
        frame[column] = frame[column].astype("string").fillna("Unknown").astype(str)

    complete = frame["earnings"].notna() & np.isfinite(frame["earnings"])
    for column in numeric_cols:
        complete &= frame[column].notna() & np.isfinite(frame[column])
    frame = frame.loc[complete].copy()
    if frame.empty:
        raise ValueError("No usable rows remain. Check the selected columns and missing values.")
    if set(frame["gender_label"].unique()) != {"Male", "Female"}:
        raise ValueError(
            "Both selected groups need usable earnings rows. Check the column mapping and missing values."
        )

    return frame, len(raw) - len(frame)


def _fit_uploaded_gam(frame, numeric_cols, category_cols):
    counts = frame["gender_label"].value_counts()
    if len(frame) < 50 or counts.get("Male", 0) < 20 or counts.get("Female", 0) < 20:
        return {
            "fit_available": False,
            "message": "The file has too few complete rows in one or both selected groups for a useful GAM. EDA is still available; use at least 50 complete rows and 20 per group.",
        }
    for column in numeric_cols:
        if frame[column].nunique() < 5:
            return {
                "fit_available": False,
                "message": f"'{column}' has fewer than five distinct numeric values, so it cannot form a useful smooth curve. Remove it from the numeric selections or treat it as a category.",
            }

    model_frame = frame.copy()
    model_frame["__gender_code"] = (model_frame["gender_label"] == "Female").astype(int)

    category_info = {}
    for column in category_cols:
        frequency = model_frame[column].value_counts()
        if len(frequency) > MAX_CATEGORY_LEVELS:
            keep = set(frequency.head(MAX_CATEGORY_LEVELS - 1).index)
            model_frame[column] = model_frame[column].map(
                lambda value: value if value in keep else "__OTHER_CATEGORIES__"
            )
        levels = list(pd.unique(model_frame[column]))
        level_codes = {level: index for index, level in enumerate(levels)}
        model_frame[f"__category_{column}"] = model_frame[column].map(level_codes).astype(int)
        category_info[column] = {
            "levels": levels,
            "codes": level_codes,
            "term_index": None,
            "counts": model_frame[column].value_counts().to_dict(),
        }

    feature_columns = list(numeric_cols)
    feature_columns.append("__gender_code")
    feature_columns.extend(f"__category_{column}" for column in category_cols)
    gender_index = len(numeric_cols)
    category_indices = {
        column: gender_index + 1 + index for index, column in enumerate(category_cols)
    }
    male_indicator_index = gender_index + 1 + len(category_cols)
    female_indicator_index = male_indicator_index + 1
    matrix_parts = [model_frame[column].to_numpy(dtype=float) for column in numeric_cols]
    matrix_parts.append(model_frame["__gender_code"].to_numpy(dtype=float))
    matrix_parts.extend(
        model_frame[f"__category_{column}"].to_numpy(dtype=float)
        for column in category_cols
    )
    gender_codes = model_frame["__gender_code"].to_numpy(dtype=float)
    matrix_parts.extend([1.0 - gender_codes, gender_codes])
    X = np.column_stack(matrix_parts)
    y = model_frame["earnings"].to_numpy(dtype=float)

    indices = np.arange(len(model_frame))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=gender_codes,
    )
    rng = np.random.default_rng(42)
    fit_indices = train_indices
    if len(fit_indices) > MAX_MODEL_TRAIN_ROWS:
        fit_indices = np.sort(rng.choice(fit_indices, MAX_MODEL_TRAIN_ROWS, replace=False))
    score_indices = test_indices
    if len(score_indices) > MAX_TEST_SCORING_ROWS:
        score_indices = np.sort(rng.choice(score_indices, MAX_TEST_SCORING_ROWS, replace=False))

    terms = []
    smooth_terms = {}
    for index, column in enumerate(numeric_cols):
        n_splines = min(10, max(5, int(model_frame[column].nunique())))
        smooth_terms[(column, "Male")] = len(terms)
        terms.append(s(index, by=male_indicator_index, n_splines=n_splines))
        smooth_terms[(column, "Female")] = len(terms)
        terms.append(s(index, by=female_indicator_index, n_splines=n_splines))

    gender_term_index = len(terms)
    terms.append(f(gender_index))
    for column in category_cols:
        category_info[column]["term_index"] = len(terms)
        terms.append(f(category_indices[column]))
    if not terms:
        raise ValueError("Choose at least one model input column in addition to gender.")

    term_expression = terms[0]
    for term in terms[1:]:
        term_expression = term_expression + term

    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        gam = LinearGAM(term_expression, max_iter=100).fit(X[fit_indices], y[fit_indices])
    predictions = gam.predict(X[score_indices])
    metrics = {
        "test_r2": float(r2_score(y[score_indices], predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y[score_indices], predictions))),
        "test_mae": float(mean_absolute_error(y[score_indices], predictions)),
        "train_rows": int(len(fit_indices)),
        "train_partition_rows": int(len(train_indices)),
        "test_rows": int(len(score_indices)),
        "test_partition_rows": int(len(test_indices)),
        "fit_warnings": [str(item.message) for item in fit_warnings[:3]],
    }

    baseline = np.zeros(X.shape[1], dtype=float)
    for index, column in enumerate(numeric_cols):
        baseline[index] = float(model_frame.iloc[fit_indices][column].median())
    baseline[gender_index] = 0.0
    for column in category_cols:
        encoded_col = f"__category_{column}"
        baseline[category_indices[column]] = float(
            model_frame.iloc[fit_indices][encoded_col].mode().iloc[0]
        )
    baseline[male_indicator_index] = 1.0
    baseline[female_indicator_index] = 0.0

    gender_predictions = {}
    for label, code in [("Male", 0.0), ("Female", 1.0)]:
        row = baseline.copy()
        row[gender_index] = code
        row[male_indicator_index] = 1.0 - code
        row[female_indicator_index] = code
        gender_predictions[label] = float(gam.predict(row.reshape(1, -1))[0])

    smooth_effects = {}
    for index, column in enumerate(numeric_cols):
        lo = float(model_frame.iloc[fit_indices][column].quantile(0.01))
        hi = float(model_frame.iloc[fit_indices][column].quantile(0.99))
        values = np.linspace(lo, hi, 80)
        curves = {}
        for label, code in [("Male", 0.0), ("Female", 1.0)]:
            grid = np.tile(baseline, (len(values), 1))
            grid[:, index] = values
            grid[:, gender_index] = code
            grid[:, male_indicator_index] = 1.0 - code
            grid[:, female_indicator_index] = code
            effect, interval = gam.partial_dependence(
                term=smooth_terms[(column, label)], X=grid, width=0.95
            )
            curves[label] = {
                "x": values,
                "effect": np.asarray(effect),
                "lower": interval[:, 0],
                "upper": interval[:, 1],
            }
        smooth_effects[column] = curves

    category_effects = {}
    for column in category_cols:
        info = category_info[column]
        levels = info["levels"]
        codes = info["codes"]
        grid = np.tile(baseline, (len(levels), 1))
        grid[:, category_indices[column]] = [codes[level] for level in levels]
        effect, interval = gam.partial_dependence(
            term=info["term_index"], X=grid, width=0.95
        )
        category_effects[column] = pd.DataFrame({
            "category": ["Other / less common" if value == "__OTHER_CATEGORIES__" else value for value in levels],
            "effect": effect,
            "lower": interval[:, 0],
            "upper": interval[:, 1],
            "workers": [info["counts"][value] for value in levels],
        })

    return {
        "fit_available": True,
        "gam": gam,
        "metrics": metrics,
        "gender_predictions": gender_predictions,
        "smooth_effects": smooth_effects,
        "category_effects": category_effects,
        "numeric_cols": list(numeric_cols),
        "category_cols": list(category_cols),
        "clean_frame": model_frame,
        "gender_term_index": gender_term_index,
        "baseline_category_values": {
            column: str(model_frame.iloc[fit_indices][column].mode().iloc[0])
            for column in category_cols
        },
    }


def _show_explanation(text):
    with st.expander("How to read this", expanded=False):
        st.write(text)


def _clear_uploaded_graph_dialog():
    st.session_state.pop("selected_upload_gam_graph", None)


@st.dialog("GAM graph details", width="large", on_dismiss=_clear_uploaded_graph_dialog)
def _show_uploaded_gam_dialog(graph):
    st.markdown(f"### {graph['title']}")
    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(graph["figure"], use_container_width=True, key=f"upload_detail_{graph['key']}")
    with right:
        for heading, paragraph in zip(
            ["A. What question does this answer?", "B. Understanding the axes", "C. How to read it", "D. Where the numbers come from", "E. What we learned"],
            graph["explanation"],
        ):
            st.markdown(f"**{heading}**")
            st.write(paragraph)
    if st.button("Close", key=f"upload_close_{graph['key']}"):
        st.session_state.pop("selected_upload_gam_graph", None)
        st.rerun()


def _render_upload_gam_gallery(model, currency_label):
    colors = {"Male": "#2563EB", "Female": "#D97757"}
    graphs = []
    for column in model["numeric_cols"]:
        curves = model["smooth_effects"][column]
        fig = go.Figure()
        explanations = []
        for group, dash in [("Male", "solid"), ("Female", "dash")]:
            curve = curves[group]
            fig.add_trace(go.Scatter(x=curve["x"], y=curve["upper"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=curve["x"], y=curve["lower"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(37,99,235,.13)" if group == "Male" else "rgba(217,119,87,.14)", showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=curve["x"], y=curve["effect"], mode="lines", name=group,
                line={"color": colors[group], "width": 2.5, "dash": dash},
                customdata=np.column_stack([curve["lower"], curve["upper"]]),
                hovertemplate=f"{group}<br>{column}: %{{x:.2f}}<br>Contribution: {currency_label} %{{y:,.0f}}<br>95% interval: %{{customdata[0]:,.0f}} to %{{customdata[1]:,.0f}}<extra></extra>",
            ))
            vals = curve["effect"]
            quarter = max(1, len(vals) // 4)
            first, last = float(np.mean(vals[:quarter])), float(np.mean(vals[-quarter:]))
            direction = "higher" if last > first else "lower" if last < first else "about the same"
            explanations.append(f"The {group} contribution is {direction} at the high end than at the low end: about {currency_label} {first:,.0f} versus {currency_label} {last:,.0f}, averaged over the first and last quarters of this graph.")
        fig.add_hline(y=0, line_color="#172B4D", line_width=1)
        curve_min, curve_max = float(np.min(curves["Male"]["x"])), float(np.max(curves["Male"]["x"]))
        fig.update_layout(title=f"How {column} relates to fitted earnings", xaxis_title=column, yaxis_title=f"Smooth contribution ({currency_label})", template="plotly_white", height=360, hovermode="closest", legend={"orientation": "h", "y": -0.2}, font={"color": "#172B4D"})
        graphs.append({
            "key": "numeric_" + str(column).replace(" ", "_"), "title": f"{column} smooth pattern", "subtitle": f"Does the modelled {column} contribution rise or fall across its range?", "figure": fig,
            "explanation": [
                f"This graph asks how the model's {column}-related pattern changes for Male and Female workers.",
                f"X is {column} in the uploaded data's units, from about {curve_min:.1f} to {curve_max:.1f} (the training sample's 1st–99th percentile range). Y is this smooth term's contribution in {currency_label}; it is not a person's full earnings.",
                "The solid and dashed lines show separate group curves; the shaded ribbons are pointwise 95% confidence intervals. Rising means this model component is larger at that part of the range; falling means it is smaller.",
                "pyGAM calculates the curve from the fitted smooth term with other selected inputs held at the reference profile. Zero is the term's reference point; the model adds this term with the other components and its intercept to form fitted earnings.",
                " ".join(explanations) + " This is an adjusted association in this upload, not proof of cause."
            ]
        })

    for column in model["category_cols"]:
        values = model["category_effects"][column].sort_values("effect").copy()
        base_level = model["baseline_category_values"][column]
        base_row = values[values["category"].astype(str) == base_level]
        base_effect = float(base_row["effect"].iloc[0]) if not base_row.empty else float(values["effect"].median())
        levels = values["category"].astype(str).tolist()
        positions = np.arange(len(values))
        fig = go.Figure()
        for group in ["Male", "Female"]:
            estimated = model["gender_predictions"][group] + values["effect"].to_numpy() - base_effect
            fig.add_trace(go.Scatter(
                x=estimated, y=positions, mode="markers", name=group,
                marker={"color": colors[group], "size": 10},
                customdata=np.column_stack([levels, [group] * len(levels)]),
                hovertemplate="%{customdata[1]}<br>Category: %{customdata[0]}<br>Fitted estimate: " + currency_label + " %{x:,.0f}<extra></extra>",
            ))
        fig.update_yaxes(tickmode="array", tickvals=positions, ticktext=levels, autorange="reversed")
        fig.update_layout(title=f"Adjusted earnings profiles by {column}", xaxis_title=f"Fitted earnings estimate ({currency_label})", yaxis_title=column, template="plotly_white", height=max(360, 35 * len(values)), hovermode="closest", legend={"orientation": "h", "y": -0.2}, font={"color": "#172B4D"})
        male_estimates = model["gender_predictions"]["Male"] + values["effect"].to_numpy() - base_effect
        female_estimates = model["gender_predictions"]["Female"] + values["effect"].to_numpy() - base_effect
        high_m = levels[int(np.argmax(male_estimates))]
        high_f = levels[int(np.argmax(female_estimates))]
        gap = float(np.mean(male_estimates - female_estimates))
        graphs.append({
            "key": "category_" + str(column).replace(" ", "_"), "title": f"{column} profiles", "subtitle": f"How do fitted profiles vary across {column} groups?", "figure": fig,
            "explanation": [
                f"This graph asks how adjusted fitted earnings profiles vary across the selected {column} categories for Male and Female workers.",
                f"X is fitted earnings in {currency_label}. Y lists categories. The points are constructed profile estimates, not actual category averages.",
                "Blue points are Male and coral points are Female. Compare points within each row for gender; compare rows to see how the category part changes.",
                "Each point starts from the GAM estimate for a typical profile and replaces the reference category contribution with the displayed category contribution. The category term is shared by both groups; the gender estimates differ by the model's gender term and gender-specific smooths. No confidence interval is computed for these constructed profiles.",
                f"The highest displayed fitted profile is {high_m} for Male and {high_f} for Female. The average Male-minus-Female distance across displayed categories is {currency_label} {gap:,.0f}; because the category term is shared, this distance comes from the model's gender profile, not a category-specific gender effect. This is association, not proof of cause."
            ]
        })

    if not graphs:
        return
    st.markdown("#### GAM graph gallery")
    st.caption("Hover or tap a point/line to inspect its value. Choose **Explore graph** for the detailed explanation.")
    for start in range(0, len(graphs), 2):
        cols = st.columns(2)
        for col, graph in zip(cols, graphs[start:start + 2]):
            with col, st.container(border=True):
                st.markdown(f"#### {graph['title']}")
                st.caption(graph["subtitle"])
                preview = go.Figure(graph["figure"])
                preview.update_layout(height=300, margin={"l": 45, "r": 15, "t": 45, "b": 55}, font={"size": 11})
                st.plotly_chart(preview, use_container_width=True, key=f"upload_preview_{graph['key']}")
                if st.button("Explore graph", key=f"upload_open_{graph['key']}", type="primary"):
                    st.session_state["selected_upload_gam_graph"] = graph["key"]
    selected_key = st.session_state.get("selected_upload_gam_graph")
    chosen = next((graph for graph in graphs if graph["key"] == selected_key), None)
    if chosen:
        _show_uploaded_gam_dialog(chosen)


def _render_uploaded_report(report, currency_label):
    full_frame = report["frame"]
    frame = full_frame.copy()
    outcome_name = report["outcome_name"]
    numeric_cols = report["numeric_cols"]
    category_cols = report["category_cols"]
    st.caption(f"Active dataset: Custom upload — {report['dataset_name']}")
    st.markdown("#### Filters")
    filter_columns = st.columns(2)
    filter_suffix = hashlib.sha256(repr(report["key"]).encode("utf-8")).hexdigest()[:10]
    selected_groups = filter_columns[0].multiselect(
        "Groups shown in descriptive charts",
        ["Male", "Female"],
        default=["Male", "Female"],
        key=f"upload_group_filter_{filter_suffix}",
    )
    frame = frame[frame["gender_label"].isin(selected_groups)].copy()
    if frame.empty:
        st.warning("Choose at least one group to display descriptive results.")
        return
    if "age" in full_frame.columns and pd.api.types.is_numeric_dtype(full_frame["age"]):
        age_min, age_max = int(full_frame["age"].min()), int(full_frame["age"].max())
        if age_min < age_max:
            age_range = filter_columns[1].slider(
                "Age range for descriptive charts",
                min_value=age_min,
                max_value=age_max,
                value=(age_min, age_max),
                key=f"upload_age_filter_{filter_suffix}",
            )
            frame = frame[frame["age"].between(*age_range)].copy()
    st.caption("These filters change EDA and descriptive findings only. GAM estimates and held-out test scores remain based on the full uploaded analysis sample.")
    if frame.empty:
        st.warning("No rows match these filters. Widen the filters to see descriptive results.")
        return
    summary = frame.groupby("gender_label")["earnings"].agg(["count", "median", "mean"])
    upload_section = st.radio("Analysis section", ["EDA", "GAM", "Conclusions"], horizontal=True, key=f"upload_analysis_section_{filter_suffix}")

    if upload_section == "EDA":
        st.subheader("Explore the uploaded data")
        top = st.columns(3)
        top[0].metric("Rows analyzed", f"{len(frame):,}")
        top[1].metric("Male rows", f"{int(summary.loc['Male', 'count']) if 'Male' in summary.index else 0:,}")
        top[2].metric("Female rows", f"{int(summary.loc['Female', 'count']) if 'Female' in summary.index else 0:,}")
        st.caption(
            f"Rows not used (other group values or missing/unusable selected fields): {report['rows_removed']:,}."
        )

        st.markdown("#### Middle earnings by gender")
        medians = summary["median"].reindex(["Male", "Female"]).dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(medians.index, medians.values, color=[{"Male": "#2563EB", "Female": "#D97757"}[label] for label in medians.index])
        ax.bar_label(bars, labels=[f"{currency_label} {value:,.0f}" for value in medians.values], padding=3)
        ax.set_ylabel(f"Middle earnings ({currency_label})")
        ax.set_title("Observed middle earnings by gender")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
        _show_explanation(
            "This compares actual earnings in the uploaded file. The middle value is the point where half the workers earn more and half earn less. It does not account for differences in age, schooling, or type of work."
        )

        if numeric_cols:
            chosen_numeric = st.selectbox("Explore earnings against a numeric field", numeric_cols, key="upload_eda_numeric")
            chart_data = frame[[chosen_numeric, "earnings", "gender_label"]].dropna()
            if len(chart_data) > 5_000:
                chart_data = chart_data.sample(5_000, random_state=42)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for label, color in [("Male", "#2563EB"), ("Female", "#D97757")]:
                subset = chart_data[chart_data["gender_label"] == label]
                ax.scatter(subset[chosen_numeric], subset["earnings"], s=14, alpha=0.25, color=color, label=label)
            ax.set_xlabel(chosen_numeric)
            ax.set_ylabel(f"Earnings ({currency_label})")
            ax.set_title(f"Observed earnings versus {chosen_numeric}")
            ax.legend(title="Group")
            st.pyplot(fig)
            plt.close(fig)
            _show_explanation(
                "Each dot is one worker from the upload (up to 5,000 dots are shown for readability). The plot shows raw values, before accounting for other characteristics."
            )

        if category_cols:
            chosen_category = st.selectbox("Compare earnings across a category", category_cols, key="upload_eda_category")
            category_counts = frame[chosen_category].value_counts().head(12)
            top_frame = frame[frame[chosen_category].isin(category_counts.index)]
            category_medians = top_frame.groupby([chosen_category, "gender_label"])["earnings"].median().unstack()
            category_medians = category_medians.reindex(category_counts.index)
            fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(category_medians) + 1.5)))
            positions = np.arange(len(category_medians))
            for label, offset, color in [("Male", -0.12, "#2563EB"), ("Female", 0.12, "#D97757")]:
                if label in category_medians:
                    ax.scatter(category_medians[label], positions + offset, label=label, color=color, s=48)
            ax.set_yticks(positions, labels=[str(value) for value in category_medians.index])
            ax.set_xlabel(f"Middle earnings ({currency_label})")
            ax.set_ylabel(chosen_category)
            ax.set_title(f"Observed middle earnings across {chosen_category} groups")
            ax.legend(title="Group")
            ax.grid(axis="x", alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
            _show_explanation(
                "Each row is one of the 12 largest groups. The dots show actual middle earnings for Male and Female workers in that group; small groups are left out of this chart."
            )

        with st.expander("Preview the uploaded rows"):
            st.dataframe(frame.drop(columns=["__gender_value"], errors="ignore").head(20), use_container_width=True)

    if upload_section == "GAM":
        st.subheader("GAM: adjusted patterns")
        if not report["model"]["fit_available"]:
            st.warning(report["model"]["message"])
        else:
            model = report["model"]
            metrics = model["metrics"]
            st.markdown("The GAM compares groups while accounting for the other columns you selected.")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Test R²", f"{metrics['test_r2']:.3f}")
            metric_cols[1].metric("Test RMSE", f"{currency_label} {metrics['test_rmse']:,.0f}")
            metric_cols[2].metric("Test MAE", f"{currency_label} {metrics['test_mae']:,.0f}")
            st.caption(
                f"Checked on {metrics['test_rows']:,} workers held aside from model fitting. "
                f"The model used {metrics['train_rows']:,} training rows."
            )
            if metrics["train_partition_rows"] > metrics["train_rows"]:
                st.info(
                    f"The upload has a large training group, so the model used a random {metrics['train_rows']:,}-row sample for speed. "
                    f"The separate test check used {metrics['test_rows']:,} of {metrics['test_partition_rows']:,} held-aside rows."
                )
            if metrics["fit_warnings"]:
                st.warning("The GAM reported a fitting note: " + "; ".join(metrics["fit_warnings"]))

            predictions = model["gender_predictions"]
            predicted_gap = predictions["Male"] - predictions["Female"]
            pred_cols = st.columns(3)
            pred_cols[0].metric("Estimated earnings: Male", f"{currency_label} {predictions['Male']:,.0f}")
            pred_cols[1].metric("Estimated earnings: Female", f"{currency_label} {predictions['Female']:,.0f}")
            pred_cols[2].metric("Difference for this profile", f"{currency_label} {predicted_gap:,.0f}")
            _show_explanation(
                "The model uses the same typical age, schooling, and category values for both groups, then compares its estimates. This is one example profile, not the average earnings of everyone in each group."
            )

            _render_upload_gam_gallery(model, currency_label)

    if upload_section == "Conclusions":
        st.subheader("Conclusions")
        st.markdown("### A. What does this dataset contain?")
        st.write(
            f"The current filtered view shows {len(frame):,} rows from {report['dataset_name']}. The analysis uses earnings and gender plus {len(numeric_cols)} selected numeric field(s) and {len(category_cols)} selected category field(s)."
        )
        st.markdown("### B. What did EDA reveal?")
        if {"Male", "Female"}.issubset(summary.index):
            male_middle = float(summary.loc["Male", "median"])
            female_middle = float(summary.loc["Female", "median"])
            st.markdown(
                f"In this file, the middle earnings value is {currency_label} {male_middle:,.0f} for Male workers and "
                f"{currency_label} {female_middle:,.0f} for Female workers. These are actual group comparisons."
            )
        else:
            st.info("Choose both Male and Female in the filters to compare their observed middle earnings. The filter does not change the GAM fit.")

        st.markdown("### C. What did the GAM analysis reveal?")
        if report["model"]["fit_available"]:
            model = report["model"]
            male_estimate = model["gender_predictions"]["Male"]
            female_estimate = model["gender_predictions"]["Female"]
            st.info(
                f"For workers with the same typical selected characteristics, the model estimates "
                f"{currency_label} {male_estimate:,.0f} for Male and {currency_label} {female_estimate:,.0f} for Female. "
                "This is a pattern in the uploaded data, not proof that gender caused it."
            )
            for column in model["numeric_cols"]:
                group_patterns = []
                for group in ["Male", "Female"]:
                    values = model["smooth_effects"][column][group]["effect"]
                    quarter = max(1, len(values) // 4)
                    first, last = float(np.mean(values[:quarter])), float(np.mean(values[-quarter:]))
                    direction = "rises" if last > first else "falls" if last < first else "stays similar"
                    group_patterns.append(f"{group}'s curve {direction} from about {currency_label} {first:,.0f} to {currency_label} {last:,.0f}")
                st.markdown(f"**{column}:** " + "; ".join(group_patterns) + ". These are model components, not full earnings.")
            for column in model["category_cols"]:
                category = model["category_effects"][column]
                high = category.loc[category["effect"].idxmax()]
                low = category.loc[category["effect"].idxmin()]
                st.markdown(
                    f"**{column}:** {high['category']} has the highest fitted category component ({currency_label} {high['effect']:,.0f}); "
                    f"{low['category']} has the lowest ({currency_label} {low['effect']:,.0f}). These are shared category associations, not actual pay amounts."
                )
            test_r2 = model["metrics"]["test_r2"]
            if test_r2 >= 0:
                st.markdown(
                    f"On workers held aside from fitting, the model accounted for about **{test_r2:.0%} of the earnings differences**. "
                    "The rest was not explained by the selected columns."
                )
            else:
                st.markdown(
                    "On workers held aside from fitting, the model did worse than simply guessing the overall average earnings. "
                    "This file may need different input columns or more data."
                )
        else:
            st.info("The file is small or incomplete for a reliable GAM, so this conclusion is descriptive only.")
        with st.expander("What these results cannot tell you"):
            st.markdown(
                "- They cannot prove what caused an earnings difference.\n"
                "- The model can only account for the columns selected for it; other important factors may be missing.\n"
                "- The estimated Male/Female difference may not be the same for every job or category.\n"
                "- Results describe the uploaded rows and may not represent workers outside that file."
            )


def render_upload_dashboard():
    st.subheader("Analyze your own data")
    st.caption(
        "Upload a CSV, choose which columns represent earnings and gender, and optionally include numeric and work-category fields. "
        "Only the two selected gender values are compared. The file is processed in this app session and is not written to the project."
    )
    st.info(
        "For company data, run the app on a machine you control. If this dashboard is hosted online, the uploaded file is sent to that hosting machine while it is analyzed."
    )
    sample_path = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "synthetic_plfs_test.csv"
    if sample_path.exists():
        st.download_button(
            "Download synthetic test CSV",
            data=sample_path.read_bytes(),
            file_name="synthetic_plfs_test.csv",
            mime="text/csv",
            help="Artificial data for testing the upload flow. It is not real PLFS data.",
        )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="user_dataset_csv")
    if uploaded_file is None:
        st.session_state.pop("user_dataset_report", None)
        st.markdown(
            "**Required:** one numeric earnings column and one column with two groups to compare. "
            "**Optional:** age or schooling as numeric columns, plus job, industry, or employment columns as categories. "
            "Do not select names, IDs, or other personal columns as model inputs."
        )
        return

    try:
        raw = _read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        return
    if len(raw.columns) < 2:
        st.error("The CSV needs at least two columns: earnings and gender.")
        return

    columns = list(raw.columns)
    default_outcome = _suggest_column(columns, ["earnings", "salary", "wage", "income", "pay"])
    default_gender = _suggest_column(columns, ["gender", "sex"])
    if default_gender == default_outcome and len(columns) > 1:
        default_gender = next(column for column in columns if column != default_outcome)

    st.markdown("#### Match your columns")
    with st.form("user_dataset_mapping"):
        outcome_col = st.selectbox("Which column contains earnings?", columns, index=_column_index(columns, default_outcome))
        gender_candidates = [column for column in columns if column != outcome_col]
        gender_col = st.selectbox("Which column identifies the two groups?", gender_candidates, index=_column_index(gender_candidates, default_gender))
        group_values = raw[gender_col].dropna().astype("string").value_counts().index.astype(str).tolist()
        if len(group_values) < 2:
            st.warning("The selected group column needs at least two different values.")
        elif len(group_values) > 50:
            st.warning("This column has too many different values to be a gender/group field. Choose another column.")

        male_guess = next(
            (value for value in group_values if value.lower() in {"male", "m", "man", "men"}),
            None,
        )
        female_guess = next(
            (value for value in group_values if value.lower() in {"female", "f", "woman", "women"}),
            None,
        )
        group_placeholder = "Choose a value"
        group_options = [group_placeholder, *group_values]
        if male_guess is None or female_guess is None:
            st.info("Check the group labels carefully. Numeric codes are not automatically interpreted as Male or Female.")
        group_cols = st.columns(2)
        male_value = group_cols[0].selectbox(
            "Which value means Male?",
            group_options,
            index=group_options.index(male_guess) if male_guess in group_options else 0,
            disabled=len(group_values) < 2 or len(group_values) > 50,
        )
        female_value = group_cols[1].selectbox(
            "Which value means Female?",
            group_options,
            index=group_options.index(female_guess) if female_guess in group_options else 0,
            disabled=len(group_values) < 2 or len(group_values) > 50,
        )

        numeric_candidates = []
        for column in columns:
            if column in {outcome_col, gender_col}:
                continue
            converted = pd.to_numeric(raw[column], errors="coerce")
            if converted.notna().mean() >= 0.9 and converted.nunique() >= 5:
                numeric_candidates.append(column)
        numeric_defaults = [column for column in numeric_candidates if any(word in str(column).lower() for word in ["age", "educ", "year", "experience"])][:4]
        numeric_cols = st.multiselect("Numeric fields for smooth curves (optional, up to 4)", numeric_candidates, default=numeric_defaults)

        category_candidates = [
            column for column in columns
            if column not in {outcome_col, gender_col, *numeric_cols}
            and raw[column].nunique(dropna=True) <= 5_000
        ]
        category_defaults = [
            column for column in category_candidates
            if any(word in str(column).lower() for word in ["job", "occupation", "industry", "employment", "department", "role", "status"])
        ][:4]
        category_cols = st.multiselect("Work/category fields for the model (optional, up to 4)", category_candidates, default=category_defaults)
        currency_label = st.text_input("Earnings unit or currency label", value="currency units", max_chars=20)
        too_many_predictors = len(numeric_cols) > 4 or len(category_cols) > 4
        if too_many_predictors:
            st.warning("Choose no more than four numeric fields and four category fields.")
        groups_mapped = male_value in group_values and female_value in group_values and male_value != female_value
        submitted = st.form_submit_button(
            "Analyze this file",
            type="primary",
            disabled=(
                len(group_values) < 2
                or len(group_values) > 50
                or not groups_mapped
                or too_many_predictors
            ),
        )

    file_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    report_key = (file_signature, uploaded_file.name, outcome_col, gender_col, male_value, female_value, tuple(numeric_cols), tuple(category_cols), currency_label)
    if submitted:
        with st.spinner("Preparing the file and fitting the GAM. Large files may take a little while..."):
            try:
                frame, rows_removed = _prepare_frame(
                    raw, outcome_col, gender_col, male_value, female_value, numeric_cols, category_cols
                )
                try:
                    model = _fit_uploaded_gam(frame, numeric_cols, category_cols)
                except Exception as model_error:
                    model = {
                        "fit_available": False,
                        "message": f"The EDA is ready, but the GAM could not be fit with these selections: {model_error}",
                    }
                st.session_state["user_dataset_report"] = {
                    "key": report_key,
                    "dataset_name": uploaded_file.name,
                    "frame": frame,
                    "rows_removed": rows_removed,
                    "outcome_name": outcome_col,
                    "numeric_cols": list(numeric_cols),
                    "category_cols": list(category_cols),
                    "model": model,
                }
            except Exception as exc:
                st.session_state.pop("user_dataset_report", None)
                st.error(f"Could not analyze this file: {exc}")

    report = st.session_state.get("user_dataset_report")
    if report and report["key"] == report_key:
        _render_uploaded_report(report, currency_label or "data units")
    elif report:
        st.session_state.pop("user_dataset_report", None)
        st.info("Your file or column choices changed. Select **Analyze this file** to refresh the results.")
