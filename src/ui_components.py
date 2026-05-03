import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from src.explainability import generate_shap_plot

# ── Beige / warm-neutral palette ──────────────────────────────────────────────
_BEIGE_BG   = "#FAF8F4"
_SAND       = "#E8DFD0"
_WARM_BROWN = "#8B7355"
_DEEP_BROWN = "#4A3728"
_HIGH_RISK  = "#C0392B"
_MED_RISK   = "#D4882A"
_LOW_RISK   = "#5D8A5E"

# Shared Altair colour scale for risk levels
_RISK_SCALE = alt.Scale(
    domain=["High Risk", "Medium Risk", "Low Risk"],
    range=[_HIGH_RISK, _MED_RISK, _LOW_RISK],
)


# ── KPI cards ─────────────────────────────────────────────────────────────────
def render_kpi_cards(df):
    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df)
    churn_count     = (df["Churn_Prediction"] == 1).sum()
    churn_rate      = (churn_count / total_customers) * 100 if total_customers > 0 else 0
    high_risk_count = (df["Risk_Level"] == "High Risk").sum()
    avg_prob        = df["Churn_Probability"].mean()

    with col1:
        st.metric("Total Customers", total_customers)
    with col2:
        st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_count} users")
    with col3:
        st.metric("High Risk Users", high_risk_count)
    with col4:
        st.metric("Avg. Churn Prob.", f"{avg_prob:.2f}")


# ── Churn distribution bar ────────────────────────────────────────────────────
def render_churn_distribution(df):
    st.subheader("Churn Distribution")

    churn_counts = (
        df["Churn_Prediction"]
        .map({0: "Retained", 1: "Churned"})
        .value_counts()
        .reset_index()
    )
    churn_counts.columns = ["Status", "Count"]

    chart = (
        alt.Chart(churn_counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Status", sort="-y", axis=alt.Axis(labelColor=_DEEP_BROWN)),
            y=alt.Y("Count", axis=alt.Axis(labelColor=_DEEP_BROWN)),
            color=alt.Color(
                "Status",
                scale=alt.Scale(
                    domain=["Retained", "Churned"],
                    range=[_LOW_RISK, _HIGH_RISK],
                ),
                legend=None,
            ),
            tooltip=["Status", "Count"],
        )
        .properties(height=300, background=_BEIGE_BG)
        .configure_axis(grid=False)
    )

    st.altair_chart(chart, use_container_width=True)


# ── Feature importance ────────────────────────────────────────────────────────
def render_feature_importance(model, feature_names):
    st.subheader("Top Churn Drivers")

    if hasattr(model, "coef_"):
        importance = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        st.warning("Model does not expose feature importance.")
        return

    feat_df = (
        pd.DataFrame({"Feature": feature_names, "Impact": importance})
        .sort_values("Impact", ascending=False)
    )
    top_10 = pd.concat([feat_df.head(5), feat_df.tail(5)])

    chart = (
        alt.Chart(top_10)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x="Impact",
            y=alt.Y("Feature", sort="-x"),
            color=alt.condition(
                alt.datum.Impact > 0,
                alt.value(_HIGH_RISK),
                alt.value(_LOW_RISK),
            ),
            tooltip=["Feature", "Impact"],
        )
        .properties(height=400, background=_BEIGE_BG)
        .configure_axis(grid=False)
    )

    st.altair_chart(chart, use_container_width=True)


# ── Tenure analysis ───────────────────────────────────────────────────────────
def render_tenure_analysis(df):
    st.subheader("Tenure vs Churn Risk")

    chart = (
        alt.Chart(df)
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("tenure:Q", bin=alt.Bin(maxbins=20), title="Tenure (months)"),
            y=alt.Y("count()", stack=None, title="Customers"),
            color=alt.Color("Risk_Level:N", scale=_RISK_SCALE),
            tooltip=["Risk_Level", "count()"],
        )
        .properties(height=300, background=_BEIGE_BG)
        .configure_axis(grid=False)
    )

    st.altair_chart(chart, use_container_width=True)


# ── Business insights ─────────────────────────────────────────────────────────
def render_business_insights(df, model, feature_names):
    st.markdown("---")
    st.subheader("Business Strategy & Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Key Findings**")
        if hasattr(model, "coef_"):
            top_feature = feature_names[np.argmax(model.coef_[0])]
            st.write(f"1. Primary churn driver: `{top_feature}`")
            if "Contract_Month-to-month" in feature_names:
                st.write("2. Month-to-month contracts significantly increase churn risk.")

        high_risk_pct = (df["Risk_Level"] == "High Risk").mean() * 100
        st.write(f"3. **{high_risk_pct:.1f}%** of customers are in the High Risk tier.")

    with col2:
        st.markdown("**Actionable Recommendations**")
        st.info("Offer loyalty discounts to high-spend, low-tenure customers.")
        st.success("Upsell month-to-month customers to annual plans with bundle incentives.")
        st.warning("Investigate Fiber Optic pricing — segment shows elevated churn.")


# ── Model baseline ────────────────────────────────────────────────────────────
def render_model_baseline(metrics):
    st.subheader("Model Baseline Performance (Test Set)")
    st.info("Metrics from the 20 % held-out test set during training.")

    if metrics is None:
        st.error("Metrics file not found. Please retrain the model.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{metrics['accuracy']}%")
    m2.metric("Precision", f"{metrics['precision']}%")
    m3.metric("Recall",    f"{metrics['recall']}%")
    m4.metric("F1 Score",  f"{metrics['f1']}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_data = pd.DataFrame({
            "Actual":    ["Actual: No",  "Actual: No",  "Actual: Yes", "Actual: Yes"],
            "Predicted": ["Pred: No", "Pred: Yes", "Pred: No",  "Pred: Yes"],
            "Count":     [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
        })
        cm_chart = (
            alt.Chart(cm_data)
            .mark_rect()
            .encode(
                x="Predicted:O",
                y="Actual:O",
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="warmgreys")),
                tooltip=["Actual", "Predicted", "Count"],
            )
            .properties(height=350)
        )
        text = cm_chart.mark_text(baseline="middle").encode(
            text="Count:Q",
            color=alt.condition(
                alt.datum.Count > 500,
                alt.value("white"),
                alt.value(_DEEP_BROWN),
            ),
        )
        st.altair_chart(cm_chart + text, use_container_width=True)

    with col2:
        st.subheader("Performance Commentary")
        st.write(f"""
- **Precision ({metrics['precision']}%)**: Correct churn prediction ~{round(metrics['precision']/10)} out of 10 times.
- **Recall ({metrics['recall']}%)**: Captures ~{round(metrics['recall'])}% of actual churners — critical for retention.
- **F1 ({metrics['f1']}%)**: Healthy precision/recall balance for this imbalanced dataset.
        """)


# ── AI Advisor results renderer ───────────────────────────────────────────────
def render_ai_advisor_insights(result, model, shap_top, shap_row, feature_names, df_scaled):
    if result.get("is_fallback"):
        st.warning("AI advisor unavailable — showing rule-based suggestions.")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.markdown("#### Risk Assessment")
        risk = result.get("risk_level", "Unknown")
        if risk == "High":
            st.error(f"**Risk Level: {risk}**")
        elif risk == "Medium":
            st.warning(f"**Risk Level: {risk}**")
        else:
            st.success(f"**Risk Level: {risk}**")

        if result.get("risk_summary"):
            st.info(result["risk_summary"])

        st.markdown("#### Contributing Factors")
        for factor in result.get("contributing_factors", []):
            st.markdown(f"- {factor}")

    with col_res2:
        st.markdown("#### Recommended Retention Actions")
        for idx, act in enumerate(result.get("recommended_actions", []), 1):
            priority = act.get("priority", "")
            color = "red" if priority == "High" else "orange" if priority == "Medium" else "green"
            with st.container():
                st.markdown(f"**{idx}. {act.get('action')}** (Priority: :{color}[{priority}])")
                st.markdown(f"> *{act.get('rationale')}*")

    st.markdown("---")

    # SHAP
    if shap_top and shap_row is not None:
        st.markdown("#### Per-customer Feature Attribution (SHAP)")
        for f in shap_top:
            arrow = "↑" if f["shap_value"] > 0 else "↓"
            st.markdown(
                f"- **{f['feature']}** {arrow} "
                f"(shap={f['shap_value']:+.3f}, value={f['value']:.3g}) — {f['direction']}"
            )
        try:
            fig = generate_shap_plot(model, shap_row, feature_names, background=df_scaled[:100], top_k=10)
            st.pyplot(fig)
        except Exception as plot_err:
            st.caption(f"SHAP plot unavailable: {plot_err}")

    with st.expander("Supporting Insights & Disclaimers"):
        if result.get("sources"):
            st.markdown("**Sources:**")
            for source in result.get("sources", []):
                st.markdown(f"- {source}")
        if result.get("disclaimers"):
            st.markdown("**Disclaimers:**")
            for disc in result.get("disclaimers", []):
                st.caption(disc)


# ── Batch analytics tab ───────────────────────────────────────────────────────
def render_batch_analytics(df):
    """Batch-level churn analytics dashboard (Issue 4)."""

    st.subheader("Risk Distribution")

    # 1. Donut / pie chart of risk tiers
    risk_counts = df["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk_Level", "Count"]

    donut = (
        alt.Chart(risk_counts)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Risk_Level:N", scale=_RISK_SCALE),
            tooltip=["Risk_Level", "Count"],
        )
        .properties(height=280, background=_BEIGE_BG)
    )
    st.altair_chart(donut, use_container_width=True)

    st.markdown("---")

    # 2. Segment breakdowns
    seg_cols = [c for c in ["Contract", "InternetService", "PaymentMethod"] if c in df.columns]
    if seg_cols:
        st.subheader("Segment Breakdown — Churn Rate")
        seg_tabs = st.tabs(seg_cols)
        for tab, col in zip(seg_tabs, seg_cols):
            with tab:
                seg_df = (
                    df.groupby(col)["Churn_Prediction"]
                    .mean()
                    .mul(100)
                    .reset_index()
                    .rename(columns={"Churn_Prediction": "Churn Rate (%)"})
                )
                bar = (
                    alt.Chart(seg_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X(f"{col}:N", sort="-y"),
                        y=alt.Y("Churn Rate (%):Q"),
                        color=alt.Color(
                            "Churn Rate (%):Q",
                            scale=alt.Scale(scheme="orangered"),
                        ),
                        tooltip=[col, "Churn Rate (%)"],
                    )
                    .properties(height=280, background=_BEIGE_BG)
                    .configure_axis(grid=False)
                )
                st.altair_chart(bar, use_container_width=True)

    st.markdown("---")

    # 3. Heatmap — tenure vs monthly charges, colour = churn probability
    if {"tenure", "MonthlyCharges", "Churn_Probability"}.issubset(df.columns):
        st.subheader("Risk Heatmap — Tenure × Monthly Charges")
        heatmap_df = df[["tenure", "MonthlyCharges", "Churn_Probability"]].copy()
        heatmap_df["tenure_bin"]   = pd.cut(heatmap_df["tenure"],         bins=10, precision=0)
        heatmap_df["charges_bin"]  = pd.cut(heatmap_df["MonthlyCharges"], bins=8,  precision=0)
        hm = (
            heatmap_df
            .groupby(["tenure_bin", "charges_bin"], observed=True)["Churn_Probability"]
            .mean()
            .reset_index()
        )
        hm["tenure_bin"]  = hm["tenure_bin"].astype(str)
        hm["charges_bin"] = hm["charges_bin"].astype(str)

        heatmap = (
            alt.Chart(hm)
            .mark_rect()
            .encode(
                x=alt.X("tenure_bin:O",  title="Tenure (months)"),
                y=alt.Y("charges_bin:O", title="Monthly Charges ($)"),
                color=alt.Color("Churn_Probability:Q", scale=alt.Scale(scheme="orangered")),
                tooltip=["tenure_bin", "charges_bin", "Churn_Probability"],
            )
            .properties(height=350, background=_BEIGE_BG)
        )
        st.altair_chart(heatmap, use_container_width=True)

    st.markdown("---")

    # 4. Top 20 at-risk customers
    st.subheader("Top 20 At-Risk Customers")
    cols_show = [c for c in ["customerID", "tenure", "MonthlyCharges", "Contract",
                              "Churn_Probability", "Risk_Level"] if c in df.columns]
    top20 = df.sort_values("Churn_Probability", ascending=False).head(20)[cols_show]
    st.dataframe(top20, use_container_width=True, hide_index=True)
