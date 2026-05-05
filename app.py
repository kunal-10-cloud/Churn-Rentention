import streamlit as st
import pandas as pd
import numpy as np
from src.model_loader import load_model, load_scaler, load_feature_names, load_metrics
from src.processor import preprocess_data, get_risk_label, validate_input, analyze_data_quality
from src.agent import run_agent
from src.explainability import explain_prediction, generate_shap_plot
from src.ui_components import (
    render_kpi_cards,
    render_churn_distribution,
    render_feature_importance,
    render_tenure_analysis,
    render_business_insights,
    render_model_baseline,
    render_ai_advisor_insights,
    render_batch_analytics,
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI | Enterprise Retention Dashboard",
    page_icon="",
    layout="wide",
)

# ── GLOBAL STYLES (beige / warm-neutral palette) ───────────────────────────────
st.markdown("""
<style>
  /* Page background */
  [data-testid="stAppViewContainer"] { background-color: #FAF8F4; }
  [data-testid="stSidebar"]          { background-color: #F0EBE1; }

  /* Metric cards */
  [data-testid="stMetric"] {
      background-color: #FFFFFF;
      border: 1px solid #E8DFD0;
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 1px 4px rgba(74,55,40,0.08);
  }

  /* Tab strip */
  [data-baseweb="tab-list"] { gap: 6px; }
  [data-baseweb="tab"] {
      background-color: #F0EBE1 !important;
      border-radius: 6px 6px 0 0 !important;
      font-weight: 500;
      color: #4A3728 !important;
  }
  [aria-selected="true"][data-baseweb="tab"] {
      background-color: #E8DFD0 !important;
      border-bottom: 2px solid #8B7355 !important;
  }

  /* Buttons */
  [data-testid="stButton"] > button {
      background-color: #8B7355;
      color: #FAF8F4;
      border: none;
      border-radius: 6px;
      padding: 8px 18px;
      font-weight: 600;
  }
  [data-testid="stButton"] > button:hover {
      background-color: #6B5A42;
  }

  /* Download button */
  [data-testid="stDownloadButton"] > button {
      background-color: #5D8A5E;
      color: #FAF8F4;
      border: none;
      border-radius: 6px;
  }
  [data-testid="stDownloadButton"] > button:hover {
      background-color: #4A7049;
  }

  h1, h2, h3 { color: #2C2416; }
  p, li      { color: #3D2E1E; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.title("ChurnGuard AI")
st.caption("Enterprise Customer Retention & Churn Analytics — Milestone 2")

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
model        = load_model()
scaler       = load_scaler()
feature_names = load_feature_names()
metrics      = load_metrics()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Data Source")
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded_file:
        st.success("File uploaded")
    else:
        st.info("Upload a Telco-format CSV to begin analysis.")

    st.markdown("---")
    st.markdown("""
**Quick guide**
1. Upload your customer dataset.
2. View predictions on the **Executive** tab.
3. Drill into an individual customer on the **AI Advisor** tab.
4. Get a PDF report with one click.
    """)

# ── MAIN FLOW ──────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    # Validate
    validation_errors = validate_input(df_raw)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()

    # Data quality report
    quality = analyze_data_quality(df_raw)
    if quality["missing"] or quality["outliers"]:
        if quality["missing"]:
            st.warning(f"{len(quality['missing'])} column(s) had missing values — imputed with median/mode.")
        if quality["outliers"]:
            out_summary = ", ".join(f"{k} ({v})" for k, v in quality["outliers"].items())
            st.info(f"Outliers detected (IQR): {out_summary}")
        with st.expander("Data quality details"):
            st.metric("Data Quality Score", f"{quality['quality_score']} / 100")
            if quality["missing"]:
                st.write("**Missing values per column:**")
                st.json(quality["missing"])
            if quality["outliers"]:
                st.write("**Outliers per column (IQR):**")
                st.json(quality["outliers"])
    else:
        st.success(f"Data quality score: {quality['quality_score']} / 100")

    # Predictions
    try:
        with st.spinner("Processing data & generating predictions…"):
            df_processed = preprocess_data(df_raw, feature_names)
            df_scaled    = scaler.transform(df_processed)
            predictions  = model.predict(df_scaled)
            probabilities = model.predict_proba(df_scaled)[:, 1]

            df_results = df_raw.copy()
            df_results["Churn_Prediction"]  = predictions
            df_results["Churn_Probability"] = probabilities
            df_results["Risk_Level"]        = df_results["Churn_Probability"].apply(get_risk_label)
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

    # Session state initialisation
    for key, default in [
        ("active_tab_index", 0),
        ("agent_result",     None),
        ("agent_customer_id", None),
        ("shap_top",         []),
        ("shap_row",         None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Tab persistence — restore active tab after rerun
    def _restore_tab():
        idx = st.session_state.active_tab_index
        if idx > 0:
            st.components.v1.html(
                f"""
                <script>
                const tryClick = () => {{
                    const tabs = window.parent.document.querySelectorAll(
                        '[data-baseweb="tab-list"] button[role="tab"]'
                    );
                    if (tabs.length > {idx}) {{ tabs[{idx}].click(); }}
                    else {{ setTimeout(tryClick, 100); }}
                }};
                tryClick();
                </script>
                """,
                height=0,
            )

    _restore_tab()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Dashboard",
        "Risk Analysis",
        "AI Retention Advisor",
        "Batch Analytics",
        "Model Performance",
        "Raw Data Explorer",
    ])

    # ━━ Tab 1 — Executive Dashboard ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        st.markdown("### Executive Summary")
        render_kpi_cards(df_results)

        col1, col2 = st.columns(2)
        with col1:
            render_churn_distribution(df_results)
        with col2:
            render_tenure_analysis(df_results)

        render_business_insights(df_results, model, feature_names)

    # ━━ Tab 2 — Risk Analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        st.header("Risk Segmentation & Drivers")
        col_a, col_b = st.columns(2)

        with col_a:
            render_feature_importance(model, feature_names)

        with col_b:
            st.subheader("High Risk Customer Profiles")
            high_risk_df = (
                df_results[df_results["Risk_Level"] == "High Risk"]
                .sort_values("Churn_Probability", ascending=False)
            )
            st.dataframe(high_risk_df.head(10), use_container_width=True, hide_index=True)
            st.caption("Top 10 most vulnerable customers.")

    # ━━ Tab 3 — AI Retention Advisor ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab3:
        st.header("AI Retention Advisor")
        st.markdown("Select an at-risk customer to generate a personalised AI retention strategy.")

        if "customerID" in df_results.columns:
            at_risk_df = (
                df_results[df_results["Risk_Level"].isin(["High Risk", "Medium Risk"])]
                .sort_values("Churn_Probability", ascending=False)
            )

            st.subheader("At-Risk Customers")
            event = st.dataframe(
                at_risk_df,
                use_container_width=True,
                selection_mode="single-row",
                on_select="rerun",
                hide_index=True,
            )

            selected_id = None
            if len(event.selection.rows) > 0:
                selected_id = at_risk_df.iloc[event.selection.rows[0]]["customerID"]

            if not selected_id:
                st.markdown("Or search via dropdown:")
                selected_id = st.selectbox(
                    "Search Customer ID",
                    options=[""] + at_risk_df["customerID"].tolist(),
                    index=0,
                    label_visibility="collapsed",
                )

            if selected_id:
                st.session_state.active_tab_index = 2
                st.markdown("---")
                st.subheader(f"Strategy for Customer: `{selected_id}`")

                row = df_results[df_results["customerID"] == selected_id].iloc[0]
                churn_prob   = float(row["Churn_Probability"])
                customer_data = row.drop(
                    labels=["Churn_Prediction", "Churn_Probability", "Risk_Level"],
                    errors="ignore",
                ).to_dict()

                if st.button("Generate Retention Strategy", key="run_agent_btn"):
                    with st.spinner("Agent is analysing customer profile…"):
                        row_idx        = df_results.index[df_results["customerID"] == selected_id][0]
                        customer_scaled = df_scaled[list(df_results.index).index(row_idx)]
                        try:
                            shap_top = explain_prediction(
                                model, customer_scaled, feature_names,
                                background=df_scaled[:100], top_k=5,
                            )
                        except Exception as shap_err:
                            shap_top = []
                            st.info(f"SHAP explanation unavailable: {shap_err}")

                        result = run_agent(customer_data, churn_prob, shap_explanation=shap_top)
                        st.session_state.agent_result      = result
                        st.session_state.agent_customer_id = selected_id
                        st.session_state.shap_top          = shap_top
                        st.session_state.shap_row          = customer_scaled

                result = st.session_state.agent_result
                if result and st.session_state.agent_customer_id == selected_id:
                    shap_top = st.session_state.get("shap_top") or []
                    shap_row = st.session_state.get("shap_row")
                    render_ai_advisor_insights(result, model, shap_top, shap_row, feature_names, df_scaled)

                    st.markdown("---")
                    from src.pdf_export import generate_pdf
                    pdf_bytes = generate_pdf(result, customer_data)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"Retention_Report_{selected_id}.pdf",
                        mime="application/pdf",
                    )
        else:
            st.info("No `customerID` column found — cannot select individual customers.")

    # ━━ Tab 4 — Batch Analytics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab4:
        st.header("Batch Churn Analytics")
        render_batch_analytics(df_results)

    # ━━ Tab 5 — Model Performance ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab5:
        st.header("Model Evaluation Metrics")
        render_model_baseline(metrics)

    # ━━ Tab 6 — Raw Data Explorer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab6:
        st.header("Dataset Explorer")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Prediction Results (CSV)",
            data=csv,
            file_name="churn_predictions_export.csv",
            mime="text/csv",
        )

else:
    # ── LANDING STATE ──────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### Welcome to ChurnGuard AI

Upload your customer dataset to unlock:
- **Real-time Churn Prediction** using Logistic Regression + SHAP.
- **Risk Segmentation** — High / Medium / Low tiers.
- **AI Retention Advisor** — LLM-generated, per-customer strategies.
- **Batch Analytics** — Portfolio-level insights at a glance.
- **PDF Export** — Shareable retention reports for stakeholders.
        """)
    with col2:
        st.image("feature_importance.png", caption="Sample Feature Importance Analysis")