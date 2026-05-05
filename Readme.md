# 🛡️ ChurnGuard AI — Customer Churn Prediction & AI Retention System

> **Live App →** _[Deploy to Streamlit Cloud and paste URL here]_

A full-stack ML + Agentic AI system that predicts customer churn and generates personalised retention strategies using LLMs, SHAP explainability, and a RAG pipeline.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Milestone 1 — ML Pipeline](#milestone-1--ml-pipeline)  
3. [Milestone 2 — AI Retention Advisor](#milestone-2--ai-retention-advisor)  
4. [Dataset](#dataset)  
5. [Model Performance](#model-performance)  
6. [System Architecture](#system-architecture)  
7. [Running Locally](#running-locally)  
8. [Environment Setup](#environment-setup)  
9. [Deployment](#deployment)  
10. [Team](#team)  

---

## Project Overview

Customer churn is one of the most expensive problems in the telecom industry. Acquiring a new customer costs 5–7× more than retaining one. **ChurnGuard AI** addresses this with two tightly integrated layers:

| Layer | What it does |
|---|---|
| **ML Model** | Logistic Regression trained on Telco data — predicts churn probability per customer |
| **AI Advisor** | LangGraph agentic workflow — generates a bespoke, structured retention strategy via LLM + RAG |

---

## Milestone 1 — ML Pipeline

- ✅ Supervised binary classification (churn / no-churn)
- ✅ Churn probability scoring (0.0 – 1.0) → High / Medium / Low risk tiers
- ✅ Feature importance via model coefficients
- ✅ Streamlit dashboard — Executive Summary, Risk Analysis, Model Performance, Data Explorer

**Preprocessing steps:**
1. Drop `customerID` (non-predictive)
2. Convert `TotalCharges` to numeric; impute 11 NaN rows with median
3. One-Hot Encode all categorical features (drop-first)
4. 80/20 stratified train–test split
5. StandardScaler normalisation (fit on train only)

---

## Milestone 2 — AI Retention Advisor

### Features added

| Feature | Detail |
|---|---|
| **AI Retention Advisor tab** | Click any at-risk customer row → LLM generates a structured retention plan |
| **LLM integration** | Groq API (`llama3-70b-8192`) via `groq` Python client |
| **SHAP explainability** | Per-customer SHAP attributions feed into the agent prompt |
| **RAG pipeline** | ChromaDB + `sentence-transformers` retrieves relevant retention playbooks |
| **Structured output** | Pydantic-validated JSON: risk level, summary, contributing factors, recommended actions, disclaimers |
| **PDF Export** (Extension 1) | `fpdf2`-generated, professional retention report — one-click download |
| **Batch Analytics** (Extension 2) | Risk donut chart, segment breakdowns, tenure × charges heatmap, top-20 at-risk table |

### Agent workflow

```
analyze_risk → identify_factors → generate_strategy → END
```

A deterministic rule-based fallback fires automatically if the Groq API is unavailable.

---

## Dataset

**IBM Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Records | 7,043 customers |
| Features | 21 (20 predictors + target) |
| Churn prevalence | ~26.5% |
| Target | `Churn` (Yes / No) |

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | **81.97%** |
| Precision | **68.31%** |
| Recall | **59.52%** |
| F1-Score | **63.61%** |

> Recall is the north-star metric — missing a churner is more costly than a false alarm.

---

## System Architecture

```
Customer CSV Upload
        │
        ▼
Preprocessing (clean / encode / scale)
        │
        ▼
Logistic Regression  ──► Churn Probability + Risk Level
        │
        ├──► Streamlit Dashboard (Milestone 1 tabs)
        │
        └──► AI Retention Advisor (Milestone 2)
                    │
                    ├── SHAP Explainer
                    ├── LangGraph Agent (Groq LLM)
                    ├── RAG Retriever (ChromaDB)
                    └── PDF Export (fpdf2)
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/ANANYA542/AI-ML-Churn_detection.git
cd AI-ML-Churn_detection
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your GROQ_API_KEY
```

### 5. Run the app

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## Environment Setup

Create a `.env` file in the project root (never commit real keys):

```
# .env.example
GROQ_API_KEY=your_groq_api_key_here
```

The app reads this automatically via `python-dotenv`. Without a valid key, the AI Advisor falls back to rule-based suggestions.

---

## Deployment

### Streamlit Community Cloud (recommended)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Repository**: `ANANYA542/AI-ML-Churn_detection`
4. Set **Main file**: `app.py`
5. Set **Python version**: 3.10
6. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
7. Click **Deploy**. Paste the resulting URL at the top of this README.

### HuggingFace Spaces (alternative)

Create a new Space → SDK: Streamlit → upload `app.py` + `requirements.txt`. Add `GROQ_API_KEY` in the Space secrets.

---

## Team

| Name | Role |
|---|---|
| Ananya | ML model, Streamlit dashboard, AI agent pipeline, PDF export, deployment |

---

## Tech Stack

`Python 3.10` · `scikit-learn` · `pandas` · `Streamlit` · `LangGraph` · `Groq API` · `SHAP` · `ChromaDB` · `sentence-transformers` · `fpdf2` · `Altair`