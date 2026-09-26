# ChurnIntel AI — Telecom Customer Churn & Retention Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask 3.0](https://img.shields.io/badge/framework-Flask%203.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![ML Models](https://img.shields.io/badge/models-Stacking%20Ensemble%20%7C%20CatBoost%20%7C%20XGBoost-success.svg)](https://scikit-learn.org/)
[![Explainable AI](https://img.shields.io/badge/XAI-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![Design System](https://img.shields.io/badge/UI%2FUX-Enterprise%20B2B%20SaaS-indigo.svg)](#-enterprise-uiux-design-system)

An enterprise-grade B2B SaaS analytics platform for telecom subscriber retention, calibrated churn risk classification, TreeSHAP feature attributions, counterfactual what-if scenario simulations, and grounded Generative AI decision support.

---

## 🏗️ System Architecture & Workflow

```text
                        TELECOM OPERATOR / ENTERPRISE TENANT
                                          │
                                          ▼
                                4-Step Ingestion Gateway
                         (CSV / XLSX / XLS Validation Pipeline)
                                          │
                                          ▼
                                 Automated Sanitization
                         (Schema Mapping, Type Check, Imputation)
                                          │
                                          ▼
                               Multi-Tenant Database Layer
                            (PostgreSQL / SQLite SQLAlchemy)
                                          │
                                          ▼
                            ML Prediction & Inference Engine
                     (Stacking Ensemble / CatBoost / XGBoost / RF)
                                          │
                                          ▼
                           Calibrated Risk Classification
                       (HIGH ≥ 48% | MEDIUM 30-48% | LOW < 30%)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
             TreeSHAP Engine                         What-If Simulator
       (Local & Global Attributions)              (Counterfactual Intervention)
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                            Grounded GenAI Retention RAG
                     (Live Telemetry & Policy Strategy Reasoning)
                                          │
                                          ▼
                                 Flask REST API Layer
                                          │
                                          ▼
                            Enterprise B2B SaaS Dashboard
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
        KPI Overview               Customer 360°               AI Assistant
      (Donut / Drivers)         (Dossier & Waterfall)       (Decision Console)
```

---

## 🌟 Key Capabilities & Modules

### 1. Enterprise UI/UX Design System & Theme Switching
* **Professional Light & Dark Themes:** Seamless toggle between a crisp enterprise light theme (`#F8FAFC` background, pure white cards, navy/slate typography) and a deep slate dark theme (`#0B0F19` canvas, `#111827` cards, `#F8FAFC` text).
* **Top Navigation Switcher:** Instant theme switching with `localStorage` persistence and zero Flash of Unstyled Content (FOUC).
* **Adaptive Chart.js:** The Risk Distribution Donut Chart dynamically updates its segment borders and legend typography colors on theme transitions.
* **Information Density:** Clean tables, subtle risk badges (`HIGH`, `MEDIUM`, `LOW`), compact KPI cards, and refined form inputs.

### 2. Machine Learning Benchmark & Stacking Ensemble
* **Champion Architecture:** Stacking Ensemble meta-learner combining CatBoost Classifier, XGBoost Classifier, and Random Forest.
* **Multi-Dataset Training:** Trained and cross-validated across **13,043 normalized subscriber records** using Stratified 5-Fold Cross Validation.
* **Tuned Decision Cutoffs:** Evaluated at a calibrated **48.0% decision threshold** to optimize recall (80.7%) and precision while minimizing false negatives.

### 3. Explainable AI (TreeSHAP Engine)
* **Local Explainability:** Identifies individual feature contributions answering *"Why is this specific subscriber predicted to churn?"*.
* **Impact Classification:** Differentiates between **Risk Accelerators** ($\uparrow$) and **Retention Anchors** ($\downarrow$).
* **Global Attributions:** Computes top dataset churn drivers across contract structures, internet service types, billing methods, and support tiers.

### 4. Interactive Counterfactual What-If Simulator
* Real-time scenario simulator allowing operators to adjust contract terms, support packages, payment methods, and monthly pricing.
* Calculates the exact model-predicted probability delta ($\Delta$) and risk tier transition without altering production database records.

### 5. Grounded RAG AI Retention Assistant
* **Fact-Grounded Reasoning:** Dynamic context injection synthesizing live SQL telemetry, TreeSHAP attributions, and cohort statistics into LLM prompts.
* **Analytical Report Blocks:** Outputs scannable reports with markdown tables, key metric callouts, executive summaries, and clear source attribution (`Source: Analytical Cohort Engine`, `Source: TreeSHAP Engine`, `Source: Grounded ML Telemetry`).
* **Multi-Provider Support:** Plug-and-play support for Google Gemini, OpenAI GPT-4o-mini, Groq Llama 3.3 70B, and the built-in Local Semantic Engine.

### 6. Bulk Ingestion & Validation Gateway
* **Supported Formats:** `.csv`, `.xlsx`, `.xls` files up to 32MB.
* **4-Step Pipeline:** Upload $\rightarrow$ Validate $\rightarrow$ Score Risk $\rightarrow$ Action Insights.
* **Health Checks:** Automatic column schema auto-mapping, type enforcement, duplicate ID resolution, and median/mode null imputation.
* **1-Click Demo Sandbox:** Instant pre-configured 150-customer dataset for zero-friction evaluation.

---

## 📊 Cross-Model Benchmark Matrix (Held-out 2,609 Test Cohort)

| Model Architecture | ROC-AUC | PR-AUC | F1-Score | Recall (Detection Rate) | Precision | Accuracy | Calibrated Cutoff | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Champion Stacking Ensemble** | **0.8312** | **0.7424** | **0.6980** | **76.3%** | **64.3%** | **74.9%** | **0.48** | **Production Deploy** |
| ⚡ **CatBoost Classifier** | **0.8312** | **0.7421** | **0.6990** | **80.7%** | **61.6%** | **73.6%** | **0.45** | Benchmark Candidate |
| 🚀 **XGBoost Classifier** | **0.8304** | **0.7410** | **0.6960** | **77.3%** | **63.3%** | **74.3%** | **0.49** | Benchmark Candidate |
| 🌲 **Gradient Boosting Machine** | **0.8283** | **0.7332** | **0.6974** | **83.9%** | **60.0%** | **72.3%** | **0.31** | Benchmark Candidate |
| 🌳 **Random Forest** | **0.8237** | **0.7330** | **0.6940** | **81.9%** | **60.2%** | **72.6%** | **0.41** | Benchmark Candidate |
| 📉 **Logistic Regression (Baseline)** | 0.8192 | 0.7175 | 0.6904 | 79.2% | 61.2% | 73.0% | 0.48 | Legacy Baseline |

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/predict` | Single customer real-time inference |
| `GET` | `/api/customer/<customer_id>` | Retrieve customer profile and calibrated prediction |
| `GET` | `/api/customer/<customer_id>/explanation` | Retrieve TreeSHAP local feature attributions |
| `POST` | `/api/customer/<customer_id>/what-if` | Execute counterfactual scenario simulation |
| `POST` | `/api/customer/<customer_id>/recommendation` | Generate GenAI retention strategy & action plan |
| `POST` | `/api/assistant` | Natural-language grounded decision support query |
| `GET` | `/api/dashboard/summary` | Executive portfolio summary & MRR at risk |
| `GET` | `/api/dashboard/risk-distribution` | Risk segmentation distribution counts |
| `GET` | `/api/dashboard/churn-drivers` | Top global TreeSHAP dataset churn drivers |
| `GET` | `/api/export/predictions` | Streaming CSV download of all predicted customers |
| `POST` | `/upload/validate` | Schema validation and health preview for uploaded files |
| `POST` | `/upload/import` | Database batch ingestion and ML inference run |

---

## 📁 Project Structure

```text
churn_app/
├── app/
│   ├── models/
│   │   └── database_models.py       # SQLAlchemy ORM schemas (Company, Customer, Prediction, Tracking)
│   ├── routes/
│   │   ├── dashboard.py             # Overview metrics, charts & MLOps benchmarking
│   │   ├── customers.py             # Filterable customer directory & Customer 360° dossier
│   │   ├── assistant.py             # AI Retention Assistant & provider settings
│   │   ├── manual_predict.py        # Scenario simulator workspace
│   │   ├── upload.py                # Dataset upload, validation, sample template & demo loader
│   │   └── api.py                   # REST API routes for predictions, SHAP, What-If, Export
│   ├── services/
│   │   ├── prediction_service.py    # Multi-model inference & probability calibration
│   │   ├── shap_service.py          # TreeSHAP explainer & global driver aggregator
│   │   ├── recommendation_service.py# Rule-based & GenAI retention action planner
│   │   ├── assistant_service.py     # Grounded analytical assistant & query routing
│   │   ├── rag_service.py           # Context retrieval & telemetry builder
│   │   ├── llm_service.py           # Multi-provider LLM connector (Gemini, OpenAI, Groq)
│   │   └── validation_service.py    # File parser, schema verifier & type cleaner
│   ├── static/
│   │   └── css/
│   │       └── main.css             # Enterprise design system tokens (Light & Dark theme)
│   └── templates/
│       ├── base.html                # Navigation, theme switcher, notifications, footer
│       ├── dashboard.html           # KPI overview, Chart.js Donut, drivers, priority queue
│       ├── customers.html           # Search, multi-facet filter bar, high-density table
│       ├── customer_detail.html     # Customer 360°, SHAP waterfall, What-If simulator
│       ├── assistant.html           # AI decision support console & suggested inquiries
│       ├── manual_predict.html      # Hypothetical scenario testing workspace
│       ├── model_info.html          # MLOps cross-model evaluation matrix & telemetry
│       ├── landing_upload.html      # 4-step ingestion gateway & demo dataset loader
│       ├── upload.html              # Drag-and-drop validation interface
│       └── customers_empty.html     # Empty state view
├── config.py                        # Centralized application configuration
├── run.py                           # Development application entry point
├── test_platform.py                 # Automated 8-suite enterprise integration test
├── requirements.txt                 # Python package dependencies
└── README.md                        # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/arpit467/Telecom_Churn_app.git
cd Telecom_Churn_app
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python run.py
```
Open your browser at **`http://127.0.0.1:5000/`**.

### 5. Run Automated Tests
```bash
python test_platform.py
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
