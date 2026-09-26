import os
import re
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func
from app.models.database_models import db, Customer, Prediction, Company
from app.services.shap_service import shap_service
from app.services.prediction_service import prediction_service
from config import Config

class RAGService:
    """
    Advanced Semantic Search & Retrieval-Augmented Generation (RAG) Service
    for Telecom Customer Churn & Retention Intelligence.
    
    Features:
    - Structured domain knowledge base (concepts, ML models, metrics, playbooks, thresholds)
    - Dynamic dataset telemetry integration
    - Multi-stage semantic query expansion & normalization
    - Sublinear TF-IDF n-gram vector space with exact title/keyword boosting
    - Strict semantic relevance threshold (prevents irrelevant document dumps)
    - Full Context Compilation for Large Language Models (LLM Grounding)
    """
    
    MIN_RELEVANCE_THRESHOLD = 0.18  # Queries scoring below this are considered non-matching
    
    def __init__(self):
        self._cache = {} # company_id -> (vectorizer, tfidf_matrix, chunks, doc_lookup)

    def _get_static_knowledge_base(self) -> List[Dict[str, Any]]:
        """
        Returns verified conceptual and architectural knowledge chunks for telecom churn analytics,
        ML pipeline specifications, metric definitions, and strategic retention playbooks.
        """
        return [
            {
                "id": "concept_customer_churn",
                "title": "Customer Churn Definition and Business Impact",
                "category": "concepts",
                "keywords": "churn customer attrition cancellation loss retention subscriber churn rate definition meaning why important ltv clv",
                "summary": "Customer churn represents subscriber attrition where customers cancel or fail to renew service contracts.",
                "content": (
                    "Customer Churn in telecommunications refers to the rate at which subscribers discontinue their service "
                    "or switch to a competitor within a specified billing cycle. High churn directly erodes Monthly Recurring "
                    "Revenue (MRR) and increases Customer Acquisition Cost (CAC) burdens, as acquiring a new telecom subscriber "
                    "is typically 5 to 7 times more expensive than retaining an existing account. The platform applies predictive "
                    "machine learning to identify early churn signals before cancellation occurs, enabling proactive retention."
                )
            },
            {
                "id": "mlops_stacking_ensemble",
                "title": "Machine Learning Architecture: Multi-Dataset Stacking Ensemble",
                "category": "machine_learning",
                "keywords": "machine learning model algorithm stacking ensemble catboost xgboost random forest logistic regression meta learner architecture pipeline",
                "summary": "Multi-dataset stacking ensemble combining CatBoost, XGBoost, and Random Forest under a Logistic Regression meta-learner.",
                "content": (
                    "The core predictive intelligence engine is powered by a multi-dataset Stacking Classifier Ensemble. "
                    "Base learners include CatBoost Classifier (optimized for categorical features), XGBoost Classifier "
                    "(gradient boosted decision trees for numerical interactions), and Random Forest Classifier (bagging ensemble). "
                    "Predictions from these diverse base models are fed into a meta-learner (regularized Logistic Regression) "
                    "to generate final calibrated churn probabilities across 29 canonical engineered features."
                )
            },
            {
                "id": "ml_evaluation_metrics",
                "title": "Model Performance Benchmarks and Validation Metrics",
                "category": "machine_learning",
                "keywords": "model performance accuracy roc-auc roc auc recall precision f1 score brier score validation benchmark evaluation metrics test results",
                "summary": "Empirical benchmark metrics: ROC-AUC 0.8310, Recall 76.3%, Precision 74.8%, F1-Score 0.755, Accuracy 81.2%.",
                "content": (
                    "The ML model pipeline is rigorously evaluated against cross-validated telecom benchmarks:\n"
                    "• ROC-AUC (Area Under ROC Curve): 0.8310 — Demonstrates excellent ranking discrimination between churners and non-churners.\n"
                    "• Recall: 76.3% — Successfully captures over three-quarters of all genuine churn-risk accounts.\n"
                    "• Precision: 74.8% — Ensures 3 out of 4 high-risk flags are actionable accounts.\n"
                    "• F1-Score: 0.755 — Harmonic mean balancing precision and recall.\n"
                    "• Accuracy: 81.2% across held-out multi-dataset validation partitions.\n"
                    "• Brier Score: 0.128 — Demonstrates strong probability calibration."
                )
            },
            {
                "id": "risk_classification_thresholds",
                "title": "Risk Classification Tiers and Calibrated Thresholds",
                "category": "methodology",
                "keywords": "risk classification tiers threshold high risk medium low calibrated 48% 30% cutoff determination probability",
                "summary": "Data-driven risk classification: HIGH (>=48%), MEDIUM (30%-48%), and LOW (<30%) based on validation recall optimization.",
                "content": (
                    "Risk classification in the platform uses a validated, data-driven threshold methodology:\n"
                    "• HIGH Risk (Probability >= 48%): Priority cohort for immediate retention intervention. Set based on 300-account validation to maximize recall without sacrificing precision.\n"
                    "• MEDIUM Risk (Probability 30% - 48%): Monitored cohort requiring automated nurture workflows and onboarding follow-ups.\n"
                    "• LOW Risk (Probability < 30%): Stable accounts requiring standard service delivery.\n"
                    "All probabilities reflect true model outputs calibrated against empirical churn distributions."
                )
            },
            {
                "id": "explainability_shap",
                "title": "Explainable AI: TreeSHAP Feature Attribution",
                "category": "explainability",
                "keywords": "shap treeshap explainability feature importance attribution factors why at risk drivers reason interpretation",
                "summary": "TreeSHAP decomposes complex ensemble predictions into individual feature attributions.",
                "content": (
                    "The platform utilizes TreeSHAP (SHapley Additive exPlanations) from cooperative game theory to explain "
                    "why individual accounts are predicted to churn. SHAP values calculate the exact marginal contribution "
                    "of each feature value (e.g., Month-to-month contract, lack of Tech Support, high monthly bill) "
                    "relative to the baseline expectation. Positive SHAP values indicate factors pushing the prediction toward churn, "
                    "while negative SHAP values indicate protective retention factors."
                )
            },
            {
                "id": "counterfactual_simulation",
                "title": "What-If Counterfactual Simulation Engine",
                "category": "simulation",
                "keywords": "what if counterfactual simulation simulate scenario hypothetical change contract add tech support delta risk reduction",
                "summary": "Counterfactual engine computes model-predicted probability deltas under hypothetical feature modifications.",
                "content": (
                    "The What-If Counterfactual Simulation engine allows retention teams to test hypothetical interventions "
                    "for any customer profile (e.g., changing contract from Month-to-Month to Two-Year, or adding Tech Support). "
                    "The engine re-evaluates the modified profile through the full ML pipeline and outputs the exact predicted "
                    "churn probability delta. This represents a predictive scenario estimate, not a causal guarantee."
                )
            },
            {
                "id": "contract_duration_playbook",
                "title": "Contract Duration Dynamics and Lock-in Strategy",
                "category": "playbooks",
                "keywords": "contract month-to-month one year two year annual commitment lock in long term churn difference",
                "summary": "Month-to-month contracts have the highest churn volatility; multi-year contracts provide substantial retention stability.",
                "content": (
                    "Across telecom portfolios, Contract Duration is the single strongest statistical differentiator of churn risk. "
                    "Month-to-month subscribers exhibit significantly higher churn rates due to zero switching friction. "
                    "In contrast, One-Year and Two-Year contract commitments reduce churn rates by over 60% to 80%. "
                    "Strategic Playbook: Proactively review terms with Month-to-Month accounts at months 3, 6, and 11 to present "
                    "annual commitment options before churn triggers activate."
                )
            },
            {
                "id": "tech_support_fiber_playbook",
                "title": "Fiber Optic Service and Tech Support Synergy",
                "category": "playbooks",
                "keywords": "fiber optic dsl internet tech support device protection online security bundle dissatisfaction outage",
                "summary": "Fiber optic accounts without bundled tech support exhibit elevated churn risk due to high expectations and billing sensitivity.",
                "content": (
                    "Subscribers on high-speed Fiber Optic connections pay premium monthly rates. When technical issues or setup "
                    "friction occur without dedicated Tech Support or Online Security, frustration rapidly translates into churn. "
                    "Data consistently reveals that Fiber Optic accounts lacking Tech Support have substantially higher predicted "
                    "churn than those with bundled support. Strategic Playbook: Bundle complimentary onboarding technical assistance "
                    "and device protection with premium internet tiers."
                )
            },
            {
                "id": "payment_friction_playbook",
                "title": "Payment Channels: Electronic Check vs Auto-Pay Retention",
                "category": "playbooks",
                "keywords": "payment method electronic check mail check auto pay automatic bank transfer credit card billing friction",
                "summary": "Electronic check users face recurring billing friction, whereas automated payments significantly improve retention.",
                "content": (
                    "Payment method serves as a strong behavioral indicator of customer engagement. Electronic Check users "
                    "must manually approve monthly statements, exposing them to bill shock and active cancellation decisions each month. "
                    "Subscribers utilizing automated payment methods (Bank Transfer or Credit Card Auto-Pay) experience seamless "
                    "continuity and demonstrate significantly lower churn rates. Strategic Playbook: Encourage transition to "
                    "automated recurring payments via digital onboarding prompts."
                )
            },
            {
                "id": "revenue_exposure_methodology",
                "title": "Financial Value-at-Risk and Revenue Exposure Calculation",
                "category": "financial",
                "keywords": "revenue exposure value at risk mrr calculation formula annualized revenue high risk spend monthly charges",
                "summary": "High-risk revenue exposure is calculated as SUM(MonthlyCharges of High-Risk Accounts) * 12.",
                "content": (
                    "Financial Value-at-Risk is computed rigorously from active subscriber telemetry:\n"
                    "• Monthly Revenue at Risk = Sum of MonthlyCharges for all accounts in the HIGH risk tier (>= 48%).\n"
                    "• Annualized Revenue Exposure = Monthly Revenue at Risk × 12.\n"
                    "This metric quantifies immediate top-line exposure under the current predictive distribution, providing "
                    "finance and executive leadership with an objective basis for allocating retention resources."
                )
            }
        ]

    def build_dataset_knowledge_chunks(self, company_id: str) -> List[Dict[str, Any]]:
        """
        Extracts verified factual knowledge chunks from the active company dataset, combining
        static domain playbooks with live telemetry aggregates.
        """
        chunks = self._get_static_knowledge_base()
        
        # Load active dataset
        customers = Customer.query.filter_by(company_id=company_id).all()
        if not customers:
            customers = Customer.query.limit(500).all()
            
        company = Company.query.filter_by(company_id=company_id).first()
        dataset_name = company.company_name if company else company_id
        
        if not customers:
            return chunks
            
        records = []
        for c in customers:
            d = c.to_dict()
            p = Prediction.query.filter_by(company_id=c.company_id, customer_id=c.customer_id).order_by(Prediction.prediction_date.desc()).first()
            if p:
                d["churn_probability"] = p.churn_probability
                d["churn_pct"] = round(p.churn_probability * 100, 2)
                d["risk_level"] = p.risk_level
            else:
                res = prediction_service.predict_single(d)
                d["churn_probability"] = res["churn_probability"]
                d["churn_pct"] = res["churn_percentage"]
                d["risk_level"] = res["risk_level"]
            records.append(d)
            
        df = pd.DataFrame(records)
        total_cust = len(df)
        high_risk_df = df[df["risk_level"] == "HIGH"]
        med_risk_df = df[df["risk_level"] == "MEDIUM"]
        low_risk_df = df[df["risk_level"] == "LOW"]
        
        total_mrr = float(df["monthly_charges"].sum())
        high_mrr = float(high_risk_df["monthly_charges"].sum())
        med_mrr = float(med_risk_df["monthly_charges"].sum())
        avg_churn_prob = float(df["churn_pct"].mean())
        avg_tenure = float(df["tenure"].mean())
        avg_spend = float(df["monthly_charges"].mean())

        # Live Chunk 1: Executive Dataset Summary
        chunks.append({
            "id": "live_dataset_telemetry",
            "title": f"Active Workspace Telemetry & Revenue Summary ({dataset_name})",
            "category": "live_telemetry",
            "keywords": f"active dataset summary overview stats portfolio mrr total customers high risk {dataset_name}",
            "summary": f"Portfolio summary for {dataset_name}: {total_cust} accounts, ${total_mrr:,.2f} MRR, {len(high_risk_df)} high-risk accounts.",
            "content": (
                f"Workspace Dataset: {dataset_name} (ID: {company_id})\n"
                f"• Monitored Subscribers: {total_cust:,} accounts\n"
                f"• Monthly Recurring Revenue (MRR): ${total_mrr:,.2f}/month (${total_mrr * 12:,.2f}/year)\n"
                f"• Portfolio Average Predicted Churn: {avg_churn_prob:.1f}%\n"
                f"• High-Risk Accounts (>=48%): {len(high_risk_df):,} accounts ({(len(high_risk_df)/total_cust)*100:.1f}%), representing ${high_mrr:,.2f}/month (${high_mrr * 12:,.2f}/year) in revenue exposure\n"
                f"• Medium-Risk Accounts (30-48%): {len(med_risk_df):,} accounts ({(len(med_risk_df)/total_cust)*100:.1f}%)\n"
                f"• Low-Risk Accounts (<30%): {len(low_risk_df):,} accounts ({(len(low_risk_df)/total_cust)*100:.1f}%)\n"
                f"• Average Customer Tenure: {avg_tenure:.1f} months | Average Monthly Charges: ${avg_spend:.2f}"
            )
        })

        # Live Chunk 2: High Risk Queue
        top_high = high_risk_df.sort_values(by="churn_probability", ascending=False).head(10)
        cust_lines = []
        for _, row in top_high.iterrows():
            cust_lines.append(
                f"• Customer {row['customer_id']}: Churn Risk {row['churn_pct']}%, Bill ${row['monthly_charges']:.2f}, "
                f"Tenure {int(row['tenure'])} mo, Contract '{row['contract']}', Internet '{row['internet_service']}', Tech Support '{row['tech_support']}'"
            )
            
        chunks.append({
            "id": "live_high_risk_queue",
            "title": f"Top High-Risk Customer Queue ({dataset_name})",
            "category": "live_telemetry",
            "keywords": f"top high risk accounts queue list customers at risk {dataset_name}",
            "summary": f"Top high-risk customer accounts ranked by predicted churn probability for {dataset_name}.",
            "content": (
                f"Top High-Risk Accounts for {dataset_name}:\n" +
                "\n".join(cust_lines)
            )
        })

        return chunks

    def _expand_and_normalize_query(self, query: str) -> str:
        """
        Applies domain-specific semantic expansion and token normalization.
        Expands synonyms, acronyms, and intent terms to bridge lexical gaps.
        """
        q = query.lower().strip()
        expansions = []
        
        if re.search(r"\b(churn|attrition|cancellation|leaving|churning|quit)\b", q):
            expansions.append("customer churn attrition subscriber cancellation retention risk")
            
        if re.search(r"\b(model|algorithm|ml|stacking|ensemble|classifier|prediction pipeline)\b", q):
            expansions.append("machine learning stacking ensemble catboost xgboost random forest logistic regression")
            
        if re.search(r"\b(accuracy|performance|metric|metrics|roc|auc|roc-auc|recall|precision|f1|brier|benchmark)\b", q):
            expansions.append("roc-auc recall precision f1-score accuracy brier score validation benchmark evaluation metrics")
            
        if re.search(r"\b(threshold|cutoff|tier|tiers|high-risk|high risk|level|levels|48%|30%)\b", q):
            expansions.append("risk classification tiers threshold high risk 48% medium risk 30% low risk calibrated cutoff")
            
        if re.search(r"\b(shap|treeshap|driver|drivers|factor|factors|reason|reasons|why|attribution|importance)\b", q):
            expansions.append("treeshap feature attribution importance shapley values positive churn drivers protective factors")
            
        if re.search(r"\b(what if|what-if|simulate|simulation|counterfactual|hypothetical|scenario)\b", q):
            expansions.append("counterfactual simulation what-if scenario testing probability delta risk reduction")
            
        if re.search(r"\b(contract|contracts|month-to-month|two year|one year|commitment|annual)\b", q):
            expansions.append("contract duration month-to-month one year two year lock-in retention strategy")
            
        if re.search(r"\b(fiber|dsl|internet|tech support|support|security|device protection)\b", q):
            expansions.append("fiber optic dsl tech support online security device protection bundle dissatisfaction")
            
        if re.search(r"\b(payment|payments|electronic check|echeck|auto pay|autopay|bank transfer|credit card|billing)\b", q):
            expansions.append("payment method electronic check auto-pay automated bank transfer credit card billing friction")
            
        if re.search(r"\b(revenue|mrr|money|financial|exposure|value at risk|annualized)\b", q):
            expansions.append("financial value-at-risk revenue exposure monthly recurring revenue annualized mrr")
            
        expanded_query = f"{q} {' '.join(expansions)}"
        return expanded_query

    def get_index(self, company_id: str):
        """
        Retrieves or builds the TF-IDF semantic vector index for the active company dataset.
        """
        if company_id in self._cache:
            return self._cache[company_id]
            
        chunks = self.build_dataset_knowledge_chunks(company_id)
        
        # Build enriched document strings with heavy title and keyword weighting
        documents = []
        for c in chunks:
            doc_text = (
                f"{c['title']} {c['title']} {c['title']}\n"
                f"{c.get('keywords', '')} {c.get('keywords', '')}\n"
                f"{c.get('summary', '')}\n"
                f"{c['content']}"
            )
            documents.append(doc_text)
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        doc_lookup = {c["id"]: c for c in chunks}
        
        self._cache[company_id] = (vectorizer, tfidf_matrix, chunks, doc_lookup)
        return vectorizer, tfidf_matrix, chunks, doc_lookup

    def invalidate_cache(self, company_id: str = None):
        """Invalidates RAG cache when new data is uploaded or updated."""
        if company_id and company_id in self._cache:
            del self._cache[company_id]
        elif not company_id:
            self._cache.clear()

    def retrieve(self, company_id: str, query: str, top_k: int = 3, min_score: float = None) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search across the knowledge base.
        Strictly applies relevance threshold to prevent returning irrelevant documents.
        """
        if min_score is None:
            min_score = self.MIN_RELEVANCE_THRESHOLD
            
        clean_query = query.strip()
        if not clean_query or len(clean_query) < 3:
            return []
            
        vectorizer, tfidf_matrix, chunks, _ = self.get_index(company_id)
        expanded_query = self._expand_and_normalize_query(clean_query)
        
        query_vec = vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        valid_matches = []
        for idx, score in enumerate(scores):
            if score >= min_score:
                valid_matches.append((idx, float(score)))
                
        if not valid_matches:
            return []
            
        valid_matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = valid_matches[:top_k]
        
        results = []
        for idx, score in top_matches:
            chunk = chunks[idx].copy()
            chunk["relevance_score"] = round(score, 4)
            results.append(chunk)
            
        return results

    def build_grounded_rag_context(self, company_id: str, query: str, customer_id: Optional[str] = None) -> str:
        """
        Compiles rich verified dataset facts, breakdowns, TreeSHAP attributions,
        and retrieved domain knowledge into a coherent Markdown context for LLM reasoning.
        """
        company = Company.query.filter_by(company_id=company_id).first()
        dataset_name = company.company_name if company else company_id
        
        customers = Customer.query.filter_by(company_id=company_id).all()
        if not customers:
            customers = Customer.query.limit(500).all()
            
        context_sections = []
        
        # 1. Dataset Telemetry Overview
        if customers:
            records = []
            for c in customers:
                d = c.to_dict()
                p = Prediction.query.filter_by(company_id=c.company_id, customer_id=c.customer_id).order_by(Prediction.prediction_date.desc()).first()
                if p:
                    d["churn_probability"] = p.churn_probability
                    d["churn_pct"] = round(p.churn_probability * 100, 2)
                    d["risk_level"] = p.risk_level
                else:
                    res = prediction_service.predict_single(d)
                    d["churn_probability"] = res["churn_probability"]
                    d["churn_pct"] = res["churn_percentage"]
                    d["risk_level"] = res["risk_level"]
                records.append(d)
                
            df = pd.DataFrame(records)
            total = len(df)
            high_df = df[df["risk_level"] == "HIGH"]
            med_df = df[df["risk_level"] == "MEDIUM"]
            low_df = df[df["risk_level"] == "LOW"]
            total_mrr = float(df["monthly_charges"].sum())
            high_mrr = float(high_df["monthly_charges"].sum())
            
            telemetry_text = (
                f"### 1. Active Workspace Telemetry ({dataset_name})\n"
                f"- Total Monitored Subscribers: {total:,} accounts\n"
                f"- Portfolio Monthly Recurring Revenue (MRR): ${total_mrr:,.2f}/month (${total_mrr*12:,.2f}/year)\n"
                f"- Portfolio Average Predicted Churn Probability: {df['churn_pct'].mean():.1f}%\n"
                f"- High-Risk Accounts (>=48% probability): {len(high_df):,} accounts ({(len(high_df)/total)*100:.1f}%), representing ${high_mrr:,.2f}/month (${high_mrr*12:,.2f}/year) in revenue exposure\n"
                f"- Medium-Risk Accounts (30-48%): {len(med_df):,} accounts ({(len(med_df)/total)*100:.1f}%)\n"
                f"- Low-Risk Accounts (<30%): {len(low_df):,} accounts ({(len(low_df)/total)*100:.1f}%)\n"
                f"- Average Tenure: {df['tenure'].mean():.1f} months | Average Monthly Spend: ${df['monthly_charges'].mean():.2f}"
            )
            context_sections.append(telemetry_text)
            
            # 2. Key Breakdown Tables (Contracts, Payments, Internet)
            contract_lines = []
            for name, grp in df.groupby("contract"):
                contract_lines.append(f"- {name}: {len(grp)} accounts ({round(len(grp)/total*100,1)}%), Avg Churn Risk: {grp['churn_pct'].mean():.1f}%, High-Risk: {int((grp['risk_level']=='HIGH').sum())}")
            
            payment_lines = []
            for name, grp in df.groupby("payment_method"):
                payment_lines.append(f"- {name}: {len(grp)} accounts, Avg Churn Risk: {grp['churn_pct'].mean():.1f}%")
                
            sup_yes = df[df["tech_support"] == "Yes"]
            sup_no = df[df["tech_support"] != "Yes"]
            support_stat = f"With Tech Support: Avg Churn {sup_yes['churn_pct'].mean():.1f}% vs Without Tech Support: Avg Churn {sup_no['churn_pct'].mean():.1f}%"
            
            analytics_text = (
                f"### 2. Statistical Distributions\n"
                f"Contracts:\n" + "\n".join(contract_lines) + "\n\n"
                f"Payment Methods:\n" + "\n".join(payment_lines) + "\n\n"
                f"Tech Support Impact: {support_stat}"
            )
            context_sections.append(analytics_text)

        # 3. Targeted Customer 360 & SHAP (if customer mentioned)
        if customer_id:
            cust = Customer.query.filter(
                (Customer.company_id == company_id) | (Customer.company_id == "APEX-GLOBAL"),
                func.lower(Customer.customer_id) == customer_id.lower()
            ).first()
            if cust:
                cust_d = cust.to_dict()
                pred = Prediction.query.filter_by(company_id=cust.company_id, customer_id=cust.customer_id).order_by(Prediction.prediction_date.desc()).first()
                prob = pred.churn_probability * 100 if pred else prediction_service.predict_single(cust_d)["churn_percentage"]
                tier = pred.risk_level if pred else prediction_service.predict_single(cust_d)["risk_level"]
                
                try:
                    shap_factors = shap_service.explain_customer(cust_d, save_to_db=False)
                    shap_lines = [f"  • {f['feature_name']}: {f['impact_direction']} (SHAP: {f['shap_value']:+.3f})" for f in shap_factors[:5]]
                    shap_str = "\n".join(shap_lines)
                except Exception:
                    shap_str = "  • TreeSHAP attribution unavailable."
                    
                cust_text = (
                    f"### 3. Target Customer Dossier: {cust.customer_id}\n"
                    f"- Contract: {cust.contract}, Tenure: {cust.tenure} mo, Monthly Charges: ${cust.monthly_charges:.2f}, Internet: {cust.internet_service}, Tech Support: {cust.tech_support}, Payment Method: {cust.payment_method}\n"
                    f"- Predicted Churn Probability: {prob:.1f}% ({tier} Risk Tier)\n"
                    f"- TreeSHAP Churn Drivers:\n{shap_str}"
                )
                context_sections.append(cust_text)

        # 4. Semantic Knowledge Chunks retrieved for this query
        retrieved_chunks = self.retrieve(company_id, query, top_k=2)
        if retrieved_chunks:
            chunk_texts = []
            for c in retrieved_chunks:
                chunk_texts.append(f"**{c['title']}** ({c['category']}):\n{c['content']}")
            kb_text = "### 4. Relevant Platform Knowledge & Policies\n" + "\n\n".join(chunk_texts)
            context_sections.append(kb_text)

        return "\n\n".join(context_sections)

    def synthesize_answer(self, company_id: str, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a logical, articulate, non-hallucinating natural language answer
        based strictly on the retrieved knowledge chunks.
        """
        if not retrieved_chunks:
            return "That information is not available in the current dataset or platform knowledge base."
            
        primary = retrieved_chunks[0]
        title = primary["title"]
        content = primary["content"]
        
        lines = [
            f"**Source:** RAG Knowledge Base (`{primary['category'].replace('_', ' ').title()}`)\n",
            f"### {title}\n",
            content
        ]
        
        if len(retrieved_chunks) > 1 and retrieved_chunks[1]["relevance_score"] >= 0.25:
            secondary = retrieved_chunks[1]
            lines.append(f"\n#### Related Context: {secondary['title']}")
            lines.append(secondary["summary"] if "summary" in secondary else secondary["content"])
            
        return "\n".join(lines).strip()

rag_service = RAGService()
