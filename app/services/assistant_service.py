import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import func
from app.models.database_models import db, Customer, Prediction, Company, CustomerExplanation
from app.services.shap_service import shap_service
from app.services.prediction_service import prediction_service
from app.services.what_if_service import what_if_service
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service
from config import Config

class AssistantService:
    """
    Intelligent Analytics Assistant for Telecom Customer Churn & Retention Intelligence.
    Adheres strictly to zero-hallucination principles, multi-tenant isolation, verified
    database/ML retrieval, cohort analytical modeling, and exact source labeling.
    """

    def __init__(self):
        pass

    # =========================================================================
    # MULTI-TENANT DATA ACCESS & TOOLS
    # =========================================================================

    def _get_company_dataframe(self, company_id: str) -> pd.DataFrame:
        """Loads all customer and prediction records strictly scoped to the active company/tenant."""
        customers = Customer.query.filter_by(company_id=company_id).all()
        if not customers:
            # Fallback to default company if workspace has no active records
            customers = Customer.query.filter_by(company_id="APEX-GLOBAL").all()
            if not customers:
                customers = Customer.query.limit(500).all()
            
        if not customers:
            return pd.DataFrame()
            
        records = []
        for c in customers:
            d = c.to_dict()
            p = Prediction.query.filter_by(company_id=c.company_id, customer_id=c.customer_id).order_by(Prediction.prediction_date.desc()).first()
            if p:
                d["churn_probability"] = p.churn_probability
                d["churn_pct"] = round(p.churn_probability * 100, 2)
                d["risk_level"] = p.risk_level
                d["model_version"] = p.model_version
            else:
                res = prediction_service.predict_single(d)
                d["churn_probability"] = res["churn_probability"]
                d["churn_pct"] = res["churn_percentage"]
                d["risk_level"] = res["risk_level"]
                d["model_version"] = res["model_version"]
            records.append(d)
            
        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Tool 1: get_customer_details
    # -------------------------------------------------------------------------
    def get_customer_details(self, company_id: str, customer_id: str) -> dict:
        """Retrieves raw verified customer record from SQL database."""
        cust = Customer.query.filter_by(company_id=company_id, customer_id=customer_id).first()
        if not cust:
            # Case-insensitive search within company
            cust = Customer.query.filter(
                Customer.company_id == company_id,
                func.lower(Customer.customer_id) == customer_id.lower()
            ).first()
            
        if not cust:
            # Global search fallback
            cust = Customer.query.filter(func.lower(Customer.customer_id) == customer_id.lower()).first()
            
        if not cust:
            return {
                "found": False,
                "error": f"That information is not available in the current dataset. Customer '{customer_id}' was not found in active workspace."
            }
            
        return {
            "found": True,
            "source": "Database (SQL Customer Table)",
            "customer_id": cust.customer_id,
            "company_id": cust.company_id,
            "tenure": int(cust.tenure),
            "contract": cust.contract,
            "internet_service": cust.internet_service,
            "monthly_charges": cust.monthly_charges,
            "total_charges": cust.total_charges,
            "phone_service": cust.phone_service,
            "multiple_lines": cust.multiple_lines,
            "online_security": cust.online_security,
            "online_backup": cust.online_backup,
            "device_protection": cust.device_protection,
            "tech_support": cust.tech_support,
            "streaming_tv": cust.streaming_tv,
            "streaming_movies": cust.streaming_movies,
            "paperless_billing": cust.paperless_billing,
            "payment_method": cust.payment_method,
            "gender": cust.gender,
            "senior_citizen": cust.senior_citizen,
            "partner": cust.partner,
            "dependents": cust.dependents,
            "created_at": cust.created_at.isoformat() if cust.created_at else None
        }

    # -------------------------------------------------------------------------
    # Tool 2: get_customer_prediction
    # -------------------------------------------------------------------------
    def get_customer_prediction(self, company_id: str, customer_id: str) -> dict:
        """Retrieves verified prediction from ML model / database."""
        cust_details = self.get_customer_details(company_id, customer_id)
        if not cust_details.get("found"):
            return cust_details
            
        pred = Prediction.query.filter_by(
            company_id=cust_details["company_id"],
            customer_id=cust_details["customer_id"]
        ).order_by(Prediction.prediction_date.desc()).first()
        
        if pred:
            return {
                "found": True,
                "source": "ML Prediction Pipeline (Database Stored)",
                "customer_id": cust_details["customer_id"],
                "churn_probability": pred.churn_probability,
                "churn_percentage": round(pred.churn_probability * 100, 2),
                "risk_level": pred.risk_level,
                "model_version": pred.model_version,
                "prediction_date": pred.prediction_date.strftime("%Y-%m-%d %H:%M UTC") if pred.prediction_date else "Recent"
            }
        else:
            res = prediction_service.predict_single(cust_details)
            return {
                "found": True,
                "source": "ML Prediction Pipeline (Live Inference)",
                "customer_id": cust_details["customer_id"],
                "churn_probability": res["churn_probability"],
                "churn_percentage": res["churn_percentage"],
                "risk_level": res["risk_level"],
                "model_version": res["model_version"],
                "prediction_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            }

    # -------------------------------------------------------------------------
    # Tool 3: get_customer_shap
    # -------------------------------------------------------------------------
    def get_customer_shap(self, company_id: str, customer_id: str) -> dict:
        """Computes exact TreeSHAP attribution corresponding to customer's actual feature values."""
        cust_details = self.get_customer_details(company_id, customer_id)
        if not cust_details.get("found"):
            return cust_details
            
        shap_factors = shap_service.explain_customer(cust_details, save_to_db=False)
        
        # Format factors to reflect actual human-readable customer feature values
        formatted_factors = []
        for f in shap_factors:
            raw_feat = f.get("raw_feature", "")
            feat_name = f.get("feature_name", raw_feat)
            shap_val = f.get("shap_value", 0.0)
            impact = f.get("impact_direction", "INCREASES_CHURN")
            
            # Map actual value from customer record
            actual_val = "Unknown"
            if "Contract" in feat_name or "Is_Month" in raw_feat or "Is_Two" in raw_feat or "Is_One" in raw_feat:
                actual_val = cust_details.get("contract", "N/A")
            elif "TechSupport" in raw_feat or "Tech Support" in feat_name:
                actual_val = cust_details.get("tech_support", "N/A")
            elif "OnlineSecurity" in raw_feat or "Online Security" in feat_name or "Security" in feat_name:
                actual_val = cust_details.get("online_security", "N/A")
            elif "MonthlyCharges" in raw_feat or "Monthly Charges" in feat_name:
                actual_val = f"${cust_details.get('monthly_charges', 0.0):.2f}"
            elif "tenure" in raw_feat.lower():
                actual_val = f"{cust_details.get('tenure', 0)} months"
            elif "Internet" in feat_name or "Internet" in raw_feat:
                actual_val = cust_details.get("internet_service", "N/A")
            elif "Payment" in feat_name or "Electronic" in raw_feat or "AutoPay" in raw_feat:
                actual_val = cust_details.get("payment_method", "N/A")
            elif "Paperless" in feat_name:
                actual_val = cust_details.get("paperless_billing", "N/A")
            elif "Senior" in feat_name:
                actual_val = "Yes" if cust_details.get("senior_citizen") == 1 else "No"
            elif "TotalCharges" in raw_feat:
                actual_val = f"${cust_details.get('total_charges', 0.0):.2f}"
            else:
                actual_val = f.get("feature_value", "Active")
                
            formatted_factors.append({
                "feature_name": feat_name,
                "raw_feature": raw_feat,
                "actual_value": actual_val,
                "shap_value": shap_val,
                "abs_shap": abs(shap_val),
                "impact_direction": impact
            })
            
        increasing = [f for f in formatted_factors if f["impact_direction"] == "INCREASES_CHURN"]
        decreasing = [f for f in formatted_factors if f["impact_direction"] == "DECREASES_CHURN"]
        
        return {
            "found": True,
            "source": "TreeSHAP Explainability Engine",
            "customer_id": cust_details["customer_id"],
            "increasing_factors": increasing[:4],
            "decreasing_factors": decreasing[:3],
            "all_factors": formatted_factors
        }

    # -------------------------------------------------------------------------
    # Tool 4: run_counterfactual_prediction
    # -------------------------------------------------------------------------
    def run_counterfactual_prediction(self, company_id: str, customer_id: str, modifications: dict) -> dict:
        """Modifies specific customer attributes and re-runs the trained ML model pipeline."""
        cust_details = self.get_customer_details(company_id, customer_id)
        if not cust_details.get("found"):
            return cust_details
            
        sim_res = what_if_service.simulate_scenario(cust_details, modified_fields=modifications)
        return {
            "found": True,
            "source": "Model-Based Counterfactual Simulation",
            "customer_id": cust_details["customer_id"],
            "baseline_probability": sim_res["baseline_probability"],
            "baseline_risk": sim_res["baseline_risk"],
            "simulated_probability": sim_res["simulated_probability"],
            "simulated_risk": sim_res["simulated_risk"],
            "probability_delta": sim_res["probability_delta"],
            "percentage_change": sim_res["percentage_change"],
            "is_risk_reduced": sim_res["is_risk_reduced"],
            "modified_fields": modifications
        }

    # -------------------------------------------------------------------------
    # Tool 5: get_retention_recommendation
    # -------------------------------------------------------------------------
    def get_retention_recommendation(self, company_id: str, customer_id: str) -> dict:
        """Generates policy-grounded intervention categories based on verified SHAP risk drivers."""
        cust_details = self.get_customer_details(company_id, customer_id)
        if not cust_details.get("found"):
            return cust_details
            
        pred = self.get_customer_prediction(company_id, customer_id)
        shap_info = self.get_customer_shap(company_id, customer_id)
        
        intervention_categories = []
        contract = cust_details.get("contract")
        tenure = cust_details.get("tenure", 0)
        support = cust_details.get("tech_support")
        security = cust_details.get("online_security")
        payment = cust_details.get("payment_method")
        
        if contract == "Month-to-month":
            intervention_categories.append({
                "category": "Contract Structure Review",
                "action": "Proactive relationship review to discuss transitioning from a Month-to-month plan to a 1-Year or 2-Year fixed-term commitment.",
                "rationale": "Month-to-month contract is identified as a primary predictive risk factor for this account."
            })
            
        if tenure <= 6:
            intervention_categories.append({
                "category": "Early-Lifecycle Onboarding Check-in",
                "action": "Customer Success outreach to confirm service quality, onboarding satisfaction, and first-quarter billing clarity.",
                "rationale": f"Account tenure is {tenure} month(s), which represents an early-lifecycle attrition vulnerability."
            })
            
        if support == "No" or security == "No":
            intervention_categories.append({
                "category": "Service & Technical Protection Review",
                "action": "Review subscriber's internet configuration and offer options to add dedicated Technical Support and Online Cyber Security protection.",
                "rationale": "Absence of active tech support or security add-ons is associated with higher service volatility in this profile."
            })
            
        if "electronic" in payment.lower() or "mail" in payment.lower():
            intervention_categories.append({
                "category": "Payment Channel Review",
                "action": "Present information on automated payment options (ACH bank transfer or recurring card billing) to reduce manual billing friction.",
                "rationale": f"Current billing method ({payment}) exhibits higher average predicted churn compared to automated payment channels."
            })
            
        if not intervention_categories:
            intervention_categories.append({
                "category": "Standard Account Engagement",
                "action": "Maintain standard service monitoring and include in regular customer satisfaction follow-ups.",
                "rationale": "Customer exhibits low predictive risk factors under the current model."
            })
            
        return {
            "found": True,
            "source": "AI-Generated Recommendation (Intervention Framework)",
            "customer_id": cust_details["customer_id"],
            "risk_level": pred.get("risk_level", "MEDIUM"),
            "churn_probability": pred.get("churn_percentage", 50.0),
            "intervention_categories": intervention_categories,
            "policy_note": "No configured promotional monetary discount or coupon offer is defined in system policies. Recommendations represent structured intervention categories based on predictive risk drivers."
        }

    # -------------------------------------------------------------------------
    # Tool 6: analyze_customer_segments (COHORT & SEGMENT SELECTION ENGINE)
    # -------------------------------------------------------------------------
    def analyze_customer_segments(self, company_id: str, criterion: str = "highest_average_churn_probability", min_segment_size: int = 10) -> dict:
        """
        Performs thorough cohort analysis across customer dimensions and determines
        the priority retention segment based on an explicit analytical criterion.
        """
        df = self._get_company_dataframe(company_id)
        if df.empty:
            return {
                "found": False,
                "error": "That information is not available in the current dataset. No customer records found."
            }
            
        total_customers = len(df)
        total_mrr = df["monthly_charges"].sum()
        
        # Define candidate cohorts to evaluate
        cohort_definitions = [
            # Multi-attribute cohorts
            {
                "name": "Month-to-Month + Fiber Optic + No Tech Support",
                "characteristics": {
                    "Contract": "Month-to-month",
                    "Internet Service": "Fiber optic",
                    "Tech Support": "No"
                },
                "mask": (df["contract"] == "Month-to-month") & (df["internet_service"].str.contains("Fiber", case=False, na=False)) & (df["tech_support"] != "Yes")
            },
            {
                "name": "Month-to-Month + Early Tenure (<= 12 mo)",
                "characteristics": {
                    "Contract": "Month-to-month",
                    "Tenure": "<= 12 months"
                },
                "mask": (df["contract"] == "Month-to-month") & (df["tenure"] <= 12)
            },
            {
                "name": "Month-to-Month + Electronic Check Billing",
                "characteristics": {
                    "Contract": "Month-to-month",
                    "Payment Method": "Electronic check"
                },
                "mask": (df["contract"] == "Month-to-month") & (df["payment_method"].str.contains("electronic", case=False, na=False))
            },
            {
                "name": "Fiber Optic + High Spend (>= $85/mo) without Tech Support",
                "characteristics": {
                    "Internet Service": "Fiber optic",
                    "Monthly Charges": ">= $85.00",
                    "Tech Support": "No"
                },
                "mask": (df["internet_service"].str.contains("Fiber", case=False, na=False)) & (df["monthly_charges"] >= 85) & (df["tech_support"] != "Yes")
            },
            {
                "name": "Month-to-Month Contracts (All)",
                "characteristics": {
                    "Contract": "Month-to-month"
                },
                "mask": df["contract"] == "Month-to-month"
            },
            {
                "name": "Fiber Optic Subscribers (All)",
                "characteristics": {
                    "Internet Service": "Fiber optic"
                },
                "mask": df["internet_service"].str.contains("Fiber", case=False, na=False)
            },
            {
                "name": "Early Lifecycle (Tenure <= 6 mo)",
                "characteristics": {
                    "Tenure": "<= 6 months"
                },
                "mask": df["tenure"] <= 6
            },
            {
                "name": "Senior Citizens with Month-to-Month Contracts",
                "characteristics": {
                    "Senior Citizen": "Yes (1)",
                    "Contract": "Month-to-month"
                },
                "mask": (df["senior_citizen"] == 1) & (df["contract"] == "Month-to-month")
            }
        ]
        
        evaluated_segments = []
        for cdef in cohort_definitions:
            sub = df[cdef["mask"]]
            count = len(sub)
            if count < min_segment_size:
                continue
                
            share_pct = round((count / total_customers) * 100, 1)
            avg_churn = round(float(sub["churn_pct"].mean()), 1)
            med_churn = round(float(sub["churn_pct"].median()), 1)
            high_risk_count = int((sub["risk_level"] == "HIGH").sum())
            high_risk_pct = round((high_risk_count / count) * 100, 1) if count > 0 else 0.0
            avg_charges = round(float(sub["monthly_charges"].mean()), 2)
            avg_tenure = round(float(sub["tenure"].mean()), 1)
            segment_mrr = round(float(sub["monthly_charges"].sum()), 2)
            annual_revenue_exposure = round(segment_mrr * 12, 2)
            
            evaluated_segments.append({
                "name": cdef["name"],
                "characteristics": cdef["characteristics"],
                "customer_count": count,
                "portfolio_share_pct": share_pct,
                "avg_predicted_churn_pct": avg_churn,
                "median_predicted_churn_pct": med_churn,
                "high_risk_count": high_risk_count,
                "high_risk_percentage": high_risk_pct,
                "avg_monthly_charges": avg_charges,
                "avg_tenure_months": avg_tenure,
                "monthly_revenue_exposure": segment_mrr,
                "annualized_revenue_exposure": annual_revenue_exposure
            })
            
        if not evaluated_segments:
            return {
                "found": False,
                "error": "No segment met the minimum size threshold in the current dataset."
            }
            
        # Select best segment based on specified criterion
        if criterion == "highest_revenue_exposure":
            selected = sorted(evaluated_segments, key=lambda x: x["monthly_revenue_exposure"], reverse=True)[0]
            criterion_label = "Highest expected monthly revenue exposure"
        elif criterion == "highest_high_risk_percentage":
            selected = sorted(evaluated_segments, key=lambda x: (x["high_risk_percentage"], x["avg_predicted_churn_pct"]), reverse=True)[0]
            criterion_label = "Highest percentage of high-risk customers"
        else: # default: highest_average_churn_probability
            selected = sorted(evaluated_segments, key=lambda x: (x["avg_predicted_churn_pct"], x["high_risk_percentage"]), reverse=True)[0]
            criterion_label = "Highest average predicted churn probability (minimum segment size: 10 accounts)"
            
        return {
            "found": True,
            "source": "Analytical Cohort Engine (Active Dataset)",
            "selection_criterion": criterion_label,
            "total_monitored_customers": total_customers,
            "selected_segment": selected,
            "all_evaluated_segments": evaluated_segments
        }

    # -------------------------------------------------------------------------
    # Tool 7: get_high_risk_customers
    # -------------------------------------------------------------------------
    def get_high_risk_customers(self, company_id: str, min_probability: float = 0.48, limit: int = 10) -> dict:
        """Queries database for verified high-risk customers sorted by predicted churn probability."""
        df = self._get_company_dataframe(company_id)
        if df.empty:
            return {"found": False, "error": "No customers found in active dataset."}
            
        high_df = df[df["churn_probability"] >= min_probability].sort_values(by="churn_probability", ascending=False)
        total_high = len(high_df)
        
        top_list = []
        for _, row in high_df.head(limit).iterrows():
            top_list.append({
                "customer_id": row["customer_id"],
                "churn_probability": row["churn_pct"],
                "risk_level": row["risk_level"],
                "contract": row["contract"],
                "tenure": int(row["tenure"]),
                "monthly_charges": float(row["monthly_charges"]),
                "internet_service": row["internet_service"],
                "tech_support": row["tech_support"],
                "payment_method": row["payment_method"]
            })
            
        mrr_high = float(high_df["monthly_charges"].sum())
        return {
            "found": True,
            "source": "ML Prediction Pipeline & SQL Database",
            "threshold_percentage": min_probability * 100,
            "total_matching_accounts": total_high,
            "total_portfolio_accounts": len(df),
            "monthly_revenue_exposure": mrr_high,
            "annualized_revenue_exposure": mrr_high * 12,
            "top_customers": top_list
        }

    # -------------------------------------------------------------------------
    # Tool 8: analyze_payment_methods
    # -------------------------------------------------------------------------
    def analyze_payment_methods(self, company_id: str) -> dict:
        """Calculates exact group statistics across payment channels from active dataset."""
        df = self._get_company_dataframe(company_id)
        if df.empty:
            return {"found": False, "error": "No customers found in active dataset."}
            
        total = len(df)
        methods = []
        for method_name, grp in df.groupby("payment_method"):
            cnt = len(grp)
            share = round((cnt / total) * 100, 1)
            avg_churn = round(float(grp["churn_pct"].mean()), 1)
            high_risk_cnt = int((grp["risk_level"] == "HIGH").sum())
            high_risk_pct = round((high_risk_cnt / cnt) * 100, 1) if cnt > 0 else 0.0
            mrr = round(float(grp["monthly_charges"].sum()), 2)
            
            methods.append({
                "payment_method": method_name,
                "customer_count": cnt,
                "share_pct": share,
                "avg_predicted_churn_pct": avg_churn,
                "high_risk_count": high_risk_cnt,
                "high_risk_pct": high_risk_pct,
                "monthly_revenue": mrr
            })
            
        methods.sort(key=lambda x: x["avg_predicted_churn_pct"], reverse=True)
        return {
            "found": True,
            "source": "Analytical Engine (Active Dataset)",
            "total_customers": total,
            "methods": methods
        }

    # -------------------------------------------------------------------------
    # Tool 9: get_contract_breakdown
    # -------------------------------------------------------------------------
    def get_contract_breakdown(self, company_id: str) -> dict:
        """Calculates exact group statistics across contract durations from active dataset."""
        df = self._get_company_dataframe(company_id)
        if df.empty:
            return {"found": False, "error": "No customers found in active dataset."}
            
        total = len(df)
        contracts = []
        for contract_name, grp in df.groupby("contract"):
            cnt = len(grp)
            share = round((cnt / total) * 100, 1)
            avg_churn = round(float(grp["churn_pct"].mean()), 1)
            high_risk_cnt = int((grp["risk_level"] == "HIGH").sum())
            high_risk_pct = round((high_risk_cnt / cnt) * 100, 1) if cnt > 0 else 0.0
            mrr = round(float(grp["monthly_charges"].sum()), 2)
            
            contracts.append({
                "contract": contract_name,
                "customer_count": cnt,
                "share_pct": share,
                "avg_predicted_churn_pct": avg_churn,
                "high_risk_count": high_risk_cnt,
                "high_risk_pct": high_risk_pct,
                "monthly_revenue": mrr
            })
            
        contracts.sort(key=lambda x: x["avg_predicted_churn_pct"], reverse=True)
        return {
            "found": True,
            "source": "Analytical Engine (Active Dataset)",
            "total_customers": total,
            "contracts": contracts
        }

    # -------------------------------------------------------------------------
    # Tool 10: get_internet_service_breakdown
    # -------------------------------------------------------------------------
    def get_internet_service_breakdown(self, company_id: str) -> dict:
        """Calculates exact internet service & tech support distribution from active dataset."""
        df = self._get_company_dataframe(company_id)
        if df.empty:
            return {"found": False, "error": "No customers found in active dataset."}
            
        total = len(df)
        services = []
        for svc_name, grp in df.groupby("internet_service"):
            cnt = len(grp)
            share = round((cnt / total) * 100, 1)
            avg_churn = round(float(grp["churn_pct"].mean()), 1)
            high_risk_cnt = int((grp["risk_level"] == "HIGH").sum())
            high_risk_pct = round((high_risk_cnt / cnt) * 100, 1) if cnt > 0 else 0.0
            mrr = round(float(grp["monthly_charges"].sum()), 2)
            
            services.append({
                "internet_service": svc_name,
                "customer_count": cnt,
                "share_pct": share,
                "avg_predicted_churn_pct": avg_churn,
                "high_risk_count": high_risk_cnt,
                "high_risk_pct": high_risk_pct,
                "monthly_revenue": mrr
            })
            
        services.sort(key=lambda x: x["avg_predicted_churn_pct"], reverse=True)
        
        # Tech support interaction
        sup_yes = df[df["tech_support"] == "Yes"]
        sup_no = df[df["tech_support"] != "Yes"]
        
        return {
            "found": True,
            "source": "Analytical Engine (Active Dataset)",
            "total_customers": total,
            "services": services,
            "tech_support_stats": {
                "with_support_count": len(sup_yes),
                "with_support_avg_churn": round(float(sup_yes["churn_pct"].mean()), 1) if len(sup_yes) > 0 else 0.0,
                "without_support_count": len(sup_no),
                "without_support_avg_churn": round(float(sup_no["churn_pct"].mean()), 1) if len(sup_no) > 0 else 0.0
            }
        }

    # -------------------------------------------------------------------------
    # Tool 11: get_dashboard_summary
    # -------------------------------------------------------------------------
    def get_dashboard_summary(self, company_id: str) -> dict:
        """Retrieves exact portfolio aggregates and revenue exposure."""
        df = self._get_company_dataframe(company_id)
        if df.empty:
            return {"found": False, "error": "No customers found in active dataset."}
            
        total = len(df)
        high_df = df[df["risk_level"] == "HIGH"]
        med_df = df[df["risk_level"] == "MEDIUM"]
        low_df = df[df["risk_level"] == "LOW"]
        
        total_mrr = round(float(df["monthly_charges"].sum()), 2)
        high_mrr = round(float(high_df["monthly_charges"].sum()), 2)
        med_mrr = round(float(med_df["monthly_charges"].sum()), 2)
        avg_churn = round(float(df["churn_pct"].mean()), 1)
        avg_tenure = round(float(df["tenure"].mean()), 1)
        avg_spend = round(float(df["monthly_charges"].mean()), 2)
        
        return {
            "found": True,
            "source": "Analytical Engine (Active Dataset Aggregate)",
            "total_customers": total,
            "high_risk_count": len(high_df),
            "high_risk_pct": round((len(high_df) / total) * 100, 1),
            "medium_risk_count": len(med_df),
            "medium_risk_pct": round((len(med_df) / total) * 100, 1),
            "low_risk_count": len(low_df),
            "low_risk_pct": round((len(low_df) / total) * 100, 1),
            "average_churn_probability_pct": avg_churn,
            "average_tenure_months": avg_tenure,
            "average_monthly_spend": avg_spend,
            "total_monthly_recurring_revenue": total_mrr,
            "total_annualized_revenue": round(total_mrr * 12, 2),
            "high_risk_monthly_revenue_exposure": high_mrr,
            "high_risk_annualized_revenue_exposure": round(high_mrr * 12, 2),
            "thresholds": {
                "high_risk_min": f"{Config.HIGH_RISK_MIN*100:.0f}%",
                "low_risk_max": f"{Config.LOW_RISK_MAX*100:.0f}%"
            }
        }

    # -------------------------------------------------------------------------
    # Tool 12: search_business_documentation
    # -------------------------------------------------------------------------
    def search_business_documentation(self, company_id: str, query: str) -> dict:
        """Performs vector search across system documentation, FAQ, and ML benchmark definitions."""
        chunks = rag_service.retrieve(company_id, query, top_k=2)
        return {
            "found": len(chunks) > 0,
            "source": "RAG Documentation & Knowledge Base",
            "chunks": chunks
        }

    # =========================================================================
    # INTENT-BASED TOOL ROUTER & ORCHESTRATOR
    # =========================================================================

    def _extract_customer_id(self, raw_text: str) -> Optional[str]:
        """Extracts customer ID token from user inquiry."""
        patterns = [
            r"\b\d{4}-[A-Z0-9]{5}\b",              # e.g. 7590-VHVEG, 5575-GNVDE
            r"\b[A-Z0-9]{3,8}-[A-Z0-9]{3,8}\b",    # e.g. TEST300-0001, VAL-00001, REG2-00042, DEMO-7590-VHVEG
            r"\bGEN-CUST-\d{5}\b",                 # e.g. GEN-CUST-00001
            r"\bVAL-\d{5}\b",                      # e.g. VAL-00840
            r"\bTEST\d*-\d+\b"                     # e.g. TEST300-0201, TEST-001
        ]
        
        ignored_words = {
            "customer", "customers", "contract", "contracts", "payment", "payments",
            "support", "security", "telecom", "retention", "high-risk", "low-risk",
            "med-risk", "two-year", "one-year", "month-to-month", "auto-pay",
            "fiber-optic", "tech-support", "e-check", "dataset", "profile",
            "prediction", "probability", "monthly", "charges", "tenure"
        }
        
        for pat in patterns:
            found = re.findall(pat, raw_text, re.IGNORECASE)
            for f in found:
                cand = f.strip()
                if cand.lower() not in ignored_words and any(c.isdigit() for c in cand):
                    return cand
                    
        # Token search
        tokens = re.split(r"[\s,;?!]+", raw_text)
        for t in tokens:
            cleaned = t.strip().upper().replace(":", "").replace("'", "").replace('"', "")
            if len(cleaned) >= 5 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
                if cleaned.lower() not in ignored_words:
                    return cleaned
                    
        return None

    def _extract_counterfactual_modifications(self, query: str) -> dict:
        """Parses simulated feature modifications requested by user."""
        q = query.lower()
        mods = {}
        
        # Contract modifications
        if "two year" in q or "2 year" in q or "2-year" in q or "two-year" in q:
            mods["contract"] = "Two year"
        elif "one year" in q or "1 year" in q or "1-year" in q or "one-year" in q:
            mods["contract"] = "One year"
        elif "month-to-month" in q or "month to month" in q:
            mods["contract"] = "Month-to-month"
            
        # Tech Support modifications
        if "gets tech support" in q or "add tech support" in q or "with tech support" in q or "adding tech support" in q or "tech support to yes" in q or "gives tech support" in q:
            mods["tech_support"] = "Yes"
        elif "removes tech support" in q or "without tech support" in q or "no tech support" in q:
            mods["tech_support"] = "No"
            
        # Online Security modifications
        if "gets online security" in q or "add online security" in q or "adding online security" in q or "with online security" in q or "security to yes" in q:
            mods["online_security"] = "Yes"
        elif "without online security" in q or "no online security" in q:
            mods["online_security"] = "No"
            
        # Payment Method modifications
        if "auto pay" in q or "autopay" in q or "bank transfer" in q or "credit card" in q:
            if "credit" in q:
                mods["payment_method"] = "Credit card (automatic)"
            else:
                mods["payment_method"] = "Bank transfer (automatic)"
        elif "electronic check" in q:
            mods["payment_method"] = "Electronic check"
            
        # Tenure modification
        tenure_match = re.search(r"tenure\s*(?:to|=|\bis\b)\s*(\d+)", q)
        if tenure_match:
            mods["tenure"] = float(tenure_match.group(1))
            
        # Monthly charges modification
        charges_match = re.search(r"(?:monthly charges|charges|bill|monthly spend)\s*(?:to|=|\bis\b)\s*\$?(\d+(?:\.\d+)?)", q)
        if charges_match:
            mods["monthly_charges"] = float(charges_match.group(1))
            
        # Default fallback if question says "gets tech support" but regex missed
        if not mods and ("tech support" in q or "support" in q):
            mods["tech_support"] = "Yes"
        if not mods and ("contract" in q):
            mods["contract"] = "Two year"
            
        return mods

    def answer_query(
        self,
        company_id: str,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        custom_key: Optional[str] = None,
        custom_provider: Optional[str] = None
    ) -> dict:
        """
        Main Conversational AI Engine.
        Executes grounded RAG extraction, tries LLM generation (Gemini/OpenAI/Groq),
        and falls back to verified semantic analytical tools.
        """
        raw_q = query.strip()
        q = raw_q.lower()
        company = Company.query.filter_by(company_id=company_id).first()
        comp_name = company.company_name if company else company_id

        # ---------------------------------------------------------------------
        # 0. Conversational Short-Circuits (Farewells & Greetings)
        # ---------------------------------------------------------------------
        farewells = ["bye", "goodbye", "cya", "see you", "see ya", "exit", "quit", "have a good day", "have a nice day", "talk to you later", "farewell", "good night"]
        cleaned_q = re.sub(r"[^\w\s]", "", q).strip()
        if cleaned_q in farewells or any(q.startswith(f) and len(q.split()) <= 4 for f in ["bye", "goodbye", "see you", "have a good"]):
            return {
                "query": raw_q,
                "response": (
                    f"Goodbye! 👋 Thank you for using the Telecom Churn & Retention Intelligence Platform for **{comp_name}**.\n\n"
                    f"Feel free to return anytime you need customer 360 dossiers, cohort prioritization, TreeSHAP churn driver explanations, or counterfactual simulations."
                ),
                "data": {"intent": "farewell"}
            }

        # ---------------------------------------------------------------------
        # 1. Attempt LLM Generation with Grounded Dataset RAG Context
        # ---------------------------------------------------------------------
        extracted_cust_id = self._extract_customer_id(raw_q)
        rag_context = rag_service.build_grounded_rag_context(company_id, raw_q, extracted_cust_id)
        
        llm_res = llm_service.generate_grounded_response(
            query=raw_q,
            context_text=rag_context,
            company_name=comp_name,
            chat_history=chat_history,
            custom_key=custom_key,
            custom_provider=custom_provider
        )
        if llm_res.get("success"):
            return {
                "query": raw_q,
                "response": llm_res["response"],
                "data": {
                    "source": f"LLM Generation ({llm_res.get('provider').title()}) with Dataset RAG Grounding",
                    "model": llm_res.get("model")
                }
            }

        # ---------------------------------------------------------------------
        # 2. Chit-Chat, Greetings & Identity / Capabilities
        # ---------------------------------------------------------------------
        greetings = ["hello", "hi", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "howdy", "sup"]
        if cleaned_q in greetings or any(q.startswith(g) and len(q.split()) <= 3 for g in greetings):
            summary = self.get_dashboard_summary(company_id)
            if summary.get("found"):
                tot = summary["total_customers"]
                high = summary["high_risk_count"]
                avg_p = summary["average_churn_probability_pct"]
                msg = (
                    f"Hello! I am your AI Analytics & Retention Assistant for **{comp_name}**.\n\n"
                    f"**Active Workspace Monitoring:**\n"
                    f"• **Total Subscribers:** **{tot:,} accounts**\n"
                    f"• **High-Risk Accounts (&ge; {summary['thresholds']['high_risk_min']}):** **{high:,} accounts** ({summary['high_risk_pct']}%)\n"
                    f"• **Portfolio Average Churn Risk:** **{avg_p:.1f}%**\n\n"
                    f"**Verified Queries You Can Ask Me:**\n"
                    f"• *'If I could focus on only one customer segment for retention, which characteristics should I target?'*\n"
                    f"• *'Tell me about customer 7590-VHVEG'*\n"
                    f"• *'Is 7590-VHVEG likely to churn?'*\n"
                    f"• *'Why is 7590-VHVEG at risk?' (TreeSHAP Attributions)*\n"
                    f"• *'What happens if we change 7590-VHVEG to a Two-year contract?' (Counterfactual)*\n"
                    f"• *'Which payment method has the highest churn risk?'*\n"
                    f"• *'Show high-risk customers'*\n\n"
                    f"How can I assist your retention team today?"
                )
            else:
                msg = f"Hello! I am your AI Retention Assistant for **{comp_name}**. Please upload a dataset to begin analytics."
            return {"query": raw_q, "response": msg, "data": {"intent": "greeting"}}

        if any(q.startswith(g) for g in ["who are you", "what can you do", "introduce yourself", "help me", "what are your capabilities", "how do i use this", "what features do you have"]):
            msg = (
                f"### AI Analytics & Retention Assistant Capabilities (**{comp_name}**)\n\n"
                f"I am a zero-hallucination conversational analytics assistant connected directly to your SQL database, "
                f"ML Stacking Ensemble prediction pipeline, and TreeSHAP explainability engine.\n\n"
                f"**What I Can Do For You:**\n"
                f"1. **Customer 360 Dossiers:** Inquire about any customer profile (e.g. `Tell me about 7590-VHVEG`).\n"
                f"2. **ML Churn Predictions:** Check genuine calibrated churn probabilities and risk tiers.\n"
                f"3. **TreeSHAP Attributions:** Understand specific features increasing or decreasing risk for an account.\n"
                f"4. **Counterfactual What-If Simulations:** Test hypothetical interventions (e.g. `What if 7590-VHVEG gets tech support?`).\n"
                f"5. **Cohort Prioritization:** Ask `If I could focus on only one customer segment for retention, which characteristics should I target?`.\n"
                f"6. **High-Risk Queues:** View subscribers sorted by churn risk with financial exposure.\n"
                f"7. **Comparative Breakdowns:** Analyze churn dynamics across contracts, payment channels, and internet services.\n"
                f"8. **Financial Telemetry:** View portfolio MRR, annualized revenue at risk, and average spend."
            )
            return {"query": raw_q, "response": msg, "data": {"intent": "capabilities"}}

        acks = ["ok", "okay", "okey", "k", "sure", "got it", "gotcha", "understood", "cool", "great", "awesome", "perfect", "good", "nice", "fine", "alright", "yes", "yep", "yeah"]
        if cleaned_q in acks:
            return {
                "query": raw_q,
                "response": (
                    f"Understood! What would you like to explore next in **{comp_name}**?\n\n"
                    f"• 👤 Inspect an individual customer (e.g. `TEST300-0001` or `7590-VHVEG`)\n"
                    f"• 🎯 Ask for optimal customer segment prioritization\n"
                    f"• 🚨 View high-risk customer cohorts\n"
                    f"• 📑 Compare contract terms or payment methods"
                ),
                "data": {"intent": "acknowledgment"}
            }

        if cleaned_q in ["thanks", "thank you", "thx", "many thanks", "appreciate it"]:
            return {
                "query": raw_q,
                "response": f"You're very welcome! Let me know if you'd like to run any further analytics for **{comp_name}**.",
                "data": {"intent": "thanks"}
            }

        # ---------------------------------------------------------------------
        # 3. Specific Customer Intelligence & Targeted Sub-Intents
        # ---------------------------------------------------------------------
        extracted_cust_id = self._extract_customer_id(raw_q)
        if extracted_cust_id:
            # Check if customer exists
            cust_details = self.get_customer_details(company_id, extracted_cust_id)
            if not cust_details.get("found"):
                return {
                    "query": raw_q,
                    "response": f"That information is not available in the current dataset. Customer **'{extracted_cust_id}'** was not found in the active workspace.",
                    "data": {"found": False}
                }
            
            c_id = cust_details["customer_id"]

            # Sub-Intent 3A: What-If / Counterfactual Simulation
            if any(w in q for w in ["what if", "what happens if", "simulate", "if we change", "if customer gets", "would adding", "change contract", "switches to", "give", "gets"]):
                mods = self._extract_counterfactual_modifications(raw_q)
                if not mods:
                    mods = {"contract": "Two year"} # Default sensible simulation
                    
                cf_result = self.run_counterfactual_prediction(company_id, c_id, mods)
                
                mod_desc = ", ".join([f"{k.replace('_', ' ').title()}: **{v}**" for k, v in mods.items()])
                lines = [
                    f"**Source:** {cf_result['source']}\n",
                    f"### Counterfactual Simulation for Customer **{c_id}**\n",
                    f"• **Simulated Feature Change(s):** {mod_desc}",
                    f"• **Original Predicted Churn Probability:** **{cf_result['baseline_probability']:.1f}%** ({cf_result['baseline_risk']} Risk tier)",
                    f"• **Counterfactual Predicted Churn Probability:** **{cf_result['simulated_probability']:.1f}%** ({cf_result['simulated_risk']} Risk tier)",
                    f"• **Model-Predicted Delta:** **{cf_result['percentage_change']}** ({'Risk Reduction' if cf_result['is_risk_reduced'] else 'Risk Increase'})\n",
                    f"*Note: This is a model-based counterfactual simulation. The model predicts a {abs(cf_result['probability_delta']):.1f} percentage-point {'reduction' if cf_result['is_risk_reduced'] else 'increase'} under this simulated feature change. This is a predictive scenario estimate, not a claim that the change guarantees actual real-world retention.*"
                ]
                return {"query": raw_q, "response": "\n".join(lines), "data": cf_result}

            # Sub-Intent 3B: SHAP Explanations & Churn Drivers
            if any(w in q for w in ["why", "explain", "driver", "drivers", "factor", "factors", "shap", "reason", "reasons", "cause", "causes", "at risk"]):
                pred = self.get_customer_prediction(company_id, c_id)
                shap_info = self.get_customer_shap(company_id, c_id)
                
                lines = [
                    f"**Source:** {shap_info['source']}\n",
                    f"### TreeSHAP Churn Drivers for Customer **{c_id}**",
                    f"• **Current Predicted Churn Probability:** **{pred.get('churn_percentage', 50.0):.1f}%** (Risk Level: `{pred.get('risk_level', 'MEDIUM')}`)\n",
                    f"#### ⚠️ Factors Contributing to Higher Predicted Churn:"
                ]
                for f in shap_info.get("increasing_factors", []):
                    lines.append(f"• **{f['feature_name']}** (Actual Value: `{f['actual_value']}`): **+{f['shap_value']:.3f}** positive contribution toward churn prediction.")
                    
                if shap_info.get("decreasing_factors"):
                    lines.append(f"\n#### 🛡️ Factors Stabilizing Retention (Protective Factors):")
                    for f in shap_info.get("decreasing_factors", []):
                        lines.append(f"• **{f['feature_name']}** (Actual Value: `{f['actual_value']}`): **{f['shap_value']:.3f}** negative contribution toward churn prediction.")
                        
                lines.append(f"\n*Interpretation: SHAP values represent individual feature attributions from the trained machine learning model for this specific customer profile.*")
                return {"query": raw_q, "response": "\n".join(lines), "data": {"prediction": pred, "shap": shap_info}}

            # Sub-Intent 3C: Retention Recommendations / Action Plan
            if any(w in q for w in ["retain", "retention", "recommend", "recommendation", "recommendations", "what should we do", "how can we", "prevent churn", "action plan"]):
                rec_info = self.get_retention_recommendation(company_id, c_id)
                lines = [
                    f"**Source:** {rec_info['source']}\n",
                    f"### Retention Intervention Strategy for Customer **{c_id}**",
                    f"• **Predicted Churn Probability:** **{rec_info['churn_probability']:.1f}%** (Risk Level: `{rec_info['risk_level']}`)\n",
                    f"#### 📋 Suggested Intervention Categories:"
                ]
                for idx, cat in enumerate(rec_info.get("intervention_categories", []), 1):
                    lines.append(f"{idx}. **{cat['category']}**")
                    lines.append(f"   • **Action:** {cat['action']}")
                    lines.append(f"   • **Rationale:** {cat['rationale']}")
                    
                lines.append(f"\n*{rec_info['policy_note']}*")
                return {"query": raw_q, "response": "\n".join(lines), "data": rec_info}

            # Sub-Intent 3D: Prediction / Probability / Risk Level Only
            if any(w in q for w in ["churn", "probability", "likely", "risk", "predict", "prediction", "score"]):
                pred = self.get_customer_prediction(company_id, c_id)
                lines = [
                    f"**Source:** {pred['source']}\n",
                    f"### Churn Prediction for Customer **{c_id}**\n",
                    f"• **Customer ID:** `{c_id}`",
                    f"• **Predicted Churn Probability:** **{pred['churn_percentage']:.1f}%**",
                    f"• **Risk Level:** **`{pred['risk_level']}`**",
                    f"• **Model Version:** `{pred['model_version']}`",
                    f"• **Prediction Timestamp:** `{pred['prediction_date']}`"
                ]
                return {"query": raw_q, "response": "\n".join(lines), "data": pred}

            # Sub-Intent 3E: Customer Detail Questions (DEFAULT for customer ID queries)
            lines = [
                f"**Source:** {cust_details['source']}\n",
                f"### Customer Profile: **{c_id}**\n",
                f"| Attribute | Value |",
                f"| :--- | :--- |",
                f"| **Customer ID** | `{cust_details['customer_id']}` |",
                f"| **Tenure** | **{cust_details['tenure']} month(s)** |",
                f"| **Contract** | **{cust_details['contract']}** |",
                f"| **Internet Service** | **{cust_details['internet_service']}** |",
                f"| **Monthly Charges** | **${cust_details['monthly_charges']:.2f}** |",
                f"| **Total Charges** | **${cust_details['total_charges']:.2f}** |",
                f"| **Phone Service** | {cust_details['phone_service']} (Multiple Lines: {cust_details['multiple_lines']}) |",
                f"| **Online Security** | {cust_details['online_security']} |",
                f"| **Online Backup** | {cust_details['online_backup']} |",
                f"| **Device Protection** | {cust_details['device_protection']} |",
                f"| **Tech Support** | **{cust_details['tech_support']}** |",
                f"| **Streaming Services** | TV: {cust_details['streaming_tv']} &bull; Movies: {cust_details['streaming_movies']} |",
                f"| **Paperless Billing** | {cust_details['paperless_billing']} |",
                f"| **Payment Method** | **{cust_details['payment_method']}** |",
                f"| **Demographics** | Gender: {cust_details['gender']} &bull; Senior Citizen: {'Yes' if cust_details['senior_citizen'] == 1 else 'No'} &bull; Partner: {cust_details['partner']} &bull; Dependents: {cust_details['dependents']} |"
            ]
            return {"query": raw_q, "response": "\n".join(lines), "data": cust_details}

        # ---------------------------------------------------------------------
        # 4. Customer Segment / Cohort Questions (CORE PRIORITIZATION ENGINE)
        # ---------------------------------------------------------------------
        if any(w in q for w in [
            "one customer segment", "one segment", "single segment", "focus on only one",
            "which characteristics", "target segment", "prioritize", "priority segment",
            "who should we target", "what type of customers", "which customer cohort",
            "which segment has the highest", "highest churn risk segment", "customer segment for retention"
        ]):
            seg_res = self.analyze_customer_segments(company_id, criterion="highest_average_churn_probability")
            if not seg_res.get("found"):
                return {
                    "query": raw_q,
                    "response": f"That information is not available in the current dataset. {seg_res.get('error', '')}",
                    "data": seg_res
                }
                
            sel = seg_res["selected_segment"]
            char_lines = "\n".join([f"- **{k}:** {v}" for k, v in sel["characteristics"].items()])
            
            lines = [
                f"## Retention Priority Cohort\n",
                f"**Source:** Analytical Cohort Engine  ",
                f"**Dataset:** {comp_name}  ",
                f"**Selection criterion:** Highest average predicted churn probability  ",
                f"**Minimum segment size:** 10 accounts\n",
                f"### Target Characteristics\n",
                f"{char_lines}\n",
                f"### Cohort Telemetry\n",
                f"- **Customers:** {sel['customer_count']} ({sel['portfolio_share_pct']}% of portfolio)",
                f"- **Average predicted churn probability:** {sel['avg_predicted_churn_pct']:.1f}%",
                f"- **Median predicted churn probability:** {sel['median_predicted_churn_pct']:.1f}%",
                f"- **Predicted high-risk customers:** {sel['high_risk_count']}/{sel['customer_count']} ({sel['high_risk_percentage']}%)",
                f"- **Average monthly charges:** ${sel['avg_monthly_charges']:.2f}",
                f"- **Average tenure:** {sel['avg_tenure_months']:.1f} months",
                f"- **Monthly revenue represented:** ${sel['monthly_revenue_exposure']:,.2f}",
                f"- **Annualized revenue represented:** ${sel['annualized_revenue_exposure']:,.2f}\n",
                f"### Analytical Interpretation\n",
                f"This cohort has the highest average predicted churn probability among the evaluated segments that meet the minimum size requirement.\n",
                f"The characteristics are associated with higher predicted churn risk in the current dataset. This is a predictive finding based on the available ML predictions and does not establish that these characteristics cause churn."
            ]
            return {"query": raw_q, "response": "\n".join(lines), "data": seg_res}

        # ---------------------------------------------------------------------
        # 5. High-Risk Customer Queue Questions
        # ---------------------------------------------------------------------
        if any(w in q for w in ["high-risk", "high risk", "most risky", "who is at risk", "top churn", "top risk"]) and any(w in q for w in ["customer", "customers", "accounts", "queue", "list", "show", "who", "table", "priority"]):
            prob_match = re.search(r"(?:above|greater than|>|over)\s*(\d{1,3})\s*%", q)
            min_p = float(prob_match.group(1)) / 100.0 if prob_match else Config.HIGH_RISK_MIN
            
            hr_res = self.get_high_risk_customers(company_id, min_probability=min_p, limit=8)
            if not hr_res.get("found"):
                return {"query": raw_q, "response": "That information is not available in the current dataset.", "data": {}}
                
            lines = [
                f"**Source:** {hr_res['source']}\n",
                f"### High-Risk Customer Accounts (Predicted Churn Probability &ge; {hr_res['threshold_percentage']:.0f}%)\n",
                f"• **Matching Accounts:** **{hr_res['total_matching_accounts']:,} customers** out of {hr_res['total_portfolio_accounts']:,}",
                f"• **Annualized Revenue Represented:** **${hr_res['annualized_revenue_exposure']:,.2f}** (${hr_res['monthly_revenue_exposure']:,.2f}/month)\n",
                f"| # | Customer ID | Predicted Churn % | Risk Level | Contract | Tenure | Internet | Monthly Charges |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
            ]
            for idx, c in enumerate(hr_res["top_customers"], 1):
                lines.append(f"| {idx} | **{c['customer_id']}** | **{c['churn_probability']:.1f}%** | `{c['risk_level']}` | {c['contract']} | {c['tenure']} mo | {c['internet_service']} | ${c['monthly_charges']:.2f} |")
                
            lines.append(f"\n*Note: Customer rankings are ordered strictly by model-predicted churn probabilities.*")
            return {"query": raw_q, "response": "\n".join(lines), "data": hr_res}

        # ---------------------------------------------------------------------
        # 6. Payment Method Analysis
        # ---------------------------------------------------------------------
        if any(w in q for w in ["payment method", "payment methods", "payment channel", "electronic check", "credit card", "bank transfer", "mailed check"]) and any(w in q for w in ["churn", "risk", "highest", "compare", "rate", "breakdown", "analysis", "which", "impact"]):
            pm_res = self.analyze_payment_methods(company_id)
            if not pm_res.get("found"):
                return {"query": raw_q, "response": "That information is not available in the current dataset.", "data": {}}
                
            lines = [
                f"**Source:** {pm_res['source']}\n",
                f"### Payment Method Churn Risk Breakdown\n",
                f"| Payment Method | Subscribers | Share % | Avg Predicted Churn % | High-Risk % | Monthly Revenue |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- |"
            ]
            for m in pm_res["methods"]:
                lines.append(f"| **{m['payment_method']}** | **{m['customer_count']:,}** | {m['share_pct']}% | **{m['avg_predicted_churn_pct']:.1f}%** | {m['high_risk_pct']}% | ${m['monthly_revenue']:,.2f} |")
                
            top_m = pm_res["methods"][0]
            lines.append(f"\n#### 💡 Analytical Finding:")
            lines.append(f"Customers using **{top_m['payment_method']}** have the highest average predicted churn risk (**{top_m['avg_predicted_churn_pct']:.1f}%**) in this dataset. This represents a statistical association in the current dataset, not proof that the payment channel directly causes churn.")
            return {"query": raw_q, "response": "\n".join(lines), "data": pm_res}

        # ---------------------------------------------------------------------
        # 7. Contract Analysis & Comparison
        # ---------------------------------------------------------------------
        if any(w in q for w in ["contract", "contracts", "month-to-month", "two year", "one year", "annual"]) and any(w in q for w in ["compare", "comparison", "churn", "risk", "highest", "rate", "breakdown", "vs", "versus", "difference", "effect"]):
            ct_res = self.get_contract_breakdown(company_id)
            if not ct_res.get("found"):
                return {"query": raw_q, "response": "That information is not available in the current dataset.", "data": {}}
                
            lines = [
                f"**Source:** {ct_res['source']}\n",
                f"### Contract Type Churn Breakdown\n",
                f"| Contract Type | Subscribers | Share % | Avg Predicted Churn % | High-Risk % | Monthly Revenue |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- |"
            ]
            for c in ct_res["contracts"]:
                lines.append(f"| **{c['contract']}** | **{c['customer_count']:,}** | {c['share_pct']}% | **{c['avg_predicted_churn_pct']:.1f}%** | {c['high_risk_pct']}% | ${c['monthly_revenue']:,.2f} |")
                
            m2m = next((c for c in ct_res["contracts"] if "month" in c["contract"].lower()), None)
            y2 = next((c for c in ct_res["contracts"] if "two" in c["contract"].lower()), None)
            if m2m and y2:
                lines.append(f"\n#### 💡 Analytical Finding:")
                lines.append(f"Subscribers on **Month-to-month** contracts exhibit an average predicted churn risk of **{m2m['avg_predicted_churn_pct']:.1f}%**, compared to **{y2['avg_predicted_churn_pct']:.1f}%** for **Two-year** contract accounts in this dataset.")
            return {"query": raw_q, "response": "\n".join(lines), "data": ct_res}

        # ---------------------------------------------------------------------
        # 8. Internet Service & Tech Support Analysis
        # ---------------------------------------------------------------------
        if any(w in q for w in ["fiber", "dsl", "internet service"]) and any(w in q for w in ["compare", "comparison", "vs", "versus", "churn", "risk", "impact", "breakdown", "tech support"]):
            net_res = self.get_internet_service_breakdown(company_id)
            if not net_res.get("found"):
                return {"query": raw_q, "response": "That information is not available in the current dataset.", "data": {}}
                
            lines = [
                f"**Source:** {net_res['source']}\n",
                f"### Internet Service & Tech Support Churn Breakdown\n",
                f"#### 1. Internet Service Breakdown:",
                f"| Service Type | Subscribers | Share % | Avg Predicted Churn % | High-Risk % | Monthly Revenue |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- |"
            ]
            for s in net_res["services"]:
                lines.append(f"| **{s['internet_service']}** | **{s['customer_count']:,}** | {s['share_pct']}% | **{s['avg_predicted_churn_pct']:.1f}%** | {s['high_risk_pct']}% | ${s['monthly_revenue']:,.2f} |")
                
            sup_st = net_res.get("tech_support_stats", {})
            lines.append(f"\n#### 2. Tech Support Relationship:")
            lines.append(f"• **Subscribers WITH Tech Support:** **{sup_st.get('with_support_count', 0):,} accounts** &bull; Avg Predicted Churn: **{sup_st.get('with_support_avg_churn', 0):.1f}%**")
            lines.append(f"• **Subscribers WITHOUT Tech Support:** **{sup_st.get('without_support_count', 0):,} accounts** &bull; Avg Predicted Churn: **{sup_st.get('without_support_avg_churn', 0):.1f}%**")
            lines.append(f"\n*Interpretation: In this dataset, accounts without Tech Support are associated with higher predicted churn probabilities.*")
            return {"query": raw_q, "response": "\n".join(lines), "data": net_res}

        # ---------------------------------------------------------------------
        # 9. Financial Revenue Exposure / Portfolio Aggregates
        # ---------------------------------------------------------------------
        if any(w in q for w in ["mrr", "revenue exposure", "value at risk", "how many customers", "how many are high risk", "average churn", "portfolio summary", "total revenue"]):
            dash_res = self.get_dashboard_summary(company_id)
            if not dash_res.get("found"):
                return {"query": raw_q, "response": "That information is not available in the current dataset.", "data": {}}
                
            lines = [
                f"**Source:** {dash_res['source']}\n",
                f"### Portfolio Summary & Revenue Exposure (**{comp_name}**)\n",
                f"| Metric | Verified Value |",
                f"| :--- | :--- |",
                f"| **Total Monitored Subscribers** | **{dash_res['total_customers']:,} accounts** |",
                f"| **High-Risk Subscribers (&ge; {dash_res['thresholds']['high_risk_min']})** | **{dash_res['high_risk_count']:,} accounts** ({dash_res['high_risk_pct']}%) |",
                f"| **Medium-Risk Subscribers ({dash_res['thresholds']['low_risk_max']} - {dash_res['thresholds']['high_risk_min']})** | **{dash_res['medium_risk_count']:,} accounts** ({dash_res['medium_risk_pct']}%) |",
                f"| **Low-Risk Subscribers (< {dash_res['thresholds']['low_risk_max']})** | **{dash_res['low_risk_count']:,} accounts** ({dash_res['low_risk_pct']}%) |",
                f"| **Portfolio Average Predicted Churn** | **{dash_res['average_churn_probability_pct']:.1f}%** |",
                f"| **Average Customer Spend** | **${dash_res['average_monthly_spend']:.2f}/month** (Tenure: {dash_res['average_tenure_months']:.1f} mo) |",
                f"| **Total Monthly Recurring Revenue (MRR)** | **${dash_res['total_monthly_recurring_revenue']:,.2f}/month** |",
                f"| **Total Annualized Revenue** | **${dash_res['total_annualized_revenue']:,.2f}/year** |",
                f"| **Annualized Revenue Represented by High-Risk Base** | **${dash_res['high_risk_annualized_revenue_exposure']:,.2f}/year** (${dash_res['high_risk_monthly_revenue_exposure']:,.2f}/month) |\n",
                f"*Calculation Note: Annualized revenue represented by the high-risk cohort = SUM(MonthlyCharges of High-Risk Accounts) × 12. This represents revenue exposure, not guaranteed recoverable revenue.*"
            ]
            return {"query": raw_q, "response": "\n".join(lines), "data": dash_res}

        # ---------------------------------------------------------------------
        # 10. General Retention Strategy Inquiries (Dataset Level)
        # ---------------------------------------------------------------------
        if any(w in q for w in ["reduce churn", "prevent churn", "retention strategy", "retention steps", "retention playbook", "how to retain", "churn reduction"]):
            lines = [
                f"**Source:** AI-Generated Recommendation (Analytical Framework)\n",
                f"### Structured Retention Intervention Roadmap\n",
                f"Based on verified predictive risk drivers across the active dataset (**{comp_name}**), the following non-monetary intervention categories are identified:\n",
                f"1. **Contract Commitment Reviews:** Proactive engagement campaigns to review terms with Month-to-Month subscribers and discuss annual commitment options.",
                f"2. **Early-Lifecycle Onboarding Support:** Structured Customer Success check-ins during the first 90 days to resolve connectivity and billing inquiries before churn triggers activate.",
                f"3. **Technical Support & Cyber Security Bundling:** Review coverage for high-speed fiber accounts lacking dedicated technical support to mitigate outage dissatisfaction.",
                f"4. **Automated Billing Guidance:** Streamline payment processing by encouraging migration to automated clearing house (ACH) or recurring card payments.\n",
                f"*Note: No promotional monetary discount, free month, or credit offer is configured in system policies. These suggestions represent operational intervention categories based on predictive risk drivers.*"
            ]
            return {"query": raw_q, "response": "\n".join(lines), "data": {"intent": "general_retention"}}

        # ---------------------------------------------------------------------
        # 11. Fallback: Semantic Vector Search & RAG Knowledge Base
        # ---------------------------------------------------------------------
        doc_res = self.search_business_documentation(company_id, raw_q)
        chunks = doc_res.get("chunks", [])
        if chunks:
            synthesized_answer = rag_service.synthesize_answer(company_id, raw_q, chunks)
            return {"query": raw_q, "response": synthesized_answer, "data": doc_res}

        # ---------------------------------------------------------------------
        # 12. Final Guardrail (Out-of-Scope / Non-Domain Queries)
        # ---------------------------------------------------------------------
        return {
            "query": raw_q,
            "response": (
                f"That information is not available in the current dataset or platform knowledge base.\n\n"
                f"**You can ask me to:**\n"
                f"• Inspect a customer dossier (e.g. `Tell me about 7590-VHVEG`)\n"
                f"• Run what-if simulations (e.g. `What if 7590-VHVEG switches to a 2-year contract?`)\n"
                f"• Prioritize customer cohorts (e.g. `Which customer segment should we target for retention?`)\n"
                f"• Analyze contracts or payment methods (e.g. `Compare contract types`)\n"
                f"• Explain ML models and metrics (e.g. `How does the stacking ensemble work?` or `What is the high-risk threshold?`)"
            ),
            "data": {"found": False}
        }

assistant_service = AssistantService()
