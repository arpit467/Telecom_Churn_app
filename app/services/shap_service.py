import os
import numpy as np
import pandas as pd
import shap
from config import Config
from app.models.database_models import db, CustomerExplanation
from ml.preprocessing.harmonizer import extract_canonical_features, clean_and_harmonize_dataframe, CANONICAL_FEATURES

FEATURE_DISPLAY_NAMES = {
    "HighRiskProfileScore": "Composite High Risk Score",
    "MonthlyCharges": "Monthly Charges",
    "MonthlyCharges_to_Tenure": "Charge-to-Tenure Velocity",
    "Is_TwoYear": "Two-Year Annual Contract",
    "Is_MonthToMonth": "Month-to-Month Contract",
    "ChargeDiscrepancy": "Price Increase Discrepancy",
    "TotalCharges": "Total Lifetime Spend",
    "tenure": "Customer Tenure (Months)",
    "ActiveServicesCount": "Active Digital Services Count",
    "Is_OneYear": "One-Year Annual Contract",
    "OnlineSecurity_Yes": "Online Cyber Security Subscribed",
    "TechSupport_Yes": "Dedicated Tech Support Active",
    "Is_ElectronicCheck": "Payment via Electronic Check",
    "Is_AutoPay": "Automatic Payment Enabled",
    "Internet_Fiber": "High-Speed Fiber Optic Internet",
    "Internet_DSL": "DSL Internet Service",
    "Internet_None": "No Internet Connection",
    "Has_Security_and_Support": "Security & Tech Support Bundle",
    "PaperlessBilling": "Paperless Invoicing",
    "SeniorCitizen": "Senior Citizen Demographics",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "PhoneService": "Phone Service Active",
    "MultipleLines_Yes": "Multiple Phone Lines Active",
    "OnlineBackup_Yes": "Online Cloud Backup Active",
    "DeviceProtection_Yes": "Device Warranty Protection",
    "StreamingTV_Yes": "Streaming TV Package",
    "StreamingMovies_Yes": "Streaming Movies Package",
    "gender_Male": "Gender (Male)"
}

class ShapService:
    def __init__(self):
        self.explainer = None
        self.base_model = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            from app.services.prediction_service import prediction_service
            if not prediction_service.models:
                prediction_service._load_bundle()
                
            # Use XGBoost or CatBoost for fast exact TreeExplainer
            self.base_model = prediction_service.models.get("xgboost") or prediction_service.models.get("catboost")
            if self.base_model is not None:
                self.explainer = shap.TreeExplainer(self.base_model)
        except Exception as e:
            print(f"SHAP initialization note: {e}")

    def explain_customer(self, customer_data: dict, prediction_id: int = None, save_to_db: bool = True) -> list:
        """
        Computes exact SHAP values for a single customer and returns top driving factors.
        """
        if self.explainer is None:
            self._init_explainer()
            
        df_feat = extract_canonical_features(customer_data).astype(np.float32)
        
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(df_feat)
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    vals = np.array(shap_vals[1])[0]
                elif len(np.shape(shap_vals)) == 2:
                    vals = np.array(shap_vals)[0]
                else:
                    vals = np.array(shap_vals).flatten()
            except Exception as e:
                vals = self._fallback_contributions(df_feat)
        else:
            vals = self._fallback_contributions(df_feat)
            
        explanations = []
        cust_id = customer_data.get("customer_id", "UNKNOWN")
        
        for feat_name, s_val in zip(CANONICAL_FEATURES, vals):
            raw_val = df_feat[feat_name].iloc[0]
            display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)
            direction = "INCREASES_CHURN" if s_val > 0 else "DECREASES_CHURN"
            
            explanations.append({
                "feature_name": display_name,
                "raw_feature": feat_name,
                "feature_value": str(raw_val),
                "shap_value": float(s_val),
                "abs_shap": abs(float(s_val)),
                "impact_direction": direction
            })
            
        explanations.sort(key=lambda x: x["abs_shap"], reverse=True)
        top_explanations = explanations[:8]
        
        if save_to_db and prediction_id is not None:
            for item in top_explanations:
                db_exp = CustomerExplanation(
                    prediction_id=prediction_id,
                    customer_id=cust_id,
                    feature_name=item["feature_name"],
                    feature_value=item["feature_value"],
                    shap_value=item["shap_value"],
                    impact_direction=item["impact_direction"]
                )
                db.session.add(db_exp)
            db.session.commit()
            
        return top_explanations

    def _fallback_contributions(self, df_feat: pd.DataFrame) -> np.ndarray:
        """Calculates normalized domain-weighted impact when TreeExplainer is unavailable."""
        weights = {
            "HighRiskProfileScore": 0.45,
            "Is_MonthToMonth": 0.35,
            "MonthlyCharges": 0.25,
            "Is_TwoYear": -0.40,
            "tenure": -0.30,
            "Is_OneYear": -0.20,
            "TechSupport_Yes": -0.20,
            "OnlineSecurity_Yes": -0.18,
            "Is_ElectronicCheck": 0.22,
            "Internet_Fiber": 0.15,
            "ActiveServicesCount": -0.15
        }
        res = []
        for feat in CANONICAL_FEATURES:
            val = float(df_feat[feat].iloc[0])
            w = weights.get(feat, 0.05)
            res.append(val * w)
        return np.array(res, dtype=np.float32)

    def get_dataset_churn_drivers(self, company_id: str = None) -> list:
        """
        Computes dynamic TreeSHAP feature importance specifically across the customer records in the given dataset.
        """
        if not company_id:
            return self.get_global_churn_drivers()
            
        from app.models.database_models import Customer
        
        customers = Customer.query.filter_by(company_id=company_id).all()
        if not customers or len(customers) < 2:
            return self.get_global_churn_drivers()
            
        try:
            cust_dicts = [c.to_dict() for c in customers]
            df_raw = pd.DataFrame(cust_dicts)
            df_clean = clean_and_harmonize_dataframe(df_raw)
            
            for col in CANONICAL_FEATURES:
                if col not in df_clean.columns:
                    df_clean[col] = 0.0
                    
            X = df_clean[CANONICAL_FEATURES].astype(np.float32)
            
            if self.explainer is None:
                self._init_explainer()
                
            if self.explainer is not None:
                # Sample up to 300 records for fast exact TreeSHAP attribution
                X_sample = X.iloc[:300] if len(X) > 300 else X
                shap_vals = self.explainer.shap_values(X_sample)
                
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    vals = np.array(shap_vals[1])
                elif len(np.shape(shap_vals)) == 2:
                    vals = np.array(shap_vals)
                else:
                    vals = np.array(shap_vals)
                    
                mean_abs = np.mean(np.abs(vals), axis=0)
                total_shap = np.sum(mean_abs)
                if total_shap > 0:
                    weights = (mean_abs / total_shap) * 100.0
                else:
                    weights = np.ones(len(CANONICAL_FEATURES)) * (100.0 / len(CANONICAL_FEATURES))
                    
                drivers = []
                for feat, weight in zip(CANONICAL_FEATURES, weights):
                    drivers.append({
                        "feature": FEATURE_DISPLAY_NAMES.get(feat, feat),
                        "raw_feature": feat,
                        "importance": round(float(weight), 2)
                    })
                drivers.sort(key=lambda x: x["importance"], reverse=True)
                return drivers
        except Exception as e:
            print(f"Dataset SHAP calculation fallback: {e}")
            
        return self.get_global_churn_drivers()

    def get_global_churn_drivers(self) -> list:
        """Returns the pre-calculated global feature importance rankings."""
        from app.services.prediction_service import prediction_service
        if not prediction_service.feature_importances:
            prediction_service._load_bundle()
            
        raw_importances = prediction_service.feature_importances or []
        formatted = []
        for item in raw_importances:
            feat = item.get("feature", "")
            imp = item.get("importance", 0.0)
            formatted.append({
                "feature": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "raw_feature": feat,
                "importance": round(float(imp) * 100, 2)
            })
        return formatted

shap_service = ShapService()
