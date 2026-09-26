import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
from app.models.database_models import db, Customer, Prediction
from ml.preprocessing.harmonizer import extract_canonical_features, clean_and_harmonize_dataframe, CANONICAL_FEATURES

class PredictionService:
    def __init__(self):
        self.bundle = None
        self.models = {}
        self.scaler = None
        self.scale_cols = []
        self.optimal_thresholds = {}
        self.feature_importances = []
        self._load_bundle()

    def _load_bundle(self):
        bundle_path = Config.MODELS_BUNDLE_PATH
        if os.path.exists(bundle_path):
            self.bundle = joblib.load(bundle_path)
            self.models = self.bundle.get('models', {})
            self.scaler = self.bundle.get('scaler', None)
            self.scale_cols = self.bundle.get('scale_cols', [])
            self.optimal_thresholds = self.bundle.get('optimal_thresholds', {})
            self.feature_importances = self.bundle.get('feature_importances', [])
        else:
            print(f"Warning: Model bundle not found at {bundle_path}")

    def classify_risk_tier(self, probability: float) -> str:
        """Categorizes probability into LOW, MEDIUM, HIGH risk levels."""
        if probability >= Config.HIGH_RISK_MIN:
            return "HIGH"
        elif probability >= Config.LOW_RISK_MAX:
            return "MEDIUM"
        else:
            return "LOW"

    def predict_single(self, customer_data: dict, model_key="champion_ensemble") -> dict:
        """Runs prediction for a single customer profile."""
        if not self.models:
            self._load_bundle()
            
        model = self.models.get(model_key, self.models.get("champion_ensemble", None))
        if model is None:
            # Fallback to first available model
            model = list(self.models.values())[0] if self.models else None
            
        if model is None:
            raise RuntimeError("No trained model loaded in PredictionService.")
            
        df_feat = extract_canonical_features(customer_data).astype(np.float32)
        
        if model_key == "logistic_regression" and self.scaler:
            df_scaled = df_feat.copy()
            df_scaled[self.scale_cols] = self.scaler.transform(df_scaled[self.scale_cols])
            prob = float(model.predict_proba(df_scaled)[0][1])
        else:
            prob = float(model.predict_proba(df_feat)[0][1])
            
        risk_level = self.classify_risk_tier(prob)
        
        # Determine model version name
        model_version = f"v2.4.0-{model_key}"
        
        return {
            "churn_probability": round(prob, 4),
            "churn_percentage": round(prob * 100, 2),
            "retention_percentage": round((1.0 - prob) * 100, 2),
            "risk_level": risk_level,
            "model_version": model_version,
            "model_key": model_key
        }

    def predict_batch_and_save(self, company_id: str, customers_list: list, model_key="champion_ensemble") -> list:
        """
        Runs high-performance vector inference on a batch of customer records and writes predictions to SQL.
        """
        if not customers_list:
            return []
            
        if not self.models:
            self._load_bundle()
            
        model = self.models.get(model_key, self.models.get("champion_ensemble", None))
        df_raw = pd.DataFrame(customers_list)
        df_clean = clean_and_harmonize_dataframe(df_raw)
        
        for col in CANONICAL_FEATURES:
            if col not in df_clean.columns:
                df_clean[col] = 0.0
                
        X = df_clean[CANONICAL_FEATURES].astype(np.float32)
        probs = model.predict_proba(X)[:, 1]
        
        results = []
        now = datetime.utcnow()
        model_version = f"v2.4.0-{model_key}"
        
        for i, cust in enumerate(customers_list):
            cust_id = cust["customer_id"]
            p = float(probs[i])
            risk = self.classify_risk_tier(p)
            
            # Create prediction record
            pred = Prediction(
                customer_id=cust_id,
                company_id=company_id,
                model_version=model_version,
                churn_probability=p,
                risk_level=risk,
                prediction_date=now
            )
            db.session.add(pred)
            
            results.append({
                "customer_id": cust_id,
                "churn_probability": round(p * 100, 2),
                "risk_level": risk,
                "model_version": model_version
            })
            
        db.session.commit()
        return results

# Singleton instance
prediction_service = PredictionService()
