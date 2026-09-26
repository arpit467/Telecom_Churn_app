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
            try:
                self.bundle = joblib.load(bundle_path)
                self.models = self.bundle.get('models', {})
                self.scaler = self.bundle.get('scaler', None)
                self.scale_cols = self.bundle.get('scale_cols', [])
                self.optimal_thresholds = self.bundle.get('optimal_thresholds', {})
                self.feature_importances = self.bundle.get('feature_importances', [])
                print(f"Successfully loaded model bundle ({len(self.models)} models available).")
                return
            except Exception as e:
                print(f"Notice: Model bundle at {bundle_path} could not be unpickled due to environment/version difference: {e}")
                print("Initiating automatic in-environment model training fallback...")
        
        # If model bundle missing or unpickle failed across python/sklearn versions, auto-train:
        self._auto_train_models()

    def _auto_train_models(self):
        """Auto-trains models directly in the current runtime environment if bundle is missing or incompatible."""
        try:
            print("Auto-training models in current environment...")
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            import xgboost as xgb
            import catboost as cb
            from ml.preprocessing.harmonizer import WeightedEnsembleClassifier

            dfs = []
            files = ["telco_churn_ibm.csv", "telco_churn_region2.csv", "telco_churn_digital.csv"]
            data_dir = Config.DATA_DIR if hasattr(Config, "DATA_DIR") else "data"
            
            for f in files:
                fpath = os.path.join(data_dir, f)
                if os.path.exists(fpath):
                    df_raw = pd.read_csv(fpath)
                    df_c = clean_and_harmonize_dataframe(df_raw)
                    dfs.append(df_c)
                    
            if not dfs:
                print("Warning: No datasets found in data/ directory for auto-training.")
                return

            df_all = pd.concat(dfs, ignore_index=True)
            X = df_all[CANONICAL_FEATURES].astype(np.float32)
            y = df_all['Target'].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_to_Tenure', 'ChargeDiscrepancy']
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])

            # Train models
            lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            lr.fit(X_train_scaled, y_train)

            rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            pos_weight = float((y_train == 0).sum()) / max(1, float((y_train == 1).sum()))
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)

            cb_model = cb.CatBoostClassifier(iterations=150, depth=5, learning_rate=0.08, auto_class_weights='Balanced', random_seed=42, verbose=False)
            cb_model.fit(X_train, y_train)

            gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
            gb_model.fit(X_train, y_train)

            ensemble = WeightedEnsembleClassifier(models=[cb_model, xgb_model, rf, lr], weights=[0.35, 0.35, 0.20, 0.10], scaler=scaler, scale_cols=scale_cols)

            self.models = {
                'champion_ensemble': ensemble,
                'catboost': cb_model,
                'xgboost': xgb_model,
                'gradient_boosting': gb_model,
                'random_forest': rf,
                'logistic_regression': lr
            }
            self.scaler = scaler
            self.scale_cols = scale_cols
            self.optimal_thresholds = {
                'champion_ensemble': 0.48,
                'catboost': 0.45,
                'xgboost': 0.49,
                'gradient_boosting': 0.31,
                'random_forest': 0.41,
                'logistic_regression': 0.48
            }
            
            # Save bundle for future speedup
            try:
                bundle = {
                    'models': self.models,
                    'scaler': self.scaler,
                    'scale_cols': self.scale_cols,
                    'optimal_thresholds': self.optimal_thresholds
                }
                joblib.dump(bundle, Config.MODELS_BUNDLE_PATH)
                print(f"Auto-trained and cached model bundle to {Config.MODELS_BUNDLE_PATH}.")
            except Exception as save_err:
                print(f"Note: Could not save model bundle to disk: {save_err}")
                
        except Exception as err:
            print(f"Auto-training fallback error: {err}")

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
        """Runs high-performance vector inference on a batch of customer records and writes predictions to SQL."""
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
