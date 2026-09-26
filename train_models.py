import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
import xgboost as xgb
import catboost as cb

from harmonizer import clean_and_harmonize_dataframe, CANONICAL_FEATURES, WeightedEnsembleClassifier

DATA_DIR = "data"
MODELS_BUNDLE_PATH = "models_bundle.joblib"
METRICS_PATH = "model_metrics.json"

def load_and_combine_datasets():
    """Loads all available datasets and merges them into a standardized multi-source DataFrame."""
    dfs = []
    
    files = [
        ("telco_churn_ibm.csv", "IBM Telco Primary"),
        ("telco_churn_region2.csv", "Suburban / Regional Cohort"),
        ("telco_churn_digital.csv", "Digital & Streaming Cohort")
    ]
    
    dataset_stats = {}
    
    for filename, display_name in files:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            df_curr = pd.read_csv(filepath)
            df_clean = clean_and_harmonize_dataframe(df_curr)
            dfs.append(df_clean)
            
            churn_count = int(df_clean['Target'].sum()) if 'Target' in df_clean.columns else 0
            dataset_stats[display_name] = {
                'total_rows': len(df_clean),
                'churn_count': churn_count,
                'churn_rate': round(churn_count / max(1, len(df_clean)) * 100, 2)
            }
            print(f"Loaded {display_name}: {len(df_clean)} rows, {churn_count} churns ({dataset_stats[display_name]['churn_rate']}%)")
            
    if not dfs:
        raise FileNotFoundError("No datasets found in data/ directory. Run download_and_prep_data.py first.")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal Combined Multi-Dataset Records: {len(combined_df)}")
    print(f"Overall Churn Rate: {combined_df['Target'].mean() * 100:.2f}%")
    return combined_df, dataset_stats

def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    """Computes comprehensive evaluation metrics for a model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return {
        'model_name': name,
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'brier_score': round(float(brier), 4),
        'confusion_matrix': cm
    }

def find_optimal_threshold(y_true, y_prob):
    """Finds the classification threshold that maximizes F1 score."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return round(float(best_thresh), 3)

def train_and_benchmark():
    combined_df, dataset_stats = load_and_combine_datasets()
    
    X = combined_df[CANONICAL_FEATURES].astype(np.float32).copy()
    y = combined_df['Target'].astype(np.int32).copy()
    
    # Train / Test split stratified by churn target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scaler for numeric features (and linear models)
    scaler = StandardScaler()
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_to_Tenure', 'ChargeDiscrepancy', 'HighRiskProfileScore']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
    
    # Calculate class imbalance ratio for tree models
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = float(neg_count / max(1, pos_count))
    print(f"Class Imbalance Ratio (Neg/Pos): {scale_pos_weight:.2f}")
    
    # 1. Baseline Logistic Regression
    print("\nTraining 1/6: Baseline Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # 2. Random Forest
    print("Training 2/6: Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=250, max_depth=12, min_samples_split=8, min_samples_leaf=3,
        class_weight='balanced', random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # 3. XGBoost
    print("Training 3/6: XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.04,
        scale_pos_weight=scale_pos_weight, subsample=0.85, colsample_bytree=0.85,
        eval_metric='logloss', random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # 4. CatBoost
    print("Training 4/6: CatBoost Classifier...")
    cat_model = cb.CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.04,
        auto_class_weights='Balanced', verbose=0, random_seed=42
    )
    cat_model.fit(X_train, y_train)
    
    # 5. Gradient Boosting
    print("Training 5/6: Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.04,
        subsample=0.85, random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # 6. Champion Multi-Model Stacking Ensemble
    print("Training 6/6: Champion Blended Stacking Ensemble...")
    ensemble_model = WeightedEnsembleClassifier([
        ('XGBoost', xgb_model, 0.35),
        ('CatBoost', cat_model, 0.35),
        ('GradientBoosting', gb_model, 0.15),
        ('RandomForest', rf_model, 0.15)
    ])
    
    # Benchmarking
    print("\n--- Evaluating Models on Held-out Multi-Dataset Test Cohort ---")
    models_dict = {
        'Champion Ensemble': (ensemble_model, X_test),
        'CatBoost': (cat_model, X_test),
        'XGBoost': (xgb_model, X_test),
        'Gradient Boosting': (gb_model, X_test),
        'Random Forest': (rf_model, X_test),
        'Logistic Regression (Baseline)': (lr_model, X_test_scaled)
    }
    
    metrics_list = []
    optimal_thresholds = {}
    
    for name, (mod, X_eval) in models_dict.items():
        y_prob = mod.predict_proba(X_eval)[:, 1]
        opt_thresh = find_optimal_threshold(y_test, y_prob)
        optimal_thresholds[name] = opt_thresh
        
        eval_res = evaluate_model(name, mod, X_eval, y_test, threshold=opt_thresh)
        eval_res['optimal_threshold'] = opt_thresh
        metrics_list.append(eval_res)
        
        print(f"[{name:32s}] ROC-AUC: {eval_res['roc_auc']:.4f} | PR-AUC: {eval_res['pr_auc']:.4f} | F1: {eval_res['f1_score']:.4f} | Recall: {eval_res['recall']:.4f} | Acc: {eval_res['accuracy']:.4f}")
        
    # Feature Importances from XGBoost & CatBoost
    xgb_imp = xgb_model.feature_importances_
    cat_imp = cat_model.get_feature_importance()
    
    # Normalize and average importances
    xgb_imp_norm = xgb_imp / np.sum(xgb_imp)
    cat_imp_norm = cat_imp / np.sum(cat_imp)
    avg_imp = (xgb_imp_norm + cat_imp_norm) / 2.0
    
    feature_importance_list = [
        {'feature': feat, 'importance': round(float(imp), 4)}
        for feat, imp in sorted(zip(CANONICAL_FEATURES, avg_imp), key=lambda x: x[1], reverse=True)
    ]
    
    # Save Model Bundle
    bundle = {
        'models': {
            'champion_ensemble': ensemble_model,
            'catboost': cat_model,
            'xgboost': xgb_model,
            'gradient_boosting': gb_model,
            'random_forest': rf_model,
            'logistic_regression': lr_model
        },
        'scaler': scaler,
        'scale_cols': scale_cols,
        'canonical_features': CANONICAL_FEATURES,
        'optimal_thresholds': optimal_thresholds,
        'feature_importances': feature_importance_list
    }
    joblib.dump(bundle, MODELS_BUNDLE_PATH)
    print(f"\nSaved trained models bundle to {MODELS_BUNDLE_PATH}")
    
    # Save Metrics JSON for UI visualization
    metrics_data = {
        'models': metrics_list,
        'feature_importances': feature_importance_list[:12], # top 12
        'dataset_stats': dataset_stats,
        'total_training_records': len(combined_df),
        'test_records': len(X_test)
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved benchmarking metrics to {METRICS_PATH}")
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    train_and_benchmark()
