import pandas as pd
import numpy as np

class WeightedEnsembleClassifier:
    """
    High-performance calibrated ensemble combining predictions from
    XGBoost, CatBoost, Gradient Boosting, and Random Forest.
    """
    def __init__(self, models_with_weights=None, models=None, weights=None, **kwargs):
        if models_with_weights is not None:
            self.models_with_weights = models_with_weights # list of (name, model, weight)
        elif models is not None:
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            self.models_with_weights = [(f"Model_{i}", m, w) for i, (m, w) in enumerate(zip(models, weights))]
        else:
            self.models_with_weights = []
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        total_prob = np.zeros((len(X), 2), dtype=np.float64)
        total_weight = sum(w for _, _, w in self.models_with_weights) or 1.0
        for _, model, weight in self.models_with_weights:
            X_in = X
            if isinstance(X, pd.DataFrame):
                if hasattr(model, 'feature_names_in_'):
                    try:
                        X_in = X[model.feature_names_in_]
                    except Exception:
                        X_in = X.to_numpy(dtype=np.float32)
                else:
                    X_in = X
            prob = model.predict_proba(X_in)
            total_prob += weight * prob
        return total_prob / total_weight

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

def clean_and_harmonize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw multi-dataset telecom/digital data, fills missing values,
    and constructs standardized canonical features.
    """
    df = df.copy()
    
    # 1. Standardize numeric columns first
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(20.0)
    elif 'monthly_charges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['monthly_charges'], errors='coerce').fillna(20.0)
    else:
        df['MonthlyCharges'] = 20.0
        
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(1.0)
    else:
        df['tenure'] = 1.0
        
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    elif 'total_charges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    else:
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        
    # 2. Binary mappings
    col_mappings = {
        'Partner': 'partner',
        'Dependents': 'dependents',
        'PhoneService': 'phone_service',
        'PaperlessBilling': 'paperless_billing'
    }
    for upper_col, lower_col in col_mappings.items():
        chosen_col = upper_col if upper_col in df.columns else (lower_col if lower_col in df.columns else None)
        if chosen_col:
            df[upper_col] = df[chosen_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        else:
            df[upper_col] = 0
            
    gender_col = 'gender' if 'gender' in df.columns else ('Gender' if 'Gender' in df.columns else None)
    if gender_col:
        df['gender_Male'] = (df[gender_col] == 'Male').astype(int)
    else:
        df['gender_Male'] = 0
        
    senior_col = 'SeniorCitizen' if 'SeniorCitizen' in df.columns else ('senior_citizen' if 'senior_citizen' in df.columns else None)
    if senior_col:
        df['SeniorCitizen'] = pd.to_numeric(df[senior_col], errors='coerce').fillna(0).astype(int)
    else:
        df['SeniorCitizen'] = 0
        
    # 3. Domain Interaction & Advanced Engineered Features
    df['MonthlyCharges_to_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1.0)
    avg_charges = df['TotalCharges'] / (df['tenure'] + 1.0)
    df['ChargeDiscrepancy'] = df['MonthlyCharges'] - avg_charges
    
    service_cols = [
        ('OnlineSecurity', 'online_security'),
        ('OnlineBackup', 'online_backup'),
        ('DeviceProtection', 'device_protection'),
        ('TechSupport', 'tech_support'),
        ('StreamingTV', 'streaming_tv'),
        ('StreamingMovies', 'streaming_movies')
    ]
    active_services = 0
    for u_col, l_col in service_cols:
        chosen = u_col if u_col in df.columns else (l_col if l_col in df.columns else None)
        if chosen:
            s_val = df[chosen].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)
            df[f"{u_col}_Yes"] = s_val
            active_services += s_val
        else:
            df[f"{u_col}_Yes"] = 0
    df['ActiveServicesCount'] = active_services
    
    # Contract encodings
    contract_col = 'Contract' if 'Contract' in df.columns else ('contract' if 'contract' in df.columns else None)
    if contract_col:
        df['Is_MonthToMonth'] = df[contract_col].str.lower().str.contains('month', na=False).astype(int)
        df['Is_OneYear'] = df[contract_col].str.lower().str.contains('one', na=False).astype(int)
        df['Is_TwoYear'] = df[contract_col].str.lower().str.contains('two', na=False).astype(int)
    else:
        df['Is_MonthToMonth'] = 1
        df['Is_OneYear'] = 0
        df['Is_TwoYear'] = 0
        
    # Internet service encodings
    internet_col = 'InternetService' if 'InternetService' in df.columns else ('internet_service' if 'internet_service' in df.columns else None)
    if internet_col:
        df['Internet_Fiber'] = df[internet_col].str.lower().str.contains('fiber', na=False).astype(int)
        df['Internet_DSL'] = df[internet_col].str.lower().str.contains('dsl', na=False).astype(int)
        df['Internet_None'] = df[internet_col].str.lower().str.contains('no', na=False).astype(int)
    else:
        df['Internet_Fiber'] = 0
        df['Internet_DSL'] = 1
        df['Internet_None'] = 0
        
    # Payment method encodings
    payment_col = 'PaymentMethod' if 'PaymentMethod' in df.columns else ('payment_method' if 'payment_method' in df.columns else None)
    if payment_col:
        df['Is_ElectronicCheck'] = df[payment_col].str.lower().str.contains('electronic', na=False).astype(int)
        df['Is_AutoPay'] = df[payment_col].str.lower().str.contains('auto', na=False).astype(int)
    else:
        df['Is_ElectronicCheck'] = 0
        df['Is_AutoPay'] = 0
        
    # Multi-lines
    lines_col = 'MultipleLines' if 'MultipleLines' in df.columns else ('multiple_lines' if 'multiple_lines' in df.columns else None)
    if lines_col:
        df['MultipleLines_Yes'] = df[lines_col].str.lower().str.contains('yes', na=False).astype(int)
    else:
        df['MultipleLines_Yes'] = 0
        
    # High risk profile score composite
    df['Has_Security_and_Support'] = ((df['OnlineSecurity_Yes'] == 1) & (df['TechSupport_Yes'] == 1)).astype(int)
    df['HighRiskProfileScore'] = (
        (df['Is_MonthToMonth'] * 2.5) +
        (df['Is_ElectronicCheck'] * 1.8) +
        (df['Internet_Fiber'] * 1.5) +
        ((df['tenure'] <= 12).astype(int) * 2.0) -
        (df['Is_TwoYear'] * 3.0) -
        (df['TechSupport_Yes'] * 1.5) -
        (df['OnlineSecurity_Yes'] * 1.2) -
        (df['Is_AutoPay'] * 1.0)
    )
    
    # Target column extraction if present
    target_col = 'Churn' if 'Churn' in df.columns else ('churn' if 'churn' in df.columns else ('Target' if 'Target' in df.columns else None))
    if target_col:
        df['Target'] = df[target_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0, 'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
        
    return df

CANONICAL_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'MonthlyCharges_to_Tenure',
    'ChargeDiscrepancy',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
    'gender_Male',
    'ActiveServicesCount',
    'OnlineSecurity_Yes',
    'OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'TechSupport_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes',
    'Is_MonthToMonth',
    'Is_OneYear',
    'Is_TwoYear',
    'Internet_Fiber',
    'Internet_DSL',
    'Internet_None',
    'Is_ElectronicCheck',
    'Is_AutoPay',
    'MultipleLines_Yes',
    'Has_Security_and_Support',
    'HighRiskProfileScore'
]

def extract_canonical_features(record: dict) -> pd.DataFrame:
    df_raw = pd.DataFrame([record])
    df_clean = clean_and_harmonize_dataframe(df_raw)
    for col in CANONICAL_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
    return df_clean[CANONICAL_FEATURES]
