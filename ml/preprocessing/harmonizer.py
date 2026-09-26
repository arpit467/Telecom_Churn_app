import pandas as pd
import numpy as np

class WeightedEnsembleClassifier:
    """
    High-performance calibrated ensemble combining predictions from
    XGBoost, CatBoost, Gradient Boosting, and Random Forest.
    """
    def __init__(self, models_with_weights):
        self.models_with_weights = models_with_weights # list of (name, model, weight)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        total_prob = np.zeros((len(X), 2), dtype=np.float64)
        total_weight = sum(w for _, _, w in self.models_with_weights)
        for _, model, weight in self.models_with_weights:
            prob = model.predict_proba(X)
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
    # (a) Monthly charges to tenure ratio
    df['MonthlyCharges_to_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1.0)
    
    # (b) Derived average monthly charges discrepancy
    avg_charges = df['TotalCharges'] / (df['tenure'] + 1.0)
    df['ChargeDiscrepancy'] = df['MonthlyCharges'] - avg_charges
    
    # (c) Active add-on services count
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
        col = u_col if u_col in df.columns else (l_col if l_col in df.columns else None)
        if col:
            df[u_col] = df[col]
            active_services += (df[col] == 'Yes').astype(int)
        else:
            df[u_col] = 'No'
    df['ActiveServicesCount'] = active_services
    
    # (d) Contract risk flags
    contract_col = 'Contract' if 'Contract' in df.columns else ('contract' if 'contract' in df.columns else None)
    if contract_col:
        df['Is_MonthToMonth'] = (df[contract_col] == 'Month-to-month').astype(int)
        df['Is_TwoYear'] = (df[contract_col] == 'Two year').astype(int)
        df['Is_OneYear'] = (df[contract_col] == 'One year').astype(int)
    else:
        df['Is_MonthToMonth'] = 1
        df['Is_TwoYear'] = 0
        df['Is_OneYear'] = 0
        
    # (e) Payment risk flag
    pm_col = 'PaymentMethod' if 'PaymentMethod' in df.columns else ('payment_method' if 'payment_method' in df.columns else None)
    if pm_col:
        df['Is_ElectronicCheck'] = (df[pm_col] == 'Electronic check').astype(int)
        df['Is_AutoPay'] = df[pm_col].astype(str).str.contains('automatic', case=False, na=False).astype(int)
    else:
        df['Is_ElectronicCheck'] = 0
        df['Is_AutoPay'] = 0
        
    # (f) High Risk Churn Combination Indicator
    tech_col = 'TechSupport' if 'TechSupport' in df.columns else ('tech_support' if 'tech_support' in df.columns else None)
    tech_sup_no = (df[tech_col] != 'Yes').astype(int) if tech_col else 1
    df['HighRiskProfileScore'] = (
        (df['Is_MonthToMonth'] * 2.0) +
        (df['Is_ElectronicCheck'] * 1.5) +
        ((df['tenure'] < 12).astype(int) * 1.5) +
        ((df['MonthlyCharges'] > 70).astype(int) * 1.0) +
        (tech_sup_no * 1.0)
    )
    
    # (g) Support & Security Bundle indicator
    sec_col = 'OnlineSecurity' if 'OnlineSecurity' in df.columns else ('online_security' if 'online_security' in df.columns else None)
    has_sec = (df[sec_col] == 'Yes').astype(int) if sec_col else 0
    has_sup = (df[tech_col] == 'Yes').astype(int) if tech_col else 0
    df['Has_Security_and_Support'] = ((has_sec == 1) & (has_sup == 1)).astype(int)
    
    # (h) Internet Service categorization
    net_col = 'InternetService' if 'InternetService' in df.columns else ('internet_service' if 'internet_service' in df.columns else None)
    if net_col:
        df['Internet_Fiber'] = (df[net_col] == 'Fiber optic').astype(int)
        df['Internet_DSL'] = (df[net_col] == 'DSL').astype(int)
        df['Internet_None'] = (df[net_col] == 'No').astype(int)
    else:
        df['Internet_Fiber'] = 0
        df['Internet_DSL'] = 0
        df['Internet_None'] = 0
        
    # (i) Multiple lines
    ml_col = 'MultipleLines' if 'MultipleLines' in df.columns else ('multiple_lines' if 'multiple_lines' in df.columns else None)
    if ml_col:
        df['MultipleLines_Yes'] = (df[ml_col] == 'Yes').astype(int)
    else:
        df['MultipleLines_Yes'] = 0
        
    # (j) Individual service flags
    for u_col, l_col in service_cols:
        col = u_col if u_col in df.columns else (l_col if l_col in df.columns else None)
        if col:
            df[f'{u_col}_Yes'] = (df[col] == 'Yes').astype(int)
        else:
            df[f'{u_col}_Yes'] = 0
            
    # Target variable mapping if present
    churn_col = 'Churn' if 'Churn' in df.columns else ('churn' if 'churn' in df.columns else ('Target' if 'Target' in df.columns else None))
    if churn_col:
        df['Target'] = df[churn_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        
    return df

# Canonical feature list used for all model training and inference
CANONICAL_FEATURES = [
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'PaperlessBilling',
    'MonthlyCharges',
    'TotalCharges',
    'gender_Male',
    'MonthlyCharges_to_Tenure',
    'ChargeDiscrepancy',
    'ActiveServicesCount',
    'Is_MonthToMonth',
    'Is_TwoYear',
    'Is_OneYear',
    'Is_ElectronicCheck',
    'Is_AutoPay',
    'HighRiskProfileScore',
    'Has_Security_and_Support',
    'Internet_Fiber',
    'Internet_DSL',
    'Internet_None',
    'MultipleLines_Yes',
    'OnlineSecurity_Yes',
    'OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'TechSupport_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes'
]

def extract_canonical_features(raw_input_dict: dict) -> pd.DataFrame:
    """
    Transforms a single user input or record dictionary into standard model-ready feature vector.
    """
    df_raw = pd.DataFrame([raw_input_dict])
    df_clean = clean_and_harmonize_dataframe(df_raw)
    
    # Ensure all canonical columns exist
    for col in CANONICAL_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = 0
            
    return df_clean[CANONICAL_FEATURES]
