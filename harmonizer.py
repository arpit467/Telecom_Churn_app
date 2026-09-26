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
    else:
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        
    # 2. Binary mappings
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        else:
            df[col] = 0
            
    if 'gender' in df.columns:
        df['gender_Male'] = (df['gender'] == 'Male').astype(int)
    else:
        df['gender_Male'] = 0
        
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
    else:
        df['SeniorCitizen'] = 0
        
    # 3. Domain Interaction & Advanced Engineered Features
    # (a) Monthly charges to tenure ratio
    df['MonthlyCharges_to_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1.0)
    
    # (b) Derived average monthly charges discrepancy
    avg_charges = df['TotalCharges'] / (df['tenure'] + 1.0)
    df['ChargeDiscrepancy'] = df['MonthlyCharges'] - avg_charges
    
    # (c) Active add-on services count
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    active_services = 0
    for sc in service_cols:
        if sc in df.columns:
            active_services += (df[sc] == 'Yes').astype(int)
    df['ActiveServicesCount'] = active_services
    
    # (d) Contract risk flag (Month-to-month has significantly higher churn)
    if 'Contract' in df.columns:
        df['Is_MonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['Is_TwoYear'] = (df['Contract'] == 'Two year').astype(int)
        df['Is_OneYear'] = (df['Contract'] == 'One year').astype(int)
    else:
        df['Is_MonthToMonth'] = 1
        df['Is_TwoYear'] = 0
        df['Is_OneYear'] = 0
        
    # (e) Payment risk flag (Electronic check has highest churn)
    if 'PaymentMethod' in df.columns:
        df['Is_ElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        df['Is_AutoPay'] = df['PaymentMethod'].astype(str).str.contains('automatic', case=False, na=False).astype(int)
    else:
        df['Is_ElectronicCheck'] = 0
        df['Is_AutoPay'] = 0
        
    # (f) High Risk Churn Combination Indicator
    # Customers on month-to-month, tenure < 12, paying > $65/mo without tech support
    tech_sup_no = (df['TechSupport'] != 'Yes').astype(int) if 'TechSupport' in df.columns else 1
    df['HighRiskProfileScore'] = (
        (df['Is_MonthToMonth'] * 2.0) +
        (df['Is_ElectronicCheck'] * 1.5) +
        ((df['tenure'] < 12).astype(int) * 1.5) +
        ((df['MonthlyCharges'] > 70).astype(int) * 1.0) +
        (tech_sup_no * 1.0)
    )
    
    # (g) Support & Security Bundle indicator
    has_sec = (df['OnlineSecurity'] == 'Yes').astype(int) if 'OnlineSecurity' in df.columns else 0
    has_sup = (df['TechSupport'] == 'Yes').astype(int) if 'TechSupport' in df.columns else 0
    df['Has_Security_and_Support'] = ((has_sec == 1) & (has_sup == 1)).astype(int)
    
    # (h) Internet Service categorization
    if 'InternetService' in df.columns:
        df['Internet_Fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
        df['Internet_DSL'] = (df['InternetService'] == 'DSL').astype(int)
        df['Internet_None'] = (df['InternetService'] == 'No').astype(int)
    else:
        df['Internet_Fiber'] = 0
        df['Internet_DSL'] = 0
        df['Internet_None'] = 0
        
    # (i) Multiple lines
    if 'MultipleLines' in df.columns:
        df['MultipleLines_Yes'] = (df['MultipleLines'] == 'Yes').astype(int)
    else:
        df['MultipleLines_Yes'] = 0
        
    # (j) Individual service flags
    for sc in service_cols:
        if sc in df.columns:
            df[f'{sc}_Yes'] = (df[sc] == 'Yes').astype(int)
        else:
            df[f'{sc}_Yes'] = 0
            
    # Target variable mapping if present
    if 'Churn' in df.columns:
        df['Target'] = df['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        
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
    Transforms a single user input from web form into standard model-ready feature vector.
    """
    df_raw = pd.DataFrame([raw_input_dict])
    df_clean = clean_and_harmonize_dataframe(df_raw)
    
    # Ensure all canonical columns exist
    for col in CANONICAL_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = 0
            
    return df_clean[CANONICAL_FEATURES]
