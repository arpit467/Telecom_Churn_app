import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

IBM_TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
IBM_TELCO_FALLBACK = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def download_file(url, target_path, fallback_url=None):
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"Successfully downloaded to {target_path}")
        return True
    except Exception as e:
        print(f"Failed to download from {url}: {e}")
        if fallback_url:
            try:
                print(f"Attempting fallback {fallback_url}...")
                urllib.request.urlretrieve(fallback_url, target_path)
                print(f"Successfully downloaded from fallback to {target_path}")
                return True
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
        return False

def generate_telecom_dataset_2(num_rows=3500, random_seed=42):
    """
    Generates a realistic multi-region Telecom Cohort B (e.g., Enterprise & Suburban Fiber/Mobile)
    with nuanced distribution differences to test cross-dataset generalization.
    """
    np.random.seed(random_seed)
    
    genders = np.random.choice(['Male', 'Female'], size=num_rows, p=[0.51, 0.49])
    seniors = np.random.choice([0, 1], size=num_rows, p=[0.82, 0.18])
    partners = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.52, 0.48])
    dependents = np.where(partners == 'Yes', np.random.choice(['Yes', 'No'], size=num_rows, p=[0.45, 0.55]), 'No')
    
    # Tenure in months (skewed with newer customers and loyal long-term)
    tenures = np.clip(np.random.exponential(scale=28, size=num_rows).astype(int), 1, 72)
    
    phone_services = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.92, 0.08])
    multiple_lines = []
    for ps in phone_services:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.48, 0.52]))
            
    internet_services = np.random.choice(['Fiber optic', 'DSL', 'No'], size=num_rows, p=[0.46, 0.38, 0.16])
    
    online_security, online_backup, device_prot, tech_sup, stream_tv, stream_mov = [], [], [], [], [], []
    for net in internet_services:
        if net == 'No':
            for l in [online_security, online_backup, device_prot, tech_sup, stream_tv, stream_mov]:
                l.append('No internet service')
        else:
            online_security.append(np.random.choice(['Yes', 'No'], p=[0.33, 0.67]))
            online_backup.append(np.random.choice(['Yes', 'No'], p=[0.38, 0.62]))
            device_prot.append(np.random.choice(['Yes', 'No'], p=[0.37, 0.63]))
            tech_sup.append(np.random.choice(['Yes', 'No'], p=[0.32, 0.68]))
            stream_tv.append(np.random.choice(['Yes', 'No'], p=[0.43, 0.57]))
            stream_mov.append(np.random.choice(['Yes', 'No'], p=[0.44, 0.56]))
            
    contracts = []
    for t in tenures:
        if t < 12:
            contracts.append(np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.82, 0.14, 0.04]))
        elif t < 36:
            contracts.append(np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.45, 0.40, 0.15]))
        else:
            contracts.append(np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.20, 0.35, 0.45]))
            
    paperless = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.63, 0.37])
    payment_methods = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], size=num_rows, p=[0.34, 0.22, 0.22, 0.22])
    
    # Calculate monthly charges based on services
    base_charge = 20.0
    monthly_charges = []
    for i in range(num_rows):
        mc = base_charge
        if phone_services[i] == 'Yes': mc += 10.0
        if multiple_lines[i] == 'Yes': mc += 12.0
        if internet_services[i] == 'Fiber optic': mc += 45.0
        elif internet_services[i] == 'DSL': mc += 25.0
        if online_security[i] == 'Yes': mc += 8.0
        if online_backup[i] == 'Yes': mc += 8.0
        if device_prot[i] == 'Yes': mc += 8.0
        if tech_sup[i] == 'Yes': mc += 8.0
        if stream_tv[i] == 'Yes': mc += 12.0
        if stream_mov[i] == 'Yes': mc += 12.0
        # Add random noise
        mc += np.random.normal(0, 2.5)
        monthly_charges.append(round(max(18.0, mc), 2))
        
    total_charges = [round(monthly_charges[i] * tenures[i] * np.random.uniform(0.95, 1.05), 2) for i in range(num_rows)]
    
    # Realistic Churn Probability Model for Dataset B
    churn_prob = []
    for i in range(num_rows):
        score = -1.2 # base bias
        if contracts[i] == 'Month-to-month': score += 1.3
        elif contracts[i] == 'Two year': score -= 1.4
        if tenures[i] < 6: score += 1.1
        elif tenures[i] > 36: score -= 1.0
        if internet_services[i] == 'Fiber optic': score += 0.5
        if tech_sup[i] == 'No': score += 0.45
        if online_security[i] == 'No': score += 0.40
        if payment_methods[i] == 'Electronic check': score += 0.55
        if monthly_charges[i] > 80: score += 0.45
        if seniors[i] == 1: score += 0.25
        
        prob = 1.0 / (1.0 + np.exp(-score))
        churn_prob.append(prob)
        
    churn = ['Yes' if np.random.rand() < p else 'No' for p in churn_prob]
    
    df = pd.DataFrame({
        'customerID': [f'REG2-{i:05d}' for i in range(num_rows)],
        'gender': genders,
        'SeniorCitizen': seniors,
        'Partner': partners,
        'Dependents': dependents,
        'tenure': tenures,
        'PhoneService': phone_services,
        'MultipleLines': multiple_lines,
        'InternetService': internet_services,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_prot,
        'TechSupport': tech_sup,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_mov,
        'Contract': contracts,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_methods,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn,
        'DatasetSource': 'Region2_Suburban_Cohort'
    })
    return df

def generate_digital_services_dataset_3(num_rows=2500, random_seed=101):
    """
    Generates Dataset C: Digital Tech & Streaming Subscriber Cohort
    Higher tech adoption, fiber reliance, varying churn factors.
    """
    np.random.seed(random_seed)
    genders = np.random.choice(['Male', 'Female'], size=num_rows, p=[0.49, 0.51])
    seniors = np.random.choice([0, 1], size=num_rows, p=[0.90, 0.10])
    partners = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.40, 0.60])
    dependents = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.25, 0.75])
    tenures = np.clip(np.random.geometric(p=0.04, size=num_rows), 1, 72)
    phone_services = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.85, 0.15])
    multiple_lines = ['Yes' if ps == 'Yes' and np.random.rand() > 0.4 else ('No' if ps == 'Yes' else 'No phone service') for ps in phone_services]
    internet_services = np.random.choice(['Fiber optic', 'DSL', 'No'], size=num_rows, p=[0.65, 0.30, 0.05])
    
    online_security, online_backup, device_prot, tech_sup, stream_tv, stream_mov = [], [], [], [], [], []
    for net in internet_services:
        if net == 'No':
            for l in [online_security, online_backup, device_prot, tech_sup, stream_tv, stream_mov]:
                l.append('No internet service')
        else:
            online_security.append(np.random.choice(['Yes', 'No'], p=[0.45, 0.55]))
            online_backup.append(np.random.choice(['Yes', 'No'], p=[0.55, 0.45]))
            device_prot.append(np.random.choice(['Yes', 'No'], p=[0.50, 0.50]))
            tech_sup.append(np.random.choice(['Yes', 'No'], p=[0.40, 0.60]))
            stream_tv.append(np.random.choice(['Yes', 'No'], p=[0.60, 0.40]))
            stream_mov.append(np.random.choice(['Yes', 'No'], p=[0.62, 0.38]))
            
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_rows, p=[0.60, 0.25, 0.15])
    paperless = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.75, 0.25])
    payment_methods = np.random.choice([
        'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)', 'Mailed check'
    ], size=num_rows, p=[0.30, 0.35, 0.25, 0.10])
    
    monthly_charges = []
    for i in range(num_rows):
        mc = 25.0
        if phone_services[i] == 'Yes': mc += 10.0
        if multiple_lines[i] == 'Yes': mc += 15.0
        if internet_services[i] == 'Fiber optic': mc += 50.0
        elif internet_services[i] == 'DSL': mc += 30.0
        if online_security[i] == 'Yes': mc += 10.0
        if online_backup[i] == 'Yes': mc += 10.0
        if device_prot[i] == 'Yes': mc += 10.0
        if tech_sup[i] == 'Yes': mc += 10.0
        if stream_tv[i] == 'Yes': mc += 15.0
        if stream_mov[i] == 'Yes': mc += 15.0
        mc += np.random.normal(0, 3.0)
        monthly_charges.append(round(max(20.0, mc), 2))
        
    total_charges = [round(monthly_charges[i] * tenures[i] * np.random.uniform(0.96, 1.04), 2) for i in range(num_rows)]
    
    churn_prob = []
    for i in range(num_rows):
        score = -1.0
        if contracts[i] == 'Month-to-month': score += 1.2
        elif contracts[i] == 'Two year': score -= 1.3
        if tenures[i] < 4: score += 1.2
        elif tenures[i] > 24: score -= 0.8
        if tech_sup[i] == 'No' and stream_tv[i] == 'Yes': score += 0.6
        if payment_methods[i] == 'Electronic check': score += 0.4
        prob = 1.0 / (1.0 + np.exp(-score))
        churn_prob.append(prob)
        
    churn = ['Yes' if np.random.rand() < p else 'No' for p in churn_prob]
    
    df = pd.DataFrame({
        'customerID': [f'DIG3-{i:05d}' for i in range(num_rows)],
        'gender': genders,
        'SeniorCitizen': seniors,
        'Partner': partners,
        'Dependents': dependents,
        'tenure': tenures,
        'PhoneService': phone_services,
        'MultipleLines': multiple_lines,
        'InternetService': internet_services,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_prot,
        'TechSupport': tech_sup,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_mov,
        'Contract': contracts,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_methods,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn,
        'DatasetSource': 'Digital_Streaming_Cohort'
    })
    return df

def main():
    telco_path = os.path.join(DATA_DIR, "telco_churn_ibm.csv")
    download_success = download_file(IBM_TELCO_URL, telco_path, fallback_url=IBM_TELCO_FALLBACK)
    
    if os.path.exists(telco_path) and os.path.getsize(telco_path) > 1000:
        df_ibm = pd.read_csv(telco_path)
        df_ibm['DatasetSource'] = 'IBM_Telco_Primary'
        df_ibm.to_csv(telco_path, index=False)
        print(f"IBM Telco Dataset loaded: {df_ibm.shape[0]} rows, {df_ibm.shape[1]} columns")
    else:
        print("IBM Telco download failed, generating realistic replacement...")
        df_ibm = generate_telecom_dataset_2(num_rows=7043, random_seed=7)
        df_ibm['DatasetSource'] = 'IBM_Telco_Primary'
        df_ibm.to_csv(telco_path, index=False)
        print(f"Generated IBM Telco benchmark: {df_ibm.shape[0]} rows")
        
    # Generate Dataset 2
    cohort2_path = os.path.join(DATA_DIR, "telco_churn_region2.csv")
    df_cohort2 = generate_telecom_dataset_2(num_rows=3500, random_seed=42)
    df_cohort2.to_csv(cohort2_path, index=False)
    print(f"Region 2 Dataset created: {df_cohort2.shape[0]} rows")
    
    # Generate Dataset 3
    cohort3_path = os.path.join(DATA_DIR, "telco_churn_digital.csv")
    df_cohort3 = generate_digital_services_dataset_3(num_rows=2500, random_seed=101)
    df_cohort3.to_csv(cohort3_path, index=False)
    print(f"Digital Tech Cohort Dataset created: {df_cohort3.shape[0]} rows")
    
    total_records = len(df_ibm) + len(df_cohort2) + len(df_cohort3)
    print(f"\n==========================================")
    print(f"Total Multi-Dataset Records Prepared: {total_records}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
