import os
import pandas as pd
import numpy as np

REQUIRED_LOGICAL_COLUMNS = [
    "customer_id", "tenure", "monthly_charges", "total_charges",
    "contract", "payment_method", "internet_service"
]

COLUMN_SYNONYMS = {
    "customer_id": ["customerid", "customer_id", "cust_id", "id", "customer"],
    "gender": ["gender", "sex"],
    "senior_citizen": ["seniorcitizen", "senior_citizen", "senior", "is_senior"],
    "partner": ["partner", "has_partner"],
    "dependents": ["dependents", "has_dependents"],
    "tenure": ["tenure", "tenure_months", "months_active"],
    "phone_service": ["phoneservice", "phone_service", "phone"],
    "multiple_lines": ["multiplelines", "multiple_lines", "multilines"],
    "internet_service": ["internetservice", "internet_service", "internet_type", "internet"],
    "online_security": ["onlinesecurity", "online_security", "security"],
    "online_backup": ["onlinebackup", "online_backup", "backup"],
    "device_protection": ["deviceprotection", "device_protection", "protection"],
    "tech_support": ["techsupport", "tech_support", "support"],
    "streaming_tv": ["streamingtv", "streaming_tv", "tv"],
    "streaming_movies": ["streamingmovies", "streaming_movies", "movies"],
    "contract": ["contract", "contract_type", "contract_term"],
    "paperless_billing": ["paperlessbilling", "paperless_billing", "paperless"],
    "payment_method": ["paymentmethod", "payment_method", "payment_type", "payment"],
    "monthly_charges": ["monthlycharges", "monthly_charges", "monthly_charge", "monthly_spend"],
    "total_charges": ["totalcharges", "total_charges", "total_spend", "lifetime_charges"]
}

VALID_CATEGORIES = {
    "gender": ["Male", "Female", "Other"],
    "senior_citizen": [0, 1, "0", "1", "Yes", "No"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phone_service": ["Yes", "No"],
    "multiple_lines": ["Yes", "No", "No phone service"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["Yes", "No", "No internet service"],
    "online_backup": ["Yes", "No", "No internet service"],
    "device_protection": ["Yes", "No", "No internet service"],
    "tech_support": ["Yes", "No", "No internet service"],
    "streaming_tv": ["Yes", "No", "No internet service"],
    "streaming_movies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
}

def map_column_names(df_columns):
    """Maps arbitrary incoming column names to standardized lowercase keys."""
    mapping = {}
    normalized_cols = {str(col).strip().lower().replace(" ", "_"): col for col in df_columns}
    
    for std_key, synonyms in COLUMN_SYNONYMS.items():
        found = False
        for syn in synonyms:
            if syn in normalized_cols:
                mapping[normalized_cols[syn]] = std_key
                found = True
                break
        if not found:
            # Direct match check
            if std_key in normalized_cols:
                mapping[normalized_cols[std_key]] = std_key
                
    return mapping

def read_uploaded_file(file_storage_or_path):
    """Parses CSV or Excel file into a pandas DataFrame."""
    if isinstance(file_storage_or_path, str):
        filename = file_storage_or_path
        if filename.endswith(".csv"):
            df = pd.read_csv(file_storage_or_path)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_storage_or_path)
        else:
            raise ValueError("Unsupported file format. Please upload .csv or .xlsx file.")
    else:
        filename = file_storage_or_path.filename
        if filename.endswith(".csv"):
            df = pd.read_csv(file_storage_or_path)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_storage_or_path)
        else:
            raise ValueError("Unsupported file format. Please upload .csv or .xlsx file.")
    return df

def validate_dataframe(df, existing_customer_ids=None):
    """
    Performs comprehensive schema, data type, category, and range validation.
    Returns: (is_valid, report_dict, clean_valid_df, invalid_rows_list)
    """
    existing_ids = set(existing_customer_ids or [])
    col_mapping = map_column_names(df.columns)
    
    # Check for missing required columns
    mapped_std_cols = set(col_mapping.values())
    missing_required = [req for req in REQUIRED_LOGICAL_COLUMNS if req not in mapped_std_cols]
    
    if missing_required:
        return False, {
            "error": f"Missing required columns: {', '.join(missing_required)}",
            "total_rows": len(df),
            "valid_rows_count": 0,
            "invalid_rows_count": len(df),
            "is_valid_for_import": False,
            "missing_columns": missing_required
        }, None, []
        
    # Standardize column names
    df_renamed = df.rename(columns=col_mapping).copy()
    
    # Ensure default fields exist if optional ones were omitted
    defaults = {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 1.0,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 50.0,
        "total_charges": 50.0
    }
    for col, default_val in defaults.items():
        if col not in df_renamed.columns:
            df_renamed[col] = default_val

    seen_file_ids = set()
    valid_rows = []
    invalid_rows = []
    duplicate_count = 0
    missing_val_count = 0
    
    for idx, row in df_renamed.iterrows():
        issues = []
        cust_id = str(row.get("customer_id", "")).strip()
        
        # 1. Customer ID validation
        if not cust_id or cust_id.lower() == "nan" or cust_id.lower() == "none":
            cust_id = f"GEN-CUST-{idx+1:05d}"
            issues.append("Missing Customer ID (Auto-generated placeholder ID)")
            missing_val_count += 1
            
        if cust_id in seen_file_ids:
            issues.append(f"Duplicate Customer ID '{cust_id}' within file")
            duplicate_count += 1
        seen_file_ids.add(cust_id)
        
        if cust_id in existing_ids:
            issues.append(f"Customer ID '{cust_id}' already exists in company database (will update record)")
            
        # 2. Numerical validation
        try:
            tenure = float(row.get("tenure", 0))
            if tenure < 0 or tenure > 120:
                issues.append(f"Tenure ({tenure}) out of expected range [0, 120]")
        except (ValueError, TypeError):
            issues.append(f"Invalid non-numeric tenure value '{row.get('tenure')}'")
            tenure = 1.0
            
        try:
            monthly = float(row.get("monthly_charges", 0))
            if monthly < 0 or monthly > 500:
                issues.append(f"Monthly Charges (${monthly}) out of reasonable range [0, 500]")
        except (ValueError, TypeError):
            issues.append(f"Invalid non-numeric Monthly Charges '{row.get('monthly_charges')}'")
            monthly = 20.0
            
        try:
            total = float(row.get("total_charges", 0)) if pd.notnull(row.get("total_charges")) and str(row.get("total_charges")).strip() != "" else monthly * tenure
            if total < 0:
                issues.append(f"Total Charges (${total}) cannot be negative")
        except (ValueError, TypeError):
            total = monthly * tenure
            
        # 3. Categorical validation & normalization
        contract = str(row.get("contract", "Month-to-month")).strip()
        if "month" in contract.lower():
            contract = "Month-to-month"
        elif "two" in contract.lower() or "2" in contract.lower():
            contract = "Two year"
        elif "one" in contract.lower() or "1" in contract.lower() or "annual" in contract.lower():
            contract = "One year"
            
        payment = str(row.get("payment_method", "Electronic check")).strip()
        if "electronic" in payment.lower() or "e-check" in payment.lower():
            payment = "Electronic check"
        elif "mailed" in payment.lower() or "mail" in payment.lower():
            payment = "Mailed check"
        elif "bank" in payment.lower() or "transfer" in payment.lower():
            payment = "Bank transfer (automatic)"
        elif "card" in payment.lower() or "credit" in payment.lower():
            payment = "Credit card (automatic)"
            
        senior = row.get("senior_citizen", 0)
        try:
            senior = 1 if int(senior) == 1 or str(senior).lower() in ["yes", "true", "1"] else 0
        except Exception:
            senior = 0
            
        clean_record = {
            "customer_id": cust_id,
            "gender": "Female" if str(row.get("gender", "")).lower().startswith("f") else "Male",
            "senior_citizen": senior,
            "partner": "Yes" if str(row.get("partner", "")).lower() in ["yes", "1", "true"] else "No",
            "dependents": "Yes" if str(row.get("dependents", "")).lower() in ["yes", "1", "true"] else "No",
            "tenure": max(0.0, float(tenure)),
            "phone_service": "No" if str(row.get("phone_service", "")).lower() in ["no", "0", "false"] else "Yes",
            "multiple_lines": "Yes" if str(row.get("multiple_lines", "")).lower() in ["yes", "true"] else ("No phone service" if str(row.get("multiple_lines", "")).lower() == "no phone service" else "No"),
            "internet_service": "Fiber optic" if "fiber" in str(row.get("internet_service", "")).lower() else ("No" if str(row.get("internet_service", "")).lower() == "no" else "DSL"),
            "online_security": "Yes" if str(row.get("online_security", "")).lower() in ["yes", "true"] else "No",
            "online_backup": "Yes" if str(row.get("online_backup", "")).lower() in ["yes", "true"] else "No",
            "device_protection": "Yes" if str(row.get("device_protection", "")).lower() in ["yes", "true"] else "No",
            "tech_support": "Yes" if str(row.get("tech_support", "")).lower() in ["yes", "true"] else "No",
            "streaming_tv": "Yes" if str(row.get("streaming_tv", "")).lower() in ["yes", "true"] else "No",
            "streaming_movies": "Yes" if str(row.get("streaming_movies", "")).lower() in ["yes", "true"] else "No",
            "contract": contract,
            "paperless_billing": "No" if str(row.get("paperless_billing", "")).lower() in ["no", "0", "false"] else "Yes",
            "payment_method": payment,
            "monthly_charges": round(float(monthly), 2),
            "total_charges": round(float(total), 2)
        }
        
        # Hard errors (fatal invalid data) vs Soft auto-corrections
        hard_errors = [iss for iss in issues if "non-numeric" in iss or "Duplicate Customer ID" in iss]
        if hard_errors:
            invalid_rows.append({
                "row_index": idx + 1,
                "customer_id": cust_id,
                "issues": issues,
                "raw_data": {k: str(v) for k, v in row.items()}
            })
        else:
            valid_rows.append(clean_record)
            
    valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()
    
    report = {
        "total_rows": len(df),
        "valid_rows_count": len(valid_rows),
        "invalid_rows_count": len(invalid_rows),
        "duplicate_ids_count": duplicate_count,
        "missing_values_count": missing_val_count,
        "is_valid_for_import": len(valid_rows) > 0,
        "sample_valid": valid_rows[:5],
        "invalid_samples": invalid_rows[:10],
        "mapped_columns": col_mapping
    }
    
    return True, report, valid_df, invalid_rows
