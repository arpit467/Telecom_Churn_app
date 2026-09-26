import os
import io
import uuid
import pandas as pd
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify, flash, send_file, Response
from werkzeug.utils import secure_filename
from config import Config
from app.models.database_models import db, Company, Customer, Prediction
from app.services.validation_service import read_uploaded_file, validate_dataframe
from app.services.prediction_service import prediction_service

upload_bp = Blueprint("upload", __name__)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

@upload_bp.route("/upload", methods=["GET"])
def upload_page():
    has_active_data = session.get("has_active_data", False)
    company_id = session.get("company_id")
    dataset_name = session.get("dataset_name", "My Dataset")
    
    current_company = None
    if company_id:
        current_company = Company.query.filter_by(company_id=company_id).first()
        
    return render_template(
        "landing_upload.html",
        has_active_data=has_active_data,
        dataset_name=dataset_name,
        current_company=current_company
    )

@upload_bp.route("/download/sample-template", methods=["GET"])
def download_sample_template():
    """Generates and serves a clean, ready-to-use sample CSV template for telecom customer churn."""
    sample_records = [
        {
            "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
        },
        {
            "customerID": "5575-GNVDE", "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 34, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "Yes", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check", "MonthlyCharges": 56.95, "TotalCharges": 1889.50
        },
        {
            "customerID": "3668-QPYBK", "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check", "MonthlyCharges": 53.85, "TotalCharges": 108.15
        },
        {
            "customerID": "7795-CFOCW", "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 45, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 42.30, "TotalCharges": 1840.75
        },
        {
            "customerID": "9237-HQITU", "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 70.70, "TotalCharges": 151.65
        },
        {
            "customerID": "9305-CDSKC", "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 8, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "Yes", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 99.65, "TotalCharges": 820.50
        },
        {
            "customerID": "1452-KIOVK", "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "Yes",
            "tenure": 22, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 89.10, "TotalCharges": 1949.40
        },
        {
            "customerID": "6713-OKOMC", "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 10, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check", "MonthlyCharges": 29.75, "TotalCharges": 301.90
        },
        {
            "customerID": "7892-POOKP", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 28, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 104.80, "TotalCharges": 3046.05
        },
        {
            "customerID": "6388-TABGU", "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "Yes",
            "tenure": 62, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 56.15, "TotalCharges": 3487.95
        }
    ]
    df_sample = pd.DataFrame(sample_records)
    output = io.BytesIO()
    df_sample.to_csv(output, index=False)
    output.seek(0)
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=sample_telecom_churn_data.csv"}
    )

@upload_bp.route("/demo/load-sample", methods=["GET", "POST"])
def load_sample_dataset():
    """1-Click demo dataset loader so users can immediately explore the full platform."""
    demo_company_id = "USER-DEMO-DATASET"
    demo_name = "Sample Telecom Dataset (Demo Sandbox)"
    
    # Ensure company exists
    company = Company.query.filter_by(company_id=demo_company_id).first()
    if not company:
        company = Company(company_id=demo_company_id, company_name=demo_name)
        db.session.add(company)
        db.session.commit()
        
    # Check if customers exist, otherwise copy from baseline
    existing_count = Customer.query.filter_by(company_id=demo_company_id).count()
    if existing_count == 0:
        seed_source = Customer.query.filter_by(company_id="APEX-GLOBAL").limit(150).all()
        if not seed_source:
            seed_source = Customer.query.limit(150).all()
            
        cust_list = []
        for s in seed_source:
            new_c = Customer(
                customer_id=f"DEMO-{s.customer_id}",
                company_id=demo_company_id,
                gender=s.gender,
                senior_citizen=s.senior_citizen,
                partner=s.partner,
                dependents=s.dependents,
                tenure=s.tenure,
                phone_service=s.phone_service,
                multiple_lines=s.multiple_lines,
                internet_service=s.internet_service,
                online_security=s.online_security,
                online_backup=s.online_backup,
                device_protection=s.device_protection,
                tech_support=s.tech_support,
                streaming_tv=s.streaming_tv,
                streaming_movies=s.streaming_movies,
                contract=s.contract,
                paperless_billing=s.paperless_billing,
                payment_method=s.payment_method,
                monthly_charges=s.monthly_charges,
                total_charges=s.total_charges
            )
            db.session.add(new_c)
            cust_list.append(new_c.to_dict())
            
        db.session.commit()
        prediction_service.predict_batch_and_save(demo_company_id, cust_list)
        
    session["company_id"] = demo_company_id
    session["dataset_name"] = demo_name
    session["has_active_data"] = True
    
    flash("✨ Loaded demo telecom dataset (150 customer accounts). Explore your AI retention insights below!", "success")
    return redirect(url_for("dashboard.index"))

@upload_bp.route("/workspace/reset", methods=["GET", "POST"])
def reset_workspace():
    """Clears active dataset from session so the user can upload a fresh new dataset."""
    session.pop("company_id", None)
    session.pop("dataset_name", None)
    session["has_active_data"] = False
    flash("Workspace reset. You can now upload a new dataset.", "info")
    return redirect(url_for("upload.upload_page"))

@upload_bp.route("/upload/validate", methods=["POST"])
def validate_upload():
    """Validates uploaded CSV/Excel and returns an instant health check preview."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400
        
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "error": "Invalid format. Please upload a .csv or .xlsx file."}), 400
        
    # Generate unique workspace token
    workspace_token = f"USER_{uuid.uuid4().hex[:8].upper()}"
    temp_path = os.path.join(Config.UPLOAD_FOLDER, f"temp_{workspace_token}_{filename}")
    file.save(temp_path)
    
    try:
        df_raw = read_uploaded_file(temp_path)
        is_valid, report, valid_df, invalid_rows = validate_dataframe(df_raw)
        
        if is_valid and valid_df is not None:
            cache_path = os.path.join(Config.UPLOAD_FOLDER, f"cache_{workspace_token}.csv")
            valid_df.to_csv(cache_path, index=False)
            report["cache_token"] = workspace_token
            report["filename"] = filename
            
        return jsonify({
            "success": True,
            "filename": filename,
            "report": report
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to process file: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception: pass

@upload_bp.route("/upload/import", methods=["POST"])
def import_valid_records():
    """Imports validated records into SQL database, runs batch ML prediction, and activates user workspace."""
    cache_token = request.form.get("cache_token", "").strip()
    dataset_name = request.form.get("dataset_name", "").strip()
    
    if not cache_token:
        flash("Upload session expired. Please upload your file again.", "warning")
        return redirect(url_for("upload.upload_page"))
        
    cache_path = os.path.join(Config.UPLOAD_FOLDER, f"cache_{cache_token}.csv")
    if not os.path.exists(cache_path):
        flash("Uploaded data cache not found. Please upload your file again.", "warning")
        return redirect(url_for("upload.upload_page"))
        
    company_id = f"WORKSPACE-{cache_token}"
    if not dataset_name:
        dataset_name = f"Uploaded Dataset ({cache_token})"
        
    # Create or update Company workspace
    company = Company.query.filter_by(company_id=company_id).first()
    if not company:
        company = Company(company_id=company_id, company_name=dataset_name)
        db.session.add(company)
    else:
        company.company_name = dataset_name
        
    db.session.commit()
    
    valid_df = pd.read_csv(cache_path)
    # Deduplicate within file if same customer_id appears multiple times
    valid_df = valid_df.drop_duplicates(subset=["customer_id"], keep="last")
    cust_list = valid_df.to_dict(orient="records")
    
    # Save Customer records safely
    seen_ids = set()
    for c_data in cust_list:
        cust_id = str(c_data["customer_id"]).strip()
        if cust_id in seen_ids:
            continue
        seen_ids.add(cust_id)
        
        existing = Customer.query.filter_by(company_id=company_id, customer_id=cust_id).first()
        
        if existing:
            for k, v in c_data.items():
                if hasattr(existing, k) and k not in ["id", "customer_id", "company_id"]:
                    setattr(existing, k, v)
        else:
            new_cust = Customer(
                customer_id=cust_id,
                company_id=company_id,
                gender=c_data.get("gender", "Male"),
                senior_citizen=int(c_data.get("senior_citizen", 0)),
                partner=c_data.get("partner", "No"),
                dependents=c_data.get("dependents", "No"),
                tenure=float(c_data.get("tenure", 1.0)),
                phone_service=c_data.get("phone_service", "Yes"),
                multiple_lines=c_data.get("multiple_lines", "No"),
                internet_service=c_data.get("internet_service", "DSL"),
                online_security=c_data.get("online_security", "No"),
                online_backup=c_data.get("online_backup", "No"),
                device_protection=c_data.get("device_protection", "No"),
                tech_support=c_data.get("tech_support", "No"),
                streaming_tv=c_data.get("streaming_tv", "No"),
                streaming_movies=c_data.get("streaming_movies", "No"),
                contract=c_data.get("contract", "Month-to-month"),
                paperless_billing=c_data.get("paperless_billing", "Yes"),
                payment_method=c_data.get("payment_method", "Electronic check"),
                monthly_charges=float(c_data.get("monthly_charges", 50.0)),
                total_charges=float(c_data.get("total_charges", 50.0))
            )
            db.session.add(new_cust)
            
    db.session.commit()
    
    # Run Batch ML Prediction
    batch_results = prediction_service.predict_batch_and_save(company_id, cust_list)
    
    # Invalidate RAG cache for this company
    from app.services.rag_service import rag_service
    rag_service.invalidate_cache(company_id)
    
    high_risk_count = sum(1 for r in batch_results if r["risk_level"] == "HIGH")
    medium_risk_count = sum(1 for r in batch_results if r["risk_level"] == "MEDIUM")
    low_risk_count = sum(1 for r in batch_results if r["risk_level"] == "LOW")
    
    try: os.remove(cache_path)
    except Exception: pass
    
    # Set Session State
    session["company_id"] = company_id
    session["dataset_name"] = dataset_name
    session["has_active_data"] = True
    
    flash(
        f"🎉 Successfully imported and analyzed {len(cust_list):,} customers for '{dataset_name}'! "
        f"Identified {high_risk_count} High-Risk, {medium_risk_count} Medium-Risk, and {low_risk_count} Low-Risk accounts.",
        "success"
    )
    return redirect(url_for("dashboard.index"))
