import os
import json
import pandas as pd
from datetime import datetime
from config import Config
from app.models.database_models import (
    db, Company, Customer, Prediction, CustomerExplanation,
    RetentionRecommendation, ModelTracking
)
from app.services.prediction_service import prediction_service
from app.services.shap_service import shap_service
from app.services.recommendation_service import recommendation_service
from app.services.validation_service import validate_dataframe

def seed_database_if_empty(app):
    """
    Initializes database tables, creates default multi-tenant companies,
    and populates customer records with pre-computed ML predictions & SHAP explanations.
    """
    with app.app_context():
        db.create_all()
        
        # Check if companies exist
        if Company.query.first() is not None:
            return # Already seeded
            
        print("Initializing enterprise database & seeding multi-tenant telecom data...")
        
        # 1. Create Companies
        companies = [
            Company(company_id="APEX-GLOBAL", company_name="Apex Telecom Global"),
            Company(company_id="METRO-FIBER", company_name="MetroFiber Broadband & Mobile"),
            Company(company_id="SKYLINE-NET", company_name="Skyline Digital Communications")
        ]
        db.session.add_all(companies)
        db.session.commit()
        
        # 2. Seed Model Benchmark Tracking Info
        metrics_file = Config.METRICS_PATH
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                
            for idx, m in enumerate(metrics_data.get("models", [])):
                is_prod = "Champion" in m["model_name"]
                track = ModelTracking(
                    run_id=f"RUN-2026-09-00{idx+1}",
                    model_name=m["model_name"],
                    version=f"v2.4.0-{m['model_name'].lower().replace(' ', '_')[:12]}",
                    accuracy=m["accuracy"],
                    precision=m["precision"],
                    recall=m["recall"],
                    f1_score=m["f1_score"],
                    roc_auc=m["roc_auc"],
                    pr_auc=m["pr_auc"],
                    hyperparameters={
                        "optimal_threshold": m.get("optimal_threshold", 0.5),
                        "brier_score": m.get("brier_score", 0.16)
                    },
                    trained_at=datetime.utcnow(),
                    is_production=is_prod
                )
                db.session.add(track)
            db.session.commit()
            
        # 3. Load sample customer cohorts from data/
        data_sources = [
            ("APEX-GLOBAL", "data/telco_churn_ibm.csv", 350),
            ("METRO-FIBER", "data/telco_churn_region2.csv", 200),
            ("SKYLINE-NET", "data/telco_churn_digital.csv", 150)
        ]
        
        for comp_id, file_rel, sample_size in data_sources:
            file_path = os.path.join(Config.BASE_DIR, file_rel)
            if not os.path.exists(file_path):
                # Fallback path
                file_path = os.path.join(Config.BASE_DIR, "..", file_rel)
                
            if os.path.exists(file_path):
                df_raw = pd.read_csv(file_path).head(sample_size)
                is_valid, report, valid_df, _ = validate_dataframe(df_raw)
                
                if is_valid and valid_df is not None:
                    cust_list = valid_df.to_dict(orient="records")
                    
                    # Add customers to DB
                    for cust_data in cust_list:
                        c = Customer(
                            customer_id=cust_data["customer_id"],
                            company_id=comp_id,
                            gender=cust_data["gender"],
                            senior_citizen=cust_data["senior_citizen"],
                            partner=cust_data["partner"],
                            dependents=cust_data["dependents"],
                            tenure=cust_data["tenure"],
                            phone_service=cust_data["phone_service"],
                            multiple_lines=cust_data["multiple_lines"],
                            internet_service=cust_data["internet_service"],
                            online_security=cust_data["online_security"],
                            online_backup=cust_data["online_backup"],
                            device_protection=cust_data["device_protection"],
                            tech_support=cust_data["tech_support"],
                            streaming_tv=cust_data["streaming_tv"],
                            streaming_movies=cust_data["streaming_movies"],
                            contract=cust_data["contract"],
                            paperless_billing=cust_data["paperless_billing"],
                            payment_method=cust_data["payment_method"],
                            monthly_charges=cust_data["monthly_charges"],
                            total_charges=cust_data["total_charges"]
                        )
                        db.session.add(c)
                    db.session.commit()
                    
                    # Run Batch Prediction & Pre-compute Top SHAP Explanations
                    prediction_service.predict_batch_and_save(comp_id, cust_list)
                    
                    # Generate deep SHAP explanations & recommendations for first 25 customers of company
                    sample_subset = cust_list[:25]
                    for cust_data in sample_subset:
                        pred_record = Prediction.query.filter_by(
                            company_id=comp_id,
                            customer_id=cust_data["customer_id"]
                        ).first()
                        
                        pred_id = pred_record.prediction_id if pred_record else None
                        shap_factors = shap_service.explain_customer(
                            cust_data,
                            prediction_id=pred_id,
                            save_to_db=True
                        )
                        
                        prob = pred_record.churn_probability if pred_record else 0.5
                        risk = pred_record.risk_level if pred_record else "MEDIUM"
                        
                        recommendation_service.generate_retention_strategy(
                            cust_data,
                            churn_probability=prob,
                            risk_level=risk,
                            shap_factors=shap_factors,
                            prediction_id=pred_id,
                            save_to_db=True
                        )
                        
        print("Enterprise database initialized and multi-tenant data seeded successfully!")
