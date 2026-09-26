import json
import os
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
from config import Config
from app.models.database_models import Company, Customer, Prediction, ModelTracking
from app.services.shap_service import shap_service

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
def index():
    has_active_data = session.get("has_active_data", False)
    selected_company_id = session.get("company_id")
    dataset_name = session.get("dataset_name", "My Customer Dataset")
    
    # If no active data in user session, show clean Welcome & Upload onboarding portal
    if not has_active_data or not selected_company_id:
        return render_template(
            "landing_upload.html",
            has_active_data=False,
            dataset_name=None
        )
        
    current_company = Company.query.filter_by(company_id=selected_company_id).first()
    
    # Compute Dashboard Aggregates for THIS user's active dataset
    total_customers = Customer.query.filter_by(company_id=selected_company_id).count()
    if total_customers == 0:
        return render_template(
            "landing_upload.html",
            has_active_data=False,
            dataset_name=None
        )
        
    predictions = Prediction.query.filter_by(company_id=selected_company_id).all()
    
    high_risk_count = sum(1 for p in predictions if p.risk_level == "HIGH")
    medium_risk_count = sum(1 for p in predictions if p.risk_level == "MEDIUM")
    low_risk_count = sum(1 for p in predictions if p.risk_level == "LOW")
    
    avg_churn_prob = (
        sum(p.churn_probability for p in predictions) / len(predictions) * 100
    ) if predictions else 0.0
    
    # Revenue at risk calculations
    high_risk_ids = [p.customer_id for p in predictions if p.risk_level == "HIGH"]
    high_risk_customers = Customer.query.filter(
        Customer.company_id == selected_company_id,
        Customer.customer_id.in_(high_risk_ids)
    ).all()
    monthly_rev_at_risk = sum(c.monthly_charges for c in high_risk_customers)
    
    # Top Churn Drivers specifically computed from active dataset
    global_drivers = shap_service.get_dataset_churn_drivers(selected_company_id)[:6]
    
    # Urgent high risk alerts (Top 8)
    top_urgent_preds = (
        Prediction.query.filter(
            Prediction.company_id == selected_company_id,
            Prediction.risk_level == "HIGH"
        )
        .order_by(Prediction.churn_probability.desc())
        .limit(8)
        .all()
    )
    
    urgent_customers = []
    for p in top_urgent_preds:
        c = Customer.query.filter_by(company_id=selected_company_id, customer_id=p.customer_id).first()
        if c:
            urgent_customers.append({
                "customer_id": c.customer_id,
                "churn_probability": round(p.churn_probability * 100, 1),
                "tenure": int(c.tenure),
                "monthly_charges": c.monthly_charges,
                "contract": c.contract,
                "payment_method": c.payment_method,
                "internet_service": c.internet_service
            })
            
    return render_template(
        "dashboard.html",
        has_active_data=True,
        dataset_name=dataset_name,
        current_company=current_company,
        total_customers=total_customers,
        high_risk_count=high_risk_count,
        medium_risk_count=medium_risk_count,
        low_risk_count=low_risk_count,
        avg_churn_prob=round(avg_churn_prob, 1),
        monthly_rev_at_risk=round(monthly_rev_at_risk, 2),
        global_drivers=global_drivers,
        urgent_customers=urgent_customers
    )

@dashboard_bp.route("/models")
def model_benchmarking():
    """Displays MLOps benchmark comparisons, hyperparameters, and drift metrics."""
    has_active_data = session.get("has_active_data", False)
    dataset_name = session.get("dataset_name", "My Dataset")
    selected_company_id = session.get("company_id")
    current_company = Company.query.filter_by(company_id=selected_company_id).first() if selected_company_id else None
    
    tracked_models = ModelTracking.query.order_by(ModelTracking.roc_auc.desc()).all()
    
    metrics_data = {}
    if os.path.exists(Config.METRICS_PATH):
        with open(Config.METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
            
    return render_template(
        "model_info.html",
        has_active_data=has_active_data,
        dataset_name=dataset_name,
        current_company=current_company,
        tracked_models=tracked_models,
        metrics_data=metrics_data
    )
