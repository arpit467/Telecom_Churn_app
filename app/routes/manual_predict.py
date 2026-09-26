from flask import Blueprint, render_template, request, session
from app.models.database_models import Company
from app.services.prediction_service import prediction_service
from app.services.shap_service import shap_service
from app.services.recommendation_service import recommendation_service

manual_predict_bp = Blueprint("manual_predict", __name__)

@manual_predict_bp.route("/manual-prediction", methods=["GET", "POST"])
def manual_prediction_page():
    has_active_data = session.get("has_active_data", False)
    dataset_name = session.get("dataset_name", "My Dataset")
    selected_company_id = session.get("company_id")
    current_company = Company.query.filter_by(company_id=selected_company_id).first() if selected_company_id else None
    
    prediction_result = None
    shap_factors = None
    ai_strategy = None
    
    if request.method == "POST":
        raw_dict = {
            "customer_id": request.form.get("customer_id", "SIMULATED-01"),
            "gender": request.form.get("gender", "Male"),
            "senior_citizen": request.form.get("senior", "0"),
            "partner": request.form.get("partner", "No"),
            "dependents": request.form.get("dependents", "No"),
            "tenure": request.form.get("tenure", "6"),
            "phone_service": request.form.get("phone", "Yes"),
            "multiple_lines": request.form.get("multiple", "No"),
            "internet_service": request.form.get("internet", "Fiber optic"),
            "online_security": request.form.get("security", "No"),
            "online_backup": request.form.get("backup", "No"),
            "device_protection": request.form.get("protection", "No"),
            "tech_support": request.form.get("support", "No"),
            "streaming_tv": request.form.get("tv", "Yes"),
            "streaming_movies": request.form.get("movies", "Yes"),
            "contract": request.form.get("contract", "Month-to-month"),
            "paperless_billing": request.form.get("paperless", "Yes"),
            "payment_method": request.form.get("payment", "Electronic check"),
            "monthly_charges": request.form.get("monthly", "85.0"),
            "total_charges": request.form.get("total", "510.0")
        }
        
        model_choice = request.form.get("model_choice", "champion_ensemble")
        pred_res = prediction_service.predict_single(raw_dict, model_key=model_choice)
        shap_factors = shap_service.explain_customer(raw_dict, save_to_db=False)
        ai_strategy = recommendation_service.generate_retention_strategy(
            raw_dict,
            churn_probability=pred_res["churn_probability"],
            risk_level=pred_res["risk_level"],
            shap_factors=shap_factors,
            save_to_db=False
        )
        
        prediction_result = {
            "inputs": raw_dict,
            "prediction": pred_res,
            "model_choice": model_choice
        }
        
    return render_template(
        "manual_predict.html",
        has_active_data=has_active_data,
        dataset_name=dataset_name,
        current_company=current_company,
        result=prediction_result,
        shap_factors=shap_factors,
        ai_strategy=ai_strategy
    )
