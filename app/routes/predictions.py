import io
import csv
from flask import Blueprint, request, jsonify, session, Response
from app.models.database_models import db, Customer, Prediction, Company
from app.services.prediction_service import prediction_service
from app.services.shap_service import shap_service
from app.services.what_if_service import what_if_service
from app.services.recommendation_service import recommendation_service

predictions_bp = Blueprint("predictions_api", __name__, url_prefix="/api")

@predictions_bp.route("/predict", methods=["POST"])
def api_predict_single():
    data = request.get_json() or request.form.to_dict()
    if not data:
        return jsonify({"success": False, "error": "No input payload provided"}), 400
        
    model_choice = data.get("model_choice", "champion_ensemble")
    try:
        res = prediction_service.predict_single(data, model_key=model_choice)
        return jsonify({
            "success": True,
            "prediction": {
                "churn_probability": res["churn_percentage"],
                "retention_probability": res["retention_percentage"],
                "risk_level": res["risk_level"],
                "model_version": res["model_version"]
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@predictions_bp.route("/customer/<customer_id>", methods=["GET"])
def api_get_customer(customer_id):
    company_id = request.args.get("company_id") or session.get("company_id")
    query = Customer.query.filter_by(customer_id=customer_id)
    if company_id:
        query = query.filter_by(company_id=company_id)
    customer = query.first()
    
    if not customer:
        return jsonify({"success": False, "error": f"Customer '{customer_id}' not found."}), 404
        
    cust_dict = customer.to_dict()
    latest_pred = Prediction.query.filter_by(
        company_id=customer.company_id,
        customer_id=customer.customer_id
    ).order_by(Prediction.prediction_date.desc()).first()
    
    return jsonify({
        "success": True,
        "customer": cust_dict,
        "prediction": latest_pred.to_dict() if latest_pred else None
    })

@predictions_bp.route("/customer/<customer_id>/explanation", methods=["GET"])
def api_get_explanation(customer_id):
    company_id = request.args.get("company_id") or session.get("company_id")
    query = Customer.query.filter_by(customer_id=customer_id)
    if company_id:
        query = query.filter_by(company_id=company_id)
    customer = query.first()
    
    if not customer:
        return jsonify({"success": False, "error": "Customer not found."}), 404
        
    cust_dict = customer.to_dict()
    shap_factors = shap_service.explain_customer(cust_dict, save_to_db=False)
    return jsonify({
        "success": True,
        "customer_id": customer_id,
        "shap_factors": shap_factors
    })

@predictions_bp.route("/customer/<customer_id>/what-if", methods=["POST"])
def api_what_if_simulation(customer_id):
    company_id = request.args.get("company_id") or session.get("company_id")
    customer = Customer.query.filter_by(customer_id=customer_id).first()
    if not customer:
        return jsonify({"success": False, "error": "Customer not found."}), 404
        
    payload = request.get_json() or {}
    modified_fields = payload.get("modifications", {})
    model_choice = payload.get("model_choice", "champion_ensemble")
    
    sim_result = what_if_service.simulate_scenario(
        customer.to_dict(),
        modified_fields=modified_fields,
        model_key=model_choice
    )
    return jsonify({"success": True, "simulation": sim_result})

@predictions_bp.route("/customer/<customer_id>/recommendation", methods=["POST"])
def api_generate_recommendation(customer_id):
    customer = Customer.query.filter_by(customer_id=customer_id).first()
    if not customer:
        return jsonify({"success": False, "error": "Customer not found."}), 404
        
    cust_dict = customer.to_dict()
    pred_res = prediction_service.predict_single(cust_dict)
    shap_factors = shap_service.explain_customer(cust_dict, save_to_db=False)
    
    rec_res = recommendation_service.generate_retention_strategy(
        cust_dict,
        churn_probability=pred_res["churn_probability"],
        risk_level=pred_res["risk_level"],
        shap_factors=shap_factors,
        save_to_db=True
    )
    return jsonify({"success": True, "recommendation": rec_res})

@predictions_bp.route("/dashboard/summary", methods=["GET"])
def api_dashboard_summary():
    company_id = request.args.get("company_id") or session.get("company_id", "APEX-GLOBAL")
    total = Customer.query.filter_by(company_id=company_id).count()
    preds = Prediction.query.filter_by(company_id=company_id).all()
    
    high = sum(1 for p in preds if p.risk_level == "HIGH")
    med = sum(1 for p in preds if p.risk_level == "MEDIUM")
    low = sum(1 for p in preds if p.risk_level == "LOW")
    avg = (sum(p.churn_probability for p in preds) / len(preds) * 100) if preds else 0.0
    
    return jsonify({
        "success": True,
        "company_id": company_id,
        "total_customers": total,
        "high_risk_count": high,
        "medium_risk_count": med,
        "low_risk_count": low,
        "average_churn_probability": round(avg, 2)
    })

@predictions_bp.route("/dashboard/risk-distribution", methods=["GET"])
def api_risk_distribution():
    company_id = request.args.get("company_id") or session.get("company_id", "APEX-GLOBAL")
    preds = Prediction.query.filter_by(company_id=company_id).all()
    
    high = sum(1 for p in preds if p.risk_level == "HIGH")
    med = sum(1 for p in preds if p.risk_level == "MEDIUM")
    low = sum(1 for p in preds if p.risk_level == "LOW")
    
    return jsonify({
        "success": True,
        "labels": ["High Risk (>=65%)", "Medium Risk (35-65%)", "Low Risk (<35%)"],
        "counts": [high, med, low],
        "colors": ["#ef4444", "#f59e0b", "#10b981"]
    })

@predictions_bp.route("/dashboard/churn-drivers", methods=["GET"])
def api_churn_drivers():
    drivers = shap_service.get_global_churn_drivers()
    return jsonify({"success": True, "drivers": drivers})

@predictions_bp.route("/export/predictions", methods=["GET"])
def export_predictions_csv():
    """Generates streaming CSV export of all customer churn predictions."""
    company_id = request.args.get("company_id") or session.get("company_id", "APEX-GLOBAL")
    
    results = (
        db.session.query(Customer, Prediction)
        .outerjoin(Prediction, Customer.customer_id == Prediction.customer_id)
        .filter(Customer.company_id == company_id)
        .order_by(Prediction.churn_probability.desc().nullslast())
        .all()
    )
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Customer_ID", "Company_ID", "Tenure_Months", "Monthly_Charges", "Total_Charges",
        "Contract", "Payment_Method", "Internet_Service", "Tech_Support", "Online_Security",
        "Churn_Probability_Pct", "Risk_Level", "Model_Version"
    ])
    
    for c, p in results:
        writer.writerow([
            c.customer_id,
            c.company_id,
            c.tenure,
            c.monthly_charges,
            c.total_charges,
            c.contract,
            c.payment_method,
            c.internet_service,
            c.tech_support,
            c.online_security,
            f"{p.churn_probability*100:.2f}%" if p else "N/A",
            p.risk_level if p else "UNSCORED",
            p.model_version if p else "N/A"
        ])
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=churn_predictions_{company_id}.csv"}
    )
