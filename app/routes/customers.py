from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from app.models.database_models import db, Company, Customer, Prediction, CustomerExplanation, RetentionRecommendation
from app.services.prediction_service import prediction_service
from app.services.shap_service import shap_service
from app.services.recommendation_service import recommendation_service
from app.services.what_if_service import what_if_service

customers_bp = Blueprint("customers", __name__)

@customers_bp.route("/customers")
def customer_list():
    has_active_data = session.get("has_active_data", False)
    selected_company_id = session.get("company_id")
    dataset_name = session.get("dataset_name", "My Dataset")
    
    if not has_active_data or not selected_company_id:
        return render_template(
            "customers_empty.html",
            has_active_data=False,
            dataset_name=None
        )
        
    current_company = Company.query.filter_by(company_id=selected_company_id).first()
    total_in_db = Customer.query.filter_by(company_id=selected_company_id).count()
    
    if total_in_db == 0:
        return render_template(
            "customers_empty.html",
            has_active_data=False,
            dataset_name=None
        )
        
    # Multi-parameter filter parsing
    search_query = request.args.get("q", "").strip()
    risk_filter = request.args.get("risk", "ALL").strip().upper()
    if risk_filter not in ["ALL", "HIGH", "MEDIUM", "LOW"]:
        risk_filter = "ALL"
        
    contract_filter = request.args.get("contract", "ALL").strip()
    internet_filter = request.args.get("internet", "ALL").strip()
    tech_support_filter = request.args.get("tech_support", "ALL").strip()
    payment_filter = request.args.get("payment", "ALL").strip()
    tenure_range_filter = request.args.get("tenure_range", "ALL").strip()
    charges_range_filter = request.args.get("charges_range", "ALL").strip()
    senior_filter = request.args.get("senior_citizen", "ALL").strip()
    sort_by = request.args.get("sort", "churn_desc").strip()
    page = request.args.get("page", 1, type=int)
    per_page = 20
    
    # Query builder strictly isolated to selected company dataset
    query = db.session.query(Customer, Prediction).outerjoin(
        Prediction,
        (Customer.customer_id == Prediction.customer_id) & (Customer.company_id == Prediction.company_id)
    ).filter(Customer.company_id == selected_company_id)
    
    if search_query:
        query = query.filter(Customer.customer_id.ilike(f"%{search_query}%"))
        
    if risk_filter in ["HIGH", "MEDIUM", "LOW"]:
        query = query.filter(Prediction.risk_level == risk_filter)
        
    if contract_filter and contract_filter != "ALL":
        query = query.filter(Customer.contract == contract_filter)
        
    if internet_filter and internet_filter != "ALL":
        if internet_filter == "No":
            query = query.filter(Customer.internet_service == "No")
        else:
            query = query.filter(Customer.internet_service.ilike(f"%{internet_filter}%"))
            
    if tech_support_filter and tech_support_filter != "ALL":
        query = query.filter(Customer.tech_support == tech_support_filter)
        
    if payment_filter and payment_filter != "ALL":
        query = query.filter(Customer.payment_method.ilike(f"%{payment_filter}%"))
        
    if senior_filter and senior_filter in ["0", "1"]:
        query = query.filter(Customer.senior_citizen == int(senior_filter))
        
    if tenure_range_filter == "0-12":
        query = query.filter(Customer.tenure <= 12)
    elif tenure_range_filter == "13-36":
        query = query.filter(Customer.tenure > 12, Customer.tenure <= 36)
    elif tenure_range_filter == "37-72":
        query = query.filter(Customer.tenure > 36)
        
    if charges_range_filter == "0-35":
        query = query.filter(Customer.monthly_charges < 35)
    elif charges_range_filter == "35-75":
        query = query.filter(Customer.monthly_charges >= 35, Customer.monthly_charges <= 75)
    elif charges_range_filter == "75+":
        query = query.filter(Customer.monthly_charges > 75)

    # Calculate summary metrics on matching records
    matching_total = query.count()
    
    # Risk distribution of filtered results
    risk_distribution_query = query.with_entities(
        Prediction.risk_level,
        db.func.count(Customer.customer_id),
        db.func.avg(Prediction.churn_probability)
    ).group_by(Prediction.risk_level).all()
    
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNSCORED": 0}
    total_prob_sum = 0.0
    scored_count = 0
    
    for r_level, count, avg_p in risk_distribution_query:
        if r_level in risk_counts:
            risk_counts[r_level] = count
        if avg_p is not None:
            total_prob_sum += (avg_p * count)
            scored_count += count
            
    avg_matching_churn = round((total_prob_sum / scored_count) * 100, 1) if scored_count > 0 else 0.0
    
    # Active filters list for chips UI
    active_filters = []
    if search_query:
        active_filters.append({"label": f"ID: {search_query}", "key": "q"})
    if risk_filter != "ALL":
        active_filters.append({"label": f"Risk: {risk_filter}", "key": "risk"})
    if contract_filter != "ALL":
        active_filters.append({"label": f"Contract: {contract_filter}", "key": "contract"})
    if internet_filter != "ALL":
        active_filters.append({"label": f"Internet: {internet_filter}", "key": "internet"})
    if tech_support_filter != "ALL":
        active_filters.append({"label": f"Tech Support: {tech_support_filter}", "key": "tech_support"})
    if payment_filter != "ALL":
        active_filters.append({"label": f"Payment: {payment_filter}", "key": "payment"})
    if tenure_range_filter != "ALL":
        t_names = {"0-12": "Tenure: 0-12 mo (New)", "13-36": "Tenure: 13-36 mo (Mid)", "37-72": "Tenure: 37+ mo (Loyal)"}
        active_filters.append({"label": t_names.get(tenure_range_filter, tenure_range_filter), "key": "tenure_range"})
    if charges_range_filter != "ALL":
        c_names = {"0-35": "Spend: < $35", "35-75": "Spend: $35 - $75", "75+": "Spend: > $75"}
        active_filters.append({"label": c_names.get(charges_range_filter, charges_range_filter), "key": "charges_range"})
    if senior_filter in ["0", "1"]:
        active_filters.append({"label": "Senior: Yes" if senior_filter == "1" else "Senior: No", "key": "senior_citizen"})

    # Sorting
    if sort_by == "churn_asc":
        query = query.order_by(Prediction.churn_probability.asc().nullslast(), Customer.customer_id.asc())
    elif sort_by == "tenure_desc":
        query = query.order_by(Customer.tenure.desc())
    elif sort_by == "tenure_asc":
        query = query.order_by(Customer.tenure.asc())
    elif sort_by == "charges_desc":
        query = query.order_by(Customer.monthly_charges.desc())
    elif sort_by == "charges_asc":
        query = query.order_by(Customer.monthly_charges.asc())
    else:  # default "churn_desc"
        query = query.order_by(Prediction.churn_probability.desc().nullslast(), Customer.customer_id.asc())

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    items = []
    for cust, pred in pagination.items:
        items.append({
            "customer": cust,
            "prediction": pred,
            "churn_prob": round(pred.churn_probability * 100, 1) if pred else 0.0,
            "risk_level": pred.risk_level if pred else "UNSCORED"
        })
        
    filter_state = {
        "q": search_query,
        "risk": risk_filter,
        "contract": contract_filter,
        "internet": internet_filter,
        "tech_support": tech_support_filter,
        "payment": payment_filter,
        "tenure_range": tenure_range_filter,
        "charges_range": charges_range_filter,
        "senior_citizen": senior_filter,
        "sort": sort_by
    }
        
    return render_template(
        "customers.html",
        has_active_data=True,
        dataset_name=dataset_name,
        current_company=current_company,
        items=items,
        pagination=pagination,
        filters=filter_state,
        active_filters=active_filters,
        matching_total=matching_total,
        risk_counts=risk_counts,
        avg_matching_churn=avg_matching_churn,
        total_customers=total_in_db
    )

@customers_bp.route("/customer/<customer_id>")
def customer_detail(customer_id):
    has_active_data = session.get("has_active_data", False)
    selected_company_id = session.get("company_id")
    dataset_name = session.get("dataset_name", "My Dataset")
    
    current_company = Company.query.filter_by(company_id=selected_company_id).first() if selected_company_id else None
    
    customer = None
    if selected_company_id:
        customer = Customer.query.filter_by(company_id=selected_company_id, customer_id=customer_id).first()
        
    if not customer:
        customer = Customer.query.filter_by(customer_id=customer_id).first()
        if not customer:
            flash(f"Customer '{customer_id}' not found in active dataset.", "warning")
            return redirect(url_for("customers.customer_list"))
            
    cust_dict = customer.to_dict()
    
    # 1. Run / Retrieve Latest Prediction
    latest_pred = Prediction.query.filter_by(
        company_id=customer.company_id,
        customer_id=customer.customer_id
    ).order_by(Prediction.prediction_date.desc()).first()
    
    if not latest_pred:
        pred_res = prediction_service.predict_single(cust_dict)
        latest_pred = Prediction(
            customer_id=customer.customer_id,
            company_id=customer.company_id,
            model_version=pred_res["model_version"],
            churn_probability=pred_res["churn_probability"],
            risk_level=pred_res["risk_level"]
        )
        db.session.add(latest_pred)
        db.session.commit()
    else:
        pred_res = {
            "churn_probability": latest_pred.churn_probability,
            "churn_percentage": round(latest_pred.churn_probability * 100, 2),
            "retention_percentage": round((1.0 - latest_pred.churn_probability) * 100, 2),
            "risk_level": latest_pred.risk_level,
            "model_version": latest_pred.model_version
        }
        
    # 2. Retrieve / Generate SHAP Explanations
    shap_factors = shap_service.explain_customer(cust_dict, prediction_id=latest_pred.prediction_id, save_to_db=False)
    
    # 3. Retrieve / Generate Retention Strategy & AI Summary
    ai_strategy = recommendation_service.generate_retention_strategy(
        cust_dict,
        churn_probability=pred_res["churn_probability"],
        risk_level=pred_res["risk_level"],
        shap_factors=shap_factors,
        prediction_id=latest_pred.prediction_id,
        save_to_db=False
    )
    
    return render_template(
        "customer_detail.html",
        has_active_data=has_active_data,
        dataset_name=dataset_name,
        current_company=current_company,
        customer=customer,
        prediction=pred_res,
        shap_factors=shap_factors,
        ai_strategy=ai_strategy
    )
