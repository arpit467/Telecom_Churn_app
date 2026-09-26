from flask import Blueprint, render_template, request, session, jsonify
from app.models.database_models import Company
from app.services.assistant_service import assistant_service
from app.services.llm_service import llm_service

assistant_bp = Blueprint("assistant", __name__)

@assistant_bp.route("/assistant")
def assistant_page():
    has_active_data = session.get("has_active_data", False)
    dataset_name = session.get("dataset_name", "My Dataset")
    selected_company_id = session.get("company_id")
    current_company = Company.query.filter_by(company_id=selected_company_id).first() if selected_company_id else None
    
    active_provider, _ = llm_service.get_active_provider(
        custom_key=session.get("custom_api_key"),
        provider_pref=session.get("custom_provider")
    )
    
    return render_template(
        "assistant.html",
        has_active_data=has_active_data,
        dataset_name=dataset_name,
        current_company=current_company,
        active_provider=active_provider,
        has_gemini_key=bool(llm_service.gemini_key or session.get("custom_api_key")),
        has_openai_key=bool(llm_service.openai_key or session.get("custom_api_key"))
    )

@assistant_bp.route("/api/assistant", methods=["POST"])
def api_query_assistant():
    payload = request.get_json() or {}
    query = payload.get("query", "").strip()
    company_id = payload.get("company_id") or session.get("company_id", "APEX-GLOBAL")
    chat_history = payload.get("chat_history", [])
    custom_key = payload.get("api_key") or session.get("custom_api_key")
    custom_provider = payload.get("provider") or session.get("custom_provider")
    
    if not query:
        return jsonify({"success": False, "error": "Query cannot be empty."}), 400
        
    result = assistant_service.answer_query(
        company_id=company_id,
        query=query,
        chat_history=chat_history,
        custom_key=custom_key,
        custom_provider=custom_provider
    )
    
    return jsonify({
        "success": True,
        "company_id": company_id,
        "query": query,
        "response": result["response"],
        "data": result.get("data", {})
    })

@assistant_bp.route("/api/assistant/test-key", methods=["POST"])
def api_test_key():
    payload = request.get_json() or {}
    provider = payload.get("provider", "gemini").lower()
    api_key = payload.get("api_key", "").strip()
    
    if not api_key:
        return jsonify({"success": False, "error": "Please provide an API key to test."}), 400
        
    res = llm_service.test_provider_connection(provider, api_key)
    if res.get("success"):
        # Store in session for this user session
        session["custom_api_key"] = api_key
        session["custom_provider"] = provider
    return jsonify(res)

@assistant_bp.route("/api/assistant/clear-key", methods=["POST"])
def api_clear_key():
    session.pop("custom_api_key", None)
    session.pop("custom_provider", None)
    return jsonify({"success": True, "message": "Custom API Key cleared. Default provider restored."})
