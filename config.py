import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get("SECRET_KEY", "telecom_churn_retention_intelligence_2026")
    
    # Database: Supports PostgreSQL via DATABASE_URL or SQLite fallback
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'churn_intelligence.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # ML Artifacts
    ML_ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
    MODELS_BUNDLE_PATH = os.path.join(ML_ARTIFACTS_DIR, "models_bundle.joblib")
    METRICS_PATH = os.path.join(ML_ARTIFACTS_DIR, "model_metrics.json")
    
    # Fallback paths for legacy location
    if not os.path.exists(MODELS_BUNDLE_PATH) and os.path.exists(os.path.join(BASE_DIR, "models_bundle.joblib")):
        MODELS_BUNDLE_PATH = os.path.join(BASE_DIR, "models_bundle.joblib")
    if not os.path.exists(METRICS_PATH) and os.path.exists(os.path.join(BASE_DIR, "model_metrics.json")):
        METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

    # Risk Thresholds (Data-Driven from Cross-Validation Performance Optimization)
    LOW_RISK_MAX = float(os.environ.get("LOW_RISK_MAX", "0.30"))
    HIGH_RISK_MIN = float(os.environ.get("HIGH_RISK_MIN", "0.48"))
    
    # LLM Settings (Optional API Key for Live OpenAI/Gemini/Ollama)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "hybrid_expert_rules")

    # Uploads
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
