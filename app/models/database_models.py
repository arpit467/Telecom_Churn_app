from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Company(db.Model):
    __tablename__ = "companies"
    
    company_id = db.Column(db.String(50), primary_key=True)
    company_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    customers = db.relationship("Customer", backref="company", cascade="all, delete-orphan", lazy="dynamic")
    predictions = db.relationship("Prediction", backref="company", cascade="all, delete-orphan", lazy="dynamic")

    def to_dict(self):
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Customer(db.Model):
    __tablename__ = "customers"
    __table_args__ = (
        db.UniqueConstraint("customer_id", "company_id", name="uq_customer_company"),
    )
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(50), nullable=False, index=True)
    company_id = db.Column(db.String(50), db.ForeignKey("companies.company_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Demographics
    gender = db.Column(db.String(10), default="Male")
    senior_citizen = db.Column(db.Integer, default=0)
    partner = db.Column(db.String(5), default="No")
    dependents = db.Column(db.String(5), default="No")
    
    # Service usage & Tenure
    tenure = db.Column(db.Float, default=1.0)
    phone_service = db.Column(db.String(5), default="Yes")
    multiple_lines = db.Column(db.String(25), default="No")
    internet_service = db.Column(db.String(25), default="DSL")
    online_security = db.Column(db.String(25), default="No")
    online_backup = db.Column(db.String(25), default="No")
    device_protection = db.Column(db.String(25), default="No")
    tech_support = db.Column(db.String(25), default="No")
    streaming_tv = db.Column(db.String(25), default="No")
    streaming_movies = db.Column(db.String(25), default="No")
    
    # Contract & Billing
    contract = db.Column(db.String(25), default="Month-to-month")
    paperless_billing = db.Column(db.String(5), default="Yes")
    payment_method = db.Column(db.String(50), default="Electronic check")
    monthly_charges = db.Column(db.Float, default=50.0)
    total_charges = db.Column(db.Float, default=50.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "customer_id": self.customer_id,
            "company_id": self.company_id,
            "gender": self.gender,
            "senior_citizen": self.senior_citizen,
            "partner": self.partner,
            "dependents": self.dependents,
            "tenure": self.tenure,
            "phone_service": self.phone_service,
            "multiple_lines": self.multiple_lines,
            "internet_service": self.internet_service,
            "online_security": self.online_security,
            "online_backup": self.online_backup,
            "device_protection": self.device_protection,
            "tech_support": self.tech_support,
            "streaming_tv": self.streaming_tv,
            "streaming_movies": self.streaming_movies,
            "contract": self.contract,
            "paperless_billing": self.paperless_billing,
            "payment_method": self.payment_method,
            "monthly_charges": self.monthly_charges,
            "total_charges": self.total_charges,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Prediction(db.Model):
    __tablename__ = "predictions"
    
    prediction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(50), nullable=False, index=True)
    company_id = db.Column(db.String(50), db.ForeignKey("companies.company_id", ondelete="CASCADE"), nullable=False, index=True)
    model_version = db.Column(db.String(50), default="v2.4.0-ensemble-prod")
    churn_probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False) # LOW, MEDIUM, HIGH
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    explanations = db.relationship("CustomerExplanation", backref="prediction", cascade="all, delete-orphan", lazy="dynamic")
    recommendations = db.relationship("RetentionRecommendation", backref="prediction", cascade="all, delete-orphan", lazy="dynamic")

    def to_dict(self):
        return {
            "prediction_id": self.prediction_id,
            "customer_id": self.customer_id,
            "company_id": self.company_id,
            "model_version": self.model_version,
            "churn_probability": round(self.churn_probability * 100, 2),
            "risk_level": self.risk_level,
            "prediction_date": self.prediction_date.isoformat() if self.prediction_date else None
        }

class CustomerExplanation(db.Model):
    __tablename__ = "customer_explanations"
    
    explanation_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.prediction_id", ondelete="CASCADE"), nullable=True, index=True)
    customer_id = db.Column(db.String(50), nullable=False, index=True)
    feature_name = db.Column(db.String(100), nullable=False)
    feature_value = db.Column(db.String(100), nullable=True)
    shap_value = db.Column(db.Float, nullable=False)
    impact_direction = db.Column(db.String(10), nullable=False) # INCREASES_CHURN / DECREASES_CHURN

    def to_dict(self):
        return {
            "explanation_id": self.explanation_id,
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "shap_value": round(self.shap_value, 4),
            "impact_direction": self.impact_direction
        }

class RetentionRecommendation(db.Model):
    __tablename__ = "retention_recommendations"
    
    recommendation_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(50), nullable=False, index=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.prediction_id", ondelete="CASCADE"), nullable=True, index=True)
    recommendation_text = db.Column(db.Text, nullable=False)
    ai_summary = db.Column(db.Text, nullable=True)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_version = db.Column(db.String(50), default="v2.4.0-genai-rules")

    def to_dict(self):
        return {
            "recommendation_id": self.recommendation_id,
            "customer_id": self.customer_id,
            "recommendation_text": self.recommendation_text,
            "ai_summary": self.ai_summary,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "model_version": self.model_version
        }

class ModelTracking(db.Model):
    __tablename__ = "model_tracking"
    
    run_id = db.Column(db.String(50), primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    version = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    roc_auc = db.Column(db.Float, nullable=False)
    pr_auc = db.Column(db.Float, nullable=False)
    hyperparameters = db.Column(db.JSON, nullable=True)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_production = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "version": self.version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "hyperparameters": self.hyperparameters,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "is_production": self.is_production
        }
