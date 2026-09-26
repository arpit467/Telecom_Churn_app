from app.routes.dashboard import dashboard_bp
from app.routes.customers import customers_bp
from app.routes.upload import upload_bp
from app.routes.predictions import predictions_bp
from app.routes.assistant import assistant_bp
from app.routes.manual_predict import manual_predict_bp

def register_routes(app):
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(customers_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(assistant_bp)
    app.register_blueprint(manual_predict_bp)
