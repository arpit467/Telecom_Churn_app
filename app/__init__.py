import os
from flask import Flask
from config import Config
from app.models.database_models import db
from app.routes import register_routes
from app.utils.db_seed import seed_database_if_empty

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize Database
    db.init_app(app)
    
    # Register Route Blueprints
    register_routes(app)
    
    @app.before_request
    def handle_company_context():
        from flask import request, session
        company_id_arg = request.args.get("company_id")
        if company_id_arg:
            session["company_id"] = company_id_arg
            session["has_active_data"] = True
            
    # Auto-Seed Database if empty
    with app.app_context():
        try:
            seed_database_if_empty(app)
        except Exception as e:
            print(f"DB initialization notice: {e}")
            
    return app
