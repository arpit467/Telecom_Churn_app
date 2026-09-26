from app.services.prediction_service import prediction_service

class WhatIfService:
    def simulate_scenario(self, original_customer_dict: dict, modified_fields: dict, model_key="champion_ensemble") -> dict:
        """
        Runs model-based counterfactual simulation by applying hypothetical modifications
        to a customer's profile and comparing original vs. simulated churn probabilities.
        """
        # 1. Baseline prediction
        baseline_res = prediction_service.predict_single(original_customer_dict, model_key=model_key)
        baseline_prob = baseline_res["churn_percentage"]
        baseline_risk = baseline_res["risk_level"]
        
        # 2. Construct simulated counterfactual profile
        simulated_dict = original_customer_dict.copy()
        for field, new_val in modified_fields.items():
            if new_val is not None and str(new_val).strip() != "":
                simulated_dict[field] = new_val
                
        # 3. Simulated prediction
        simulated_res = prediction_service.predict_single(simulated_dict, model_key=model_key)
        simulated_prob = simulated_res["churn_percentage"]
        simulated_risk = simulated_res["risk_level"]
        
        prob_delta = round(simulated_prob - baseline_prob, 2)
        is_reduction = prob_delta < 0
        
        return {
            "customer_id": original_customer_dict.get("customer_id", "Simulated Customer"),
            "model_used": model_key,
            "baseline_probability": baseline_prob,
            "baseline_risk": baseline_risk,
            "simulated_probability": simulated_prob,
            "simulated_risk": simulated_risk,
            "probability_delta": prob_delta,
            "is_risk_reduced": is_reduction,
            "percentage_change": f"{prob_delta:+.2f}%",
            "modified_fields": modified_fields,
            "disclaimer": "Model-based simulation for scenario planning. Actual customer retention depends on real-world adoption and service execution."
        }

what_if_service = WhatIfService()
