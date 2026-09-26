from app import create_app
from app.models.database_models import Company, Customer, Prediction

app = create_app()
client = app.test_client()

print('--- 1. Testing Database & Companies ---')
with app.app_context():
    companies = Company.query.all()
    print('Companies in DB:', [c.company_name for c in companies])
    cust_count = Customer.query.count()
    pred_count = Prediction.query.count()
    print(f'Total Seeded Customers: {cust_count}, Predictions: {pred_count}')
    assert cust_count > 0, 'No customers seeded'

print('\n--- 2. Testing Dashboard Route ---')
res = client.get('/')
assert res.status_code == 200, f'Dashboard failed: {res.status_code}'
assert b'Retention' in res.data
print('GET / passed -> Status 200')

print('\n--- 3. Testing Customers Directory Route ---')
res_cust = client.get('/customers')
assert res_cust.status_code == 200, f'Customers list failed: {res_cust.status_code}'
print('GET /customers passed -> Status 200')

print('\n--- 4. Testing Customer 360 & SHAP Route ---')
with app.app_context():
    first_cust = Customer.query.first()
    first_id = first_cust.customer_id
res_detail = client.get(f'/customer/{first_id}')
assert res_detail.status_code == 200, f'Customer detail failed: {res_detail.status_code}'
assert b'SHAP Feature Attributions' in res_detail.data
assert b'Counterfactual What-If' in res_detail.data
print(f'GET /customer/{first_id} passed -> Status 200')

print('\n--- 5. Testing What-If Simulation API ---')
res_whatif = client.post(f'/api/customer/{first_id}/what-if', json={
    'modifications': {
        'contract': 'Two year',
        'tech_support': 'Yes'
    }
})
assert res_whatif.status_code == 200, f'What-if API failed: {res_whatif.status_code}'
json_data = res_whatif.get_json()
assert json_data['success'] is True
sim = json_data['simulation']
print(f"POST /api/customer/{first_id}/what-if passed -> Baseline: {sim['baseline_probability']}% -> Simulated: {sim['simulated_probability']}% (Delta: {sim['probability_delta']}%)")

print('\n--- 6. Testing AI Assistant API ---')
res_ai = client.post('/api/assistant', json={
    'query': 'How many high-risk customers do we have?'
})
assert res_ai.status_code == 200
print('POST /api/assistant passed')

print('\n--- 7. Testing MLOps & Benchmark Models Route ---')
res_models = client.get('/models')
assert res_models.status_code == 200
assert b'Evaluation Matrix' in res_models.data
print('GET /models passed -> Status 200')

print('\n--- 8. Testing CSV Export ---')
res_export = client.get('/api/export/predictions')
assert res_export.status_code == 200
assert b'Customer_ID,Company_ID' in res_export.data
print('GET /api/export/predictions passed -> Status 200')

print('\n=================================================================')
print('>>> ALL 8 ENTERPRISE PLATFORM TEST SUITES PASSED FLAWLESSLY! <<<')
print('=================================================================')
