from src.web_app import app

def test_homepage_loads():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"House Price Predictor" in response.data

def test_invalid_json_returns_400():
    client = app.test_client()
    response = client.post("/api/predict", json={"area": -2})
    assert response.status_code == 400
    assert "error" in response.get_json()
