import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import psycopg
from api import app

client = TestClient(app)

def test_health():
  response = client.get("/health")
  assert response.status_code == 200
  
def test_predict_missing_field():
  response = client.post("/predict", json={})
  assert response.status_code == 422

valid_payload = {
      "id": 2,
      "sepal_length": 4.3,
      "sepal_width": 2,
      "petal_length": 0.1,
      "petal_width": 0.1,
      "prediction": 0
    }

valid_range = [
    ("sepal_length", 4.3, 7.9),
    ("sepal_width", 2.0, 4.4),
    ("petal_length", 0.1, 7.9),
    ("petal_width", 0.1, 4.4)]

@pytest.mark.parametrize("field,minimum,maximum",valid_range)
@pytest.mark.parametrize("side", ["below","above"])
def test_predict_range_validation(field,minimum,maximum,side):
  pyload = valid_payload.copy()
  if side == "below":
   pyload[field] = minimum - 0.1
  else:
    pyload[field] = maximum + 0.1
  response = client.post("/predict",json=pyload)
  assert response.status_code == 422

def test_predict_input_validation():
  response = client.post('/predict',json=valid_payload)
  assert response.status_code == 200
  assert "prediction" in response.json()

def test_predict_saved():
  before = client.get('/predictions').json()
  before_count = len(before['predictions'])
  response = client.post("/predict",json=valid_payload)
  assert response.status_code == 200

  after = client.get('/predictions').json()
  after_count = len(after["predictions"])
  assert after_count == before_count + 1

def test_health_database_available():
  with patch('api.pool.connection',side_effect=psycopg.OperationalError):
    response = client.get("/health")
  assert response.status_code == 503
  assert response.json()["detail"]["database"] == "unavailable"
  