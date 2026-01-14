from modules.calcul import calcul
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_calcul_2():
    assert calcul(2) == 4

def test_calcul_negatif():
    assert calcul(-3) == 9

def test_calcul_0():
    assert calcul(0) == 0

def test_calcul_refuse_string():
    response = client.post("/calcul", json={"n": "abc"})
    assert response.status_code == 422


def test_calcul_refuse_float():
    response = client.post("/calcul", json={"n": 3.14})
    assert response.status_code == 422
