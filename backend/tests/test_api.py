from fastapi.testclient import TestClient

from app.main import app
from app.query_interpreter import QueryInterpreter


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_total_revenue(monkeypatch):
    monkeypatch.setattr(QueryInterpreter, "_build_client", lambda self: None)
    response = TestClient(app).post(
        "/query",
        json={"query": "What is the total revenue?"},
    )

    assert response.status_code == 200
    assert response.json()["result"] == [
        {"revenue": 6134.4}
    ]
