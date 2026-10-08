from fastapi.testclient import TestClient

from app import feedback as feedback_module
from app.main import app


def test_feedback_endpoint_persists_and_reuses_generic_record(tmp_path, monkeypatch):
    feedback_path = tmp_path / "feedback_log.csv"
    monkeypatch.setattr(feedback_module.settings, "dataset_dir", tmp_path)
    monkeypatch.setattr(feedback_module.settings, "feedback_file", feedback_path.name)

    response = TestClient(app).post(
        "/feedback",
        json={
            "query": "Show total revenue by month.",
            "incorrect_interpretation": "Grouped revenue by all records.",
            "correct_interpretation": "Group revenue by month.",
            "notes": "Regression feedback",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "saved"
    assert set(payload["feedback"]) == {
        "query",
        "incorrect_interpretation",
        "correct_interpretation",
        "notes",
        "created_at",
    }

    saved_rows = feedback_module.FeedbackStore().get_relevant(
        "Show total revenue by month."
    )
    assert len(saved_rows) == 1
    assert saved_rows[0]["correct_interpretation"] == "Group revenue by month."
    assert saved_rows[0]["notes"] == "Regression feedback"
