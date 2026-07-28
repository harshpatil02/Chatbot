from fastapi.testclient import TestClient

from app.main import app


def test_chat_returns_structured_response() -> None:
    client = TestClient(app)
    response = client.post("/api/chat", json={"query": "What is in the document?"})

    assert response.status_code == 200
    payload = response.json()
    assert "guardrail_status" in payload
    assert "blocked" in payload
    assert "reason" in payload
    assert "answer" in payload
    assert "documents" in payload
    assert "citations" in payload
