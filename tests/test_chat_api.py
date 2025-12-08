from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# Invalid test check for now
def test_invalid_model_name():
    payload = {
        "model_name": "invalid-model",
        "model_provider": "Groq",
        "system_prompt": "You are a test bot",
        "messages": ["Hello"],
        "allow_search": False,
    }
    response = client.post("/chat", json=payload)
    print(response.status_code)
    assert response.status_code == 400
