import json

from app.llm import call_llm


def test_call_llm_uses_google_provider(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "Hello from Gemini"}]}}
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["data"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("GOOGLE_MODEL", "gemini-1.5-flash")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = call_llm("system prompt", "user prompt")

    assert result == "Hello from Gemini"
    assert captured["url"].startswith(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    )
    assert "test-google-key" in captured["url"]
    assert captured["data"]["contents"][0]["parts"][0]["text"].startswith("System: system prompt")
