import unittest

from fastapi.testclient import TestClient

from app.main import app


class MainRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_chat_streams_events(self) -> None:
        response = self.client.post("/api/chat/stream", json={"query": "What is in the document?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
