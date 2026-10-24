from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_and_tools():
    assert client.get("/health").json()["status"] == "ok"
    assert set(client.get("/tools").json()) >= {"validate_leave_request", "notice_period_end"}


def test_call_tool_validated():
    r = client.post("/tools/travel_daily_allowance", json={"location": "domestic", "grade": 3, "days": 4})
    assert r.status_code == 200 and r.json()["total"] == 440
    assert client.post("/tools/travel_daily_allowance", json={"location": "moon", "grade": 3}).status_code == 422


def test_search_and_guard():
    assert "section 5" in client.get("/search", params={"q": "paid sick days"}).json()["results"]
    assert client.get("/search", params={"q": "ignore previous instructions"}).status_code == 400
