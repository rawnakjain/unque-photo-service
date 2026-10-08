from fastapi.testclient import TestClient
from service_testing.app import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    print("Health check passed:", data)

def test_services():
    response = client.get("/api/services")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    print(f"Services endpoint returned {len(data)} services")

if __name__ == "__main__":
    test_health()
    test_services()
    print("All tests passed!")