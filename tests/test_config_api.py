from datetime import datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app  # noqa: E402
from api.routes import config as config_routes  # noqa: E402
from database import get_db  # noqa: E402


class _StubProfile:
    def __init__(self, well_id: str):
        self.id = 1
        self.well_id = well_id
        self.rig_id = "RIG-01"
        self.total_depth = 2500.0
        self.kick_off_point = 120.0
        self.build_rate = 2.5
        self.max_inclination = 45.0
        self.target_zone_start = 1800.0
        self.target_zone_end = 2200.0
        self.geological_data = {"rock": "shale"}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()


class _StubQuery:
    def __init__(self, results):
        self._results = results

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return self._results

    def first(self):
        return self._results[0] if self._results else None


class _StubSession:
    def __init__(self, results):
        self._results = results

    def query(self, _model):
        return _StubQuery(self._results)


client = TestClient(app)


def _override_get_db():
    session = _StubSession([_StubProfile("WELL-01")])
    try:
        yield session
    finally:
        pass


def test_get_well_profiles(monkeypatch):
    app.dependency_overrides[get_db] = _override_get_db

    response = client.get("/api/v1/config/well-profiles")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["count"] == 1
    assert body["profiles"][0]["well_id"] == "WELL-01"

    app.dependency_overrides.pop(get_db, None)


def test_get_system_config():
    response = client.get("/api/v1/config/system")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "config" in body
    assert body["config"]["api_version"] == "1.0.0"
