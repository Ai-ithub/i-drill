from pathlib import Path
import sys
from datetime import datetime

from fastapi.testclient import TestClient

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app  # noqa: E402


class _StubUser:
    def __init__(self, username: str, role: str = "viewer"):
        self.id = 1
        self.username = username
        self.email = f"{username}@example.com"
        self.full_name = "Test User"
        self.role = role
        self.is_active = True
        self.created_at = datetime.now()
        self.hashed_password = "hashed"


class _StubAuthService:
    def __init__(self, allow_registration: bool = True, allow_login: bool = True):
        self.allow_registration = allow_registration
        self.allow_login = allow_login

    def create_user(self, username, email, password, full_name=None, role="viewer"):
        if not self.allow_registration or username == "duplicate":
            return None
        return _StubUser(username=username, role=role)

    def authenticate_user(self, username, password):
        if not self.allow_login or password != "secret":
            return None
        return _StubUser(username=username)

    def create_access_token(self, data, expires_delta):
        return "stub-token"

    def verify_password(self, password, hashed_password):
        return password == "current"

    def update_user_password(self, user_id, new_password):
        return True

    def get_user_by_id(self, user_id):
        return _StubUser(username="admin", role="admin") if user_id == 1 else None


client = TestClient(app)


def test_register_user_success(monkeypatch):
    from api.routes import auth as auth_routes

    stub_service = _StubAuthService()
    monkeypatch.setattr(auth_routes, "auth_service", stub_service, raising=False)

    payload = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "secretpass",
        "full_name": "New User",
        "role": "viewer"
    }

    response = client.post("/api/v1/auth/register", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["user"]["username"] == "newuser"


def test_register_user_duplicate(monkeypatch):
    from api.routes import auth as auth_routes

    stub_service = _StubAuthService(allow_registration=False)
    monkeypatch.setattr(auth_routes, "auth_service", stub_service, raising=False)

    payload = {
        "username": "duplicate",
        "email": "dup@example.com",
        "password": "secretpass"
    }

    response = client.post("/api/v1/auth/register", json=payload)

    assert response.status_code == 400


def test_login_success(monkeypatch):
    from api.routes import auth as auth_routes

    stub_service = _StubAuthService()
    monkeypatch.setattr(auth_routes, "auth_service", stub_service, raising=False)

    response = client.post(
        "/api/v1/auth/login",
        data={"username": "tester", "password": "secret"},
        headers={"content-type": "application/x-www-form-urlencoded"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "stub-token"


def test_login_failure(monkeypatch):
    from api.routes import auth as auth_routes

    stub_service = _StubAuthService(allow_login=False)
    monkeypatch.setattr(auth_routes, "auth_service", stub_service, raising=False)

    response = client.post(
        "/api/v1/auth/login",
        data={"username": "tester", "password": "wrong"},
        headers={"content-type": "application/x-www-form-urlencoded"}
    )

    assert response.status_code == 401
