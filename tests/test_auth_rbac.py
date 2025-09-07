import os
from jose import jwt
from app.core.config import JWT_SECRET, JWT_ALG

def test_login_and_users_list(client, ensure_admin):
    # 1) login
    resp = client.post("/api/v1/auth/login", json=ensure_admin)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "access_token" in data and data["access_token"], data
    token = data["access_token"]

    # 2) (debug) декод без проверки exp — нам важно, что структура токена ок
    claims = jwt.get_unverified_claims(token)
    assert claims.get("type") == "access", claims
    assert "sub" in claims, claims

    # Если хочешь проверить подпись, но игнорировать exp:
    decoded = jwt.decode(
        token,
        os.environ.get("JWT_SECRET", JWT_SECRET),
        algorithms=[os.environ.get("JWT_ALG", JWT_ALG)],
        options={"verify_exp": False},
    )
    assert decoded["sub"] == claims["sub"]
    assert decoded["type"] == "access"

    # 3) admin-only endpoint
    r2 = client.get("/api/v1/users/", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200, f"/users/ failed: {r2.status_code} {r2.text}"
    users = r2.json()
    assert any(u["email"] == ensure_admin["email"] for u in users), users

def test_alerts_route_exists(client, ensure_admin):
    # login
    resp = client.post("/api/v1/auth/login", json=ensure_admin)
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]

    # alerts должен существовать и отдавать 200 (список)
    r2 = client.get("/api/v1/alerts/?limit=5", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200, f"Expected 200, got {r2.status_code}: {r2.text}"
    assert isinstance(r2.json(), list)
