Auth/RBAC module for your FastAPI backend

1) Install extra deps:
   pip install passlib[bcrypt] PyJWT SQLAlchemy

2) Create tables (simple SQL migration):
   psql "$DATABASE_URL" -f migrations/sql/0001_init.sql
   # or: psql -h localhost -U postgres -d postgres -f migrations/sql/0001_init.sql

3) Wire the router into your FastAPI app (server.py):
   # --- RBAC/AUTH integration start ---
from app.api.v1.router import api_router as rbac_api_router  # new
app.include_router(rbac_api_router, prefix="/api/v1")
# --- RBAC/AUTH integration end ---

4) Env vars (example):
   export DATABASE_URL=postgresql+psycopg2://postgres:1@localhost:5432/postgres
   export JWT_SECRET=$(python - <<<'import secrets; print(secrets.token_urlsafe(48))')
   export ACCESS_TOKEN_EXPIRES_MIN=15
   export REFRESH_TOKEN_EXPIRES_DAYS=30

5) Admin bootstrap (create first admin user manually):
   INSERT INTO users (id,email,full_name,password_hash,role,is_active)
   VALUES (gen_random_uuid(),'admin@example.com','Admin',
           '$2b$12$examplehashPLEASECHANGE','admin', true);
   # You can generate a bcrypt hash with Python:
   # python - << 'PY'
   # from passlib.context import CryptContext
   # print(CryptContext(schemes=['bcrypt']).hash('YourStrongPassword'))
   # PY

6) Endpoints:
   POST /api/v1/auth/login        {email,password} -> access/refresh
   POST /api/v1/auth/refresh      {refresh_token}  -> rotated pair
   POST /api/v1/auth/register     (admin)            -> create user (officer/citizen)
   GET  /api/v1/users/            (admin)            -> list users
   POST /api/v1/devices/register  (admin)            -> returns device_id + device_secret (show once)

7) Device auth (for ingest endpoints):
   Expect headers: X-Device-Id, X-Device-Secret
   Use dependency: get_current_device in app.core.deps

8) Notes:
   - Passwords & device secrets are stored hashed (bcrypt).
   - Refresh tokens are stored hashed (sha256) & rotated on each refresh.
   - Roles available: admin, officer, citizen (device is pseudo-role).
