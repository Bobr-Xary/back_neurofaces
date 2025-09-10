
"""RBAC media + alert ACL + system settings + hidden flag"""
from alembic import op
import sqlalchemy as sa

# Adjust down_revision to your current head
revision = "20250910_rbac_acl_settings"
down_revision = None  # set to your current head manually if you track revisions strictly
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    # alerts.hidden + alerts.user_id (if not exists)
    op.execute("""
        ALTER TABLE alerts
        ADD COLUMN IF NOT EXISTS hidden boolean NOT NULL DEFAULT false
    """)
    op.execute("""
        ALTER TABLE alerts
        ADD COLUMN IF NOT EXISTS user_id uuid NULL
    """)

    # alert_access table
    op.execute("""
        CREATE TABLE IF NOT EXISTS alert_access (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            alert_id uuid NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            can_view_face boolean NOT NULL DEFAULT false,
            created_at timestamp without time zone NOT NULL DEFAULT (now() at time zone 'utc')
        )
    """)
    # unique constraint (safe upsert)
    try:
        op.create_unique_constraint("uq_alert_access_alert_user", "alert_access", ["alert_id", "user_id"])
    except Exception:
        pass

    # system_settings table (single row, id=1)
    op.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            id integer PRIMARY KEY,
            telegram_enabled boolean NOT NULL DEFAULT true,
            telegram_chat_id text NULL,
            updated_at timestamp without time zone NOT NULL DEFAULT (now() at time zone 'utc')
        )
    """)
    # ensure singleton row exists
    op.execute("""
        INSERT INTO system_settings (id, telegram_enabled)
        VALUES (1, true)
        ON CONFLICT (id) DO NOTHING
    """)

def downgrade():
    # reversible enough if needed
    try:
        op.drop_constraint("uq_alert_access_alert_user", "alert_access", type_="unique")
    except Exception:
        pass
    op.execute("DROP TABLE IF EXISTS alert_access")
    op.execute("DROP TABLE IF EXISTS system_settings")
    # keep alerts.hidden and alerts.user_id columns (non-destructive)
