"""add alert detail fields (device/user/paths/geo/time)"""
from alembic import op
import sqlalchemy as sa

# set your ids:
revision = "20250907_add_alert_detail_fields"
down_revision = "<CURRENT_HEAD>"  # <-- поменяй на актуальный head

def upgrade():
    with op.batch_alter_table("alerts") as b:
        b.add_column(sa.Column("device_id", sa.UUID(), nullable=True))
        b.add_column(sa.Column("user_id", sa.UUID(), nullable=True))
        b.add_column(sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False))
        b.add_column(sa.Column("lat", sa.Float(), nullable=True))
        b.add_column(sa.Column("lon", sa.Float(), nullable=True))
        b.add_column(sa.Column("address", sa.Text(), nullable=True))
        b.add_column(sa.Column("zone", sa.Text(), nullable=True))
        b.add_column(sa.Column("raw_path", sa.Text(), nullable=True))
        b.add_column(sa.Column("face_path", sa.Text(), nullable=True))
    op.create_index("ix_alerts_captured_at", "alerts", ["captured_at"])

def downgrade():
    op.drop_index("ix_alerts_captured_at", table_name="alerts")
    with op.batch_alter_table("alerts") as b:
        b.drop_column("face_path")
        b.drop_column("raw_path")
        b.drop_column("zone")
        b.drop_column("address")
        b.drop_column("lon")
        b.drop_column("lat")
        b.drop_column("captured_at")
        b.drop_column("user_id")
        b.drop_column("device_id")
