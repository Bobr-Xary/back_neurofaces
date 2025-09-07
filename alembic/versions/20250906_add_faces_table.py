"""Add faces table

Revision ID: <set_me>
Revises: <set_current_head>
Create Date: 2025-09-06 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic.
revision = "<set_me>"            # TODO: replace
down_revision = "<set_current_head>"  # TODO: replace
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "faces",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("embedding", sa.LargeBinary, nullable=False),
        sa.Column("file_path", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    # Опционально индексы:
    op.create_index("ix_faces_created_at", "faces", ["created_at"])

def downgrade():
    op.drop_index("ix_faces_created_at", table_name="faces")
    op.drop_table("faces")
