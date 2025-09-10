from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config, pool
import os, sys

# --- Alembic Config ---
config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

# чтобы работали импорты вида "from app...."
sys.path.insert(0, os.path.abspath("."))

# --- твои настройки и метаданные моделей ---
from app.core.config_env import settings
# ВАЖНО: импортируй метаданные всех моделей
# если у тебя Base лежит в app.db.base (или где-то ещё) — скорректируй импорт:
target_metadata = None

def run_migrations_offline():
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = settings.DATABASE_URL
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
