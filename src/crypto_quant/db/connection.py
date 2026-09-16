"""Database connection and session management."""

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, db_path: str = "data/crypto_quant.db", in_memory: bool = False):
        """Initialize database manager."""
        self.in_memory = in_memory
        if in_memory:
            self.db_path = Path(":memory:")
            self.engine = create_engine(
                "sqlite://",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False,
            )
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                connect_args={"check_same_thread": False, "timeout": 60.0},
                echo=False,
            )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self) -> None:
        """Create all missing tables and migrate existing ones.

        SQLite's ``create_all`` adds new tables but does NOT add new columns to
        tables that already exist, so we run a lightweight ALTER pass for any
        column that is in the model but missing from an existing table. This
        upgrades a pre-existing project DB in place without a rebuild.
        """
        Base.metadata.create_all(bind=self.engine)
        self._migrate_columns()

    def _migrate_columns(self) -> None:
        """Add any model columns missing from existing tables (idempotent)."""
        from sqlalchemy import inspect, text
        inspector = inspect(self.engine)
        existing = {t: set(c["name"] for c in inspector.get_columns(t))
                    for t in inspector.get_table_names()}
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                cols = existing.get(table.name)
                if cols is None:
                    continue
                for col in table.columns:
                    if col.name in cols:
                        continue
                    # Match SQLite type from the column definition.
                    coltype = col.type.compile(dialect=self.engine.dialect)
                    conn.execute(text(
                        f"ALTER TABLE {table.name} ADD COLUMN {col.name} {coltype}"
                    ))

    def drop_tables(self) -> None:
        """Drop all database tables (for testing)."""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def session_generator(self) -> Generator[Session, None, None]:
        """Generate database sessions (for dependency injection)."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def execute_raw(self, query: str) -> list:
        """Execute raw SQL query."""
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return result.fetchall()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table."""
        with self.engine.connect() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            return result.scalar()

    def backup(self, backup_path: str) -> None:
        """Backup database to file."""
        import shutil
        shutil.copy2(self.db_path, backup_path)


# Global database instance
_db_manager: DatabaseManager | None = None


def get_db_manager(db_path: str | None = None) -> DatabaseManager:
    """Get or create database manager singleton."""
    global _db_manager
    if _db_manager is None or db_path:
        if db_path is None:
            db_path = os.getenv("DB_PATH", "data/crypto_quant.db")
        _db_manager = DatabaseManager(db_path)
    return _db_manager


def init_database(db_path: str | None = None) -> DatabaseManager:
    """Initialize database with tables."""
    db = get_db_manager(db_path)
    db.create_tables()
    return db
