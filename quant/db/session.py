"""SQLAlchemy engine and session factory."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None  # type: ignore[type-arg]


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def get_session() -> sessionmaker:  # type: ignore[type-arg]
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager for a transactional database session."""
    session = get_session()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
