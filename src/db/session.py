from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import settings
from src.db.models import Base

_engine = None
_SessionLocal = None


def get_engine(database_url: str | None = None):
    global _engine
    if database_url is not None:
        return create_engine(database_url, pool_pre_ping=True)
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine


def get_session_factory(database_url: str | None = None):
    global _SessionLocal
    if database_url is not None:
        return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def get_session(database_url: str | None = None):
    """Usage: with get_session() as session: ..."""
    factory = get_session_factory(database_url)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_tables(database_url: str | None = None) -> None:
    Base.metadata.create_all(bind=get_engine(database_url))


def drop_all_tables(database_url: str | None = None) -> None:
    Base.metadata.drop_all(bind=get_engine(database_url))