import os
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from psycopg.rows import dict_row
from sqlalchemy import JSON, Column, Index, Integer, String, Text, create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

try:
    import psycopg2  # noqa: F401
except Exception:  # pragma: no cover - optional fallback
    psycopg2 = None

load_dotenv()


def get_database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/guardrail_rag",
    )


def _build_engine() -> Any:
    url = get_database_url()
    if url.startswith("postgresql+psycopg://") and psycopg2 is not None:
        return create_engine(url.replace("postgresql+psycopg://", "postgresql+psycopg2://"), pool_pre_ping=True, pool_size=5, max_overflow=10)
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def get_async_database_url() -> str:
    return os.getenv("ASYNC_DATABASE_URL", get_database_url().replace("postgresql+psycopg://", "postgresql+asyncpg://"))


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    file_name = Column(String(255), nullable=True)
    page_number = Column(Integer, nullable=True)
    chunk_metadata = Column(JSON, default=dict)
    embedding = Column(Vector(1536), nullable=True)

    __table_args__ = (
        Index(
            "hnsw_index_document_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


try:
    async_engine = create_async_engine(get_async_database_url(), pool_pre_ping=True, pool_size=5, max_overflow=10)
    AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False)
except Exception:  # pragma: no cover - optional async dependency
    async_engine = None
    AsyncSessionLocal = None


def get_session() -> Session:
    return SessionLocal()


def get_async_session_factory() -> async_sessionmaker[AsyncSession] | None:
    return AsyncSessionLocal


def init_db() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=engine)
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS hnsw_index_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )


def get_connection() -> Any:
    return psycopg.connect(get_database_url(), row_factory=dict_row)


def create_tables() -> None:
    init_db()


def add_document_chunk(content: str, *, metadata: dict[str, Any] | None = None, embedding: list[float] | None = None) -> DocumentChunk:
    metadata = metadata or {}
    with get_session() as session:
        chunk = DocumentChunk(content=content, chunk_metadata=metadata, embedding=embedding)
        session.add(chunk)
        session.commit()
        session.refresh(chunk)
        return chunk
