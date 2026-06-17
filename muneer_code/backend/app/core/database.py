import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.core.config import settings

logger = logging.getLogger("database")

# Build connection string
postgres_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
sqlite_url = "sqlite:///./doc_intelligence.db"

# Resilient connection logic
try:
    # Try PostgreSQL first
    engine = create_engine(
        postgres_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3}  # Fail fast to fall back quickly if Postgres is down
    )
    # Check connection
    with engine.connect() as conn:
        logger.info("Successfully connected to PostgreSQL database.")
except Exception as e:
    logger.warning(
        f"PostgreSQL connection failed: {e}. Falling back to SQLite for hackathon resiliency."
    )
    # SQLite fallback
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
