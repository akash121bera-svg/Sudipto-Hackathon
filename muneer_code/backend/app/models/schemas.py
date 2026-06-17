import os
import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import TypeDecorator, CHAR
import uuid

from backend.app.core.database import Base
from pydantic import BaseModel, Field

# Resilient UUID column that works on both Postgres (using native UUID) and SQLite (using String/CHAR)
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(32), storing as string without dashes.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value

# ==========================================
# SQLALCHEMY MODELS
# ==========================================

class DBModelBase(Base):
    __abstract__ = True

class DocumentModel(DBModelBase):
    __tablename__ = "documents"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="pending")  # 'pending', 'processing', 'completed', 'requires_review'
    file_path = Column(String(500), nullable=False)
    preprocessed_path = Column(String(500), nullable=True)
    confidence_score = Column(Float, default=0.0)
    retry_count = Column(Integer, default=0)
    extracted_data = Column(JSON, nullable=True)
    validation_report = Column(JSON, nullable=True)
    reasoning_log = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    @property
    def file_url(self):
        return f"/storage/uploads/{os.path.basename(self.file_path)}" if self.file_path else None

    @property
    def preprocessed_url(self):
        return f"/storage/preprocessed/{os.path.basename(self.preprocessed_path)}" if self.preprocessed_path else None

class HumanReviewModel(DBModelBase):
    __tablename__ = "human_reviews"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    document_id = Column(GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    fields_flagged = Column(JSON, nullable=False)  # List of keys or details
    corrected_data = Column(JSON, nullable=True)
    reviewed_by = Column(String(100), nullable=True)
    status = Column(String(50), default="pending")  # 'pending', 'resolved'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class FeedbackMemoryModel(DBModelBase):
    __tablename__ = "feedback_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(GUID(), nullable=True)
    field_key = Column(String(100), nullable=False)
    original_value = Column(String(500), nullable=True)
    corrected_value = Column(String(500), nullable=True)
    handwriting_patch_path = Column(String(500), nullable=True)
    embedding_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class DocumentBase(BaseModel):
    filename: str
    status: str
    file_path: str
    preprocessed_path: Optional[str] = None
    file_url: Optional[str] = None
    preprocessed_url: Optional[str] = None
    confidence_score: float = 0.0
    retry_count: int = 0
    extracted_data: Optional[Dict[str, Any]] = None
    validation_report: Optional[Dict[str, Any]] = None
    reasoning_log: Optional[List[Dict[str, Any]]] = None

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: uuid.UUID
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        from_attributes = True

class HumanReviewResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    fields_flagged: List[str]
    corrected_data: Optional[Dict[str, Any]] = None
    reviewed_by: Optional[str] = None
    status: str
    created_at: datetime.datetime
    resolved_at: Optional[datetime.datetime] = None

    class Config:
        from_attributes = True

class HumanReviewSubmit(BaseModel):
    corrected_data: Dict[str, Any]
    reviewed_by: str = "human_operator"

class MemoryEntryResponse(BaseModel):
    id: int
    document_id: Optional[uuid.UUID]
    field_key: str
    original_value: Optional[str]
    corrected_value: Optional[str]
    handwriting_patch_path: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True
