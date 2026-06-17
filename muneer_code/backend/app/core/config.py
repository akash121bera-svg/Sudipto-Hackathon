import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Autonomous Document Intelligence Platform"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field(default="change-me-in-production")
    
    # Storage settings
    UPLOAD_DIR: str = Field(default=str(BASE_DIR / "storage" / "uploads"))
    PREPROCESSED_DIR: str = Field(default=str(BASE_DIR / "storage" / "preprocessed"))
    PATCHES_DIR: str = Field(default=str(BASE_DIR / "storage" / "patches"))

    # PostgreSQL config
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: str = Field(default="5432")
    POSTGRES_DB: str = Field(default="doc_intelligence")

    # Qdrant config
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_API_KEY: str = Field(default="")

    # LLM Settings (Ollama + Llama3)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3")
    
    # OCR Fallback Configuration
    # Set to True to allow the system to mock complex OCR models (TrOCR) and LLM logic 
    # if they are not installed or cannot run locally on CPU.
    ENABLE_MOCK_FALLBACK: bool = Field(default=True)
    
    # Gemini Vision API (best OCR quality — set this for accurate handwriting recognition)
    GEMINI_API_KEY: Optional[str] = Field(default=None)

    # Hugging Face Settings (for TrOCR/all-MiniLM-L6-v2)
    HF_CACHE_DIR: str = Field(default=str(BASE_DIR / "storage" / "hf_cache"))

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Silently ignore unknown env vars

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PREPROCESSED_DIR, exist_ok=True)
os.makedirs(settings.PATCHES_DIR, exist_ok=True)
