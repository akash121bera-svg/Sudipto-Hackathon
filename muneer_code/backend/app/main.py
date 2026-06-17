import os
import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse

from backend.app.core.config import settings, BASE_DIR
from backend.app.core.database import Base, engine
from backend.app.core.vector_store import init_vector_db
from backend.app.routes import documents, review, memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# Create Database tables
try:
    logger.info("Initializing database schemas...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schemas initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize database tables: {e}")

# Initialize Vector DB collections
try:
    logger.info("Initializing vector database collections...")
    init_vector_db()
    logger.info("Vector database initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize Qdrant collections: {e}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Autonomous Multi-Agent Document Intelligence Platform with LangGraph routing, hybrid OCR, and long-term learning memory.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(documents.router, prefix=settings.API_V1_STR)
app.include_router(review.router, prefix=settings.API_V1_STR)
app.include_router(memory.router, prefix=settings.API_V1_STR)

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PREPROCESSED_DIR, exist_ok=True)
os.makedirs(settings.PATCHES_DIR, exist_ok=True)

# Mount local storage folder under /storage to serve patches, raw uploads and preprocessed documents to the frontend
storage_dir = BASE_DIR / "storage"
os.makedirs(storage_dir, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_dir)), name="storage")

# Create folder for frontend assets
frontend_dir = BASE_DIR / "static"
os.makedirs(frontend_dir, exist_ok=True)

@app.get("/ui/index.html")
def redirect_ui_index():
    """Redirects direct calls to static index.html to the root route to prevent cache issues."""
    return RedirectResponse(url="/", status_code=307)

# Mount frontend directory
app.mount("/ui", StaticFiles(directory=str(frontend_dir)), name="ui")

@app.get("/")
def read_root():
    """Serves the dashboard interface with no-cache headers."""
    index_file = frontend_dir / "index.html"
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(
            content=content,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        )
    return HTMLResponse(content="<h1>Frontend static index.html not found</h1>", status_code=404)

