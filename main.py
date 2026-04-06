"""
UFM Admission Bot — FastAPI Server.

Khởi chạy:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    - /api/v1/chat/message  → Chat API (public, domain-locked)
    - /admin                → Admin Upload UI
    - /api/v1/admin/*       → Admin Ingestion API (JWT protected)
    - /docs                 → Swagger UI
    - /health               → Health check
"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env trước khi import bất kỳ module nào
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routers.admin_router import router as admin_router
from app.api.routers.chat_router import router as chat_router
from app.core.config.chat_config import chat_cfg

# ── App Factory ──
app = FastAPI(
    title="UFM Admission Bot API",
    description="API Server cho Bot Tư Vấn Tuyển Sinh UFM",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — Domain-locked (chỉ cho phép từ domain đã đăng ký) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=chat_cfg.security.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Static Files ──
_static_dir = _PROJECT_ROOT / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Mount Routers ──
app.include_router(chat_router)      # Public chat (domain-locked)
app.include_router(admin_router)     # Admin (JWT protected)


# ── Admin UI ──
@app.get("/admin", tags=["Admin UI"], include_in_schema=False)
async def admin_ui():
    """Serve Admin Upload UI."""
    html_path = _static_dir / "admin.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return {"error": "Admin UI not found. Check static/admin.html"}


# ── Demo Chat UI ──
@app.get("/chat", tags=["Chat UI"], include_in_schema=False)
async def chat_ui():
    """Serve Demo Chat UI."""
    html_path = _static_dir / "chat.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return {"error": "Chat UI not found. Check static/chat.html"}


# ── Document Viewer UI ──
@app.get("/view-document/{session_id}", tags=["Doc Viewer"], include_in_schema=False)
async def view_document_page(session_id: str):
    """Serve Document Viewer UI."""
    html_path = _static_dir / "doc_viewer.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return {"error": "Doc Viewer UI not found. Check static/doc_viewer.html"}


# ── Health Check ──
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "ufm-admission-bot"}


# ── Startup Event ──
@app.on_event("startup")
async def on_startup():
    from app.utils.logger import get_logger
    logger = get_logger("main")
    logger.info("=" * 60)
    logger.info("  UFM Admission Bot API — Started")
    logger.info("  Chat API:  POST /api/v1/chat/message (public)")
    logger.info("  Chat UI:   http://localhost:8000/chat")
    logger.info("  Admin UI:  http://localhost:8000/admin")
    logger.info("  Swagger:   http://localhost:8000/docs")
    logger.info("  CORS Origins: %s", chat_cfg.security.allowed_origins)
    logger.info("=" * 60)
