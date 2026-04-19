"""
Admin Router — FastAPI Endpoints cho Admin Ingestion API.

Endpoints:
  POST /api/v1/admin/login          → Lấy JWT token
  POST /api/v1/admin/ingest         → Upload .md files → Background Task
  GET  /api/v1/admin/tasks          → Liệt kê tất cả tasks
  GET  /api/v1/admin/tasks/{id}     → Poll trạng thái 1 task
  DELETE /api/v1/admin/documents    → Soft-delete chunks theo file
"""

import os
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form,
    HTTPException, Request, UploadFile, status,
)
from fastapi.responses import JSONResponse

from app.core.config.admin_config import admin_cfg
from app.core.security import (
    admin_rate_limiter,
    create_access_token,
    get_current_admin,
)
from app.services.admin.task_store import task_store, TaskStatus
from app.services.admin.ingestion_worker import process_ingestion
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Router ──
router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])

# Allowed extensions
_ALLOWED_EXT = set(admin_cfg.rate_limit.allowed_extensions)
_MAX_FILE_BYTES = admin_cfg.rate_limit.max_file_size_mb * 1024 * 1024


# ══════════════════════════════════════════════════════════
# LOGIN — Lấy JWT Token
# ══════════════════════════════════════════════════════════
@router.post("/login", summary="Đăng nhập Admin → JWT Token")
async def admin_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """
    Xác thực Admin và trả về JWT token.

    Body (form-data):
        - username: Tên đăng nhập
        - password: Mật khẩu
    """
    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    admin_rate_limiter.check(client_ip)

    # Validate credentials
    cred = admin_cfg.credentials
    if username != cred.username or password != cred.password:
        logger.warning("Admin Login FAILED: username='%s' from %s", username, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tên đăng nhập hoặc mật khẩu.",
        )

    token = create_access_token(subject=username)
    logger.info("Admin Login OK: username='%s' from %s", username, client_ip)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": admin_cfg.jwt.access_token_expire_minutes,
    }


# ══════════════════════════════════════════════════════════
# INGEST — Upload files → Background Processing
# ══════════════════════════════════════════════════════════
@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Nạp file Markdown vào VectorDB",
)
async def ingest_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Danh sách file .md"),
    program_level: Optional[str] = Form(None, description="Bậc học: thac_si/tien_si/dai_hoc"),
    program_name: Optional[str] = Form(None, description="Ngành: VD: Marketing"),
    academic_year: Optional[str] = Form(None, description="Năm học: VD: 2025-2026"),
    reference_url: Optional[str] = Form(None, description="Đường dẫn tham khảo"),
):
    if not files:
        raise HTTPException(status_code=400, detail="Chưa có file nào được upload.")

    # ── Validate & Fallback null cho metadata ──
    _VALID_LEVELS = {"thac_si", "tien_si", "dai_hoc"}
    clean_level = program_level.strip() if program_level else None
    if clean_level and clean_level not in _VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"program_level không hợp lệ: '{clean_level}'. "
                   f"Chỉ chấp nhận: {', '.join(_VALID_LEVELS)} hoặc để trống.",
        )
    clean_program = program_name.strip() if program_name else None
    clean_year = academic_year.strip() if academic_year else None
    clean_url = reference_url.strip() if reference_url else None

    logger.info(
        "Ingest - Metadata override: level=%s, program=%s, year=%s, url=%s",
        clean_level, clean_program, clean_year, clean_url
    )

    tasks_created = []

    for upload_file in files:
        filename = upload_file.filename or "unknown.md"

        ext = os.path.splitext(filename)[1].lower()
        if ext not in _ALLOWED_EXT:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": f"Chỉ chấp nhận file {', '.join(_ALLOWED_EXT)}",
            })
            continue

        content_bytes = await upload_file.read()
        if len(content_bytes) > _MAX_FILE_BYTES:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": f"File quá lớn",
            })
            continue

        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            tasks_created.append({
                "file_name": filename,
                "status": "rejected",
                "reason": "File không phải UTF-8 encoding",
            })
            continue

        task = task_store.create(file_name=filename)

        background_tasks.add_task(
            process_ingestion,
            file_name=filename,
            file_content=content,
            task=task,
            override_level=clean_level,
            override_program=clean_program,
            override_year=clean_year,
            override_url=clean_url,
        )

        tasks_created.append({
            "file_name": filename,
            "task_id": task.task_id,
            "status": "accepted",
        })
        logger.info("Ingest - Queued: '%s' → task_id=%s", filename, task.task_id)

    return {
        "total_files": len(files),
        "accepted": sum(1 for t in tasks_created if t.get("status") == "accepted"),
        "rejected": sum(1 for t in tasks_created if t.get("status") == "rejected"),
        "tasks": tasks_created,
    }


# ══════════════════════════════════════════════════════════
# COMPOSE — Convert HTML to MD and Ingest
# ══════════════════════════════════════════════════════════
from pydantic import BaseModel

class ComposeRequest(BaseModel):
    title: str
    html_content: str
    file_name: Optional[str] = ""
    program_level: Optional[str] = ""
    program_name: Optional[str] = ""
    academic_year: Optional[str] = ""
    reference_url: Optional[str] = ""

@router.post(
    "/compose",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Chuyển đổi HTML sang Markdown bằng AI và nạp vào VectorDB"
)
async def compose_content(req: ComposeRequest, background_tasks: BackgroundTasks):
    from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
    from app.core.config import query_flow_config

    sys_prompt = "Bạn là một công cụ chuyển đổi HTML sang Markdown chuyên nghiệp. Hãy chuyển nội dung tôi cung cấp sang Markdown chuẩn. Trả về kết quả dưới dạng Markdown thuần (không bọc trong thẻ ```markdown), KHÔNG bình luận thêm."
    # --- Chunking HTML for large content ---
    # Vì nội dung HTML có thể cực lớn vượt quá giới hạn output token của Gemini (8192), 
    # ta chia nhỏ HTML thành các đoạn khoảng 20000 ký tự (theo dòng) để xử lý tuần tự.
    html_lines = req.html_content.split('\n')
    html_chunks = []
    current_chunk_lines = []
    current_len = 0
    max_chunk_chars = 20000

    for line in html_lines:
        line_len = len(line)
        if current_len + line_len > max_chunk_chars and current_chunk_lines:
            html_chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_len = 0
        current_chunk_lines.append(line)
        current_len += line_len
    if current_chunk_lines:
        html_chunks.append("\n".join(current_chunk_lines))

    import re
    markdown_body_parts = []
    try:
        for idx, chunk_html in enumerate(html_chunks):
            logger.info("Compose - Đang xử lý chunk %d/%d (chiều dài: %d ký tự)", idx+1, len(html_chunks), len(chunk_html))
            user_prompt = f"Tiêu đề: {req.title} (Phần {idx+1}/{len(html_chunks)})\n\nNội dung HTML phần này:\n{chunk_html}"
            
            # Sử dụng config của main_bot (Gemini Flash) vì max_tokens=9000 đủ để chứa bài nội dung dài
            part_md = _call_gemini_api_with_fallback(
                system_prompt=sys_prompt,
                user_content=user_prompt,
                config_section=query_flow_config.main_bot,
            )
            
            # Xóa dư thừa markdown code block markers trên mỗi phần bằng regex
            part_md = part_md.strip()
            part_md = re.sub(r'^```[a-zA-Z]*\n', '', part_md, flags=re.IGNORECASE)
            part_md = re.sub(r'\n```$', '', part_md)
            part_md = part_md.strip()
            
            markdown_body_parts.append(part_md)

        markdown_body = "\n\n".join(markdown_body_parts)
    except Exception as e:
        logger.error(f"Lỗi AI chuyển đổi: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi AI chuyển đổi HTML sang Markdown: {str(e)}")

    # Xóa toàn bộ markdown delimiters còn sót
    markdown_body = markdown_body.strip()
    markdown_body = re.sub(r'^```[a-zA-Z]*\n', '', markdown_body, flags=re.IGNORECASE)
    markdown_body = re.sub(r'\n```$', '', markdown_body)
    markdown_body = markdown_body.strip()
    
    # Tạo metadata frontmatter để ingestion parser đọc được
    frontmatter_lines = ["---"]
    frontmatter_lines.append(f'title: "{req.title}"')
    if req.program_level:
        frontmatter_lines.append(f'program_level: "{req.program_level}"')
    if req.program_name:
        frontmatter_lines.append(f'program_name: "{req.program_name}"')
    if req.academic_year:
        frontmatter_lines.append(f'academic_year: "{req.academic_year}"')
    if req.reference_url:
        frontmatter_lines.append(f'reference_url: "{req.reference_url}"')
    frontmatter_lines.append("---\n-start-")
    
    final_content = "\n".join(frontmatter_lines) + "\n\n" + markdown_body

    file_name = req.file_name.strip() if req.file_name else "Ban_Nhaps.md"
    if not file_name.endswith(".md"):
        file_name += ".md"

    # Tạo task nạp
    task = task_store.create(file_name=file_name)
    background_tasks.add_task(
        process_ingestion,
        file_name=file_name,
        file_content=final_content,
        task=task,
        override_level=req.program_level,
        override_program=req.program_name,
        override_year=req.academic_year,
        override_url=req.reference_url,
    )
    
    logger.info("Compose - Queued: '%s' → task_id=%s", file_name, task.task_id)

    return {
        "file_name": file_name,
        "markdown_length": len(markdown_body),
        "markdown_preview": markdown_body[:300] + "...",
        "task_id": task.task_id,
        "status": "accepted"
    }


# ══════════════════════════════════════════════════════════
# TASK STATUS — Poll trạng thái
# ══════════════════════════════════════════════════════════
@router.get("/tasks", summary="Liệt kê tất cả tasks")
async def list_tasks():
    """Trả về danh sách tất cả ingestion tasks (mới nhất trước)."""
    return {"tasks": task_store.list_all()}


@router.get("/tasks/{task_id}", summary="Xem trạng thái 1 task")
async def get_task_status(task_id: str):
    """Poll trạng thái chi tiết của 1 ingestion task."""
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' không tồn tại.",
        )
    return task.to_dict()


# ══════════════════════════════════════════════════════════
# DOCUMENTS — Quản lý dữ liệu VectorDB
# ══════════════════════════════════════════════════════════
@router.get("/documents", summary="Lấy danh sách các tài liệu đã train")
async def list_documents():
    """Lấy danh sách các tài liệu hiện có trong VectorDB"""
    from app.services.admin.ingestion_worker import _get_db_connection
    conn = None
    try:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    kc.source,
                    MAX(kc.program_level) as program_level,
                    MAX(kc.program_name) as program_name,
                    MAX(kc.academic_year) as academic_year,
                    COUNT(*) as chunk_count,
                    SUM(kc.char_count) as total_chars,
                    bool_and(kc.is_active) as is_active,
                    MIN(kc.created_at) as created_at,
                    MAX(kc.updated_at) as updated_at
                FROM knowledge_chunks kc
                WHERE kc.is_active = TRUE
                GROUP BY kc.source
                ORDER BY MAX(kc.updated_at) DESC
            """)
            rows = cur.fetchall()
            documents = []
            for r in rows:
                documents.append({
                    "source": r[0],
                    "program_level": r[1],
                    "program_name": r[2],
                    "academic_year": r[3],
                    "chunk_count": r[4],
                    "total_chars": r[5] or 0,
                    "is_active": r[6],
                    "created_at": r[7].isoformat() if r[7] else None,
                    "updated_at": r[8].isoformat() if r[8] else None,
                })

            # Also pull files that are in ingestion_logs but might have 0 chunks or errors
            cur.execute("""
                SELECT file_name, status, chunks_count, created_at, updated_at 
                FROM ingestion_logs 
                WHERE file_name NOT IN (SELECT DISTINCT source FROM knowledge_chunks WHERE is_active = TRUE)
                ORDER BY updated_at DESC
            """)
            log_rows = cur.fetchall()
            for r in log_rows:
                documents.append({
                    "source": r[0],
                    "program_level": None,
                    "program_name": None,
                    "academic_year": None,
                    "chunk_count": r[2] or 0,
                    "total_chars": 0,
                    "is_active": False,
                    "created_at": r[3].isoformat() if r[3] else None,
                    "updated_at": r[4].isoformat() if r[4] else None,
                    "status": r[1]
                })

            return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@router.get("/documents/stats", summary="Lấy thống kê VectorDB")
async def get_documents_stats():
    """Lấy thống kê Database knowledge_chunks và ingestion_logs"""
    from app.services.admin.ingestion_worker import _get_db_connection
    conn = None
    try:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            # Stats ingestion_logs
            cur.execute("SELECT COUNT(*) FROM ingestion_logs WHERE status = 'completed'")
            total_documents = cur.fetchone()[0]

            # Stats knowledge_chunks
            cur.execute("SELECT COUNT(*), SUM(char_count) FROM knowledge_chunks WHERE is_active = TRUE")
            row = cur.fetchone()
            active_chunks = row[0] or 0
            total_characters = row[1] or 0
            
            cur.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE is_active = TRUE AND embedding IS NOT NULL")
            has_embeddings = cur.fetchone()[0] or 0

            # Group by program_level
            cur.execute("""
                SELECT program_level as level, COUNT(DISTINCT source), COUNT(*)
                FROM knowledge_chunks 
                WHERE is_active = TRUE 
                GROUP BY level
            """)
            level_rows = cur.fetchall()
            by_level = {}
            for row in level_rows:
                lvl = row[0] or "unknown"
                by_level[lvl] = {
                    "documents": row[1],
                    "chunks": row[2]
                }

            return {
                "db_status": "connected",
                "total_documents": total_documents,
                "active_chunks": active_chunks,
                "total_characters": total_characters,
                "has_embeddings": has_embeddings,
                "by_level": by_level
            }
    except Exception as e:
        logger.error("Stats Error: %s", e)
        return {"db_status": "offline", "total_documents": 0, "active_chunks": 0, "total_characters": 0, "has_embeddings": 0, "by_level": {}}
    finally:
        if conn:
            conn.close()

@router.get("/documents/detail", summary="Lấy chi tiết chunk của 1 tài liệu")
async def get_document_detail(source: str):
    from app.services.admin.ingestion_worker import _get_db_connection
    conn = None
    try:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, content, char_count, chunk_level, section_name, section_path, program_level, program_name, academic_year
                FROM knowledge_chunks 
                WHERE source = %s AND is_active = TRUE
                ORDER BY created_at ASC
            """, (source,))
            rows = cur.fetchall()
            chunks = []
            for r in rows:
                chunks.append({
                    "chunk_id": r[0],
                    "content": r[1],
                    "char_count": r[2],
                    "chunk_level": r[3] or "child",
                    "section_name": r[4],
                    "section_path": r[5],
                    "program_level": r[6],
                    "program_name": r[7],
                    "academic_year": r[8]
                })
            return {"source": source, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@router.post("/tasks/{task_id}/cancel", summary="Hủy một task ingestion")
async def cancel_task(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.update(TaskStatus.FAILED, "Bị hủy bởi Admin")
    return {"message": "Đã hủy task", "task_id": task_id}

@router.delete("/documents", summary="Soft-delete chunks theo tên file")
async def delete_document(file_name: str):
    """
    Soft-delete tất cả chunks của 1 file (is_active = FALSE).
    """
    from app.services.admin.dedup_service import DedupService
    from app.services.admin.ingestion_worker import _get_db_connection

    conn = None
    try:
        conn = _get_db_connection()
        conn.autocommit = False
        dedup = DedupService(conn)
        count = dedup.soft_delete_old_chunks(file_name)
        dedup.remove_old_log(file_name)

        logger.info("Admin DELETE: '%s' → %d chunks soft-deleted", file_name, count)

        return {
            "file_name": file_name,
            "chunks_deleted": count,
            "message": f"Đã soft-delete {count} chunks.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa: {str(e)}",
        )
    finally:
        if conn:
            conn.close()
