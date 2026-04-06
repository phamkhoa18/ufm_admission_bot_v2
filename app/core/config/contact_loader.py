"""
Contact Info Loader — Đọc thông tin liên hệ tĩnh từ file Markdown.

File contact_info.md chứa toàn bộ SĐT, email, địa chỉ phòng ban UFM.
Được cache 1 lần duy nhất khi module load — KHÔNG gọi LLM, KHÔNG tốn token.

Sử dụng:
    from app.core.config.contact_loader import get_contact_block

    # Lấy toàn bộ thông tin liên hệ (để gắn vào fallback message)
    full_contact = get_contact_block()
"""

from pathlib import Path

_CONTACT_FILE = Path(__file__).parent / "contact_info.md"

# Cache nội dung khi import module (1 lần duy nhất)
_contact_cache: str | None = None


def _load_contact() -> str:
    """Đọc file contact_info.md, cache lại."""
    global _contact_cache
    if _contact_cache is not None:
        return _contact_cache
    try:
        _contact_cache = _CONTACT_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        _contact_cache = (
            "Hotline Tuyển sinh: (028) 3772 0406 | Email: tuyensinh@ufm.edu.vn\n"
            "Website: https://tuyensinh.ufm.edu.vn"
        )
    return _contact_cache


def get_contact_block() -> str:
    """Trả toàn bộ nội dung contact_info.md (gắn vào fallback message)."""
    return _load_contact()

"""Utility to load static contact info from contact_info.md."""
