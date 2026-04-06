import uuid
from cachetools import TTLCache

# Cache lưu trữ dữ liệu tài liệu mỗi session. (TTLCache max 500 items, TTL 300s = 5 phút)
document_cache = TTLCache(maxsize=500, ttl=300)

def create_document_session(document_data: dict, session_id: str = None) -> str:
    """
    Tạo session cho nội dung tài liệu (tồn tại 5 phút).
    Nếu session_id được truyền vào (vd: chunk_id) → dùng luôn, không random.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    # Lưu toàn bộ dữ liệu (bao gồm nội dung 'content' và metadata) vào cache
    document_cache[session_id] = document_data
    return session_id

def get_document_session(session_id: str) -> dict:
    """
    Lấy thông tin tài liệu từ cache bằng session_id.
    """
    return document_cache.get(session_id)
