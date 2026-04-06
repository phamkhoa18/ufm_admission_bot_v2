import time
import uuid

# Lưu trữ tĩnh tạm thời (không dùng cachetools để tránh lỗi thư viện chưa cài trong Docker)
document_cache = {}

def create_document_session(document_data: dict, session_id: str = None) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Simple manual clear if it grows too large
    if len(document_cache) > 1000:
        document_cache.clear()

    document_cache[session_id] = {
        "timestamp": time.time(),
        "data": document_data
    }
    return session_id

def get_document_session(session_id: str) -> dict:
    item = document_cache.get(session_id)
    if not item:
        return None
    
    # TTL 5 phút
    if time.time() - item["timestamp"] > 300:
        del document_cache[session_id]
        return None
        
    return item["data"]
