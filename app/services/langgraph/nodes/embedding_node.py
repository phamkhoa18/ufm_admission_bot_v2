"""
Embedding Node — Nhúng standalone_query + multi_queries thành vectors.

Vị trí trong Graph:
  [multi_query_node] → [embedding_node] → [rag_node] → ...

Nhiệm vụ:
  1. Gom standalone_query + multi_queries thành 1 batch
  2. Gọi API BGE-M3 (OpenRouter) 1 lần duy nhất
  3. Trả về list vectors 1024D trong state["query_embeddings"]

Output convention:
  query_embeddings[0] = standalone_query (chính)
  query_embeddings[1..N] = multi_queries (biến thể)
  Nếu multi_queries rỗng → chỉ có 1 vector.
"""

import json
import time
import urllib.request
import urllib.error
from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _embed_batch(
    texts: list,
    api_key: str,
    base_url: str,
    model: str,
    dimensions: int,
    max_retries: int = 3,
    timeout: int = 15,
) -> list:
    """
    Gọi API Embedding cho 1 batch texts.
    Retry với exponential backoff khi gặp lỗi tạm thời.

    Returns: List[List[float]] — mỗi phần tử là 1 vector 1024D
    """
    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UFM-Admission-Bot/1.0",
        "HTTP-Referer": "https://ufm.edu.vn",
    }
    data = {
        "model": model,
        "input": texts,
        "dimensions": dimensions,
    }

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # Sort theo index để đảm bảo thứ tự trả về đúng thứ tự input
            raw_data = sorted(result["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in raw_data]

        except urllib.error.HTTPError as e:
            if e.code in {429, 500, 502, 503} and attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("Embedding API %d, retry %d/%d sau %ds...", e.code, attempt, max_retries, wait)
                time.sleep(wait)
                continue
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            raise RuntimeError(f"Embedding API Error {e.code}: {error_body}") from e

        except (urllib.error.URLError, OSError) as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("Embedding Network error, retry %d/%d sau %ds...", attempt, max_retries, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Embedding Network Error: {e}") from e


def embedding_node(state: GraphState) -> GraphState:
    """
    Embedding Node — Nhúng batch [standalone + biến thể] → vectors.

    Input:
      - state["standalone_query"]: Câu hỏi đã reformulate
      - state["multi_queries"]:    List biến thể (có thể rỗng)

    Output:
      - state["query_embeddings"]: List vectors 1024D

    Logic:
      1. Gom standalone_query + multi_queries thành 1 batch
      2. Gọi API BGE-M3 1 lần duy nhất (batch embedding)
      3. Nếu API lỗi → Fallback: query_embeddings = [] (Cache/RAG sẽ tự xử lý)
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    multi_queries = state.get("multi_queries", [])
    config = query_flow_config.embedding
    start_time = time.time()

    # ── Gom batch: standalone + các biến thể ──
    batch_texts = [standalone_query] + multi_queries
    batch_size = len(batch_texts)

    logger.info("Embedding Node - Nhung %d cau (1 goc + %d bien the)...", batch_size, len(multi_queries))

    # ── Lấy API key ──
    api_key = query_flow_config.api_keys.get_key(config.provider)
    base_url = query_flow_config.api_keys.get_base_url(config.provider)

    if not api_key:
        elapsed = time.time() - start_time
        logger.warning("Embedding [%.3fs] Chua cau hinh API Key cho '%s'", elapsed, config.provider)
        return {
            **state,
            "query_embeddings": [],
        }

    # ── Gọi API Embedding (1 batch duy nhất) ──
    try:
        embeddings = _embed_batch(
            texts=batch_texts,
            api_key=api_key,
            base_url=base_url,
            model=config.model,
            dimensions=config.dimensions,
            timeout=15,
        )

        elapsed = time.time() - start_time

        # Validate kết quả
        if len(embeddings) != batch_size:
            logger.warning("Embedding [%.3fs] Ky vong %d vectors, nhan %d", elapsed, batch_size, len(embeddings))

        logger.info(
            "Embedding [%.3fs] OK: model=%s, dims=%d, vectors=%d",
            elapsed, config.model, config.dimensions, len(embeddings)
        )

        return {
            **state,
            "query_embeddings": embeddings,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Embedding [%.3fs] Loi: %s", elapsed, e, exc_info=True)
        return {
            **state,
            "query_embeddings": [],
        }

