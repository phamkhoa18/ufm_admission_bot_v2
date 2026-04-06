"""
RAG Node — Truy xuất Context từ VectorDB nội bộ + LLM Context Curator.

Vị trí trong Graph:
  [embedding_node] → [rag_node] → [response_node] hoặc [rag_search]

Nhiệm vụ:
  1. Nhận query_embeddings từ Embedding Node
  2. Gọi Hybrid Retriever: Vector + BM25 → RRF → Top N Parents
  3. Context Curator (LLM) lọc giữ info liên quan, loại noise
  4. Ghi kết quả đã curate vào state["rag_context"]

Fallback: Nếu DB chưa sẵn sàng → rag_context = ""
"""

import time

from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.services.retriever_service import hybrid_retrieve
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)

_EMPTY_RAG = {"rag_context": "", "retrieved_chunks": []}


def rag_node(state: GraphState) -> GraphState:
    """
    RAG Node — Hybrid Retrieval (Vector + BM25).

    Input:  state["standalone_query"], state["query_embeddings"]
    Output: state["rag_context"], state["retrieved_chunks"]
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    query_embeddings = state.get("query_embeddings", [])
    start_time = time.time()

    # Không có embedding -> Vẫn tiếp tục thực hiện BM25 Search (Fallback)
    primary_embedding = query_embeddings[0] if query_embeddings else []
    
    if not query_embeddings:
        logger.warning("RAG Node - Embedding API lỗi, tự động Fallback sang BM25-only Search.")

    try:
        program_level = state.get("program_level_filter")
        program_name = state.get("program_name_filter")

        result = hybrid_retrieve(
            query_text=standalone_query,
            query_embedding=primary_embedding,
            program_level=program_level,
            program_name=program_name,
            query_embeddings=query_embeddings,
        )

        raw_context = result["rag_context"]
        retrieved_chunks = result["retrieved_chunks"]
        top1_cosine = result.get("top1_cosine_score", 0.0)
        elapsed_db = time.time() - start_time

        logger.info(
            "RAG Node [%.3fs] Hybrid OK: vec=%d, bm25=%d, parents=%d, ctx=%d chars, top1=%.4f",
            elapsed_db, result['vector_count'], result['bm25_count'],
            len(result['parent_ids']), len(raw_context), top1_cosine
        )

        return {
            **state,
            "rag_context": raw_context,
            "retrieved_chunks": retrieved_chunks,
        }

    except ImportError as e:
        elapsed = time.time() - start_time
        logger.warning("RAG Node [%.3fs] psycopg2 chua cai: %s", elapsed, e)
        return {**state, **_EMPTY_RAG}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("RAG Node [%.3fs] DB error: %s", elapsed, e, exc_info=True)
        return {**state, **_EMPTY_RAG}
