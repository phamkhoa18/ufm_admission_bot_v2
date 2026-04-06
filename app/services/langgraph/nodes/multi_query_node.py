"""
Multi-Query Node — Sinh biến thể câu hỏi để tăng Recall cho Vector Search.

Vị trí trong Graph:
  [intent_node] → [multi_query_node] → [embedding_node] → [rag_node]

Nhiệm vụ:
  Nhận standalone_query → Gọi LLM → Sinh N biến thể đa từ vựng.

Ví dụ:
  "Học phí ngành Marketing là bao nhiêu?"
  → ["Chi phí đào tạo ngành Marketing tại UFM",
      "Mức học phí chương trình cử nhân Marketing UFM",
      "Ngành Marketing UFM thu bao nhiêu tiền một kỳ"]
"""

import re
import time

from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_variants(raw_output: str) -> list:
    """Parse output dạng "1. ...\\n2. ..." thành list[str]."""
    variants = []
    for line in raw_output.strip().split("\n"):
        cleaned = re.sub(r"^\s*[\d]+[\.)\]]\s*", "", line.strip())
        cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned).strip()
        if cleaned and len(cleaned) > 5:
            variants.append(cleaned)
    return variants


def multi_query_node(state: GraphState) -> GraphState:
    """
    Multi-Query Node — Sinh biến thể câu hỏi.

    Input:  state["standalone_query"]
    Output: state["multi_queries"] (list biến thể, rỗng nếu tắt/lỗi)
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    config = query_flow_config.multi_query
    start_time = time.time()

    # Multi-Query bị tắt
    if not config.enabled:
        return {**state, "multi_queries": []}

    try:
        # Render system prompt (thay {{ num_variants }})
        sys_prompt_raw = prompt_manager.get_system("multi_query_node")
        sys_prompt = sys_prompt_raw.replace(
            "{{ num_variants }}", str(config.num_variants)
        )

        user_content = prompt_manager.render_user(
            "multi_query_node",
            standalone_query=standalone_query,
        )

        raw_output = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=config,
            node_key="multi_query",
        )

        variants = _parse_variants(raw_output)[:config.num_variants]

        elapsed = time.time() - start_time
        logger.info("Multi-Query [%.3fs] Sinh %d bien the", elapsed, len(variants))
        return {**state, "multi_queries": variants}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Multi-Query [%.3fs] Loi: %s", elapsed, e, exc_info=True)
        return {**state, "multi_queries": []}
