"""
Intent Router Node — Phân loại ý định người dùng bằng LLM.

Vị trí trong Graph:
  [context_node] → [guard_node] → [intent_node] → {agent nodes} → [response_node]

Nhiệm vụ:
  Gọi IntentService (LLM) để xác định intent của standalone_query.
  Ghi kết quả vào State và điều hướng sang Node tương ứng.

Routing (graph_builder quyết định đường đi thực tế):
  PROCEED_RAG*       → multi_query → embedding → rag → ...
  PROCEED_FORM       → form → response
  PROCEED_CARE       → care → response
  GREET              → response (Template chào hỏi — $0, 0ms)
  CLARIFY            → response (Template hỏi lại — $0, 0ms)
  BLOCK_FALLBACK     → response (Fallback cứng từ YAML — $0, 0ms)
"""

import time

from app.services.langgraph.state import GraphState
from app.services.intent_service import classify_intent
from app.core.config import query_flow_config
from app.core.config.contact_loader import get_contact_block
from app.utils.query_analyzer import extract_all
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Map intent_action → next_node (dùng cho state metadata) ──
# Lưu ý: graph_builder._intent_router có logic override riêng cho RAG actions
_ACTION_TO_NODE: dict = {
    "PROCEED_RAG":            "response",
    "PROCEED_RAG_UFM_SEARCH": "rag_search",
    "PROCEED_RAG_PR_SEARCH":  "rag_search",
    "PROCEED_FORM":           "form",
    "PROCEED_PR":             "rag_search",
    "PROCEED_CARE":           "care",
    "GREET":                  "response",
    "CLARIFY":                "response",
    "BLOCK_FALLBACK":         "response",
}


def _build_state(
    state: dict,
    *,
    intent: str,
    intent_summary: str,
    intent_action: str,
    next_node: str,
    program_level: str,
    program_name: str,
    final_response: str = "",
    response_source: str = "",
) -> dict:
    """Helper: Tạo state output chuẩn cho intent_node."""
    return {
        **state,
        "intent": intent,
        "intent_summary": intent_summary,
        "intent_action": intent_action,
        "program_level_filter": program_level,
        "program_name_filter": program_name,
        "next_node": next_node,
        "final_response": final_response,
        "response_source": response_source,
    }


def _resolve_instant_response(intent_action: str, intent: str) -> tuple:
    """
    Trả (final_response, response_source) cho các intent không cần RAG.
    Trả None nếu intent cần đi qua agent nodes.
    """
    if intent_action == "GREET":
        return query_flow_config.response_templates.get_greet(), "greet_template"

    if intent_action == "CLARIFY":
        return query_flow_config.response_templates.get_clarify(), "clarify_template"

    if intent_action == "BLOCK_FALLBACK":
        semantic_cfg = query_flow_config.semantic_router
        fallback_msg = semantic_cfg.fallbacks.get(
            intent,
            semantic_cfg.fallback_out_of_scope,
        ).strip()
        return f"{fallback_msg}\n{get_contact_block()}", "intent_block"

    return None


def intent_node(state: GraphState) -> GraphState:
    """
    Intent Router Node — Phân loại intent và điều hướng.

    Input:  state["standalone_query"]
    Output: state["intent"], state["intent_action"], state["next_node"],
            state["final_response"] (nếu GREET/CLARIFY/BLOCK)
    """
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    # Phân loại intent
    result = classify_intent(standalone_query=standalone_query)
    intent = result["intent"]
    intent_summary = result["intent_summary"]
    intent_action = result["intent_action"]
    next_node = _ACTION_TO_NODE.get(intent_action, "response")
    elapsed = time.time() - start_time

    # Trích xuất metadata filter
    query_meta = extract_all(standalone_query)
    program_level = query_meta["program_level"]
    program_name = query_meta["program_name"]

    if program_level:
        logger.info("Intent Node - program_level_filter='%s'", program_level)
    if program_name:
        logger.info("Intent Node - program_name_filter='%s'", program_name)

    logger.info(
        "Intent Node [%.3fs] intent='%s' action='%s' -> %s",
        elapsed, intent, intent_action, next_node,
    )

    common = dict(
        intent=intent, intent_summary=intent_summary,
        intent_action=intent_action,
        program_level=program_level, program_name=program_name,
    )

    # Instant response: GREET / CLARIFY / BLOCK_FALLBACK
    instant = _resolve_instant_response(intent_action, intent)
    if instant is not None:
        final_response, response_source = instant
        logger.info("Intent Node - %s -> template/fallback", intent_action)
        return _build_state(
            state, **common,
            next_node="response",
            final_response=final_response,
            response_source=response_source,
        )

    # Proceed: RAG / Form / Care
    return _build_state(
        state, **common,
        next_node=next_node,
        final_response=state.get("final_response", ""),
        response_source=state.get("response_source", ""),
    )


