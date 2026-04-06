"""
Response Node — LLM Phản Hồi Chính (Node cuối cùng).

Vị trí trong Graph:
  [rag_search / rag_node / intent_node / care / form] → [response_node] → END

Nhiệm vụ:
  1. Nếu đã có final_response (GREET/CLARIFY/BLOCK/Guard/Care/Form) → Bypass
  2. Nếu cần sinh câu trả lời từ context → Gọi Main Bot LLM
  3. Ghi kết quả vào state["final_response"]
"""

import time

from app.services.langgraph.state import GraphState
from app.services.langgraph.nodes.context_node import _call_gemini_api_with_fallback
from app.core.config import query_flow_config
from app.core.config.contact_loader import get_contact_block
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Response sources không cần gọi LLM
_BYPASS_SOURCES = {
    "greet_template",
    "clarify_template",
    "care_template",
    "form_template",
    "intent_block",
    "fast_scan",
    "contextual_guard",
    "keyword_filter",
    "input_validation",
}


def _fallback_response(existing: str, rag_context: str, error: bool = False) -> str:
    """Tạo fallback response khi LLM rỗng hoặc lỗi."""
    if existing:
        return existing
    if rag_context:
        return rag_context
    if error:
        return (
            "Xin lỗi bạn, hệ thống đang bảo trì tạm thời. "
            "Bạn vui lòng thử lại sau giây lát hoặc liên hệ trực tiếp:\n"
            f"{get_contact_block()}"
        )
    return (
        "Xin lỗi bạn, mình chưa tìm thấy thông tin phù hợp. "
        "Bạn có thể diễn đạt cụ thể hơn hoặc liên hệ trực tiếp:\n"
        f"{get_contact_block()}"
    )


def response_node(state: GraphState) -> GraphState:
    """
    Response Node — Bypass hoặc gọi Main Bot LLM.

    Input:  state["standalone_query"], state["final_response"], state["response_source"]
    Output: state["final_response"], state["response_source"]
    """
    start_time = time.time()
    response_source = state.get("response_source", "")
    standalone_query = state.get("standalone_query", state.get("user_query", ""))
    existing_response = state.get("final_response", "")

    logger.info("RESPONSE NODE - Source: %s, Existing: %d ky tu", response_source, len(existing_response))

    # Bypass — đã có câu trả lời sẵn
    if response_source in _BYPASS_SOURCES:
        elapsed = time.time() - start_time
        logger.info("RESPONSE NODE [%.3fs] BYPASS -> %s", elapsed, response_source)
        return state

    # Generate — gọi Main Bot LLM
    config = query_flow_config.main_bot

    if not config.enabled:
        elapsed = time.time() - start_time
        logger.info("RESPONSE NODE [%.3fs] Main Bot disabled", elapsed)
        return {
            **state,
            "final_response": existing_response or state.get("rag_context", ""),
            "response_source": "main_bot_disabled",
        }

    rag_context = state.get("rag_context") or ""
    chat_history_text = state.get("chat_history_text") or ""
    web_citations = state.get("web_search_citations") or []

    try:
        sys_prompt = prompt_manager.get_system("main_bot")
        user_content = prompt_manager.render_user(
            "main_bot",
            standalone_query=standalone_query,
            final_response=existing_response,
            rag_context=rag_context,
            chat_history_text=chat_history_text,
            web_citations=web_citations,
        )

        logger.info("RESPONSE NODE - LLM %s/%s, query='%s'",
                    config.provider, config.model, standalone_query[:80])

        generated = _call_gemini_api_with_fallback(
            system_prompt=sys_prompt,
            user_content=user_content,
            config_section=config,
            node_key="main_bot",
        )

        elapsed = time.time() - start_time

        if generated and generated.strip():
            logger.info("RESPONSE NODE [%.3fs] LLM OK: %d ky tu", elapsed, len(generated))
            return {
                **state,
                "final_response": generated.strip(),
                "response_source": f"{response_source}_generated",
            }

        # LLM trả rỗng → fallback
        logger.warning("RESPONSE NODE [%.3fs] LLM tra rong -> fallback", elapsed)
        return {
            **state,
            "final_response": _fallback_response(existing_response, rag_context),
            "response_source": f"{response_source}_fallback",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("RESPONSE NODE [%.3fs] LLM error: %s", elapsed, e, exc_info=True)
        return {
            **state,
            "final_response": _fallback_response(existing_response, rag_context, error=True),
            "response_source": "main_bot_error",
        }
