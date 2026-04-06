"""
Contextual Guard Node — Chốt 2: Chặn tinh SAU khi có ngữ cảnh.

Vị trí trong Graph:
  [context_node] → [contextual_guard_node] → [intent_node] → ...
                                            ↘ [END] (nếu bị chặn)

Nhiệm vụ:
  Quét standalone_query bằng Layer 2 (chạy song song 2a Llama Guard và 2b Deep Qwen/Gemini).
  UNSAFE → block ngay. SAFE → chuyển sang Intent Node.
"""

import time
import asyncio

from app.services.langgraph.state import GraphState
from app.core.config.contact_loader import get_contact_block
from app.utils.guardian_utils import GuardianService
from app.utils.logger import get_logger

logger = get_logger(__name__)


def contextual_guard_node(state: GraphState) -> GraphState:
    """
    Contextual Guard — Quét standalone_query bằng Layer 2a và 2b song song.
    UNSAFE → block + trả contact info. SAFE → pass sang Intent Node.
    """
    query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    logger.info("CONTEXTUAL GUARD — Quet Layer 2 (2a + 2b concurrent)...")

    try:
        is_valid, msg = asyncio.run(GuardianService.check_layer_2_concurrent(query))
    except Exception as e:
        logger.error("CONTEXTUAL GUARD — Loi khi chay concurrent: %s", e)
        is_valid, msg = True, "Lỗi quét bảo mật, cho phép đi tiếp"

    elapsed = time.time() - start_time

    if not is_valid:
        logger.warning(
            "CONTEXTUAL GUARD [%.3fs] BLOCKED: %s",
            elapsed, msg
        )
        return {
            **state,
            "contextual_guard_passed": False,
            "contextual_guard_blocked_layer": "2_concurrent",
            "contextual_guard_message": f"[Contextual-Guard L2 — {elapsed:.3f}s] {msg}",
            "final_response": f"{msg}\n{get_contact_block()}",
            "response_source": "contextual_guard",
        }

    # PASS
    logger.info("CONTEXTUAL GUARD [%.3fs] PASS", elapsed)
    return {
        **state,
        "contextual_guard_passed": True,
        "contextual_guard_blocked_layer": None,
        "contextual_guard_message": f"[Contextual-Guard PASS — {elapsed:.3f}s] standalone_query an toan",
    }
