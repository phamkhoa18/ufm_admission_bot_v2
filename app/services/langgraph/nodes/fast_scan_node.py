"""
Fast-Scan Node — Chốt 1: Chặn thô TRƯỚC khi gọi LLM.

Vị trí trong Graph:
  [START] → [fast_scan_node] → [context_node] → ...
                              ↘ [END] (nếu bị chặn)

Nhiệm vụ:
  Quét user_query thô bằng Regex + kiểm tra độ dài.
  KHÔNG GỌI API (trừ khi query dài cần tóm tắt).

Layers:
  0a: Input Validation — Hard limit (chống DoS)
  0b: Long Query Summarizer — Tóm tắt query dài bằng LLM nhẹ
  1a: Keyword Filter — Từ cấm nhạy cảm
  1b: Injection Filter — Prompt Injection / Jailbreak pattern
"""

import time

from app.services.langgraph.state import GraphState
from app.utils.guardian_utils import GuardianService
from app.utils.query_summarizer import summarize_long_query
from app.core.config import query_flow_config
from app.core.config.contact_loader import get_contact_block
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _blocked_state(state, query, normalized, original_query,
                   query_was_summarized, layer, label, msg, elapsed):
    """Helper tạo state bị chặn — tránh duplicate giữa các layer."""
    return {
        **state,
        "user_query": query,
        "normalized_query": normalized,
        "original_query": original_query,
        "query_was_summarized": query_was_summarized,
        "fast_scan_passed": False,
        "fast_scan_blocked_layer": layer,
        "fast_scan_message": f"[Fast-Scan {label} — {elapsed:.3f}s] {msg}",
        "final_response": f"{msg}\n{get_contact_block()}",
        "response_source": "fast_scan",
    }


def fast_scan_node(state: GraphState) -> GraphState:
    """
    Fast-Scan — Chặn thô trên user_query gốc.

    Input:  state["user_query"]
    Output: state["fast_scan_passed"], state["normalized_query"],
            state["original_query"], state["query_was_summarized"]
    """
    query = state.get("user_query", "")
    start_time = time.time()
    iv_config = query_flow_config.input_validation
    original_query = ""
    query_was_summarized = False

    # ── Layer 0a: Hard Limit — Chống DoS ──
    if len(query) > iv_config.max_input_chars:
        elapsed = time.time() - start_time
        logger.info("FastScan L0a BLOCKED: %d chars > %d max (%.3fs)",
                    len(query), iv_config.max_input_chars, elapsed)
        return {
            **state,
            "normalized_query": query[:200].lower(),
            "original_query": "",
            "query_was_summarized": False,
            "fast_scan_passed": False,
            "fast_scan_blocked_layer": 0,
            "fast_scan_message": f"[Fast-Scan L0a — {elapsed:.3f}s] Chan DoS: {len(query)} chars",
            "final_response": f"{iv_config.fallback_too_long}\n{get_contact_block()}",
            "response_source": "fast_scan",
        }

    # ── Layer 0b: Long Query Summarizer ──
    if len(query) >= iv_config.summarize_threshold:
        logger.info("FastScan L0b: Query dai %d chars (>= %d) -> tom tat...",
                    len(query), iv_config.summarize_threshold)
        summarized, success = summarize_long_query(query)

        original_query = query
        query = summarized
        query_was_summarized = True

        elapsed_sum = time.time() - start_time
        if success:
            logger.info("FastScan L0b: OK (%.3fs) | %d -> %d chars",
                        elapsed_sum, len(original_query), len(summarized))
        else:
            logger.warning("FastScan L0b: LLM loi -> fallback cat cung (%.3fs)",
                           elapsed_sum)

    # ── Chuẩn hóa teencode ──
    normalized = GuardianService.normalize_text(query)

    # ── Layer 1a: Keyword Filter ──
    is_valid, msg = GuardianService.check_layer_1_keyword_filter(normalized)
    if not is_valid:
        elapsed = time.time() - start_time
        return _blocked_state(state, query, normalized, original_query,
                              query_was_summarized, 1, "L1a", msg, elapsed)

    # ── Layer 1b: Injection Filter ──
    is_valid, msg = GuardianService.check_layer_1b_injection_filter(normalized)
    if not is_valid:
        elapsed = time.time() - start_time
        return _blocked_state(state, query, normalized, original_query,
                              query_was_summarized, 1, "L1b", msg, elapsed)

    # ── PASS ──
    elapsed = time.time() - start_time
    scan_msg = f"[Fast-Scan PASS — {elapsed:.3f}s] Sach, cho qua Context Node"
    if query_was_summarized:
        scan_msg = (
            f"[Fast-Scan PASS — {elapsed:.3f}s] "
            f"Query da tom tat ({len(original_query)} -> {len(query)} chars)"
        )

    return {
        **state,
        "user_query": query,
        "normalized_query": normalized,
        "original_query": original_query,
        "query_was_summarized": query_was_summarized,
        "fast_scan_passed": True,
        "fast_scan_blocked_layer": None,
        "fast_scan_message": scan_msg,
    }
