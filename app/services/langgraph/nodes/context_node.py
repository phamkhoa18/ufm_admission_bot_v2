"""
Context Node — Query Reformulation.

Vị trí trong Graph:
  [fast_scan_node] → [context_node] → [contextual_guard_node] → ...

Nhiệm vụ:
  1. Đọc chat_history + user_query
  2. Gọi Gemini Flash Lite reformulate câu hỏi lửng lơ
     thành câu hỏi độc lập (standalone_query).
  3. Nếu không có lịch sử → skip reformulation (tiết kiệm API call).

Fallback: Nếu API lỗi → standalone_query = user_query (fail-safe)
"""

import json
import time
import urllib.request
import urllib.error
from types import SimpleNamespace

from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _summarize_message(content: str) -> str:
    """Tóm tắt message dài bằng Gemini Flash Lite. Fallback: cắt cứng."""
    config = query_flow_config.memory.auto_summarize
    try:
        summary = _call_gemini_api_with_fallback(
            system_prompt=prompt_manager.get_system("auto_summarize"),
            user_content=content,
            config_section=config,
        )
        return summary[:config.target_length]
    except Exception:
        return content[:config.target_length] + "..."


def _build_history_prompt(chat_history: list, max_turns: int) -> str:
    """Xây chuỗi lịch sử hội thoại. Message quá dài được tóm tắt tự động."""
    if not chat_history:
        return ""

    recent = chat_history[-(max_turns * 2):]
    summarize_cfg = query_flow_config.memory.auto_summarize
    max_len = query_flow_config.memory.max_tokens_per_message

    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if summarize_cfg.enabled and len(content) > summarize_cfg.trigger_length:
            content = _summarize_message(content)
        elif len(content) > max_len:
            content = content[:max_len] + "..."

        prefix = "Người dùng" if role == "user" else "Bot"
        lines.append(f"{prefix}: {content}")

    return "\n".join(lines)


def _call_gemini_api(
    system_prompt: str,
    user_content: str,
    config_section,
) -> str:
    """Gọi API LLM (OpenRouter). Dùng chung cho Reformulation, Multi-Query, etc."""
    api_key = query_flow_config.api_keys.get_key(config_section.provider)
    base_url = query_flow_config.api_keys.get_base_url(config_section.provider)

    if not api_key:
        raise ValueError(
            f"Chưa cấu hình API Key cho provider '{config_section.provider}'"
        )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UFM-Admission-Bot/1.0",
        "HTTP-Referer": "https://ufm.edu.vn",
    }
    data = {
        "model": config_section.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": config_section.temperature,
        "max_tokens": config_section.max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config_section.timeout_seconds) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


def _call_gemini_api_with_fallback(
    system_prompt: str,
    user_content: str,
    config_section,
    node_key: str = "",
) -> str:
    """
    Gọi API LLM với cơ chế fallback tự động.

    Luồng:
      1. Thử primary model từ config_section
      2. Nếu lỗi + có node_key → đọc fallbacks[] từ models_config.yaml
      3. Thử lần lượt từng fallback model
    """
    fb_settings = query_flow_config.fallback_models.settings

    # ── Primary model ──
    try:
        return _call_gemini_api(system_prompt, user_content, config_section)
    except Exception as primary_error:
        if fb_settings.log_fallback:
            logger.warning(
                "Primary FAIL (%s/%s): %s",
                getattr(config_section, 'provider', '?'),
                getattr(config_section, 'model', '?'),
                primary_error,
            )
        if not node_key:
            raise
        last_error = primary_error

    # ── Fallback models ──
    from app.core.config import models_yaml_data
    fallbacks_raw = models_yaml_data.get(node_key, {}).get("fallbacks", []) or []

    if not fallbacks_raw:
        raise last_error

    for i, fb_entry in enumerate(fallbacks_raw[:fb_settings.max_retries]):
        try:
            temp = SimpleNamespace(
                provider=fb_entry.get("provider", "openrouter"),
                model=fb_entry.get("model", ""),
                temperature=getattr(config_section, 'temperature', 0.0),
                max_tokens=getattr(config_section, 'max_tokens', 500),
                timeout_seconds=getattr(config_section, 'timeout_seconds', 15),
            )

            time.sleep(fb_settings.retry_delay_ms / 1000)
            result = _call_gemini_api(system_prompt, user_content, temp)

            if fb_settings.log_fallback:
                logger.info(
                    "Fallback #%d OK: %s/%s (node_key='%s')",
                    i + 1, temp.provider, temp.model, node_key,
                )
            return result

        except Exception as e:
            last_error = e
            if fb_settings.log_fallback:
                logger.warning(
                    "Fallback #%d FAIL (%s/%s): %s",
                    i + 1, fb_entry.get("provider", "?"),
                    fb_entry.get("model", "?"), e,
                )

    raise last_error


def _reformulate_query(user_query: str, chat_history: list) -> str:
    """Gọi Gemini Flash Lite để reformulate câu hỏi lửng lơ thành câu hỏi độc lập."""
    config = query_flow_config.query_reformulation
    max_turns = query_flow_config.memory.max_history_turns
    history_text = _build_history_prompt(chat_history, max_turns)

    user_content = prompt_manager.render_user(
        "context_node",
        chat_history_text=history_text,
        user_query=user_query,
    )

    return _call_gemini_api(
        system_prompt=prompt_manager.get_system("context_node"),
        user_content=user_content,
        config_section=config,
    )


def context_node(state: GraphState) -> GraphState:
    """
    Context Node — Query Reformulation.

    Input:
      - state["user_query"]: Câu hỏi thô (đã qua Fast-Scan)
      - state["chat_history"]: Lịch sử hội thoại

    Output:
      - state["standalone_query"]: Câu hỏi đã reformulate (hoặc giữ nguyên)
    """
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    config = query_flow_config.query_reformulation
    start_time = time.time()
    
    # Chuẩn bị history_text dùng chung cho state
    max_turns = query_flow_config.memory.max_history_turns
    history_text = _build_history_prompt(chat_history, max_turns)

    # Reformulation bị tắt
    if not config.enabled:
        return {**state, "standalone_query": user_query, "chat_history_text": history_text}

    # Không có lịch sử → skip
    if config.skip_if_no_history and not chat_history:
        elapsed = time.time() - start_time
        logger.info("Context Node [%.3fs] Khong co history -> giu nguyen", elapsed)
        return {**state, "standalone_query": user_query, "chat_history_text": history_text}

    # Có lịch sử → reformulate
    try:
        # Vẫn cần truyền vào _reformulate_query nếu nó được gọi bằng chuỗi
        # (Ở đây ta tối ưu: Truyền trực tiếp history_text thay vì build lại)
        user_content = prompt_manager.render_user(
            "context_node",
            chat_history_text=history_text,
            user_query=user_query,
        )

        standalone = _call_gemini_api(
            system_prompt=prompt_manager.get_system("context_node"),
            user_content=user_content,
            config_section=config,
        )
        
        elapsed = time.time() - start_time
        logger.info(
            "Context Node [%.3fs] Reformulated: '%s' -> '%s'",
            elapsed, user_query[:50], standalone[:80]
        )
        return {**state, "standalone_query": standalone, "chat_history_text": history_text}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Context Node [%.3fs] Error: %s -> Fallback", elapsed, e, exc_info=True)
        return {**state, "standalone_query": user_query, "chat_history_text": history_text}
