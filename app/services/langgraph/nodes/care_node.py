"""
Care Node — Xử lý nhóm intent Chăm sóc / Khiếu nại / Tâm lý.

Vị trí trong Graph:
  [intent_node] → [care_node] → [response_node] → END

Nhiệm vụ:
  Gọi Care LLM sinh câu trả lời đồng cảm, tự động gắn thông tin liên hệ
  từ contact_loader (Single Source of Truth).

Config sources:
  - Model config  → models_config.yaml section "care:"
  - System prompt → prompts_config.yaml section "care_node:"
  - Contact info  → contact_loader.get_contact_block()
  - API keys      → .env (qua query_flow_config.api_keys)

Fallback: Nếu LLM lỗi → trả thẳng contact info (bypass cứng)
"""

import json
import time
import urllib.request

from jinja2 import Environment, BaseLoader

from app.services.langgraph.state import GraphState
from app.core.config import query_flow_config, prompts_yaml_data
from app.core.config.care import CareConfig
from app.core.config.contact_loader import get_contact_block
from app.core.prompts import prompt_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Module-level singletons ──
_care_config = CareConfig()
_tone_guides = prompts_yaml_data.get("care_node", {}).get("tone_guides", {})
_jinja_env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)


def _get_tone_guide(intent: str) -> str:
    """Lấy hướng dẫn giọng điệu theo intent từ prompts_config.yaml."""
    return str(
        _tone_guides.get(intent.lower(), _tone_guides.get("default", ""))
    ).strip()


def _call_care_llm(system_prompt: str, user_query: str) -> str:
    """Gọi Care LLM qua OpenRouter/Groq API."""
    provider = _care_config.provider
    api_key = query_flow_config.api_keys.get_key(provider)
    base_url = query_flow_config.api_keys.get_base_url(provider)

    if not api_key:
        raise ValueError(f"Chưa cấu hình API Key cho provider '{provider}'")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "UFM-Admission-Bot/1.0",
        "HTTP-Referer": "https://ufm.edu.vn",
    }
    payload = {
        "model": _care_config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": _care_config.temperature,
        "max_tokens": _care_config.max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body["choices"][0]["message"]["content"].strip()


def care_node(state: GraphState) -> GraphState:
    """
    Care Node — Gọi Care LLM sinh câu trả lời đồng cảm.

    Input:
      - state["intent"]: Tên intent (HO_TRO_SINH_VIEN, KHIEU_NAI_GOP_Y)
      - state["standalone_query"]: Câu hỏi của sinh viên

    Output:
      - state["final_response"]: Câu trả lời đồng cảm + thông tin liên hệ
      - state["response_source"]: "care_template"
      - state["next_node"]: "response"
    """
    intent = state.get("intent", "")
    intent_action = state.get("intent_action", "")
    user_query = state.get("standalone_query", state.get("user_query", ""))
    start_time = time.time()

    # ── Guard: Chỉ chạy khi intent đúng là PROCEED_CARE ──
    if intent_action != "PROCEED_CARE":
        logger.info("Care Node - SKIP (intent_action='%s' != 'PROCEED_CARE')", intent_action)
        return state

    contact_text = get_contact_block()
    tone_guide = _get_tone_guide(intent)

    try:
        # Render system prompt (Jinja2: {{ tone_guide }})
        sys_prompt_raw = prompt_manager.get_system("care_node")
        system_prompt = _jinja_env.from_string(sys_prompt_raw).render(
            tone_guide=tone_guide,
        )

        # Gọi LLM sinh 2-3 câu an ủi
        care_response_raw = _call_care_llm(system_prompt, user_query)

        # Gắn thông tin liên hệ tĩnh (tiết kiệm token + chống hallucination)
        care_response = f"{care_response_raw.strip()}\n\n{contact_text}"

        elapsed = time.time() - start_time
        logger.info(
            "Care Node [%.3fs] OK, intent='%s', %d ky tu",
            elapsed, intent, len(care_response)
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "Care Node [%.3fs] LLM loi: %s -> Fallback",
            elapsed, e, exc_info=True
        )
        # Fallback: render từ prompts_config.yaml hoặc hardcoded
        fallback_raw = prompts_yaml_data.get("care_node", {}).get("fallback_message", "")
        if fallback_raw and "{{ contact_text }}" in str(fallback_raw):
            care_response = _jinja_env.from_string(str(fallback_raw)).render(
                contact_text=contact_text
            )
        else:
            care_response = (
                "Mình hiểu bạn đang cần hỗ trợ. "
                "Bạn có thể liên hệ trực tiếp qua các kênh sau:\n\n"
                f"{contact_text}\n\n"
                "Đừng ngại liên hệ nhé, đội ngũ UFM luôn sẵn sàng hỗ trợ bạn!"
            )

    return {
        **state,
        "final_response": care_response,
        "response_source": "care_template",
        "next_node": "response",
    }
