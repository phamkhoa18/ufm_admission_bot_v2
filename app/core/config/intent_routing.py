# app/core/config/intent_routing.py
# Intent → Action mapping, response templates, edge-case threshold.
# Config sources:
#   intent_routing_config.yaml → action mapping, thresholds
#   prompts_config.yaml        → response templates (GREET / CLARIFY)

import random
from pydantic import BaseModel, Field
from typing import Dict
from app.core.config import intent_routing_yaml_data, prompts_yaml_data


# ── Edge Case Threshold ──
_thr = intent_routing_yaml_data.get("thresholds", {})

class IntentThresholdConfig(BaseModel):
    min_query_length: int = Field(
        default=_thr.get("min_query_length", 5),
        ge=1,
        description="Câu < min_query_length ký tự → CHAO_HOI ngay, không gọi LLM."
    )


# ── Intent → Action Mapping ──
_actions_raw = intent_routing_yaml_data.get("intent_actions", {})

class IntentActionConfig(BaseModel):
    mapping: Dict[str, str] = Field(
        default=_actions_raw,
        description="intent_name → action_string"
    )

    def get_action(self, intent: str) -> str:
        """Trả về action cho 1 intent. Mặc định: CLARIFY."""
        return self.mapping.get(intent, "CLARIFY")


# ── Response Templates (GREET & CLARIFY — không cần gọi LLM) ──
_tmpl = prompts_yaml_data.get("response_templates", {})

class ResponseTemplateConfig(BaseModel):
    greet_messages: list[str] = Field(
        default=_tmpl.get("GREET", [
            "Chào bạn! 😊 Tôi là trợ lý tư vấn tuyển sinh UFM. Bạn muốn hỏi gì?"
        ])
    )
    clarify_messages: list[str] = Field(
        default=_tmpl.get("CLARIFY", [
            "Xin lỗi bạn, bạn có thể nói rõ hơn về câu hỏi không ạ?"
        ])
    )

    def _with_contact(self, msg: str) -> str:
        """Gắn thông tin liên hệ vào cuối message."""
        from app.core.config.contact_loader import get_contact_block
        return f"{msg}\n---\n{get_contact_block()}"

    def get_greet(self) -> str:
        """Trả về 1 template chào hỏi ngẫu nhiên + thông tin liên hệ."""
        return self._with_contact(random.choice(self.greet_messages))

    def get_clarify(self) -> str:
        """Trả về 1 template hỏi lại ngẫu nhiên + thông tin liên hệ."""
        return self._with_contact(random.choice(self.clarify_messages))
