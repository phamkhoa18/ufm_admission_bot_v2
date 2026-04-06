# app/core/config/care.py
# Pydantic config cho Care Node — CHỈ chứa model config.
# Model config → models_config.yaml section "care:"
# Prompts      → prompts_config.yaml section "care_node:"
# Contact info → contact_loader.get_contact_block()  (Single Source of Truth)

from pydantic import BaseModel, Field
from app.core.config import models_yaml_data


_care = models_yaml_data.get("care", {})


class CareConfig(BaseModel):
    """Config cho Care Node — chỉ giữ model config."""
    provider: str = _care.get("provider", "openrouter")
    model: str = _care.get("model", "qwen/qwen3.5-flash-02-23")
    temperature: float = Field(
        default=_care.get("temperature", 0.6),
        ge=0.0, le=2.0,
    )
    max_tokens: int = _care.get("max_tokens", 300)
