"""
Pydantic config cho Form Agent.

- FormSettings: Model config (selector / extractor / drafter) từ models_config.yaml
- FormFieldDef: Key trích xuất từ form_config.yaml
- form_cfg: Singleton instance
"""

from typing import List
from pydantic import BaseModel
from app.core.config import _load_yaml, models_yaml_data
from app.utils.logger import get_logger

_logger = get_logger(__name__)

_fm_settings = models_yaml_data.get("form", {})


class FormSettings(BaseModel):
    """Model config cho 3 bước: Selector → Extractor → Drafter."""
    provider: str = _fm_settings.get("provider", "openrouter")

    # Selector (chọn mẫu đơn)
    selector_model: str = _fm_settings.get("selector", {}).get("model", "google/gemini-2.0-flash-001")
    selector_temperature: float = _fm_settings.get("selector", {}).get("temperature", 0.0)
    selector_max_tokens: int = _fm_settings.get("selector", {}).get("max_tokens", 50)
    selector_timeout: int = _fm_settings.get("selector", {}).get("timeout_seconds", 5)

    # Extractor (trích xuất thông tin cá nhân)
    extractor_model: str = _fm_settings.get("extractor", {}).get("model", "google/gemini-2.5-flash")
    extractor_temperature: float = _fm_settings.get("extractor", {}).get("temperature", 0.0)
    extractor_max_tokens: int = _fm_settings.get("extractor", {}).get("max_tokens", 800)
    extractor_timeout: int = _fm_settings.get("extractor", {}).get("timeout_seconds", 10)

    # Drafter (soạn thảo văn bản)
    drafter_model: str = _fm_settings.get("drafter", {}).get("model", "google/gemini-2.5-flash")
    drafter_temperature: float = _fm_settings.get("drafter", {}).get("temperature", 0.4)
    drafter_temperature_no_template: float = _fm_settings.get("drafter", {}).get("temperature_no_template", 0.3)
    drafter_max_tokens: int = _fm_settings.get("drafter", {}).get("max_tokens", 4000)
    drafter_timeout: int = _fm_settings.get("drafter", {}).get("timeout_seconds", 20)


class FormFieldDef(BaseModel):
    """Định nghĩa 1 key trích xuất cho Extractor."""
    key: str
    label: str
    extract_hint: str = ""


class FormConfig(BaseModel):
    """Gộp settings + fields."""
    settings: FormSettings = FormSettings()
    fields: List[FormFieldDef] = []


# ── Singleton — Load 1 lần khi import ──
try:
    _form_data = _load_yaml("form_config.yaml")
    form_cfg = FormConfig(
        fields=[FormFieldDef(**f) for f in _form_data.get("fields", [])],
    )
except Exception as e:
    _logger.warning("FormConfig - Loi khi load form_config.yaml: %s", e)
    form_cfg = FormConfig()
