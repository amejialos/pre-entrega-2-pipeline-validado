"""Construcción del modelo de chat según la configuración.

Es el único módulo que lee variables de entorno: el resto del pipeline recibe el
modelo por parámetro (inyección de dependencias), lo que permite probarlo sin red.
Ambos modelos se crean con `max_retries=0`: la única capa de reintento es la de
`chain.py` (`with_retry`). Dos capas apiladas son impredecibles e intesteables.
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

DEFAULT_PROVIDER = "anthropic"
SUPPORTED_PROVIDERS = ("openai", "anthropic")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
# ChatAnthropic pone 128000 por defecto y el SDK rechaza pedidos sin streaming tan
# largos. Una extracción entra de sobra en 2048.
ANTHROPIC_MAX_TOKENS = 2048


def _require_env(var: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise ValueError(f"Falta la variable de entorno {var}. Copiá .env.example a .env y completala.")
    return value


def build_model(provider: str | None = None) -> BaseChatModel:
    """Devuelve ChatOpenAI o ChatAnthropic. Sin argumento usa LLM_PROVIDER (default: anthropic)."""
    name = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name == "openai":
        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
            api_key=_require_env("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            temperature=0,  # extracción: lo más determinista posible
            max_retries=0,
        )
    if name == "anthropic":
        # Sin temperature: Claude Opus 5 y Sonnet 5 rechazan el parámetro.
        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL,
            api_key=_require_env("ANTHROPIC_API_KEY"),
            max_tokens=ANTHROPIC_MAX_TOKENS,
            max_retries=0,
        )
    raise ValueError(f"Proveedor desconocido: {name!r}. Opciones: {', '.join(SUPPORTED_PROVIDERS)}.")
