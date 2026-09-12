"""build_model: elige el proveedor por argumento o por entorno, sin tocar la red."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from providers import (
    ANTHROPIC_MAX_TOKENS,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    build_model,
)

VARIABLES = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
)


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    """Ningún test depende del .env de la máquina."""
    for var in VARIABLES:
        monkeypatch.delenv(var, raising=False)


def test_openai_por_argumento(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-openai")
    model = build_model("openai")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == DEFAULT_OPENAI_MODEL
    assert model.openai_api_key.get_secret_value() == "key-openai"
    assert model.openai_api_base is None
    assert model.temperature == 0
    assert model.max_retries == 0


def test_openai_toma_modelo_y_base_url_del_entorno(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-gemini")
    monkeypatch.setenv("OPENAI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    model = build_model("openai")
    assert model.model_name == "gemini-3.6-flash"
    assert model.openai_api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_anthropic_por_defecto(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-anthropic")
    model = build_model()
    assert isinstance(model, ChatAnthropic)
    assert model.model == DEFAULT_ANTHROPIC_MODEL
    assert model.anthropic_api_key.get_secret_value() == "key-anthropic"
    assert model.max_retries == 0
    assert model.max_tokens == ANTHROPIC_MAX_TOKENS
    # Sin temperature: los Claude actuales rechazan el parámetro.
    assert model.temperature is None


def test_anthropic_toma_modelo_del_entorno(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    assert build_model("anthropic").model == "claude-haiku-4-5"


def test_elige_por_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "key-openai")
    assert isinstance(build_model(), ChatOpenAI)


def test_normaliza_el_nombre_del_proveedor(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-openai")
    assert isinstance(build_model("  OpenAI "), ChatOpenAI)


def test_proveedor_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        build_model("gemini")


def test_key_faltante_de_openai():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_model("openai")


def test_key_faltante_de_anthropic():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_model("anthropic")


def test_key_en_blanco_cuenta_como_faltante(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_model("anthropic")
