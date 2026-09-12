"""La cadena LCEL: prompt, contador de intentos, validador, reintentos y process_text."""

import logging

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from chain import PROMPT, registrar_intento, validar_salida
from errors import SalidaIncompletaError, SalidaInvalidaError
from schemas import Criticidad, ExtraccionTecnica

OBJETO = ExtraccionTecnica(
    tecnologias=["FastAPI", "Redis", "PostgreSQL"],
    nivel_de_criticidad=Criticidad.alta,
    resumen_tecnico="API con caché y persistencia.",
)


def salida(*, parsed=OBJETO, parsing_error=None, **meta):
    """Arma el dict que produce with_structured_output(include_raw=True)."""
    return {"raw": AIMessage(content="", response_metadata=meta), "parsed": parsed, "parsing_error": parsing_error}


# --- Prompt -----------------------------------------------------------------------


def test_prompt_es_un_chat_prompt_template_con_una_sola_variable():
    assert isinstance(PROMPT, ChatPromptTemplate)
    assert PROMPT.input_variables == ["texto"]


def test_prompt_tiene_roles_system_y_human_y_coloca_el_texto():
    mensajes = PROMPT.format_messages(texto="Redis se cayó")
    assert [m.type for m in mensajes] == ["system", "human"]
    assert "Redis se cayó" in mensajes[1].content
    assert "nivel_de_criticidad" in mensajes[0].content


# --- registrar_intento --------------------------------------------------------------


def test_registrar_intento_cuenta_sobre_el_mismo_dict_y_loguea(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    entrada = {"texto": "hola"}
    assert registrar_intento(entrada) is entrada
    registrar_intento(entrada)
    assert entrada["intento"] == 2
    assert "Intento 1: enviando 4 caracteres" in caplog.text
    assert "Intento 2: enviando 4 caracteres" in caplog.text


# --- validar_salida -----------------------------------------------------------------


def test_validar_salida_devuelve_el_objeto_y_loguea(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    assert validar_salida(salida(finish_reason="stop")) is OBJETO
    assert "Salida válida (3 tecnologías, criticidad alta)" in caplog.text


def test_corte_por_tokens_en_openai(caplog):
    caplog.set_level(logging.WARNING, logger="pipeline")
    with pytest.raises(SalidaIncompletaError, match="finish_reason=length"):
        validar_salida(salida(finish_reason="length"))
    assert "Salida incompleta" in caplog.text


def test_corte_por_tokens_en_anthropic():
    with pytest.raises(SalidaIncompletaError, match="stop_reason=max_tokens"):
        validar_salida(salida(stop_reason="max_tokens"))


def test_corte_por_tokens_gana_aunque_haya_parsed():
    """Un JSON truncado puede parsear por casualidad: el corte se revisa primero."""
    with pytest.raises(SalidaIncompletaError):
        validar_salida(salida(parsed=OBJETO, finish_reason="length"))


def test_parsing_error_lanza_salida_invalida(caplog):
    caplog.set_level(logging.WARNING, logger="pipeline")
    try:
        ExtraccionTecnica(tecnologias=[], nivel_de_criticidad="alta", resumen_tecnico="x")
    except ValidationError as error:
        error_real = error
    with pytest.raises(SalidaInvalidaError, match="tecnologias"):
        validar_salida(salida(parsed=None, parsing_error=error_real, finish_reason="stop"))
    assert "Salida inválida" in caplog.text


def test_parsed_none_sin_parsing_error_tambien_es_invalida():
    with pytest.raises(SalidaInvalidaError, match="no devolvió la estructura"):
        validar_salida(salida(parsed=None, finish_reason="stop"))


def test_sin_metadata_no_es_corte():
    assert validar_salida(salida()) is OBJETO
