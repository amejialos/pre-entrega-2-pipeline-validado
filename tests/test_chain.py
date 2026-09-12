"""La cadena LCEL: prompt, contador de intentos, validador, reintentos y process_text."""

import logging

import anthropic
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from chain import (
    MAX_INTENTOS,
    PROMPT,
    build_chain,
    cadena_por_defecto,
    process_text,
    registrar_intento,
    validar_salida,
)
from errors import SalidaIncompletaError, SalidaInvalidaError
from schemas import Criticidad, ExtraccionTecnica
from tests.fakes import FakeChatModel, error_de_conexion, error_http, mensaje_con_herramienta

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


# --- La cadena completa, con el fake atravesando el parseo real -----------------------

ARGS_OK = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché y persistencia.",
}
ARGS_MAL = {"tecnologias": [], "nivel_de_criticidad": "urgente", "resumen_tecnico": "x"}
ENTRADA = {"texto": "FastAPI con Redis y PostgreSQL; el pool se agota."}


def cadena_con(*respuestas):
    modelo = FakeChatModel(respuestas=list(respuestas))
    return build_chain(modelo, con_espera=False), modelo


async def test_exito_al_primer_intento():
    chain, modelo = cadena_con(mensaje_con_herramienta(ARGS_OK, finish_reason="stop"))
    resultado = await chain.ainvoke(dict(ENTRADA))
    assert isinstance(resultado, ExtraccionTecnica)
    assert resultado.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert resultado.nivel_de_criticidad is Criticidad.alta
    assert modelo.llamadas == 1


async def test_salida_invalida_y_despues_valida(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    chain, modelo = cadena_con(
        mensaje_con_herramienta(ARGS_MAL, finish_reason="stop"),
        mensaje_con_herramienta(ARGS_OK, finish_reason="stop"),
    )
    resultado = await chain.ainvoke(dict(ENTRADA))
    assert resultado.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert modelo.llamadas == 2
    assert "Intento 1:" in caplog.text
    assert "Salida inválida" in caplog.text
    assert "Intento 2:" in caplog.text
    assert "Salida válida" in caplog.text


async def test_cortada_y_despues_valida():
    chain, modelo = cadena_con(
        mensaje_con_herramienta(ARGS_OK, finish_reason="length"),
        mensaje_con_herramienta(ARGS_OK, finish_reason="stop"),
    )
    assert (await chain.ainvoke(dict(ENTRADA))).nivel_de_criticidad is Criticidad.alta
    assert modelo.llamadas == 2


async def test_cortada_en_anthropic_y_despues_valida():
    chain, modelo = cadena_con(
        mensaje_con_herramienta(ARGS_OK, stop_reason="max_tokens"),
        mensaje_con_herramienta(ARGS_OK, stop_reason="tool_use"),
    )
    await chain.ainvoke(dict(ENTRADA))
    assert modelo.llamadas == 2


async def test_invalida_tres_veces_agota_los_reintentos():
    chain, modelo = cadena_con(*[mensaje_con_herramienta(ARGS_MAL)] * MAX_INTENTOS)
    with pytest.raises(SalidaInvalidaError):
        await chain.ainvoke(dict(ENTRADA))
    assert modelo.llamadas == MAX_INTENTOS


async def test_rate_limit_se_reintenta_y_queda_en_el_log(caplog):
    caplog.set_level(logging.WARNING, logger="pipeline")
    chain, modelo = cadena_con(error_http(openai.RateLimitError, 429, "429 demasiadas requests"), mensaje_con_herramienta(ARGS_OK))
    await chain.ainvoke(dict(ENTRADA))
    assert modelo.llamadas == 2
    # Los errores del proveedor no pasan por el validador: los loguea el callback on_llm_error.
    assert "El proveedor falló: RateLimitError: 429 demasiadas requests" in caplog.text


async def test_error_de_conexion_de_anthropic_se_reintenta():
    chain, modelo = cadena_con(error_de_conexion(anthropic), mensaje_con_herramienta(ARGS_OK))
    await chain.ainvoke(dict(ENTRADA))
    assert modelo.llamadas == 2


async def test_error_de_autenticacion_no_se_reintenta():
    chain, modelo = cadena_con(error_http(openai.AuthenticationError, 401), mensaje_con_herramienta(ARGS_OK))
    with pytest.raises(openai.AuthenticationError):
        await chain.ainvoke(dict(ENTRADA))
    assert modelo.llamadas == 1


async def test_bad_request_de_anthropic_no_se_reintenta():
    chain, modelo = cadena_con(error_http(anthropic.BadRequestError, 400), mensaje_con_herramienta(ARGS_OK))
    with pytest.raises(anthropic.BadRequestError):
        await chain.ainvoke(dict(ENTRADA))
    assert modelo.llamadas == 1


async def test_cada_invocacion_cuenta_intentos_desde_uno(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    chain, _ = cadena_con(mensaje_con_herramienta(ARGS_OK), mensaje_con_herramienta(ARGS_OK))
    await chain.ainvoke(dict(ENTRADA))
    await chain.ainvoke(dict(ENTRADA))
    assert caplog.text.count("Intento 1:") == 2
    assert "Intento 2:" not in caplog.text


# --- process_text -------------------------------------------------------------------


@pytest.mark.parametrize("texto", ["", "   ", "\n\t"])
async def test_process_text_rechaza_texto_vacio_sin_llamar_al_modelo(texto):
    chain, modelo = cadena_con(mensaje_con_herramienta(ARGS_OK))
    with pytest.raises(ValueError, match="vacío"):
        await process_text(texto, chain)
    assert modelo.llamadas == 0


async def test_process_text_devuelve_el_objeto_y_loguea_inicio_y_fin(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    chain, _ = cadena_con(mensaje_con_herramienta(ARGS_OK))
    resultado = await process_text(ENTRADA["texto"], chain)
    assert isinstance(resultado, ExtraccionTecnica)
    assert f"Procesando texto de {len(ENTRADA['texto'])} caracteres" in caplog.text
    assert "Listo en" in caplog.text


async def test_process_text_loguea_error_y_propaga_si_se_agotan_los_reintentos(caplog):
    caplog.set_level(logging.ERROR, logger="pipeline")
    chain, _ = cadena_con(*[mensaje_con_herramienta(ARGS_MAL)] * MAX_INTENTOS)
    with pytest.raises(SalidaInvalidaError):
        await process_text(ENTRADA["texto"], chain)
    assert "Reintentos agotados o error permanente: SalidaInvalidaError" in caplog.text


async def test_process_text_sin_cadena_construye_la_real_desde_el_entorno(monkeypatch):
    """Sin keys, la cadena por defecto falla con un error de configuración claro."""
    for var in ("LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cadena_por_defecto.cache_clear()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        await process_text("un texto")
