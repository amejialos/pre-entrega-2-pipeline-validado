"""La cadena LCEL: prompt -> modelo con salida estructurada -> validación, con reintentos.

    registrar_intento | PROMPT | model.with_structured_output(ExtraccionTecnica, include_raw=True) | validar_salida

todo envuelto en `.with_retry(retry_if_exception_type=RECUPERABLES, stop_after_attempt=3)`.
`process_text` es la puerta de entrada: recibe un string y devuelve un
`ExtraccionTecnica` validado, o lanza una excepción si se agotaron los intentos.
"""

import logging
import time
from functools import cache

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from errors import RECUPERABLES, SalidaIncompletaError, SalidaInvalidaError
from providers import build_model
from schemas import ExtraccionTecnica

logger = logging.getLogger("pipeline")

SYSTEM_PROMPT = """Sos un analista técnico senior. Recibís un texto crudo (una descripción de \
arquitectura de software, un log de error, un reporte de incidente) y extraés información \
estructurada.

Completá los tres campos:
- tecnologias: cada tecnología, framework, servicio, base de datos o herramienta mencionada, \
con su nombre canónico (por ejemplo "FastAPI", no "fastapi" ni "una API en Python"). No \
inventes tecnologías que no aparezcan en el texto. Si el texto no menciona ninguna con \
claridad, indicá la más probable a partir del contexto.
- nivel_de_criticidad: "alta" si describe una caída, pérdida de datos o un error activo en \
producción; "media" si describe degradación, riesgo latente o un problema acotado; "baja" si \
es informativo, una mejora o no tiene impacto operativo.
- resumen_tecnico: una o dos oraciones, en español, que resuman el sistema o el problema en \
términos técnicos.

Respondé únicamente con la estructura pedida."""

# La única variable de entrada es `texto`. Sin f-strings: LangChain gestiona la sustitución.
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Texto a analizar:\n\n{texto}"),
    ]
)

# Cómo avisa cada proveedor que cortó la respuesta por falta de tokens.
CORTES_POR_TOKENS = {"finish_reason": "length", "stop_reason": "max_tokens"}

MAX_INTENTOS = 3


def registrar_intento(entrada: dict) -> dict:
    """Primer eslabón: cuenta el intento y lo loguea.

    `with_retry` reutiliza el mismo diccionario de entrada en cada intento, así
    que el contador sobrevive entre reintentos de una misma invocación.
    """
    entrada["intento"] = entrada.get("intento", 0) + 1
    logger.info("Intento %d: enviando %d caracteres al modelo", entrada["intento"], len(entrada["texto"]))
    return entrada


def validar_salida(salida: dict) -> ExtraccionTecnica:
    """Último eslabón: convierte {raw, parsed, parsing_error} en un objeto o en una excepción.

    Primero revisa el corte por tokens: un JSON truncado a veces parsea igual por
    casualidad, así que no alcanza con mirar `parsed`.
    """
    meta = salida["raw"].response_metadata or {}
    for clave, valor in CORTES_POR_TOKENS.items():
        if meta.get(clave) == valor:
            logger.warning("Salida incompleta: el modelo cortó por falta de tokens (%s=%s)", clave, valor)
            raise SalidaIncompletaError(f"El modelo cortó la respuesta por falta de tokens ({clave}={valor})")

    if salida.get("parsing_error") is not None or salida.get("parsed") is None:
        detalle = salida.get("parsing_error") or "el modelo no devolvió la estructura pedida"
        logger.warning("Salida inválida, no cumple el esquema: %s", detalle)
        raise SalidaInvalidaError(f"La salida no cumple el esquema: {detalle}")

    resultado: ExtraccionTecnica = salida["parsed"]
    logger.info(
        "Salida válida (%d tecnologías, criticidad %s)",
        len(resultado.tecnologias),
        resultado.nivel_de_criticidad.value,
    )
    return resultado


class LogDeErroresDelModelo(BaseCallbackHandler):
    """Loguea los errores que lanza el proveedor (429, 5xx, red).

    Esos errores no pasan por `validar_salida`, así que sin este callback un reintento
    por rate limit no dejaría rastro en los logs. `on_llm_error` es el hook estándar
    de LangChain para fallos del modelo.
    """

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        logger.warning("El proveedor falló: %s: %s", type(error).__name__, error)


def build_chain(model: BaseChatModel, *, con_espera: bool = True) -> Runnable:
    """Arma la cadena completa: entrada {"texto": str}, salida ExtraccionTecnica.

    `with_retry` vuelve a ejecutar toda la cadena ante una excepción de RECUPERABLES,
    hasta MAX_INTENTOS veces, con espera exponencial y jitter. `con_espera=False`
    quita la espera (para tests). Un error permanente (auth, 400) sale al primer intento.
    """
    return (
        (
            RunnableLambda(registrar_intento)
            | PROMPT
            | model.with_structured_output(ExtraccionTecnica, include_raw=True)
            | RunnableLambda(validar_salida)
        )
        .with_retry(
            retry_if_exception_type=RECUPERABLES,
            stop_after_attempt=MAX_INTENTOS,
            wait_exponential_jitter=con_espera,
        )
        .with_config(callbacks=[LogDeErroresDelModelo()])
    )


@cache
def cadena_por_defecto() -> Runnable:
    """La cadena real, construida una sola vez a partir del entorno."""
    return build_chain(build_model())


async def process_text(text: str, chain: Runnable | None = None) -> ExtraccionTecnica:
    """Procesa un texto y devuelve el objeto validado.

    Lanza ValueError si el texto está vacío (sin llamar al modelo) y deja propagar
    la última excepción de la cadena si se agotaron los reintentos o el error es
    permanente: el llamador decide qué hacer.
    """
    if not text or not text.strip():
        raise ValueError("El texto no puede estar vacío")
    if chain is None:
        chain = cadena_por_defecto()

    logger.info("Procesando texto de %d caracteres", len(text))
    inicio = time.perf_counter()
    try:
        resultado = await chain.ainvoke({"texto": text})
    except Exception as error:
        logger.error("Reintentos agotados o error permanente: %s: %s", type(error).__name__, error)
        raise
    logger.info("Listo en %.1f s", time.perf_counter() - inicio)
    return resultado
