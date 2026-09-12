"""Excepciones del pipeline y la lista explícita de lo que vale la pena reintentar.

`with_retry` reintenta cualquier `Exception` si no se le indica otra cosa. Acá
se define qué es recuperable: una salida que el modelo puede corregir en otro
intento, y los errores transitorios del proveedor. Lo permanente (key inválida,
request mal armado, configuración faltante) queda afuera y falla al primer intento.
"""

import anthropic
import openai


class PipelineError(Exception):
    """Base de los errores propios del pipeline."""


class SalidaIncompletaError(PipelineError):
    """El modelo cortó la respuesta por falta de tokens (finish_reason / stop_reason)."""


class SalidaInvalidaError(PipelineError):
    """La respuesta no cumple el esquema `ExtraccionTecnica`."""


# Errores que un nuevo intento puede resolver. Todo lo demás es permanente.
# APITimeoutError hereda de APIConnectionError en ambos SDKs.
RECUPERABLES: tuple[type[BaseException], ...] = (
    SalidaIncompletaError,
    SalidaInvalidaError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)
