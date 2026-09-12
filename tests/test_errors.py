"""Jerarquía de errores del pipeline y la tupla de lo que se reintenta."""

import anthropic
import langchain_anthropic.chat_models as wrappers_anthropic
import langchain_openai.chat_models.base as wrappers_openai
import openai
import pytest

from errors import RECUPERABLES, PipelineError, SalidaIncompletaError, SalidaInvalidaError


def test_jerarquia():
    assert issubclass(SalidaIncompletaError, PipelineError)
    assert issubclass(SalidaInvalidaError, PipelineError)
    assert issubclass(PipelineError, Exception)


@pytest.mark.parametrize(
    "excepcion",
    [
        SalidaIncompletaError,
        SalidaInvalidaError,
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,  # hereda de APIConnectionError
        openai.InternalServerError,
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
        anthropic.OverloadedError,  # 529
        # Las clases con las que LangChain envuelve los errores de los SDKs:
        wrappers_openai.OpenAIAPIError,          # 5xx (así llegó un 503 de Gemini en la prueba real)
        wrappers_openai.OpenAIRateLimitError,
        wrappers_openai.OpenAIConnectionError,
        wrappers_openai.OpenAITimeoutError,
        wrappers_anthropic.AnthropicAPIError,
        wrappers_anthropic.AnthropicOverloadedError,
        wrappers_anthropic.AnthropicRateLimitError,
        wrappers_anthropic.AnthropicConnectionError,
        wrappers_anthropic.AnthropicTimeoutError,
    ],
)
def test_se_reintenta(excepcion):
    assert issubclass(excepcion, RECUPERABLES)


@pytest.mark.parametrize(
    "excepcion",
    [
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.BadRequestError,
        openai.NotFoundError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.BadRequestError,
        anthropic.NotFoundError,
        ValueError,  # configuración: key faltante, proveedor desconocido
        wrappers_openai.OpenAIAuthenticationError,
        wrappers_openai.OpenAIInvalidRequestError,
        wrappers_openai.OpenAIModelNotFoundError,
        wrappers_openai.OpenAIContextOverflowError,
        wrappers_anthropic.AnthropicInvalidRequestError,
        wrappers_anthropic.AnthropicModelNotFoundError,
        wrappers_anthropic.AnthropicContextOverflowError,
    ],
)
def test_no_se_reintenta(excepcion):
    assert not issubclass(excepcion, RECUPERABLES)
