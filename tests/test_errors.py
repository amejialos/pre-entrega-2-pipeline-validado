"""Jerarquía de errores del pipeline y la tupla de lo que se reintenta."""

import anthropic
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
    ],
)
def test_no_se_reintenta(excepcion):
    assert not issubclass(excepcion, RECUPERABLES)
