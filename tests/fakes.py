"""Dobles de prueba: un modelo de chat guionado y excepciones reales de los SDKs.

`FakeChatModel` hereda del modelo base de LangChain e implementa `bind_tools`,
que es lo que `with_structured_output` necesita. Así el parseo real de LangChain
(PydanticToolsParser) corre en los tests; solo se falsea la llamada al proveedor.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Los SDKs actuales están construidos sobre httpx2; versiones anteriores usan httpx.
try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

# with_structured_output nombra la herramienta igual que la clase del esquema.
NOMBRE_HERRAMIENTA = "ExtraccionTecnica"


class FakeChatModel(BaseChatModel):
    """Devuelve las respuestas guionadas en orden. Una excepción en la lista se lanza."""

    respuestas: list[Any]
    llamadas: int = 0

    def bind_tools(self, tools, **kwargs):  # firma libre, como en LangChain
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.llamadas += 1
        if not self.respuestas:
            raise AssertionError("El fake se quedó sin respuestas guionadas")
        respuesta = self.respuestas.pop(0)
        if isinstance(respuesta, BaseException):
            raise respuesta
        return ChatResult(generations=[ChatGeneration(message=respuesta)])

    @property
    def _llm_type(self) -> str:
        return "fake"


def mensaje_con_herramienta(
    args: dict, *, finish_reason: str | None = None, stop_reason: str | None = None
) -> AIMessage:
    """Lo que devuelve un proveedor con tool calling: un AIMessage con tool_calls."""
    meta = {}
    if finish_reason:
        meta["finish_reason"] = finish_reason
    if stop_reason:
        meta["stop_reason"] = stop_reason
    return AIMessage(
        content="",
        response_metadata=meta,
        tool_calls=[{"name": NOMBRE_HERRAMIENTA, "args": args, "id": "call_1", "type": "tool_call"}],
    )


def error_http(cls, status: int, mensaje: str = "error"):
    """Construye una excepción real de un SDK (openai.* o anthropic.*) con un status HTTP."""
    request = httpx.Request("POST", "https://example.invalid/v1")
    return cls(mensaje, response=httpx.Response(status, request=request), body=None)


def error_de_conexion(modulo):
    """openai.APIConnectionError o anthropic.APIConnectionError, según el módulo."""
    return modulo.APIConnectionError(request=httpx.Request("POST", "https://example.invalid/v1"))
