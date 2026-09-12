# Pipeline de extracción de entidades técnicas: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un pipeline LCEL asíncrono que recibe un párrafo técnico y devuelve un `ExtraccionTecnica` validado con Pydantic, con reintentos solo ante salidas inválidas/incompletas o errores transitorios, y logs que muestran cada intento.

**Architecture:** Cuatro eslabones (`registrar_intento | PROMPT | model.with_structured_output(ExtraccionTecnica, include_raw=True) | validar_salida`) envueltos en `.with_retry(retry_if_exception_type=RECUPERABLES, stop_after_attempt=3)`. El modelo se inyecta (`build_chain(model)`), solo `providers.py` lee el entorno, y los tests usan un fake de modelo de chat que atraviesa el parseo real de LangChain.

**Tech Stack:** Python 3.12, `uv`, langchain-core 1.6.x, langchain-openai 1.6.x, langchain-anthropic 1.7.x, pydantic 2.13, python-dotenv, pytest 9 + pytest-asyncio 1.4 (modo `auto`), httpx2 (solo tests, para construir excepciones reales de los SDKs).

**Spec:** `docs/superpowers/specs/2026-09-12-pipeline-extraccion-tecnica-design.md`

## Global Constraints

- Python `>=3.12`, entorno con `uv` (`uv sync`, `uv run ...`). No usar `pip` a mano.
- Prosa, docstrings, comentarios y mensajes de commit en **español, voz "vos"**. Commits en imperativo ("Agregar...", "Corregir...") y con la línea final `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Módulos en la **raíz** del repo: `schemas.py`, `errors.py`, `providers.py`, `chain.py`, `main.py`. Tests en `tests/`.
- Los tests **no tocan la red ni necesitan keys**. Verificar siempre con `uv run pytest -q` antes de dar una tarea por terminada.
- Ninguna key en git: `.env` ignorado, `.env.example` versionado.
- Solo `providers.py` lee `os.environ`. Los modelos se crean con `max_retries=0`.
- Logger `logging.getLogger("pipeline")`; los módulos no configuran handlers, solo `main.py`.
- Rama de trabajo: `feature/pipeline-validado`, creada desde `main` en la Task 1. Un commit por tarea como mínimo.
- Regla del `CLAUDE.md` global: al terminar cada tarea con éxito, actualizar el vault de Obsidian (registro de sesión, conceptos, lecciones, nota del proyecto, índice).

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `pyproject.toml` | dependencias, `[tool.uv] package = false`, config de pytest |
| `.env.example` | variables de entorno documentadas |
| `schemas.py` | `Criticidad`, `Tecnologia`, `ExtraccionTecnica`: el contrato |
| `errors.py` | `PipelineError`, `SalidaIncompletaError`, `SalidaInvalidaError`, `RECUPERABLES` |
| `providers.py` | `build_model(provider=None)`: el único lector del entorno |
| `chain.py` | `SYSTEM_PROMPT`, `PROMPT`, `registrar_intento`, `validar_salida`, `build_chain`, `cadena_por_defecto`, `process_text` |
| `main.py` | mini-script asíncrono con tres textos, `--provider` |
| `tests/fakes.py` | `FakeChatModel`, `mensaje_con_herramienta`, `error_http`, `error_de_conexion` |
| `tests/test_*.py` | un archivo por módulo |
| `README.md` | instalación, variables, cómo correr, cómo reintenta, salida real |

---

### Task 1: Entorno, dependencias y esqueleto del proyecto

**Files:**
- Create: `pyproject.toml`, `.env.example`, `.python-version` (lo genera `uv`), `README.md` (stub), `tests/__init__.py`

**Interfaces:**
- Produces: un entorno donde `uv run pytest -q` corre (0 tests) y `uv run python -c "import langchain_openai, langchain_anthropic"` funciona.

- [ ] **Step 1: Crear la rama de trabajo**

```bash
cd "/Users/flaviasendino/Documents/Proyectos Claude/Curso AI Engineering/Pre Entrega 2"
git checkout -b feature/pipeline-validado
```

- [ ] **Step 2: Fijar Python 3.12 y escribir `pyproject.toml`**

```bash
uv python pin 3.12
```

Contenido de `pyproject.toml`:

```toml
[project]
name = "pipeline-validado"
version = "0.1.0"
description = "Pre-entrega 2: pipeline de extracción de entidades técnicas con LCEL, Pydantic y reintentos"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langchain-core>=1.6.3",
    "langchain-openai>=1.6.2",
    "langchain-anthropic>=1.7.2",
    "pydantic>=2.13.5",
    "python-dotenv>=1.2.3",
]

# No es un paquete instalable: los módulos viven en la raíz y se importan desde ahí.
[tool.uv]
package = false

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"

[dependency-groups]
dev = [
    "pytest>=9.1.1",
    "pytest-asyncio>=1.4.0",
    "httpx2>=2.12.0",
]
```

- [ ] **Step 3: Escribir `.env.example`**

```bash
# Copiá este archivo a .env y completá las keys. El .env NUNCA se sube a git.

# Proveedor por defecto cuando el código no indica uno: openai | anthropic
LLM_PROVIDER=anthropic

# --- OpenAI ---
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# Opcional: endpoint alternativo compatible con la API de OpenAI.
# Para probar gratis con una key de Gemini (Google AI Studio):
#   OPENAI_API_KEY=<tu key de Gemini, empieza con AIza>
#   OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
#   OPENAI_MODEL=gemini-3.6-flash
# OPENAI_BASE_URL=

# --- Anthropic ---
ANTHROPIC_API_KEY=sk-ant-...
# claude-opus-5 es el default; claude-haiku-4-5 es el más barato.
ANTHROPIC_MODEL=claude-opus-5
```

- [ ] **Step 4: Stub del README y paquete de tests**

`README.md`:

```markdown
# Pipeline de extracción de entidades técnicas

Pre-entrega 2 del curso AI Engineering. (README completo en la última tarea del plan.)
```

`tests/__init__.py`: archivo vacío.

- [ ] **Step 5: Instalar y verificar**

```bash
uv sync
uv run python -c "import langchain_core, langchain_openai, langchain_anthropic; print(langchain_core.__version__, langchain_openai.__version__, langchain_anthropic.__version__)"
uv run pytest -q
```

Esperado: versiones `1.6.x 1.6.x 1.7.x` (o superiores) y `no tests ran` con código de salida 5 (pytest sin tests). Debe existir `.venv/` y `uv.lock`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version .env.example README.md tests/__init__.py
git commit -m "Crear entorno Python 3.12 con uv y esqueleto del proyecto

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: El contrato de datos (`schemas.py`)

**Files:**
- Create: `schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `Criticidad(str, Enum)` con `baja`, `media`, `alta`; `Tecnologia` (alias de tipo); `ExtraccionTecnica(BaseModel)` con `tecnologias: list[Tecnologia]`, `nivel_de_criticidad: Criticidad`, `resumen_tecnico: str`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_schemas.py`:

```python
"""El contrato de datos: qué acepta y qué rechaza ExtraccionTecnica."""

import json

import pytest
from pydantic import ValidationError

from schemas import Criticidad, ExtraccionTecnica

VALIDO = {
    "tecnologias": ["FastAPI", "Redis"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis.",
}


def test_objeto_valido():
    obj = ExtraccionTecnica(**VALIDO)
    assert obj.tecnologias == ["FastAPI", "Redis"]
    assert obj.nivel_de_criticidad is Criticidad.alta
    assert obj.resumen_tecnico == "API con caché en Redis."


def test_lista_de_tecnologias_vacia_falla():
    with pytest.raises(ValidationError, match="tecnologias"):
        ExtraccionTecnica(**{**VALIDO, "tecnologias": []})


def test_tecnologia_vacia_o_solo_espacios_falla():
    with pytest.raises(ValidationError, match="tecnologias"):
        ExtraccionTecnica(**{**VALIDO, "tecnologias": ["FastAPI", "   "]})


def test_tecnologia_se_recorta():
    obj = ExtraccionTecnica(**{**VALIDO, "tecnologias": ["  Redis "]})
    assert obj.tecnologias == ["Redis"]


def test_criticidad_fuera_del_enum_falla():
    with pytest.raises(ValidationError, match="nivel_de_criticidad"):
        ExtraccionTecnica(**{**VALIDO, "nivel_de_criticidad": "urgente"})


def test_criticidad_es_sensible_a_mayusculas():
    with pytest.raises(ValidationError):
        ExtraccionTecnica(**{**VALIDO, "nivel_de_criticidad": "ALTA"})


def test_resumen_vacio_falla():
    with pytest.raises(ValidationError, match="resumen_tecnico"):
        ExtraccionTecnica(**{**VALIDO, "resumen_tecnico": ""})


def test_campo_extra_falla():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExtraccionTecnica(**VALIDO, comentario="no debería estar")


def test_serializa_el_enum_como_texto():
    datos = json.loads(ExtraccionTecnica(**VALIDO).model_dump_json())
    assert datos == VALIDO


def test_las_descripciones_viajan_en_el_esquema_json():
    """Las description de los Field son las instrucciones de formato para el modelo."""
    props = ExtraccionTecnica.model_json_schema()["properties"]
    for campo in ("tecnologias", "nivel_de_criticidad", "resumen_tecnico"):
        assert props[campo].get("description"), f"{campo} sin description"
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `uv run pytest tests/test_schemas.py -q`
Esperado: `ModuleNotFoundError: No module named 'schemas'`.

- [ ] **Step 3: Implementar `schemas.py`**

```python
"""El contrato de datos del pipeline: la forma exacta que garantiza la salida.

El modelo de lenguaje responde en texto libre; este esquema es el molde que
`with_structured_output` le impone y que Pydantic valida. Las `description` de
cada campo viajan al proveedor dentro del esquema de la herramienta, así que
son parte de las instrucciones que recibe el modelo.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class Criticidad(str, Enum):
    """Nivel de criticidad. Hereda de `str` para serializarse como texto plano."""

    baja = "baja"
    media = "media"
    alta = "alta"


# Un nombre de tecnología: sin espacios sobrantes y nunca vacío.
Tecnologia = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExtraccionTecnica(BaseModel):
    """Resultado validado de analizar un texto técnico."""

    # Un campo inventado por el modelo es un error, no se ignora en silencio.
    model_config = ConfigDict(extra="forbid")

    tecnologias: list[Tecnologia] = Field(
        min_length=1,
        description=(
            "Tecnologías, frameworks, servicios, bases de datos o herramientas "
            "mencionados en el texto, con su nombre canónico (por ejemplo 'FastAPI', "
            "'Redis', 'PostgreSQL'). Al menos una."
        ),
    )
    nivel_de_criticidad: Criticidad = Field(
        description=(
            "'alta' si describe una caída, pérdida de datos o un error activo en "
            "producción; 'media' si describe degradación, riesgo latente o un problema "
            "acotado; 'baja' si es informativo, una mejora o no tiene impacto operativo."
        ),
    )
    resumen_tecnico: str = Field(
        min_length=1,
        description="Una o dos oraciones en español que resuman el sistema o el problema en términos técnicos.",
    )
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `uv run pytest tests/test_schemas.py -q`
Esperado: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add schemas.py tests/test_schemas.py
git commit -m "Agregar el contrato de datos ExtraccionTecnica con Pydantic

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Errores propios y qué se reintenta (`errors.py`)

**Files:**
- Create: `errors.py`
- Test: `tests/test_errors.py`

**Interfaces:**
- Produces: `PipelineError(Exception)`, `SalidaIncompletaError(PipelineError)`, `SalidaInvalidaError(PipelineError)`, `RECUPERABLES: tuple[type[BaseException], ...]`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_errors.py`:

```python
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
        openai.APITimeoutError,       # hereda de APIConnectionError
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
        ValueError,                    # configuración: key faltante, proveedor desconocido
    ],
)
def test_no_se_reintenta(excepcion):
    assert not issubclass(excepcion, RECUPERABLES)
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `uv run pytest tests/test_errors.py -q`
Esperado: `ModuleNotFoundError: No module named 'errors'`.

- [ ] **Step 3: Implementar `errors.py`**

```python
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
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `uv run pytest tests/test_errors.py -q`
Esperado: `20 passed`.

- [ ] **Step 5: Commit**

```bash
git add errors.py tests/test_errors.py
git commit -m "Agregar excepciones del pipeline y la tupla RECUPERABLES

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Construcción del modelo según el entorno (`providers.py`)

**Files:**
- Create: `providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Produces: `build_model(provider: str | None = None) -> BaseChatModel`; constantes `DEFAULT_PROVIDER = "anthropic"`, `SUPPORTED_PROVIDERS = ("openai", "anthropic")`, `DEFAULT_OPENAI_MODEL = "gpt-4o-mini"`, `DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"`, `ANTHROPIC_MAX_TOKENS = 2048`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_providers.py`:

```python
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
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `uv run pytest tests/test_providers.py -q`
Esperado: `ModuleNotFoundError: No module named 'providers'`.

- [ ] **Step 3: Implementar `providers.py`**

```python
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
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `uv run pytest tests/test_providers.py -q`
Esperado: `10 passed`. Si `ChatAnthropic` rechaza `max_tokens` como nombre de parámetro, usar `max_tokens_to_sample=ANTHROPIC_MAX_TOKENS` (es el alias) y anotarlo en el README.

- [ ] **Step 5: Commit**

```bash
git add providers.py tests/test_providers.py
git commit -m "Agregar build_model: ChatOpenAI o ChatAnthropic según LLM_PROVIDER

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Prompt, contador de intentos y validador (`chain.py`, parte 1)

**Files:**
- Create: `chain.py` (solo `SYSTEM_PROMPT`, `PROMPT`, `CORTES_POR_TOKENS`, `registrar_intento`, `validar_salida`; `build_chain` y `process_text` llegan en las tasks 6 y 7)
- Test: `tests/test_chain.py` (se amplía en las tasks 6 y 7)

**Interfaces:**
- Consumes: `ExtraccionTecnica`, `Criticidad` (Task 2); `SalidaIncompletaError`, `SalidaInvalidaError` (Task 3).
- Produces: `PROMPT: ChatPromptTemplate` con `input_variables == ["texto"]`; `registrar_intento(entrada: dict) -> dict`; `validar_salida(salida: dict) -> ExtraccionTecnica` donde `salida` tiene las claves `raw` (`AIMessage`), `parsed`, `parsing_error`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_chain.py`:

```python
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
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `uv run pytest tests/test_chain.py -q`
Esperado: `ModuleNotFoundError: No module named 'chain'`.

- [ ] **Step 3: Implementar la primera parte de `chain.py`**

```python
"""La cadena LCEL: prompt -> modelo con salida estructurada -> validación, con reintentos.

    registrar_intento | PROMPT | model.with_structured_output(ExtraccionTecnica, include_raw=True) | validar_salida

todo envuelto en `.with_retry(retry_if_exception_type=RECUPERABLES, stop_after_attempt=3)`.
`process_text` es la puerta de entrada: recibe un string y devuelve un
`ExtraccionTecnica` validado, o lanza una excepción si se agotaron los intentos.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from errors import SalidaIncompletaError, SalidaInvalidaError
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
```

- [ ] **Step 4: Correr y ver que pasan**

Run: `uv run pytest tests/test_chain.py -q`
Esperado: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add chain.py tests/test_chain.py
git commit -m "Agregar el prompt, el contador de intentos y el validador de la cadena

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Fake de modelo de chat y `build_chain` con reintentos (`chain.py`, parte 2)

**Files:**
- Create: `tests/fakes.py`
- Modify: `chain.py` (agregar imports, `MAX_INTENTOS`, `build_chain`)
- Modify: `tests/test_chain.py` (agregar la sección de la cadena)

**Interfaces:**
- Consumes: `RECUPERABLES` (Task 3); `PROMPT`, `registrar_intento`, `validar_salida` (Task 5).
- Produces: `build_chain(model: BaseChatModel, *, con_espera: bool = True) -> Runnable` (entrada `{"texto": str}`, salida `ExtraccionTecnica`); `MAX_INTENTOS = 3`. En tests: `FakeChatModel(respuestas=[...])` con atributo `llamadas`; `mensaje_con_herramienta(args, *, finish_reason=None, stop_reason=None) -> AIMessage`; `error_http(cls, status, mensaje="error")`; `error_de_conexion(modulo)`.

- [ ] **Step 1: Escribir `tests/fakes.py`**

```python
"""Dobles de prueba: un modelo de chat guionado y excepciones reales de los SDKs.

`FakeChatModel` hereda del modelo base de LangChain e implementa `bind_tools`,
que es lo que `with_structured_output` necesita. Así el parseo real de LangChain
(PydanticToolsParser) corre en los tests; solo se falsea la llamada al proveedor.
"""

from typing import Any

import anthropic
import openai
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

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001 - firma libre, como en LangChain
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


def mensaje_con_herramienta(args: dict, *, finish_reason: str | None = None, stop_reason: str | None = None) -> AIMessage:
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
```

- [ ] **Step 2: Agregar los tests de la cadena a `tests/test_chain.py`**

Al principio del archivo, sumar imports:

```python
import anthropic
import openai

from chain import MAX_INTENTOS, build_chain
from tests.fakes import FakeChatModel, error_de_conexion, error_http, mensaje_con_herramienta
```

Al final del archivo:

```python
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


async def test_rate_limit_se_reintenta():
    chain, modelo = cadena_con(error_http(openai.RateLimitError, 429), mensaje_con_herramienta(ARGS_OK))
    await chain.ainvoke(dict(ENTRADA))
    assert modelo.llamadas == 2


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
```

- [ ] **Step 3: Correr y ver que fallan**

Run: `uv run pytest tests/test_chain.py -q`
Esperado: `ImportError: cannot import name 'MAX_INTENTOS' from 'chain'`.

- [ ] **Step 4: Agregar `build_chain` a `chain.py`**

Reemplazar los imports del principio por:

```python
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from errors import RECUPERABLES, SalidaIncompletaError, SalidaInvalidaError
from schemas import ExtraccionTecnica
```

Debajo de `CORTES_POR_TOKENS`:

```python
MAX_INTENTOS = 3
```

Al final del archivo:

```python
def build_chain(model: BaseChatModel, *, con_espera: bool = True) -> Runnable:
    """Arma la cadena completa: entrada {"texto": str}, salida ExtraccionTecnica.

    `with_retry` vuelve a ejecutar toda la cadena ante una excepción de RECUPERABLES,
    hasta MAX_INTENTOS veces, con espera exponencial y jitter. `con_espera=False`
    quita la espera (para tests). Un error permanente (auth, 400) sale al primer intento.
    """
    return (
        RunnableLambda(registrar_intento)
        | PROMPT
        | model.with_structured_output(ExtraccionTecnica, include_raw=True)
        | RunnableLambda(validar_salida)
    ).with_retry(
        retry_if_exception_type=RECUPERABLES,
        stop_after_attempt=MAX_INTENTOS,
        wait_exponential_jitter=con_espera,
    )
```

- [ ] **Step 5: Correr y ver que pasan**

Run: `uv run pytest tests/test_chain.py -q`
Esperado: `20 passed`. Los tests de reintento no deben tardar: `con_espera=False` elimina el backoff.

- [ ] **Step 6: Commit**

```bash
git add chain.py tests/fakes.py tests/test_chain.py
git commit -m "Agregar build_chain con with_structured_output y reintentos selectivos

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `process_text` y la cadena por defecto (`chain.py`, parte 3)

**Files:**
- Modify: `chain.py` (agregar `cadena_por_defecto`, `process_text`)
- Modify: `tests/test_chain.py` (agregar la sección de `process_text`)

**Interfaces:**
- Consumes: `build_model` (Task 4), `build_chain` (Task 6).
- Produces: `cadena_por_defecto() -> Runnable` (memoizada con `functools.cache`); `async def process_text(text: str, chain: Runnable | None = None) -> ExtraccionTecnica`.

- [ ] **Step 1: Agregar los tests**

Imports adicionales al principio de `tests/test_chain.py`:

```python
from chain import cadena_por_defecto, process_text
```

Al final del archivo:

```python
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
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `uv run pytest tests/test_chain.py -q`
Esperado: `ImportError: cannot import name 'cadena_por_defecto' from 'chain'`.

- [ ] **Step 3: Agregar `cadena_por_defecto` y `process_text` a `chain.py`**

Sumar a los imports:

```python
import time
from functools import cache

from providers import build_model
```

Al final del archivo:

```python
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
```

- [ ] **Step 4: Correr toda la suite**

Run: `uv run pytest -q`
Esperado: `66 passed` (10 schemas + 20 errors + 10 providers + 26 chain).

- [ ] **Step 5: Commit**

```bash
git add chain.py tests/test_chain.py
git commit -m "Agregar process_text con ainvoke, logs y cadena por defecto

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Mini-script de prueba (`main.py`) y prueba real

**Files:**
- Create: `main.py`
- Modify: `.env` (local, ignorado por git)

**Interfaces:**
- Consumes: `build_model`, `SUPPORTED_PROVIDERS` (Task 4); `build_chain`, `process_text` (Tasks 6 y 7).

- [ ] **Step 1: Escribir `main.py`**

```python
"""Mini-script de prueba: procesa tres textos en paralelo y muestra el objeto validado.

Los tres casos ejercitan situaciones distintas: una descripción de arquitectura,
un log de error y un texto ambiguo (la "prueba de estrés" de la consigna). Corren
con asyncio.gather; un caso que falla imprime "error controlado" y no frena a los
otros. Los logs del logger "pipeline" muestran cada intento y cada validación.
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from chain import build_chain, process_text
from providers import SUPPORTED_PROVIDERS, build_model

TEXTOS = {
    "Descripción de arquitectura": (
        "Nuestra API pública está construida con FastAPI y corre detrás de un balanceador "
        "Nginx. Las sesiones y el caché de respuestas viven en Redis, y la persistencia "
        "principal es PostgreSQL 16 con SQLAlchemy. Desde el último despliegue, cuando "
        "superamos las 500 conexiones concurrentes el pool de PostgreSQL se agota y los "
        "tiempos de respuesta pasan de 80 ms a más de 4 segundos."
    ),
    "Log de error": (
        "2026-09-12 03:14:07 ERROR [order-service] kafka.errors.KafkaTimeoutError: Failed to "
        "update metadata after 60.0 secs.\n"
        "    at KafkaProducer.send (producer.py:412)\n"
        "    at OrderPublisher.publish (publisher.py:58)\n"
        "2026-09-12 03:14:08 CRITICAL [order-service] 1240 pedidos sin publicar; el consumidor "
        "en Spark no recibe eventos desde las 03:02. Reinicio automático del pod en Kubernetes "
        "fallido (CrashLoopBackOff)."
    ),
    "Texto ambiguo (prueba de estrés)": (
        "El equipo quiere modernizar el sistema de reportes que hoy corre en una planilla "
        "compartida. Se habló de usar algo en la nube y quizás Python, pero todavía no hay "
        "decisión tomada. Nadie se quejó del sistema actual."
    ),
}


async def procesar(titulo: str, texto: str, chain) -> None:
    try:
        resultado = await process_text(texto, chain)
        print(f"\n=== {titulo} ===\n{resultado.model_dump_json(indent=2)}")
    except Exception as error:  # un caso que falla no frena a los otros
        print(f"\n=== {titulo} ===\nerror controlado: {type(error).__name__}: {error}")


async def main(provider: str | None) -> None:
    try:
        model = build_model(provider)
    except ValueError as error:
        print(f"error de configuración: {error}")
        return
    print(f"Proveedor: {model._llm_type}, modelo: {getattr(model, 'model_name', None) or getattr(model, 'model', '?')}")
    chain = build_chain(model)
    await asyncio.gather(*(procesar(titulo, texto, chain) for titulo, texto in TEXTOS.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba el pipeline con tres textos de ejemplo.")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, help="openai o anthropic (default: LLM_PROVIDER)")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()  # carga .env al entorno; el código nunca tiene keys escritas
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # el SDK loguea cada request en INFO
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    asyncio.run(main(parse_args().provider))
```

- [ ] **Step 2: Verificar que el script arranca sin keys**

Run: `uv run python main.py --provider openai` (sin `.env`)
Esperado: `error de configuración: Falta la variable de entorno OPENAI_API_KEY. ...` y código de salida 0.

- [ ] **Step 3: Prueba real con Gemini (camino OpenAI)**

La Pre-entrega 1 ya tiene un `.env` con las mismas variables (key de Gemini + `OPENAI_BASE_URL`, y key de Anthropic):

```bash
cp "../Pre Entrega 1/.env" .env
uv run python main.py --provider openai
```

Esperado: tres bloques `=== ... ===` con JSON válido (o `error controlado` en el ambiguo si el modelo no logra cumplir el esquema tres veces), y logs `INFO pipeline: Intento 1: enviando N caracteres al modelo` / `Salida válida (...)` por cada texto. Si el endpoint de Gemini rechaza el método `json_schema` de `ChatOpenAI` (error 400 mencionando `response_format`), cambiar en `providers.py` la construcción de OpenAI para pasar `model_kwargs` no sirve; en cambio, en `build_chain` usar `model.with_structured_output(ExtraccionTecnica, include_raw=True, method="function_calling")` **solo si** `isinstance(model, ChatOpenAI)` con `openai_api_base` distinto de `None`, y documentarlo en el README como limitación de los endpoints compatibles.

- [ ] **Step 4: Prueba real con Anthropic**

```bash
uv run python main.py --provider anthropic
```

Esperado: igual que el paso anterior. Si `claude-opus-5` no está disponible en la cuenta o resulta caro, poner `ANTHROPIC_MODEL=claude-haiku-4-5` en `.env` y repetir. Guardar la salida completa de ambas corridas (copiar de la terminal) para el README de la Task 9.

- [ ] **Step 5: Ajustes que salgan de la prueba real**

Cualquier cambio en el código por lo que se vio en las pruebas reales va en su propio commit, con el motivo en el mensaje (como en la PE1: `e25f519`). Correr `uv run pytest -q` después.

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "Agregar main.py: tres textos de prueba en paralelo con logs de cada intento

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: README, revisión contra la spec, repositorio en GitHub y PR

**Files:**
- Modify: `README.md` (completo)
- Modify: `docs/superpowers/specs/2026-09-12-pipeline-extraccion-tecnica-design.md` (solo si algo cambió en la implementación)

- [ ] **Step 1: Escribir el README completo**

```markdown
# Pipeline de extracción de entidades técnicas

Pipeline asíncrono en Python 3.12 + LangChain que recibe un párrafo técnico (una
descripción de arquitectura o un log de error) y devuelve un objeto **validado con
Pydantic**: tecnologías mencionadas, nivel de criticidad y resumen técnico. Si el
modelo devuelve algo que no cumple el esquema o corta la respuesta por tokens, el
pipeline lo detecta y reintenta.

Pre-entrega 2 del curso AI Engineering: "Pipeline de procesamiento validado".

## Estructura

```
schemas.py      # Criticidad y ExtraccionTecnica: el contrato de datos (Pydantic)
errors.py       # SalidaIncompletaError, SalidaInvalidaError y RECUPERABLES (qué se reintenta)
providers.py    # build_model(): ChatOpenAI o ChatAnthropic según LLM_PROVIDER
chain.py        # PROMPT, validador, build_chain() y process_text()
main.py         # mini-script asíncrono: tres textos en paralelo
tests/          # pytest, sin red y sin keys (el modelo se reemplaza por un fake)
.env.example    # variables de entorno
docs/superpowers/  # especificación de diseño y plan de implementación
```

## La cadena

```
{"texto": ...}
  -> registrar_intento                                   log "Intento N: enviando M caracteres"
  -> PROMPT (ChatPromptTemplate: system + human)          la única variable es {texto}
  -> model.with_structured_output(ExtraccionTecnica, include_raw=True)
                                                          tool calling -> {"raw", "parsed", "parsing_error"}
  -> validar_salida                                       corte por tokens -> SalidaIncompletaError
                                                          esquema inválido -> SalidaInvalidaError
  -> ExtraccionTecnica
todo envuelto en .with_retry(retry_if_exception_type=RECUPERABLES, stop_after_attempt=3)
```

`process_text(text)` es la puerta de entrada: valida que el texto no esté vacío,
ejecuta `await chain.ainvoke({"texto": text})` y loguea inicio, intentos, validación
y duración.

## Instalación

Requiere Python 3.12 o superior.

**Con `uv` (recomendado):**

```bash
uv sync
```

**Con `venv` clásico:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install langchain-core langchain-openai langchain-anthropic pydantic python-dotenv
pip install pytest pytest-asyncio httpx2
```

## Variables de entorno

Copiá `.env.example` a `.env` y completá lo que vayas a usar. El `.env` está en
`.gitignore`: nunca se sube.

| Variable | Obligatoria | Default | Para qué |
|---|---|---|---|
| `LLM_PROVIDER` | No | `anthropic` | Proveedor cuando el código no indica uno (`openai` o `anthropic`) |
| `OPENAI_API_KEY` | Para usar OpenAI | | Key de OpenAI (o de Gemini, ver abajo) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Modelo de OpenAI |
| `OPENAI_BASE_URL` | No | | Endpoint alternativo compatible con la API de OpenAI |
| `ANTHROPIC_API_KEY` | Para usar Anthropic | | Key de Anthropic |
| `ANTHROPIC_MODEL` | No | `claude-opus-5` | Modelo de Anthropic (`claude-haiku-4-5` es el más barato) |

### Probar gratis con Gemini

Google expone un endpoint compatible con la API de OpenAI y una key gratuita en
[Google AI Studio](https://aistudio.google.com/apikey):

```
OPENAI_API_KEY=<tu key de Gemini>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-3.6-flash
```

## Correr el script de prueba

```bash
uv run python main.py                       # proveedor según LLM_PROVIDER
uv run python main.py --provider openai     # uno en particular
```

Procesa tres textos en paralelo: una descripción de arquitectura, un log de error y
un texto ambiguo (la "prueba de estrés"). Cada uno imprime el JSON validado o
`error controlado: ...` si el pipeline agotó los reintentos. Un caso que falla no
frena a los otros; el script siempre termina con código 0.

### Salida real

Corrida del <FECHA> con `--provider openai` (Gemini):

```
<PEGAR ACÁ LA SALIDA COMPLETA DE LA TERMINAL, LOGS INCLUIDOS>
```

Corrida con `--provider anthropic` (`<MODELO>`):

```
<PEGAR ACÁ LA SALIDA COMPLETA DE LA TERMINAL, LOGS INCLUIDOS>
```

## Correr los tests

```bash
uv run pytest -q
```

Los tests no tocan la red ni necesitan keys. `tests/fakes.py` define un
`FakeChatModel` que hereda del modelo base de LangChain, implementa `bind_tools` y
devuelve mensajes con `tool_calls` guionados: el parseo real de LangChain
(`with_structured_output`) corre en los tests, solo se falsea la llamada al proveedor.
Las excepciones de los SDKs se construyen reales (`openai.RateLimitError`, etc.).

## Uso desde código

```python
import asyncio
from dotenv import load_dotenv
from chain import process_text

async def main():
    load_dotenv()
    resultado = await process_text("FastAPI con Redis; el pool de PostgreSQL se agota.")
    print(resultado.tecnologias, resultado.nivel_de_criticidad.value)
    print(resultado.model_dump_json(indent=2))

asyncio.run(main())
```

Para inyectar un modelo propio: `build_chain(build_model("openai"))` y pasar la
cadena a `process_text(texto, chain)`.

## Qué se reintenta y qué no

| Situación | Excepción | ¿Se reintenta? |
|---|---|---|
| El modelo cortó por falta de tokens (`finish_reason=length` / `stop_reason=max_tokens`) | `SalidaIncompletaError` | Sí |
| La salida no cumple el esquema (lista vacía, criticidad inválida, campo extra) | `SalidaInvalidaError` | Sí |
| Rate limit (429), conexión caída, timeout, 5xx | `RateLimitError`, `APIConnectionError`, `InternalServerError` | Sí |
| Key inválida (401/403), request inválido (400), modelo inexistente (404) | `AuthenticationError`, `BadRequestError`, `NotFoundError` | No |
| Key faltante o proveedor desconocido | `ValueError` | No (falla antes de llamar) |

Tres intentos en total con espera exponencial y jitter. Los modelos se construyen con
`max_retries=0` para que `with_retry` sea la única capa de reintento.

## Decisiones de diseño

- `include_raw=True` en `with_structured_output`: sin el mensaje crudo no se puede
  ver `finish_reason`, y un JSON truncado a veces parsea igual por casualidad. El
  corte se revisa antes que el esquema.
- Reintentos por tipo de excepción, no por defecto: `with_retry` reintenta
  cualquier `Exception` si no se le indica otra cosa, y eso incluiría un 401.
- Las `description` de los campos Pydantic viajan al proveedor dentro del esquema de
  la herramienta: son las instrucciones de formato. El prompt no repite el esquema.
- Sin `temperature` para Anthropic (los Claude actuales la rechazan); `temperature=0`
  para OpenAI.
- Solo `providers.py` lee el entorno; el modelo y la cadena se inyectan por parámetro.

## Limitaciones conocidas

- Sin fallback a otro modelo: si los tres intentos fallan, la excepción llega al llamador.
- `with_structured_output` depende de tool calling o JSON mode del proveedor.
- El contador de intentos vive en el diccionario de entrada: `process_text` crea uno
  nuevo por llamada, pero si invocás la cadena directamente con el mismo dict dos
  veces, el conteo continúa.
- Solo texto plano; sin chunking para textos muy largos.
```

Reemplazar los tres marcadores `<...>` de la sección "Salida real" por la fecha, el
modelo y la salida capturada en la Task 8 (pasos 3 y 4). Si alguna salida trae la
key o datos personales, recortarla.

- [ ] **Step 2: Releer la spec contra el código**

Recorrer la tabla "La consigna y dónde se resuelve" de la spec y verificar cada fila
en el código. Si la implementación se apartó de la spec (por ejemplo, el método
`function_calling` para endpoints compatibles, el `max_tokens` de Anthropic, o el
formato de los logs), actualizar la spec en su propio commit:

```bash
git add docs/superpowers/specs/2026-09-12-pipeline-extraccion-tecnica-design.md
git commit -m "Docs: la spec refleja lo implementado

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 3: Verificación final**

```bash
uv run pytest -q
uv run python main.py --provider openai
git status --short
```

Esperado: todos los tests pasan; el script corre; `git status` no muestra `.env`.

- [ ] **Step 4: Commit del README**

```bash
git add README.md
git commit -m "Agregar README con instalación, variables, uso y salida real

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 5: Crear el repo público en GitHub y abrir el PR**

```bash
git checkout main
gh repo create amejialos/pre-entrega-2-pipeline-validado --public --source=. --remote=origin --push \
  --description "Pre-entrega 2 AI Engineering: pipeline de extracción técnica con LCEL, Pydantic y reintentos"
git checkout feature/pipeline-validado
git push -u origin feature/pipeline-validado
gh pr create --title "Pipeline de extracción de entidades técnicas (Pre-entrega 2)" --body "$(cat <<'EOF'
## Qué hay

- `schemas.py`: `ExtraccionTecnica` (Pydantic, `extra="forbid"`, lista no vacía, enum de criticidad).
- `chain.py`: `PROMPT` (`ChatPromptTemplate`, sin f-strings) | `with_structured_output(include_raw=True)` | validador, con `.with_retry()` solo sobre errores recuperables; `process_text()` con `.ainvoke()` y logs.
- `providers.py`: `ChatOpenAI` o `ChatAnthropic` por `LLM_PROVIDER`, `max_retries=0`.
- `main.py`: tres textos en paralelo. README con salida real.
- Tests sin red: fake de modelo de chat con `bind_tools`.

## Cómo probar

```bash
uv sync && uv run pytest -q
cp .env.example .env   # completar keys
uv run python main.py
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Merge y verificación del repo público**

```bash
gh pr merge --merge --delete-branch
git checkout main && git pull
gh repo view --web   # verificar que el repo es público y el README se ve
```

- [ ] **Step 7: Actualizar el vault de Obsidian**

Según el `CLAUDE.md` global: nota de sesión en `Registro/`, conceptos nuevos en
`Conceptos/`, lecciones en `Lecciones/Lecciones aprendidas - Pre-entrega 2.md`, nota
del proyecto (`Proyectos/Pre-entrega 2 - ...`: estado "entregable listo", URL del repo,
cómo correr) y `00 - Inicio.md`. Después, entregar la URL del repo en la plataforma.

---

## Self-review (hecha al escribir el plan)

**Cobertura de la spec:** esquema (Task 2), errores y `RECUPERABLES` (Task 3), proveedores con `max_retries=0` y sin `temperature` en Anthropic (Task 4), prompt sin f-strings + validador con `finish_reason` primero (Task 5), cadena con `include_raw=True` y `with_retry` selectivo (Task 6), `process_text` con `ainvoke`, logs y cadena memoizada (Task 7), `main.py` con tres textos en paralelo y prueba real (Task 8), README con salida real, repo y PR (Task 9). Tests: cada fila de la tabla de la sección 7 de la spec tiene su test en las tasks 2 a 7.

**Desvíos respecto de la spec, a reflejar en la Task 9:** (1) `ANTHROPIC_MAX_TOKENS = 2048` no estaba en la spec; (2) el log de éxito es `Salida válida (...)` sin el número de intento, porque el validador no lo conoce (el intento se lee en la línea anterior del log); (3) `build_chain` tiene el parámetro `con_espera` para que los tests no esperen el backoff.

**Consistencia de nombres:** `build_model`, `build_chain(model, *, con_espera)`, `process_text(text, chain=None)`, `cadena_por_defecto`, `registrar_intento`, `validar_salida`, `PROMPT`, `MAX_INTENTOS`, `RECUPERABLES`, `FakeChatModel(respuestas=...)`, `mensaje_con_herramienta`, `error_http`, `error_de_conexion` se usan con la misma firma en todas las tasks.
