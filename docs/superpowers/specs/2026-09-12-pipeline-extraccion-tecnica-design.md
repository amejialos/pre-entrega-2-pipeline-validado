# Pipeline de extracción de entidades técnicas: diseño

Pre-entrega 2 del curso AI Engineering: "Pipeline de procesamiento validado".
Fecha: 2026-09-12.

## 1. Objetivo

Un pipeline que recibe un párrafo de texto crudo (una descripción de arquitectura o un
log de error) y devuelve un objeto validado con las tecnologías mencionadas, un nivel
de criticidad y un resumen técnico. Construido con LangChain Expression Language (LCEL),
salida estructurada con Pydantic, ejecución asíncrona y reintentos ante salidas
inválidas o incompletas.

Ejemplo de salida:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

### La consigna y dónde se resuelve

| Requisito | Dónde |
|---|---|
| Esquema Pydantic: `tecnologias` (lista no vacía), `nivel_de_criticidad` (enum baja/media/alta), `resumen_tecnico` | `schemas.py` |
| `ChatPromptTemplate` modular con el texto de entrada y las instrucciones de formato, sin f-strings | `chain.py` (`PROMPT`) |
| Cadena LCEL `prompt \| model.with_structured_output(Schema)` | `chain.py` (`build_chain`) |
| Al menos un reintento automático ante JSON mal formado o incompleto (`.with_retry()`) | `chain.py` (`build_chain`) + `errors.py` |
| Detectar el corte por `finish_reason` antes de transformar | `chain.py` (`validar_salida`) |
| `async def process_text(text: str)` que ejecuta con `.ainvoke()` | `chain.py` |
| Cliente `ChatOpenAI` o `ChatAnthropic` reutilizando la lógica del Módulo 1 | `providers.py` (`build_model`) |
| Logs para observar el flujo, la validación y los reintentos | logger `pipeline` en `chain.py` |
| Mini-script de prueba asíncrono | `main.py` |
| README con instrucciones y ejemplo de salida | `README.md` |

### Criterios de evaluación (mínimo 70 %)

| Criterio | Peso | Se cubre con |
|---|---|---|
| Esquema y salida estructurada | 30 % | `schemas.py`, `with_structured_output` |
| Cadena LCEL y prompting | 30 % | `PROMPT`, `build_chain` |
| Ejecución asíncrona y resiliencia | 25 % | `process_text`, `with_retry`, logs |
| Estructura del repo y script de prueba | 15 % | archivos en la raíz, `main.py`, tests |

## 2. Estructura del repositorio

```
schemas.py           # Criticidad (enum) y ExtraccionTecnica (el contrato de datos)
errors.py            # SalidaIncompletaError, SalidaInvalidaError, RECUPERABLES
providers.py         # build_model(): ChatOpenAI o ChatAnthropic según LLM_PROVIDER
chain.py             # PROMPT, validar_salida, build_chain(), process_text()
main.py              # mini-script asíncrono con tres textos de prueba
tests/
    __init__.py
    fakes.py         # FakeChatModel: modelo de chat guionado, sin red
    test_schemas.py
    test_errors.py
    test_providers.py
    test_chain.py
docs/superpowers/    # esta spec y el plan de implementación
.env.example         # variables de entorno necesarias
.gitignore           # .env, .venv, __pycache__, .claude
pyproject.toml       # uv, Python 3.12, dependencias y config de pytest
README.md
```

Los módulos van en la raíz (no en un paquete) porque el rubro nombra literalmente
`schemas.py` y `chain.py`. Regla que se conserva de la Pre-entrega 1: cada módulo
responde una sola pregunta y se prueba sin leer los demás. Solo `providers.py` lee
el entorno.

Stack: Python 3.12 con `uv`; `langchain-core`, `langchain-openai`,
`langchain-anthropic`, `pydantic`, `python-dotenv`; `pytest` y `pytest-asyncio` en
modo `auto` para los tests. Versiones verificadas el 2026-09-12: langchain-core 1.6.3,
langchain-openai 1.6.2, langchain-anthropic 1.7.2.

## 3. Componentes

### 3.1 `schemas.py`: el contrato

```python
class Criticidad(str, Enum):
    baja = "baja"
    media = "media"
    alta = "alta"


class ExtraccionTecnica(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tecnologias: list[Tecnologia] = Field(min_length=1, description="...")
    nivel_de_criticidad: Criticidad = Field(description="...")
    resumen_tecnico: str = Field(min_length=1, description="...")
```

- `tecnologias`: lista de al menos un elemento; cada elemento es
  `Tecnologia = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]`,
  así un string vacío o solo espacios dentro de la lista también falla.
- `nivel_de_criticidad`: el enum. Cualquier otro valor ("urgente", "ALTA") falla la
  validación. `str, Enum` para que se serialice como texto.
- `resumen_tecnico`: string no vacío.
- Las `description` de cada campo viajan al proveedor dentro del esquema JSON de la
  herramienta: son parte de las instrucciones de formato. El prompt no repite el
  esquema a mano.
- `extra="forbid"`: un campo inventado por el modelo es un error, no se ignora.

### 3.2 `errors.py`: qué se reintenta

```python
class PipelineError(Exception): ...
class SalidaIncompletaError(PipelineError): ...   # el modelo cortó por tokens
class SalidaInvalidaError(PipelineError): ...     # la salida no cumple el esquema

RECUPERABLES: tuple[type[BaseException], ...] = (
    SalidaIncompletaError,
    SalidaInvalidaError,
    openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError,
    anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError,
    anthropic.OverloadedError,   # 529, no hereda de InternalServerError
)
```

`APITimeoutError` hereda de `APIConnectionError` en ambos SDKs, así que queda cubierto.
LangChain envuelve los errores de los SDKs en clases propias (`OpenAIAPIError`,
`AnthropicOverloadedError`, `AnthropicAuthenticationError`...) que heredan de las del
SDK, así que la clasificación se conserva; `test_errors.py` lo verifica para cada
wrapper. (Visto en la prueba real: un 503 de Gemini llegó como `OpenAIAPIError`.)
Errores de autenticación (401/403), `BadRequestError` (400), key faltante o proveedor
desconocido **no** están en la tupla: fallan al primer intento. Es la regla de la
consigna: no reintentar errores permanentes.

### 3.3 `providers.py`: el modelo

```python
def build_model(provider: str | None = None) -> BaseChatModel:
```

- `provider` explícito, o `LLM_PROVIDER` del entorno, o `"anthropic"` por defecto.
  Se normaliza (`strip().lower()`).
- `openai`: `ChatOpenAI(model=OPENAI_MODEL o "gpt-4o-mini", api_key=OPENAI_API_KEY,
  base_url=OPENAI_BASE_URL o None, temperature=0, max_retries=0)`.
  `OPENAI_BASE_URL` permite probar gratis con el endpoint compatible de Gemini.
- `anthropic`: `ChatAnthropic(model=ANTHROPIC_MODEL o "claude-opus-5",
  api_key=ANTHROPIC_API_KEY, max_tokens=2048, max_retries=0)`. Sin `temperature`: los
  modelos Claude actuales (Opus 5, Sonnet 5) rechazan el parámetro. `max_tokens=2048`
  porque el default de langchain-anthropic (128000) hace que el SDK exija streaming.
- Key faltante o proveedor desconocido lanzan `ValueError` con un mensaje que dice
  qué variable falta y que hay que copiar `.env.example` a `.env`.
- `max_retries=0` en ambos: `.with_retry()` es la única capa de reintento. Dos capas
  apiladas son impredecibles e intesteables (decisión heredada de la Pre-entrega 1).

### 3.4 `chain.py`: la cadena

**Prompt** (`PROMPT`, constante de módulo):

```python
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Texto a analizar:\n\n{texto}"),
])
```

`SYSTEM_PROMPT` explica el rol (analista técnico), qué va en cada campo y el criterio
de criticidad (alta: caída, pérdida de datos o error en producción; media: degradación
o riesgo latente; baja: informativo o mejora). La única variable de entrada es `texto`
(`PROMPT.input_variables == ["texto"]`). No hay f-strings: LangChain gestiona la
sustitución.

**Validador** (`validar_salida(salida: dict) -> ExtraccionTecnica`):

Recibe el dict que produce `with_structured_output(..., include_raw=True)`:
`{"raw": AIMessage, "parsed": ExtraccionTecnica | None, "parsing_error": Exception | None}`.

1. Si `raw.response_metadata` dice que el modelo cortó por tokens
   (`finish_reason == "length"` en OpenAI, `stop_reason == "max_tokens"` en Anthropic),
   loguea `WARNING` y lanza `SalidaIncompletaError`. Se revisa **antes** de mirar
   `parsed`, porque un JSON truncado puede parsear por casualidad.
2. Si `parsing_error` no es `None` o `parsed` es `None`, loguea `WARNING` con el
   detalle del error de validación y lanza `SalidaInvalidaError`.
3. Si todo está bien, loguea `INFO` con la cantidad de tecnologías y la criticidad,
   y devuelve `parsed`.

**Registro de intento** (`registrar_intento(entrada: dict) -> dict`): primer eslabón.
Incrementa `entrada["intento"]` y loguea `INFO` "Intento N: enviando M caracteres al
modelo". Funciona porque `with_retry` reutiliza el mismo diccionario de entrada en cada
intento (verificado en un spike). El prompt ignora la clave extra.

**La cadena** (`build_chain(model: BaseChatModel, *, con_espera: bool = True) -> Runnable`):

```python
return (
    RunnableLambda(registrar_intento)
    | PROMPT
    | model.with_structured_output(ExtraccionTecnica, include_raw=True)
    | RunnableLambda(validar_salida)
).with_retry(
    retry_if_exception_type=RECUPERABLES,
    stop_after_attempt=3,
    wait_exponential_jitter=con_espera,
).with_config(callbacks=[LogDeErroresDelModelo()])
```

Tres intentos en total (dos reintentos), con backoff exponencial y jitter
(`con_espera=False` lo quita, para tests). Entrada: `{"texto": str}`. Salida:
`ExtraccionTecnica`.

**Callback `LogDeErroresDelModelo`** (`BaseCallbackHandler` con `on_llm_error`): loguea
`WARNING` "El proveedor falló: <tipo>: <mensaje>". Los errores del proveedor (429, 5xx,
red) no pasan por `validar_salida`, así que sin esto un reintento por rate limit no
dejaría rastro en los logs. Se descubrió en la prueba real con un 503 de Gemini.

**`process_text`**:

```python
async def process_text(text: str, chain: Runnable | None = None) -> ExtraccionTecnica:
```

- Texto vacío o solo espacios: `ValueError` sin llamar al modelo.
- Sin `chain`, usa la cadena por defecto, que se construye una sola vez con
  `build_chain(build_model())` (memoizada con `functools.cache`). La inyección es lo
  que permite testear sin red y sin keys.
- Loguea inicio (`INFO`, largo del texto), ejecuta `await chain.ainvoke({"texto": text})`,
  loguea fin con la duración. Si los reintentos se agotan, loguea `ERROR` "reintentos
  agotados" y deja propagar la excepción: el llamador decide.

**Logs**: logger `logging.getLogger("pipeline")`. El módulo no configura handlers;
`main.py` llama a `logging.basicConfig(level=INFO)`. Secuencia típica de una corrida
con un reintento:

```
INFO  pipeline: Procesando texto de 312 caracteres
INFO  pipeline: Intento 1: enviando 312 caracteres al modelo
WARNING pipeline: Salida inválida, no cumple el esquema: 1 validation error for ExtraccionTecnica (tecnologias: List should have at least 1 item)
INFO  pipeline: Intento 2: enviando 312 caracteres al modelo
INFO  pipeline: Salida válida (3 tecnologías, criticidad alta)
INFO  pipeline: Listo en 2.4 s
```

## 4. Flujo de una llamada

```
main.py
  -> process_text(texto)
     -> chain.ainvoke({"texto": texto})            # with_retry, hasta 3 intentos
        -> registrar_intento                      # log "Intento N"
        -> PROMPT                                 # dict -> ChatPromptValue
        -> model.with_structured_output(...)      # tool calling -> {"raw", "parsed", "parsing_error"}
        -> validar_salida                         # ExtraccionTecnica, o excepción
           -> SalidaIncompletaError / SalidaInvalidaError -> with_retry reintenta
           -> AuthenticationError / BadRequestError    -> with_retry NO reintenta
     <- ExtraccionTecnica
```

## 5. Configuración (`.env.example`)

| Variable | Obligatoria | Default | Para qué |
|---|---|---|---|
| `LLM_PROVIDER` | No | `anthropic` | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Para OpenAI | | Key de OpenAI o de Gemini |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Modelo de OpenAI |
| `OPENAI_BASE_URL` | No | | Endpoint compatible (Gemini: `https://generativelanguage.googleapis.com/v1beta/openai/`) |
| `ANTHROPIC_API_KEY` | Para Anthropic | | Key de Anthropic |
| `ANTHROPIC_MODEL` | No | `claude-opus-5` | Modelo de Anthropic (`claude-haiku-4-5` es el más barato) |

`.env` está en `.gitignore`. `main.py` carga `.env` con `python-dotenv`.

## 6. `main.py`

Mini-script de prueba asíncrono, con `--provider` opcional como en la Pre-entrega 1.
Tres textos de ejemplo, elegidos para ejercitar casos distintos:

1. Descripción de arquitectura (FastAPI + Redis + PostgreSQL con un cuello de botella).
2. Log de error (stack trace con Kafka y una excepción de timeout).
3. Texto ambiguo, con pocas menciones técnicas: la "prueba de estrés" de la consigna.

Corren en paralelo con `asyncio.gather`. Cada uno imprime el título del caso y el
JSON validado (`model_dump_json(indent=2)`), o `error controlado: ...` si el pipeline
agotó los reintentos o falló la configuración. Un caso que falla no frena a los
otros (cada corrutina atrapa `Exception`, como en la Pre-entrega 1). Termina con
código 0.

## 7. Tests

Sin red y sin keys. `tests/fakes.py` define `FakeChatModel(BaseChatModel)`:

- Recibe una lista de respuestas guionadas: `AIMessage` con `tool_calls` (lo que un
  proveedor devuelve con tool calling) o excepciones para lanzar.
- Implementa `bind_tools` devolviendo `self`, lo que habilita
  `with_structured_output` del `BaseChatModel` (método `function_calling`), así el
  **parseo real de LangChain** corre en los tests.
- Cuenta las llamadas (`llamadas`), para verificar cuántos intentos hubo.
- Helper `mensaje_con_herramienta(args, *, finish_reason=None, stop_reason=None)`
  que arma el `AIMessage` con `response_metadata`.

Verificado en un spike el 2026-09-12: con este fake la cadena completa produce
`ExtraccionTecnica`, `parsing_error` llega como `ValidationError` y `with_retry`
respeta `retry_if_exception_type`.

Casos:

| Archivo | Qué prueba |
|---|---|
| `test_schemas.py` | Objeto válido; lista vacía falla; string vacío en la lista falla; enum inválido falla; campo extra falla; `model_dump_json` serializa el enum como texto |
| `test_errors.py` | Jerarquía (`PipelineError`); `RECUPERABLES` incluye las propias y las transitorias de ambos SDKs, y no incluye `AuthenticationError` ni `BadRequestError` |
| `test_providers.py` | Elige por argumento y por `LLM_PROVIDER`; default anthropic; proveedor desconocido y key faltante lanzan `ValueError`; `max_retries == 0` en ambos; OpenAI recibe `base_url` y `temperature=0`; Anthropic no recibe `temperature` (queda en su default) |
| `test_chain.py` | `PROMPT.input_variables == ["texto"]` y roles system/human; `validar_salida` con corte por tokens (OpenAI y Anthropic), con `parsing_error`, con `parsed is None`, y con salida válida; cadena: éxito al primer intento (1 llamada); inválida y luego válida (2 llamadas); cortada y luego válida (2 llamadas); inválida tres veces lanza `SalidaInvalidaError` (3 llamadas); `AuthenticationError` real del SDK lanza sin reintentar (1 llamada); `process_text` rechaza texto vacío sin llamar; `process_text` con cadena inyectada devuelve el objeto; los logs de intento y validación aparecen (`caplog`) |

Las excepciones de los SDKs se construyen reales (como en la Pre-entrega 1), con
`httpx2.Request`/`Response` mínimos.

## 8. README

Instalación con `uv` y con `venv`, tabla de variables, cómo probar gratis con Gemini,
cómo correr `main.py` y los tests, cómo funciona la cadena (diagrama del flujo), qué
se reintenta y qué no, y una **salida real** capturada de la prueba con Gemini y con
Anthropic. Sin PDF ni informe: la consigna pide que todo esté en el README.

## 9. Decisiones tomadas y alternativas descartadas

1. **`include_raw=True` + validador propio** en vez de `with_structured_output` a
   secas con `.with_retry()`. Sin el mensaje crudo no se puede ver `finish_reason`, y
   reintentar toda excepción reintentaría un 401. Alternativa descartada:
   `PydanticOutputParser` + `OutputFixingParser` (una llamada extra por error y no es
   el camino que prefiere la consigna).
2. **Reintentar por tipo de excepción**, no por defecto. `with_retry` reintenta
   `Exception` si no se le indica otra cosa; la tupla `RECUPERABLES` deja afuera lo
   permanente.
3. **`max_retries=0` en los modelos**: una sola capa de reintento (heredado de la
   Pre-entrega 1).
4. **Contador de intentos en el diccionario de entrada.** `with_retry` no expone el
   número de intento a los eslabones (solo lo etiqueta en los callbacks, y `on_retry`
   no se dispara en langchain-core 1.6). Como reutiliza el mismo dict, un primer
   eslabón que lo incrementa es la forma más simple y visible. Alternativa
   descartada: un callback handler propio.
5. **Modelo inyectable en `build_chain` y cadena inyectable en `process_text`**: los
   tests no tocan la red. Solo `providers.py` lee el entorno.
6. **Fake propio de modelo de chat** en vez de `GenericFakeChatModel`: los fakes de
   LangChain no implementan `bind_tools`, y sin eso `with_structured_output` lanza
   `NotImplementedError`.
7. **Módulos en la raíz** en vez de un paquete: el rubro nombra `schemas.py` y
   `chain.py`; un paquete `pipeline/` obligaría al corrector a buscar.
8. **Sin `temperature` en Anthropic**, `temperature=0` en OpenAI: los Claude actuales
   rechazan el parámetro; para extracción conviene determinismo donde se puede.
9. **`claude-opus-5` por defecto en Anthropic**, según la referencia actual de la API;
   `ANTHROPIC_MODEL=claude-haiku-4-5` lo baja al más barato.
10. **No se importa el paquete `llm_client` de la Pre-entrega 1.** Sus clientes
    envuelven los SDKs crudos y no son Runnables; lo que se reutiliza es la lógica
    (elección por entorno, una capa de reintento, tests con fakes, `.env.example`).
11. **Callback `on_llm_error` para loguear errores del proveedor** (agregado tras la
    prueba real). `with_listeners(on_error=...)` no se dispara con `ainvoke` y
    `on_retry` no lo llama `with_retry`; `on_llm_error` es el hook que sí se ejecuta en
    cada fallo del modelo, verificado con el fake.
12. **`anthropic.OverloadedError` (529) en `RECUPERABLES`**: no hereda de
    `InternalServerError`, y una sobrecarga momentánea es exactamente lo que un
    reintento con espera resuelve (la PE1 también lo reintentaba).

## 10. Limitaciones conocidas

- No hay fallback a otro modelo (`with_fallbacks`); si los tres intentos fallan, la
  excepción llega al llamador.
- `with_structured_output` depende de tool calling o JSON mode del proveedor; un
  endpoint compatible que no los soporte falla con `BadRequestError` sin reintento.
- El contador de intentos vive en el dict de entrada: si el llamador reutiliza el
  mismo dict en dos invocaciones, el conteo continúa. `process_text` crea uno nuevo
  por llamada.
- Solo texto plano en español o inglés; no hay chunking para textos muy largos.
