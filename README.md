# Pipeline de extracción de entidades técnicas

Pipeline asíncrono en Python 3.12 + LangChain que recibe un párrafo técnico (una
descripción de arquitectura o un log de error) y devuelve un objeto **validado con
Pydantic**: tecnologías mencionadas, nivel de criticidad y resumen técnico. Si el
modelo devuelve algo que no cumple el esquema o corta la respuesta por tokens, el
pipeline lo detecta y reintenta. Si el error es permanente (key inválida), falla al
primer intento.

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
y duración. Un callback `on_llm_error` loguea además los errores del proveedor (429,
5xx, red), que no pasan por el validador.

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
[Google AI Studio](https://aistudio.google.com/apikey). `ChatOpenAI` habla con
Gemini sin cambiar código:

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

Procesa tres textos en paralelo (`asyncio.gather`): una descripción de arquitectura,
un log de error y un texto ambiguo (la "prueba de estrés"). Cada uno imprime el JSON
validado o `error controlado: ...` si el pipeline agotó los reintentos. Un caso que
falla no frena a los otros; el script siempre termina con código 0.

### Salida real

Corrida del 2026-09-12 con `--provider openai` (Gemini 3.6 Flash por el endpoint
compatible). Los logs van a stderr y el resultado a stdout:

```
Proveedor: openai-chat, modelo: gemini-3.6-flash
INFO    pipeline: Procesando texto de 373 caracteres
INFO    pipeline: Procesando texto de 408 caracteres
INFO    pipeline: Procesando texto de 214 caracteres
INFO    pipeline: Intento 1: enviando 373 caracteres al modelo
INFO    pipeline: Intento 1: enviando 408 caracteres al modelo
INFO    pipeline: Intento 1: enviando 214 caracteres al modelo
INFO    pipeline: Salida válida (5 tecnologías, criticidad media)
INFO    pipeline: Listo en 5.4 s

=== Descripción de arquitectura ===
{
  "tecnologias": [
    "FastAPI",
    "Nginx",
    "Redis",
    "PostgreSQL",
    "SQLAlchemy"
  ],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "Al superar las 500 conexiones concurrentes, el pool de conexiones de PostgreSQL se agota, provocando una degradación severa en los tiempos de respuesta de la API."
}
INFO    pipeline: Salida válida (4 tecnologías, criticidad alta)
INFO    pipeline: Listo en 5.7 s

=== Log de error ===
{
  "tecnologias": [
    "Apache Kafka",
    "Apache Spark",
    "Kubernetes",
    "Python"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio order-service falló al publicar eventos en Kafka debido a un error de timeout en los metadatos, lo que dejó 1240 pedidos sin procesar en Spark y provocó que el pod de Kubernetes entrara en CrashLoopBackOff."
}
INFO    pipeline: Salida válida (1 tecnologías, criticidad baja)
INFO    pipeline: Listo en 5.9 s

=== Texto ambiguo (prueba de estrés) ===
{
  "tecnologias": [
    "Python"
  ],
  "nivel_de_criticidad": "baja",
  "resumen_tecnico": "Se plantea la modernización de un sistema de reportes basado en planillas compartidas para migrarlo a la nube, evaluando el uso de Python como lenguaje principal."
}
```

**Un reintento visto en vivo.** En una corrida anterior del mismo día, Gemini respondió
`503 UNAVAILABLE` (sobrecarga) para uno de los textos. El error llegó envuelto por
LangChain como `OpenAIAPIError`, que hereda de `openai.InternalServerError`, así que
entró en `RECUPERABLES`: se reintentó dos veces y, al persistir, terminó como error
controlado sin frenar a los otros dos textos:

```
INFO    pipeline: Intento 1: enviando 373 caracteres al modelo
INFO    pipeline: Intento 2: enviando 373 caracteres al modelo
INFO    pipeline: Intento 3: enviando 373 caracteres al modelo
ERROR   pipeline: Reintentos agotados o error permanente: OpenAIAPIError: Error code: 503 - [...'status': 'UNAVAILABLE'...]

=== Descripción de arquitectura ===
error controlado: OpenAIAPIError: Error code: 503 - [...'This model is currently experiencing high demand'...]
```

Esa corrida mostró un hueco: entre `Intento 1` e `Intento 2` no había ningún log
que dijera por qué se reintentaba, porque los errores del proveedor no pasan por el
validador. De ahí salió el callback `on_llm_error`, que desde entonces escribe
`WARNING pipeline: El proveedor falló: OpenAIAPIError: Error code: 503 ...` antes de
cada reintento (verificado en los tests con un 429 real del SDK).

**Un error permanente no se reintenta.** Con el placeholder `sk-ant-...` en lugar de
una key de Anthropic, cada texto hizo una sola llamada (`Intento 1` y nada más) y
salió como error controlado:

```
Proveedor: anthropic-chat, modelo: claude-haiku-4-5
INFO    pipeline: Intento 1: enviando 373 caracteres al modelo
ERROR   pipeline: Reintentos agotados o error permanente: AnthropicAuthenticationError: Error code: 401 - {... 'message': 'API key is invalid.'}

=== Descripción de arquitectura ===
error controlado: AnthropicAuthenticationError: Error code: 401 - {... 'message': 'API key is invalid.'}
```

## Correr los tests

```bash
uv run pytest -q          # 83 passed
```

Los tests no tocan la red ni necesitan keys. `tests/fakes.py` define un
`FakeChatModel` que hereda del modelo base de LangChain, implementa `bind_tools` y
devuelve mensajes con `tool_calls` guionados: el parseo real de LangChain
(`with_structured_output`) corre en los tests, solo se falsea la llamada al proveedor.
Las excepciones de los SDKs se construyen reales (`openai.RateLimitError`, etc.), y se
verifica que las clases con las que LangChain las envuelve conserven la clasificación.

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

Para elegir el proveedor desde código: `build_chain(build_model("openai"))` y pasar
la cadena a `process_text(texto, chain)`.

## Qué se reintenta y qué no

| Situación | Excepción | ¿Se reintenta? |
|---|---|---|
| El modelo cortó por falta de tokens (`finish_reason=length` / `stop_reason=max_tokens`) | `SalidaIncompletaError` | Sí |
| La salida no cumple el esquema (lista vacía, criticidad inválida, campo extra) | `SalidaInvalidaError` | Sí |
| Rate limit (429), conexión caída, timeout, 5xx, sobrecarga (529) | `RateLimitError`, `APIConnectionError`, `InternalServerError`, `OverloadedError` | Sí |
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
  para OpenAI. `max_tokens=2048` para Anthropic: el default de LangChain (128000)
  hace que el SDK exija streaming.
- Solo `providers.py` lee el entorno; el modelo y la cadena se inyectan por parámetro.
- El número de intento se cuenta en el primer eslabón sobre el diccionario de entrada,
  que `with_retry` reutiliza en cada intento; los errores del proveedor se loguean con
  un callback `on_llm_error`, porque no pasan por el validador.

## Limitaciones conocidas

- Sin fallback a otro modelo: si los tres intentos fallan, la excepción llega al llamador.
- `with_structured_output` depende de tool calling o JSON mode del proveedor.
- El contador de intentos vive en el diccionario de entrada: `process_text` crea uno
  nuevo por llamada, pero si invocás la cadena directamente con el mismo dict dos
  veces, el conteo continúa.
- Solo texto plano; sin chunking para textos muy largos.
