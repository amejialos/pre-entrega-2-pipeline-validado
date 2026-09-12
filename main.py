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
        print(f"\n=== {titulo} ===\n{resultado.model_dump_json(indent=2)}", flush=True)
    except Exception as error:  # un caso que falla no frena a los otros
        print(f"\n=== {titulo} ===\nerror controlado: {type(error).__name__}: {error}", flush=True)


async def main(provider: str | None) -> None:
    try:
        model = build_model(provider)
    except ValueError as error:
        print(f"error de configuración: {error}")
        return
    nombre_modelo = getattr(model, "model_name", None) or getattr(model, "model", "?")
    # flush: que el encabezado salga antes que los logs (stderr) si se redirige la salida
    print(f"Proveedor: {model._llm_type}, modelo: {nombre_modelo}", flush=True)
    chain = build_chain(model)
    await asyncio.gather(*(procesar(titulo, texto, chain) for titulo, texto in TEXTOS.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba el pipeline con tres textos de ejemplo.")
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        help="openai o anthropic (por defecto, LLM_PROVIDER)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()  # carga .env al entorno; el código nunca tiene keys escritas
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # los SDKs loguean cada request en INFO
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    asyncio.run(main(parse_args().provider))
