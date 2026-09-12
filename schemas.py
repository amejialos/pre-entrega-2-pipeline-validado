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
