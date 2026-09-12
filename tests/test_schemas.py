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
