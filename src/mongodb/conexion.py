"""
Conexión a MongoDB e inserción del corpus listo con los análisis
en una BBDD ya creada
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
NOMBRE_BASE_DATOS = "TurisPOS"


def obtener_cliente(uri: str = MONGO_URI) -> MongoClient:

    return MongoClient(uri)


def obtener_base_datos(nombre_bd: str = NOMBRE_BASE_DATOS, uri: str = MONGO_URI):

    cliente = obtener_cliente(uri)
    return cliente[nombre_bd]


def obtener_coleccion(nombre_coleccion: str, nombre_bd: str = NOMBRE_BASE_DATOS,
                       uri: str = MONGO_URI):

    bd = obtener_base_datos(nombre_bd, uri)
    return bd[nombre_coleccion]


def probar_conexion(uri: str = MONGO_URI) -> bool:

    cliente = obtener_cliente(uri)
    cliente.admin.command("ping")
    return True


def _dataframe_a_documentos(df: pd.DataFrame) -> list:

    df_limpio = df.replace({np.nan: None})
    return df_limpio.to_dict(orient="records")


def insertar_dataframe(df: pd.DataFrame, nombre_coleccion: str,
                        nombre_bd: str = NOMBRE_BASE_DATOS, uri: str = MONGO_URI,
                        reemplazar: bool = False) -> dict:

    coleccion = obtener_coleccion(nombre_coleccion, nombre_bd, uri)

    if reemplazar:
        coleccion.delete_many({})

    documentos = _dataframe_a_documentos(df)
    if not documentos:
        return {"insertados": 0, "coleccion": nombre_coleccion}

    resultado = coleccion.insert_many(documentos)
    return {"insertados": len(resultado.inserted_ids), "coleccion": nombre_coleccion}


def contar_documentos(nombre_coleccion: str, nombre_bd: str = NOMBRE_BASE_DATOS,
                       uri: str = MONGO_URI) -> int:

    coleccion = obtener_coleccion(nombre_coleccion, nombre_bd, uri)
    return coleccion.count_documents({})