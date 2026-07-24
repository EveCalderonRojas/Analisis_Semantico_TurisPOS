"""
Conexión a MongoDB e inserción del corpus (Criterio 1: persistencia del
corpus ampliado -- Proyecto 1 + reseñas nuevas de Web Scraping).

Servidor: local (MongoDB Community en localhost).
Base de datos: TurisPOS (ya creada en el servidor).

Las pruebas/demo de este módulo viven en `resultados_analisis/conexion.ipynb`
(el módulo solo tiene los procedimientos).
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
NOMBRE_BASE_DATOS = "TurisPOS"


def obtener_cliente(uri: str = MONGO_URI) -> MongoClient:
    """Crea el cliente de conexión a MongoDB."""
    return MongoClient(uri)


def obtener_base_datos(nombre_bd: str = NOMBRE_BASE_DATOS, uri: str = MONGO_URI):
    """Devuelve la referencia a la base de datos (TurisPOS por defecto)."""
    cliente = obtener_cliente(uri)
    return cliente[nombre_bd]


def obtener_coleccion(nombre_coleccion: str, nombre_bd: str = NOMBRE_BASE_DATOS,
                       uri: str = MONGO_URI):
    """Devuelve la referencia a una colección específica dentro de la base de datos."""
    bd = obtener_base_datos(nombre_bd, uri)
    return bd[nombre_coleccion]


def probar_conexion(uri: str = MONGO_URI) -> bool:
    """
    Verifica que el servidor de MongoDB responda (ping). Lanza la excepción
    de pymongo tal cual si no hay conexión, para que el mensaje de error
    sea explícito sobre qué falló (servidor apagado, puerto incorrecto, etc.).
    """
    cliente = obtener_cliente(uri)
    cliente.admin.command("ping")
    return True


def _dataframe_a_documentos(df: pd.DataFrame) -> list:
    """
    Convierte un DataFrame a una lista de diccionarios insertable en MongoDB.
    Reemplaza NaN/NaT (numpy) por None, porque MongoDB/BSON no acepta NaN.
    Las columnas que ya son listas (ej. tokens, POS parseados) se insertan tal cual.
    """
    df_limpio = df.replace({np.nan: None})
    return df_limpio.to_dict(orient="records")


def insertar_dataframe(df: pd.DataFrame, nombre_coleccion: str,
                        nombre_bd: str = NOMBRE_BASE_DATOS, uri: str = MONGO_URI,
                        reemplazar: bool = False) -> dict:
    """
    Inserta un DataFrame completo en una colección de MongoDB.

    Parameters
    ----------
    reemplazar : si True, borra todos los documentos existentes en la
        colección antes de insertar (útil para repetir pruebas sin
        acumular duplicados).

    Returns
    -------
    dict con la cantidad de documentos insertados y el nombre de la colección.
    """
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
    """Cuenta los documentos en una colección (para verificar la inserción)."""
    coleccion = obtener_coleccion(nombre_coleccion, nombre_bd, uri)
    return coleccion.count_documents({})