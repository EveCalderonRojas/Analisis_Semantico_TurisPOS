"""
Limpieza y normalización del dataset scrapeado de Booking.com
con el mismo formato del corpus del proyecto 1
para juntarlos posteriormente
"""

import pandas as pd

RUTA_ENTRADA = "../web_scrapping/resenas_booking.csv"
RUTA_SALIDA = "../../data/processed/corpus_clean.csv"


def combinar_texto(row) -> str | None:

    partes = []
    for columna in ("texto_positivo", "texto_negativo"):
        valor = row.get(columna)
        if pd.notna(valor) and str(valor).strip():
            partes.append(str(valor).strip())
    return " ".join(partes) if partes else None


def convertir_puntuacion_a_calificacion(puntuacion) -> int | None:
    """
    Convierte una puntuación de Booking (1-10, puede venir con coma
    decimal, ej. "9,1") a una calificación 1-5, agrupada en pares:
        (0, 2] -> 1 | (2, 4] -> 2 | (4, 6] -> 3 | (6, 8] -> 4 | (8, 10] -> 5
    """
    if pd.isna(puntuacion):
        return None
    try:
        valor = float(str(puntuacion).replace(",", "."))
    except ValueError:
        return None

    if valor <= 2:
        return 1
    elif valor <= 4:
        return 2
    elif valor <= 6:
        return 3
    elif valor <= 8:
        return 4
    else:
        return 5


def limpiar_dataset_booking(path_entrada: str) -> pd.DataFrame:

    df = pd.read_csv(path_entrada)

    columnas_requeridas = ["texto_positivo", "texto_negativo", "puntuacion", "lugar", "categoria", "fuente"]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Al CSV le faltan columnas requeridas: {faltantes}. Columnas encontradas: {list(df.columns)}")

    df["texto"] = df.apply(combinar_texto, axis=1)
    df["calificacion"] = df["puntuacion"].apply(convertir_puntuacion_a_calificacion)

    columnas_finales = ["texto", "calificacion", "lugar", "categoria", "fuente"]
    return df[columnas_finales]


if __name__ == "__main__":
    df_limpio = limpiar_dataset_booking(RUTA_ENTRADA)

    print(f"Filas totales: {len(df_limpio)}")
    print(f"Filas sin texto: {df_limpio['texto'].isna().sum()}")
    print(f"\nDistribución de calificación (1-5):")
    print(df_limpio["calificacion"].value_counts(dropna=False).sort_index())
    print(f"\nDistribución por categoría:")
    print(df_limpio["categoria"].value_counts())

    df_limpio.to_csv(RUTA_SALIDA, index=False)
    print(f"\nGuardado en: {RUTA_SALIDA}")