"""
Procesamiento del corpus con todas las reseñas

Proyecto 1 + Proyecto 2
"""

import ast
import re

import pandas as pd
import nltk


CORPUS_PATH_DEFAULT = "../../data/processed/corpus_final.csv"

CORPUS_PATH_DEFAULT = "../../data/processed/corpus_final.csv"


TEXT_COLUMN = "comentarios_espanol"  # texto ya traducido/limpio -- usar SIEMPRE este campo
RATING_COLUMN = "calificacion"
PLACE_TYPE_COLUMN = "categoria"


# ---------------------------------------------------------------------------
# 1. Carga del corpus + parseo de columnas POS + derivación de polaridad
# ---------------------------------------------------------------------------
def _parse_pos_column(value):
    """Convierte el string "[('La', 'DET', 'el'), ...]" en lista real de tuplas."""
    if pd.isna(value) or value == "":
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def derive_polaridad(calificacion):
    """
    Deriva la polaridad a partir de la calificación (1-5 estrellas):
      1-2 -> negativa | 3 -> neutral | 4-5 -> positiva
    Filas sin calificación quedan como None.
    """
    if pd.isna(calificacion):
        return None
    if calificacion <= 2:
        return "negativa"
    if calificacion == 3:
        return "neutral"
    return "positiva"


def cargar_corpus(path: str = CORPUS_PATH_DEFAULT) -> pd.DataFrame:

    df = pd.read_csv(path)

    columnas_requeridas = [TEXT_COLUMN, RATING_COLUMN, PLACE_TYPE_COLUMN]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Al corpus le faltan columnas requeridas: {faltantes}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    if "penntreebank" in df.columns:
        df["pos_nltk"] = df["penntreebank"].apply(_parse_pos_column)
    if "universalpos" in df.columns:
        df["pos_spacy"] = df["universalpos"].apply(_parse_pos_column)

    df["polaridad"] = df[RATING_COLUMN].apply(derive_polaridad)
    df["texto_valido"] = df[TEXT_COLUMN].notna() & (df[TEXT_COLUMN].str.strip() != "")

    return df



# Tokenización con los lemas de spaCy ya calculados

try:
    from nltk.corpus import stopwords
    STOPWORDS_ES = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS_ES = set(stopwords.words("spanish"))

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# Categorías gramaticales sin contenido semántico (se descartan para
# Word2Vec/BoW; se conservan solo para análisis morfosintáctico)

UPOS_DESCARTABLES = {"PUNCT", "SPACE", "SYM", "X"}

# Stopwords de dominio adicionales a las de NLTK:
# - conectores sueltos que NLTK no cubre bien
# - verbos auxiliares/copulativos muy frecuentes que no aportan significado
#   distintivo para campos semánticos (dominan el vocabulario si no se filtran)
STOPWORDS_DOMINIO = {
    "si", "así", "aquí", "ahí", "the",
    "ser", "estar", "haber", "poder", "ir", "hacer",
}
STOPWORDS_TOTAL = STOPWORDS_ES | STOPWORDS_DOMINIO


def tokenizar_desde_pos_spacy(pos_spacy: list) -> list:

    if not pos_spacy:
        return []

    lemas = []
    for tupla in pos_spacy:
        if len(tupla) != 3:
            continue
        _token, upos, lema = tupla
        if upos in UPOS_DESCARTABLES:
            continue
        lema_limpio = lema.lower().strip()
        if not lema_limpio or lema_limpio in STOPWORDS_TOTAL:
            continue
        if not re.match(r"^[a-záéíóúñü]+$", lema_limpio):
            continue
        lemas.append(lema_limpio)
    return lemas


def tokenizar_texto_plano(texto: str) -> list:

    if not isinstance(texto, str) or not texto.strip():
        return []
    tokens = nltk.word_tokenize(texto.lower(), language="spanish")
    return [t for t in tokens if t not in STOPWORDS_TOTAL and re.match(r"^[a-záéíóúñü]+$", t)]


def tokenizar_corpus(df: pd.DataFrame, columna_texto: str = TEXT_COLUMN) -> pd.DataFrame:
    """Agrega la columna `tokens` reutilizando pos_spacy si existe, o el fallback de NLTK."""
    if "pos_spacy" in df.columns:
        df["tokens"] = df["pos_spacy"].apply(tokenizar_desde_pos_spacy)
        sin_pos = df["pos_spacy"].apply(lambda x: len(x) == 0)
        df.loc[sin_pos, "tokens"] = df.loc[sin_pos, columna_texto].apply(tokenizar_texto_plano)
    else:
        df["tokens"] = df[columna_texto].apply(tokenizar_texto_plano)

    df["num_tokens"] = df["tokens"].apply(len)
    return df


# Función de conveniencia: todo en un solo llamado

def cargar_y_preparar_corpus(path: str = CORPUS_PATH_DEFAULT) -> pd.DataFrame:

    df = cargar_corpus(path)
    df = tokenizar_corpus(df)
    return df


def resumen_corpus(df: pd.DataFrame) -> None:
    """Imprime un resumen rápido del corpus cargado (validación manual)."""
    print(f"Total de reseñas: {len(df)}")
    print(f"Reseñas con texto válido: {df['texto_valido'].sum()}")
    print(f"\nDistribución por categoría (tipo de lugar):")
    print(df[PLACE_TYPE_COLUMN].value_counts())
    print(f"\nDistribución por polaridad:")
    print(df["polaridad"].value_counts(dropna=False))
    print(f"\nLugares únicos: {df['lugar'].nunique()}")
    if "num_tokens" in df.columns:
        print(f"\nPromedio de tokens útiles por reseña: {df['num_tokens'].mean():.1f}")


if __name__ == "__main__":
    corpus = cargar_y_preparar_corpus()
    resumen_corpus(corpus)
