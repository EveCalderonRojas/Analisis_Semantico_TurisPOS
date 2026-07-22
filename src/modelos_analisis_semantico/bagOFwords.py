"""
Representación dispersa: Bag of Words (BoW) y TF-IDF.

Línea base (baseline) para comparar contra Word2Vec y BETO en
`resultados_analisis/comparacion_modelos.ipynb` (Criterio 4, 15 pts).
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from procesamiento_corpus.preprocesamiento import cargar_y_preparar_corpus


def _tokens_a_texto(tokens_list):
    """CountVectorizer/TfidfVectorizer esperan strings; unimos los tokens
    ya limpios con espacio (ya vienen preprocesados, no hace falta
    re-tokenizar ni quitar stopwords de nuevo)."""
    return [" ".join(tokens) for tokens in tokens_list]


def construir_bow(tokens_por_reseña, min_df=2, max_df=0.95):
    """
    Construye la matriz Bag of Words (conteos de frecuencia).

    Returns
    -------
    matriz (scipy sparse), vectorizer (CountVectorizer entrenado)
    """
    textos = _tokens_a_texto(tokens_por_reseña)
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    matriz = vectorizer.fit_transform(textos)
    return matriz, vectorizer


def construir_tfidf(tokens_por_reseña, min_df=2, max_df=0.95):
    """
    Construye la matriz TF-IDF (frecuencia ponderada por rareza en el corpus).

    Returns
    -------
    matriz (scipy sparse), vectorizer (TfidfVectorizer entrenado)
    """
    textos = _tokens_a_texto(tokens_por_reseña)
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    matriz = vectorizer.fit_transform(textos)
    return matriz, vectorizer


def top_terminos_tfidf(matriz_tfidf, vectorizer, indices_grupo, top_n=15):
    """
    Términos con mayor peso TF-IDF promedio dentro de un subconjunto de filas
    (ej. solo reseñas de 'parque', o solo 'negativa'). Útil para el análisis
    de vocabulario exclusivo por categoría/polaridad.
    """
    submatriz = matriz_tfidf[indices_grupo]
    promedio = np.asarray(submatriz.mean(axis=0)).flatten()
    vocabulario = np.array(vectorizer.get_feature_names_out())

    top_idx = promedio.argsort()[::-1][:top_n]
    return list(zip(vocabulario[top_idx], promedio[top_idx]))


if __name__ == "__main__":
    corpus = cargar_y_preparar_corpus()

    matriz_tfidf, vectorizer = construir_tfidf(corpus["tokens"])
    print(f"Matriz TF-IDF: {matriz_tfidf.shape} (reseñas x vocabulario)")
    print(f"Tamaño del vocabulario: {len(vectorizer.get_feature_names_out())}")

    idx_parque = corpus.index[corpus["categoria"] == "parque"].tolist()
    idx_parque_pos = [corpus.index.get_loc(i) for i in idx_parque]
    print("\nTop términos TF-IDF en reseñas de categoría 'parque':")
    for termino, peso in top_terminos_tfidf(matriz_tfidf, vectorizer, idx_parque_pos):
        print(f"  {termino}: {peso:.4f}")
