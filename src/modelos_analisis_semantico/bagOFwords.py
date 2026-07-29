
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _tokens_a_texto(tokens_list):

    return [" ".join(tokens) for tokens in tokens_list]


def construir_bow(tokens_por_reseña, min_df=2, max_df=0.95):
    """
    Construye la matriz Bag of Words (conteos de frecuencia)
    """
    textos = _tokens_a_texto(tokens_por_reseña)
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    matriz = vectorizer.fit_transform(textos)
    return matriz, vectorizer


def construir_tfidf(tokens_por_reseña, min_df=2, max_df=0.95):
    """
    Construye la matriz TF-IDF (frecuencia ponderada por rareza en el corpus)
    """
    textos = _tokens_a_texto(tokens_por_reseña)
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    matriz = vectorizer.fit_transform(textos)
    return matriz, vectorizer


def top_terminos_tfidf(matriz_tfidf, vectorizer, indices_grupo, top_n=15):

    submatriz = matriz_tfidf[indices_grupo]
    promedio = np.asarray(submatriz.mean(axis=0)).flatten()
    vocabulario = np.array(vectorizer.get_feature_names_out())

    top_idx = promedio.argsort()[::-1][:top_n]
    return list(zip(vocabulario[top_idx], promedio[top_idx]))


def matriz_similitud_documentos(matriz_documentos):
    
    from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(matriz_documentos)