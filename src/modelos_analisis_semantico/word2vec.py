"""
Word2Vec: representaciones semánticas estáticas (Mikolov et al., 2013).

Entrena ambas variantes (CBOW y Skip-Gram) sobre el corpus propio de
reseñas, y ofrece utilidades para explorar campos semánticos, analogías
y similitud entre categorías (Criterio 2, 25 pts).

Nota: con 1,200-3,000 reseñas cortas el modelo va a ser "modesto" (es
esperado) -- algunos vecindarios semánticos pueden ser ruidosos. Eso
también es un hallazgo válido para la interpretación.
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from procesamiento_corpus.preprocesamiento import cargar_y_preparar_corpus


def entrenar_word2vec(tokens_por_reseña, sg, vector_size=100, window=5,
                       min_count=2, epochs=30, seed=42):
    """
    sg=0 -> CBOW (predice palabra central desde el contexto)
    sg=1 -> Skip-Gram (predice el contexto desde la palabra central)
    """
    modelo = Word2Vec(
        sentences=tokens_por_reseña,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=1,  # workers=1 + seed fijo => resultados reproducibles
    )
    return modelo


def entrenar_cbow_y_skipgram(tokens_por_reseña, **kwargs):
    """Entrena ambas variantes de una vez y las devuelve en un dict."""
    cbow = entrenar_word2vec(tokens_por_reseña, sg=0, **kwargs)
    skipgram = entrenar_word2vec(tokens_por_reseña, sg=1, **kwargs)
    return {"cbow": cbow, "skipgram": skipgram}


def palabras_similares(modelo, palabra, top_n=10):
    """Top_n palabras más similares (campo semántico) a `palabra`."""
    if palabra not in modelo.wv:
        return f"'{palabra}' no está en el vocabulario del modelo (revisa min_count o si aparece en el corpus)."
    return modelo.wv.most_similar(palabra, topn=top_n)


def analogia(modelo, positivos, negativos, top_n=5):
    """Analogía vectorial estilo 'rey - hombre + mujer = reina'."""
    faltantes = [p for p in positivos + negativos if p not in modelo.wv]
    if faltantes:
        return f"Palabras fuera de vocabulario: {faltantes}"
    return modelo.wv.most_similar(positive=positivos, negative=negativos, topn=top_n)


def vector_promedio_reseña(modelo, tokens):
    """
    Promedia los vectores de las palabras de una reseña para obtener un
    único vector representativo del documento (necesario para
    clasificación/clustering/t-SNE con Word2Vec).
    """
    vectores = [modelo.wv[t] for t in tokens if t in modelo.wv]
    if not vectores:
        return np.zeros(modelo.vector_size)
    return np.mean(vectores, axis=0)


def matriz_vectores_documentos(modelo, tokens_por_reseña):
    """Aplica vector_promedio_reseña a todo el corpus -> matriz (n_docs, vector_size)."""
    return np.vstack([vector_promedio_reseña(modelo, tokens) for tokens in tokens_por_reseña])


def similitud_entre_grupos(modelo, tokens_por_reseña, etiquetas_grupo):
    """
    Similitud coseno entre los vectores centroides de cada categoría (ej.
    tipo_lugar o polaridad). Responde "¿qué tan distintas son
    semánticamente las reseñas de parques vs. hoteles?".
    """
    from sklearn.metrics.pairwise import cosine_similarity

    df_temp = pd.DataFrame({"tokens": list(tokens_por_reseña), "grupo": list(etiquetas_grupo)})
    centroides = {}
    for grupo, sub in df_temp.groupby("grupo"):
        vectores = matriz_vectores_documentos(modelo, sub["tokens"])
        centroides[grupo] = vectores.mean(axis=0)

    grupos = list(centroides.keys())
    resultado = {}
    for i, g1 in enumerate(grupos):
        for g2 in grupos[i + 1:]:
            sim = cosine_similarity([centroides[g1]], [centroides[g2]])[0][0]
            resultado[(g1, g2)] = sim
    return resultado


if __name__ == "__main__":
    corpus = cargar_y_preparar_corpus()
    modelos = entrenar_cbow_y_skipgram(corpus["tokens"])

    print(f"Vocabulario CBOW: {len(modelos['cbow'].wv)} palabras")
    print(f"Vocabulario Skip-Gram: {len(modelos['skipgram'].wv)} palabras")

    for nombre, modelo in modelos.items():
        print(f"\n=== {nombre.upper()} ===")
        for palabra in ["playa", "hotel", "servicio", "comida"]:
            print(f"Similares a '{palabra}': {palabras_similares(modelo, palabra, top_n=5)}")

    print("\n=== Similitud entre categorías (Skip-Gram) ===")
    print(similitud_entre_grupos(modelos["skipgram"], corpus["tokens"], corpus["categoria"]))
