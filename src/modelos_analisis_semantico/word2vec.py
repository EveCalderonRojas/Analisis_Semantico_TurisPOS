

import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. Vocabulario y distribución de muestreo negativo
# ---------------------------------------------------------------------------
class Vocabulario:

    def __init__(self, tokens_por_reseña, min_count=2):
        contador = Counter()
        for tokens in tokens_por_reseña:
            contador.update(tokens)

        self.palabra_a_indice = {}
        frecuencias = []
        for palabra, freq in contador.items():
            if freq >= min_count:
                self.palabra_a_indice[palabra] = len(self.palabra_a_indice)
                frecuencias.append(freq)

        self.indice_a_palabra = {i: p for p, i in self.palabra_a_indice.items()}
        self.tamano = len(self.palabra_a_indice)

        frecuencias = np.array(frecuencias, dtype=np.float64)
        dist = frecuencias ** 0.75
        self.dist_muestreo = dist / dist.sum()

    def __contains__(self, palabra):
        return palabra in self.palabra_a_indice

    def __len__(self):
        return self.tamano

"""
Generación de pares de entrenamiento (ventana dinámica, como el original)
"""

def generar_pares_skipgram(tokens_por_reseña, vocab: Vocabulario, window=5, seed=42):
    """Cada par es (palabra_centro, palabra_contexto)."""
    rng = random.Random(seed)
    pares = []
    for tokens in tokens_por_reseña:
        indices = [vocab.palabra_a_indice[t] for t in tokens if t in vocab]
        n = len(indices)
        for i, centro in enumerate(indices):
            ventana_din = rng.randint(1, window)  # ventana dinámica (subsampling de contexto)
            inicio, fin = max(0, i - ventana_din), min(n, i + ventana_din + 1)
            for j in range(inicio, fin):
                if j != i:
                    pares.append((centro, indices[j]))
    return pares


def generar_pares_cbow(tokens_por_reseña, vocab: Vocabulario, window=5, seed=42):
    """Cada par es (lista_de_contexto, palabra_centro)."""
    rng = random.Random(seed)
    pares = []
    for tokens in tokens_por_reseña:
        indices = [vocab.palabra_a_indice[t] for t in tokens if t in vocab]
        n = len(indices)
        for i, centro in enumerate(indices):
            ventana_din = rng.randint(1, window)
            inicio, fin = max(0, i - ventana_din), min(n, i + ventana_din + 1)
            contexto = [indices[j] for j in range(inicio, fin) if j != i]
            if contexto:
                pares.append((contexto, centro))
    return pares


class DatasetSkipGram(Dataset):
    def __init__(self, pares):
        self.pares = pares

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        centro, contexto = self.pares[idx]
        return centro, contexto


class DatasetCBOW(Dataset):
    """Rellena (padding) el contexto a longitud fija para poder batchear,
    con una máscara que indica qué posiciones son reales vs. relleno."""

    def __init__(self, pares, max_contexto):
        self.pares = pares
        self.max_contexto = max_contexto

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        contexto, centro = self.pares[idx]
        contexto = contexto[: self.max_contexto]
        n_reales = len(contexto)
        contexto_pad = contexto + [0] * (self.max_contexto - n_reales)
        mascara = [1.0] * n_reales + [0.0] * (self.max_contexto - n_reales)
        return (
            torch.tensor(contexto_pad, dtype=torch.long),
            torch.tensor(mascara, dtype=torch.float32),
            centro,
        )


"""
Modelo: dos tablas de embeddings (entrada/salida) + Negative Sampling
"""

class SGNS(nn.Module):
    """Skip-Gram / CBOW con Negative Sampling. Misma arquitectura de dos
    tablas de embeddings del word2vec original (Mikolov et al., 2013)."""

    def __init__(self, tamano_vocab, dim):
        super().__init__()
        self.embed_entrada = nn.Embedding(tamano_vocab, dim)  # vectores finales (palabra "centro")
        self.embed_salida = nn.Embedding(tamano_vocab, dim)   # vectores auxiliares de contexto
        rango = 0.5 / dim
        nn.init.uniform_(self.embed_entrada.weight, -rango, rango)
        nn.init.zeros_(self.embed_salida.weight)

    def forward_skipgram(self, centro, contexto_pos, contexto_neg):
        v_centro = self.embed_entrada(centro)          # (B, D)
        v_pos = self.embed_salida(contexto_pos)         # (B, D)
        v_neg = self.embed_salida(contexto_neg)         # (B, K, D)

        score_pos = torch.sum(v_centro * v_pos, dim=1)
        perdida_pos = torch.nn.functional.logsigmoid(score_pos)

        score_neg = torch.bmm(v_neg, v_centro.unsqueeze(2)).squeeze(2)  # (B, K)
        perdida_neg = torch.nn.functional.logsigmoid(-score_neg).sum(dim=1)

        return -(perdida_pos + perdida_neg).mean()

    def forward_cbow(self, contexto_idx, mascara, centro_pos, centro_neg):
        v_contexto = self.embed_entrada(contexto_idx)             # (B, C, D)
        v_contexto = v_contexto * mascara.unsqueeze(-1)
        v_promedio = v_contexto.sum(dim=1) / mascara.sum(dim=1, keepdim=True).clamp(min=1)

        v_pos = self.embed_salida(centro_pos)
        v_neg = self.embed_salida(centro_neg)

        score_pos = torch.sum(v_promedio * v_pos, dim=1)
        perdida_pos = torch.nn.functional.logsigmoid(score_pos)

        score_neg = torch.bmm(v_neg, v_promedio.unsqueeze(2)).squeeze(2)
        perdida_neg = torch.nn.functional.logsigmoid(-score_neg).sum(dim=1)

        return -(perdida_pos + perdida_neg).mean()



class VectoresPalabra:
    """Imita la interfaz de gensim `modelo.wv`: __contains__, __getitem__,
    most_similar(palabra o positive/negative, topn)."""

    def __init__(self, vocab: Vocabulario, matriz_vectores):
        self.vocab = vocab
        self.vectores = matriz_vectores
        normas = np.linalg.norm(matriz_vectores, axis=1, keepdims=True)
        normas[normas == 0] = 1e-9
        self.vectores_normalizados = matriz_vectores / normas

    def __contains__(self, palabra):
        return palabra in self.vocab.palabra_a_indice

    def __getitem__(self, palabra):
        return self.vectores[self.vocab.palabra_a_indice[palabra]]

    def __len__(self):
        return self.vocab.tamano

    def most_similar(self, palabra=None, positive=None, negative=None, topn=10):
        positive = list(positive) if positive else ([palabra] if palabra else [])
        negative = list(negative) if negative else []

        vector_consulta = np.zeros(self.vectores.shape[1])
        for p in positive:
            vector_consulta = vector_consulta + self.vectores[self.vocab.palabra_a_indice[p]]
        for n in negative:
            vector_consulta = vector_consulta - self.vectores[self.vocab.palabra_a_indice[n]]

        norma = np.linalg.norm(vector_consulta)
        if norma > 0:
            vector_consulta = vector_consulta / norma

        similitudes = self.vectores_normalizados @ vector_consulta
        excluir = {self.vocab.palabra_a_indice[p] for p in positive + negative}

        resultado = []
        for idx in similitudes.argsort()[::-1]:
            if idx in excluir:
                continue
            resultado.append((self.vocab.indice_a_palabra[idx], float(similitudes[idx])))
            if len(resultado) == topn:
                break
        return resultado


class ModeloWord2VecTorch:
    """Envoltorio final: expone `.wv` y `.vector_size` igual que
    gensim.models.Word2Vec, para que el resto del pipeline no note la diferencia."""

    def __init__(self, vocab: Vocabulario, matriz_vectores):
        self.wv = VectoresPalabra(vocab, matriz_vectores)
        self.vector_size = matriz_vectores.shape[1]



def entrenar_word2vec_torch(tokens_por_reseña, modo="skipgram", vector_size=100, window=5,
                             min_count=2, epochs=15, negative_samples=5, batch_size=512,
                             lr=0.01, seed=42, verbose=True):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    vocab = Vocabulario(tokens_por_reseña, min_count=min_count)

    if modo == "skipgram":
        pares = generar_pares_skipgram(tokens_por_reseña, vocab, window=window, seed=seed)
        dataset = DatasetSkipGram(pares)
    elif modo == "cbow":
        pares = generar_pares_cbow(tokens_por_reseña, vocab, window=window, seed=seed)
        dataset = DatasetCBOW(pares, max_contexto=2 * window)
    else:
        raise ValueError("modo debe ser 'skipgram' o 'cbow'")

    cargador = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = SGNS(vocab.tamano, vector_size).to(device)
    optimizador = optim.Adam(modelo.parameters(), lr=lr)

    for epoca in range(epochs):
        perdida_total, n_batches = 0.0, 0
        for batch in cargador:
            optimizador.zero_grad()
            b = batch[0].size(0)
            neg_idx = np.random.choice(vocab.tamano, size=(b, negative_samples), p=vocab.dist_muestreo)
            negativos = torch.tensor(neg_idx, dtype=torch.long, device=device)

            if modo == "skipgram":
                centro, contexto_pos = batch
                perdida = modelo.forward_skipgram(
                    centro.to(device), contexto_pos.to(device), negativos
                )
            else:
                contexto_idx, mascara, centro_pos = batch
                perdida = modelo.forward_cbow(
                    contexto_idx.to(device), mascara.to(device), centro_pos.to(device), negativos
                )

            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item()
            n_batches += 1

        if verbose:
            print(f"  [{modo}] época {epoca + 1}/{epochs} - pérdida promedio: {perdida_total / n_batches:.4f}")

    matriz_vectores = modelo.embed_entrada.weight.detach().cpu().numpy()
    return ModeloWord2VecTorch(vocab, matriz_vectores)


def entrenar_cbow_y_skipgram(tokens_por_reseña, **kwargs):
    """Entrena ambas variantes de una vez y las devuelve en un dict."""
    cbow = entrenar_word2vec_torch(tokens_por_reseña, modo="cbow", **kwargs)
    skipgram = entrenar_word2vec_torch(tokens_por_reseña, modo="skipgram", **kwargs)
    return {"cbow": cbow, "skipgram": skipgram}



def palabras_similares(modelo, palabra, top_n=10):
    if palabra not in modelo.wv:
        return f"'{palabra}' no está en el vocabulario del modelo (revisa min_count o si aparece en el corpus)."
    return modelo.wv.most_similar(palabra, topn=top_n)


def analogia(modelo, positivos, negativos, top_n=5):
    faltantes = [p for p in positivos + negativos if p not in modelo.wv]
    if faltantes:
        return f"Palabras fuera de vocabulario: {faltantes}"
    return modelo.wv.most_similar(positive=positivos, negative=negativos, topn=top_n)


def vector_promedio_reseña(modelo, tokens):
    vectores = [modelo.wv[t] for t in tokens if t in modelo.wv]
    if not vectores:
        return np.zeros(modelo.vector_size)
    return np.mean(vectores, axis=0)


def matriz_vectores_documentos(modelo, tokens_por_reseña):
    return np.vstack([vector_promedio_reseña(modelo, tokens) for tokens in tokens_por_reseña])


def matriz_similitud_documentos(modelo, tokens_por_reseña):
    """
    Similitud coseno entre documentos, usando el promedio de vectores
    Word2Vec de cada reseña
    """
    from sklearn.metrics.pairwise import cosine_similarity

    matriz_vectores = matriz_vectores_documentos(modelo, tokens_por_reseña)
    return cosine_similarity(matriz_vectores)


def similitud_entre_grupos(modelo, tokens_por_reseña, etiquetas_grupo):
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