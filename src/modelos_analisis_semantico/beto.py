
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

_tokenizer = None
_modelo = None
_modelo_mlm = None


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cargar_beto():
    """Carga (una sola vez, cacheado) el tokenizer y el modelo base de BETO."""
    global _tokenizer, _modelo
    if _modelo is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _modelo = AutoModel.from_pretrained(MODEL_NAME)
        _modelo.to(_device())
        _modelo.eval()
    return _tokenizer, _modelo


def cargar_beto_mlm():
    """Carga la variante con cabeza de Masked Language Model (completar [MASK])."""
    global _tokenizer, _modelo_mlm
    if _modelo_mlm is None:
        if _tokenizer is None:
            cargar_beto()
        _modelo_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
        _modelo_mlm.to(_device())
        _modelo_mlm.eval()
    return _tokenizer, _modelo_mlm


@torch.no_grad()
def embeddings_de_textos(textos, modo="cls", batch_size=100, max_length=128):
    """
    Genera embeddings de una lista de textos
    """
    tokenizer, modelo = cargar_beto()
    device = _device()
    todos_los_vectores = []

    for i in range(0, len(textos), batch_size):
        lote = [str(t) for t in textos[i:i + batch_size]]
        encoded = tokenizer(
            lote, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)

        salida = modelo(**encoded)
        ultima_capa = salida.last_hidden_state

        if modo == "cls":
            vectores = ultima_capa[:, 0, :]
        elif modo == "mean":
            mascara = encoded["attention_mask"].unsqueeze(-1).expand(ultima_capa.size()).float()
            suma = torch.sum(ultima_capa * mascara, dim=1)
            conteo = torch.clamp(mascara.sum(dim=1), min=1e-9)
            vectores = suma / conteo
        else:
            raise ValueError("modo debe ser 'cls' o 'mean'")

        todos_los_vectores.append(vectores.cpu().numpy())
        print(f"  procesadas {min(i + batch_size, len(textos))}/{len(textos)} reseñas...")

    return np.vstack(todos_los_vectores)


@torch.no_grad()
def embedding_de_palabra_en_contexto(palabra, oracion, modo="mean"):
    """
    Extrae el embedding contextual de UNA palabra específica dentro de una
    oración (para el análisis de polisemia contextual, ej. "rico").
    """
    tokenizer, modelo = cargar_beto()
    device = _device()

    encoded = tokenizer(oracion, return_tensors="pt", truncation=True).to(device)
    salida = modelo(**encoded)
    ultima_capa = salida.last_hidden_state[0]

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    palabra_lower = palabra.lower()
    indices_palabra = [
        idx for idx, tok in enumerate(tokens)
        if palabra_lower in tok.lower().replace("##", "")
    ]

    if not indices_palabra:
        return None  # palabra no encontrada tokenizada tal cual; revisar oración

    if modo == "mean":
        return ultima_capa[indices_palabra].mean(dim=0).cpu().numpy()
    return ultima_capa[indices_palabra[0]].cpu().numpy()


def analizar_polisemia(palabra, oraciones_positivas, oraciones_negativas):
    """
    Similitud coseno intra-grupo vs. entre-grupos del embedding contextual
    de una palabra polisémica en oraciones positivas vs. negativas
    """
    from sklearn.metrics.pairwise import cosine_similarity

    vectores_pos = [embedding_de_palabra_en_contexto(palabra, o) for o in oraciones_positivas]
    vectores_neg = [embedding_de_palabra_en_contexto(palabra, o) for o in oraciones_negativas]
    vectores_pos = [v for v in vectores_pos if v is not None]
    vectores_neg = [v for v in vectores_neg if v is not None]

    if not vectores_pos or not vectores_neg:
        return "No se pudo localizar la palabra en suficientes oraciones de ambos grupos."

    return {
        "palabra": palabra,
        "similitud_intra_positivas": float(cosine_similarity(vectores_pos).mean()),
        "similitud_intra_negativas": float(cosine_similarity(vectores_neg).mean()),
        "similitud_entre_positiva_negativa": float(cosine_similarity(vectores_pos, vectores_neg).mean()),
    }


def buscador_semantico(consulta, corpus_textos, corpus_embeddings, top_n=5):
    """
    Buscador semántico: dada una consulta en lenguaje natural, encuentra las
    reseñas más similares del corpus por similitud coseno de embeddings BETO.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    vector_consulta = embeddings_de_textos([consulta], modo="mean")
    similitudes = cosine_similarity(vector_consulta, corpus_embeddings)[0]
    top_indices = similitudes.argsort()[::-1][:top_n]

    return [
        {"reseña": corpus_textos[i], "similitud": float(similitudes[i])}
        for i in top_indices
    ]


def completar_mask(oracion_con_mask, top_n=5):
    """Usa el MLM de BETO para predecir la palabra que falta en '[MASK]'."""
    tokenizer, modelo_mlm = cargar_beto_mlm()
    device = _device()

    encoded = tokenizer(oracion_con_mask, return_tensors="pt").to(device)
    with torch.no_grad():
        salida = modelo_mlm(**encoded)

    mask_token_id = tokenizer.mask_token_id
    posiciones_mask = (encoded["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0]

    if len(posiciones_mask) == 0:
        raise ValueError(f"La oración debe incluir el token de máscara '{tokenizer.mask_token}'")

    logits_mask = salida.logits[0, posiciones_mask[0]]
    top_ids = logits_mask.topk(top_n).indices.tolist()
    return [tokenizer.decode([tid]).strip() for tid in top_ids]


def matriz_similitud_documentos(embeddings_documentos):
    """
    Similitud coseno entre documentos a partir de una matriz de embeddings
    """
    from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(embeddings_documentos)


def extraer_oraciones_con_palabra(corpus, palabra, columna_texto="comentarios_espanol",
                                   columna_polaridad="polaridad", max_oraciones=5, seed=42):
    """
    Busca oraciones REALES del corpus que contengan `palabra` (coincidencia
    de palabra completa, sin importar mayúsculas/tildes de caja), separadas
    por polaridad ("positiva"/"negativa"). Pensada para armar ejemplos de
    polisemia contextual sin inventar oraciones a mano.

    """
    import random
    import re

    patron = re.compile(rf"\b{re.escape(palabra)}\b", re.IGNORECASE)
    resultado = {}

    for polaridad_objetivo in ("positiva", "negativa"):
        sub_corpus = corpus[corpus[columna_polaridad] == polaridad_objetivo]
        oraciones_encontradas = []

        for texto in sub_corpus[columna_texto].dropna():
            for oracion in re.split(r"(?<=[.!?])\s+", str(texto)):
                oracion = oracion.strip()
                if oracion and patron.search(oracion):
                    oraciones_encontradas.append(oracion)

        random.Random(seed).shuffle(oraciones_encontradas)
        resultado[polaridad_objetivo] = oraciones_encontradas[:max_oraciones]

    return resultado