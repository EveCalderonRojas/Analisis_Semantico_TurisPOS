import json

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, ctx, dcc, html
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from src.modelos_analisis_semantico.bagOFwords import (
    construir_tfidf, matriz_similitud_documentos as similitud_tfidf, top_terminos_tfidf,
)
from src.modelos_analisis_semantico.beto import cargar_beto, embeddings_de_textos
from src.modelos_analisis_semantico.beto import matriz_similitud_documentos as similitud_beto
from src.modelos_analisis_semantico.word2vec import (
    entrenar_word2vec_torch, matriz_similitud_documentos as similitud_w2v,
    matriz_vectores_documentos, palabras_similares,
)
from src.procesamiento_corpus.preprocesamiento import cargar_y_preparar_corpus

dash.register_page(__name__, path="/modelos", name="Modelos")

VERDE_OSCURO = "#1B4332"
VERDE_MEDIO = "#2D6A4F"
VERDE_CLARO = "#52B788"
DORADO = "#D4A017"
DORADO_CLARO = "#F4D03F"
CREMA = "#F9F5EC"
GRIS_SUAVE = "#F0EDE6"
BLANCO = "#FFFFFF"

RUTA_CORPUS = "../../data/processed/corpus_final.csv"
RUTA_EMBEDDINGS_BETO = "../../data/processed/beto_embeddings_corpus.npy"
RUTA_RESULTADOS_BETO = "../../data/processed/beto_resultados_precalculados.json"

SECCIONES = ["bagofwords", "word2vec", "beto", "comparacion"]
NOMBRES_SECCION = {
    "bagofwords": "Bag of Words",
    "word2vec": "Word2Vec",
    "beto": "BETO",
    "comparacion": "Comparación de Modelos",
}

corpus = cargar_y_preparar_corpus(RUTA_CORPUS)
CATEGORIAS = sorted(corpus["categoria"].dropna().unique().tolist())

matriz_tfidf, vectorizer_tfidf = construir_tfidf(corpus["tokens"])

print("Entrenando Word2Vec (Skip-Gram) para el dashboard...")
modelo_w2v = entrenar_word2vec_torch(corpus["tokens"], modo="skipgram", epochs=15)
print("Word2Vec listo.")

try:
    print("Cargando BETO...")
    cargar_beto()
    import numpy as np
    embeddings_corpus_beto = np.load(RUTA_EMBEDDINGS_BETO)
    with open(RUTA_RESULTADOS_BETO, "r", encoding="utf-8") as f:
        resultados_beto_precalculados = json.load(f)
    BETO_DISPONIBLE = True
    print("BETO listo.")
except Exception as e:
    print(f"BETO no disponible ({e}). Corre beto.ipynb primero para generar los archivos precalculados.")
    BETO_DISPONIBLE = False
    embeddings_corpus_beto = None
    resultados_beto_precalculados = {"polisemia": {}, "mlm_ejemplos": []}

muestra_comparacion = pd.concat([
    grupo.sample(min(5, len(grupo)), random_state=42)
    for _, grupo in corpus.groupby("categoria")
]).reset_index(drop=True)
etiquetas_muestra = [f"{cat[0].upper()}{i}" for i, cat in enumerate(muestra_comparacion["categoria"])]


def _figura_heatmap(matriz_similitud, titulo):
    fig = px.imshow(
        matriz_similitud, x=etiquetas_muestra, y=etiquetas_muestra,
        color_continuous_scale="YlOrRd", zmin=0, zmax=1,
        text_auto=".2f", title=titulo,
    )
    fig.update_layout(paper_bgcolor=BLANCO, plot_bgcolor=BLANCO, height=420)
    return fig


_matriz_tfidf_muestra, _ = construir_tfidf(muestra_comparacion["tokens"], min_df=1)
FIG_HEATMAP_TFIDF = _figura_heatmap(similitud_tfidf(_matriz_tfidf_muestra), "TF-IDF")

FIG_HEATMAP_W2V = _figura_heatmap(
    similitud_w2v(modelo_w2v, muestra_comparacion["tokens"]), "Word2Vec (Skip-Gram)",
)

if BETO_DISPONIBLE:
    _posiciones_muestra = [corpus.index.get_loc(i) for i in muestra_comparacion.index]
    _embeddings_muestra_beto = embeddings_corpus_beto[_posiciones_muestra]
    FIG_HEATMAP_BETO = _figura_heatmap(similitud_beto(_embeddings_muestra_beto), "BETO (promedio de tokens)")
else:
    FIG_HEATMAP_BETO = None

"""
Gráficos de t-SNE
Mostrados por polaridad de comentarios y categoría de los lugares
"""

def _proyeccion_tsne(matriz, seed=42, perplexity=30):
    if hasattr(matriz, "toarray"):
        matriz = matriz.toarray()
    n_muestras = matriz.shape[0]
    perplexity_ajustada = min(perplexity, max(5, n_muestras // 3))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity_ajustada, init="pca")
    return tsne.fit_transform(matriz)


print("Calculando proyecciones t-SNE sobre el corpus completo...")

_representaciones_tsne = {
    "TF-IDF": matriz_tfidf,
    "Word2Vec (Skip-Gram)": matriz_vectores_documentos(modelo_w2v, corpus["tokens"]),
}
if BETO_DISPONIBLE:
    _representaciones_tsne["BETO"] = embeddings_corpus_beto

_filas_tsne = []
_polaridad_rellena_tsne = corpus["polaridad"].fillna("neutral")
for _nombre_rep, _matriz_rep in _representaciones_tsne.items():
    _coords = _proyeccion_tsne(_matriz_rep)
    for _i in range(len(corpus)):
        _filas_tsne.append({
            "x": _coords[_i, 0], "y": _coords[_i, 1],
            "representacion": _nombre_rep,
            "categoria": corpus["categoria"].iloc[_i],
            "polaridad": _polaridad_rellena_tsne.iloc[_i],
        })

DF_TSNE = pd.DataFrame(_filas_tsne)
print("t-SNE listo.")

FIG_TSNE_CATEGORIA = px.scatter(
    DF_TSNE, x="x", y="y", color="categoria", facet_col="representacion",
    color_discrete_map={"parque": VERDE_MEDIO, "restaurante": DORADO, "alojamiento": "#5B8DB8"},
    title="t-SNE por representación (color = tipo de lugar)",
    opacity=0.7,
)
FIG_TSNE_CATEGORIA.update_layout(paper_bgcolor=BLANCO, plot_bgcolor=BLANCO, height=450)

FIG_TSNE_POLARIDAD = px.scatter(
    DF_TSNE, x="x", y="y", color="polaridad", facet_col="representacion",
    color_discrete_map={"negativa": "#C0392B", "neutral": "#B8B8B8", "positiva": "#2D6A4F"},
    title="t-SNE por representación (color = polaridad)",
    opacity=0.7,
)
FIG_TSNE_POLARIDAD.update_layout(paper_bgcolor=BLANCO, plot_bgcolor=BLANCO, height=450)


def _titulo_seccion(texto):
    return html.H2(texto, style={
        "color": VERDE_OSCURO, "fontSize": "26px",
        "borderBottom": f"3px solid {DORADO}", "paddingBottom": "12px",
        "margin": "0 0 12px",
    })


def _parrafo(texto):
    return html.P(texto, style={
        "color": "#666", "fontFamily": "Arial, sans-serif", "fontSize": "14px",
        "margin": "0 0 24px",
    })


def _tarjeta(children, borde=VERDE_CLARO):
    return html.Div(children, style={
        "backgroundColor": BLANCO, "borderRadius": "12px", "padding": "24px",
        "boxShadow": "0 2px 12px rgba(0,0,0,0.08)", "borderTop": f"4px solid {borde}",
        "margin": "0 0 24px",
    })


def _estilo_boton_sidebar(activo):
    return {
        "display": "block", "width": "100%", "textAlign": "left",
        "padding": "16px 18px", "marginBottom": "10px",
        "border": f"2px solid {VERDE_MEDIO}", "borderRadius": "10px", "cursor": "pointer",
        "backgroundColor": VERDE_MEDIO if activo else BLANCO,
        "color": BLANCO if activo else VERDE_MEDIO,
        "fontFamily": "Arial, sans-serif", "fontSize": "15px", "fontWeight": "bold",
    }


def _boton_sidebar(seccion, activo):
    return html.Button(
        NOMBRES_SECCION[seccion], id=f"btn-seccion-{seccion}", n_clicks=0,
        style=_estilo_boton_sidebar(activo),
    )


def _estilo_seccion(activa):
    return {"display": "block"} if activa else {"display": "none"}


def _seccion_bagofwords():
    return html.Div([
        _titulo_seccion("Bag of Words / TF-IDF"),
        _parrafo("Terminos mas caracteristicos (mayor peso TF-IDF promedio) por tipo de lugar."),

        html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "24px"}, children=[
            html.Button(
                cat.capitalize(), id=f"btn-tfidf-{cat}", n_clicks=0,
                style={
                    "padding": "10px 22px", "border": f"2px solid {VERDE_MEDIO}",
                    "borderRadius": "25px", "cursor": "pointer",
                    "backgroundColor": VERDE_MEDIO if cat == CATEGORIAS[0] else BLANCO,
                    "color": BLANCO if cat == CATEGORIAS[0] else VERDE_MEDIO,
                    "fontFamily": "Arial, sans-serif", "fontSize": "14px", "fontWeight": "bold",
                },
            ) for cat in CATEGORIAS
        ]),

        dcc.Store(id="categoria-tfidf-activa", data=CATEGORIAS[0]),
        _tarjeta(dcc.Graph(id="grafico-tfidf", config={"displayModeBar": False}), borde=DORADO),
    ])


def _seccion_word2vec():
    return html.Div([
        _titulo_seccion("Word2Vec - Explorador de Campos Semanticos"),
        _parrafo("Ingresa una palabra para ver sus vecinos semánticos según el modelo entrenado sobre este corpus:"),

        _tarjeta([
            html.Label("Palabra:", style={"fontFamily": "Arial, sans-serif", "fontWeight": "bold", "color": VERDE_OSCURO}),
            dcc.Input(
                id="input-palabra-w2v", type="text", value="playa", debounce=True,
                style={
                    "width": "100%", "padding": "10px 14px", "borderRadius": "8px",
                    "border": f"1px solid {VERDE_MEDIO}", "fontSize": "15px", "marginTop": "8px",
                    "marginBottom": "16px", "boxSizing": "border-box",
                },
            ),
            dcc.Graph(id="grafico-w2v", config={"displayModeBar": False}),
        ], borde=VERDE_CLARO),
    ])


def _seccion_beto():
    return html.Div([
        _titulo_seccion("BETO - Embeddings Contextuales"),
        _parrafo("Busqueda semántica en lenguaje natural sobre las reseñas del corpus." if BETO_DISPONIBLE
                  else "BETO no esta disponible en este momento -- corre resultados_analisis/beto.ipynb "
                       "primero para generar los archivos precalculados."),

        _tarjeta([
            html.Label("Buscar resenas similares a:", style={
                "fontFamily": "Arial, sans-serif", "fontWeight": "bold", "color": VERDE_OSCURO,
            }),
            dcc.Input(
                id="input-busqueda-beto", type="text",
                value="un lugar tranquilo rodeado de naturaleza", debounce=True,
                disabled=not BETO_DISPONIBLE,
                style={
                    "width": "100%", "padding": "10px 14px", "borderRadius": "8px",
                    "border": f"1px solid {VERDE_MEDIO}", "fontSize": "15px", "marginTop": "8px",
                    "marginBottom": "16px", "boxSizing": "border-box",
                },
            ),
            html.Div(id="resultados-busqueda-beto"),
        ], borde=DORADO),

        html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}, children=[
            _tarjeta([
                html.H3('Polisemia contextual: "rico"', style={"color": VERDE_OSCURO, "fontSize": "18px"}),
                html.Div(id="tarjeta-polisemia"),
            ], borde=VERDE_CLARO),

            _tarjeta([
                html.H3("Masked Language Model", style={"color": VERDE_OSCURO, "fontSize": "18px"}),
                html.Div(id="tarjeta-mlm"),
            ], borde=VERDE_CLARO),
        ]),
    ])


def _estilo_boton_toggle(activo):
    return {
        "padding": "10px 22px", "border": f"2px solid {VERDE_MEDIO}",
        "borderRadius": "25px", "cursor": "pointer",
        "backgroundColor": VERDE_MEDIO if activo else BLANCO,
        "color": BLANCO if activo else VERDE_MEDIO,
        "fontFamily": "Arial, sans-serif", "fontSize": "14px", "fontWeight": "bold",
    }


def _seccion_comparacion():
    figuras_heatmap = [FIG_HEATMAP_TFIDF, FIG_HEATMAP_W2V]
    if BETO_DISPONIBLE:
        figuras_heatmap.append(FIG_HEATMAP_BETO)

    return html.Div([
        _titulo_seccion("Comparacion de Modelos"),

        html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "24px"}, children=[
            html.Button("Heatmaps (muestra de 15)", id="btn-comp-heatmap", n_clicks=0,
                        style=_estilo_boton_toggle(True)),
            html.Button("t-SNE (corpus completo)", id="btn-comp-tsne", n_clicks=0,
                        style=_estilo_boton_toggle(False)),
        ]),

        dcc.Store(id="vista-comparacion-activa", data="heatmap"),

        html.Div(id="contenido-comparacion-heatmap", style={"display": "block"}, children=[
            _parrafo(
                "Similitud coseno entre las mismas 15 resenas (5 por categoria: "
                "P=Parque, R=Restaurante, A=Alojamiento) segun cada representacion. "
                "Cuanto mas agrupados los bloques de una misma letra, mejor separa esa "
                "representacion las categorias."
            ),
            *[_tarjeta(dcc.Graph(figure=fig, config={"displayModeBar": False}), borde=DORADO)
              for fig in figuras_heatmap],
        ]),

        html.Div(id="contenido-comparacion-tsne", style={"display": "none"}, children=[
            _parrafo(
                "Proyeccion 2D (t-SNE) de TODAS las resenas del corpus segun cada "
                "representacion, coloreadas primero por tipo de lugar y despues por "
                "polaridad. Cuanto mas separados los colores, mejor esa representacion "
                "distingue esa dimension."
            ),
            _tarjeta(dcc.Graph(figure=FIG_TSNE_CATEGORIA, config={"displayModeBar": False}), borde=VERDE_CLARO),
            _tarjeta(dcc.Graph(figure=FIG_TSNE_POLARIDAD, config={"displayModeBar": False}), borde=VERDE_CLARO),
        ]),
    ])


layout = html.Div(children=[

    html.Section(style={
        "background": f"linear-gradient(135deg, {VERDE_OSCURO} 0%, {VERDE_MEDIO} 60%, {VERDE_CLARO} 100%)",
        "padding": "60px 60px 40px", "textAlign": "center",
    }, children=[
        html.P("ANALISIS SEMANTICO", style={
            "color": DORADO_CLARO, "fontSize": "13px",
            "letterSpacing": "4px", "marginBottom": "16px", "fontFamily": "Arial, sans-serif",
        }),
        html.H1("Comparacion de Representaciones", style={
            "color": BLANCO, "fontSize": "44px", "margin": "0 0 16px",
        }),
        html.P(
            "Bag of Words / TF-IDF, Word2Vec y BETO sobre el mismo corpus de resenas turisticas.",
            style={"color": "#D5E8D4", "fontSize": "17px", "maxWidth": "640px", "margin": "0 auto"},
        ),
    ]),

    html.Div(style={
        "display": "flex", "alignItems": "flex-start", "gap": "32px",
        "maxWidth": "1300px", "margin": "0 auto", "padding": "40px 20px 60px",
    }, children=[

        html.Div(style={"width": "240px", "flexShrink": "0", "position": "sticky", "top": "90px"}, children=[
            _boton_sidebar("bagofwords", True),
            _boton_sidebar("word2vec", False),
            _boton_sidebar("beto", False),
            _boton_sidebar("comparacion", False),
        ]),

        dcc.Store(id="seccion-activa", data="bagofwords"),
        html.Div(style={"flex": "1", "minWidth": "0"}, children=[
            html.Div(_seccion_bagofwords(), id="contenedor-bagofwords", style=_estilo_seccion(True)),
            html.Div(_seccion_word2vec(), id="contenedor-word2vec", style=_estilo_seccion(False)),
            html.Div(_seccion_beto(), id="contenedor-beto", style=_estilo_seccion(False)),
            html.Div(_seccion_comparacion(), id="contenedor-comparacion", style=_estilo_seccion(False)),
        ]),
    ]),
])


@callback(
    Output("seccion-activa", "data"),
    *[Input(f"btn-seccion-{s}", "n_clicks") for s in SECCIONES],
)
def _seleccionar_seccion(*_args):
    boton = ctx.triggered_id
    if not boton:
        return SECCIONES[0]
    return boton.replace("btn-seccion-", "")


@callback(
    *[Output(f"contenedor-{s}", "style") for s in SECCIONES],
    *[Output(f"btn-seccion-{s}", "style") for s in SECCIONES],
    Input("seccion-activa", "data"),
)
def _cambiar_seccion(seccion_activa):
    estilos_contenedores = [_estilo_seccion(s == seccion_activa) for s in SECCIONES]
    estilos_botones = [_estilo_boton_sidebar(s == seccion_activa) for s in SECCIONES]
    return estilos_contenedores + estilos_botones


@callback(
    Output("categoria-tfidf-activa", "data"),
    *[Input(f"btn-tfidf-{cat}", "n_clicks") for cat in CATEGORIAS],
)
def _seleccionar_categoria_tfidf(*_args):
    boton = ctx.triggered_id
    if not boton:
        return CATEGORIAS[0]
    return boton.replace("btn-tfidf-", "")


@callback(
    Output("vista-comparacion-activa", "data"),
    Input("btn-comp-heatmap", "n_clicks"),
    Input("btn-comp-tsne", "n_clicks"),
)
def _seleccionar_vista_comparacion(*_args):
    boton = ctx.triggered_id
    return "tsne" if boton == "btn-comp-tsne" else "heatmap"


@callback(
    Output("contenido-comparacion-heatmap", "style"),
    Output("contenido-comparacion-tsne", "style"),
    Output("btn-comp-heatmap", "style"),
    Output("btn-comp-tsne", "style"),
    Input("vista-comparacion-activa", "data"),
)
def _mostrar_vista_comparacion(vista):
    es_heatmap = vista != "tsne"
    return (
        _estilo_seccion(es_heatmap),
        _estilo_seccion(not es_heatmap),
        _estilo_boton_toggle(es_heatmap),
        _estilo_boton_toggle(not es_heatmap),
    )


@callback(
    Output("grafico-tfidf", "figure"),
    *[Output(f"btn-tfidf-{cat}", "style") for cat in CATEGORIAS],
    Input("categoria-tfidf-activa", "data"),
)
def _actualizar_tfidf(categoria_activa):
    idx_categoria = corpus.index[corpus["categoria"] == categoria_activa].tolist()
    idx_categoria_pos = [corpus.index.get_loc(i) for i in idx_categoria]
    top_terminos = top_terminos_tfidf(matriz_tfidf, vectorizer_tfidf, idx_categoria_pos, top_n=15)

    df_top = pd.DataFrame(top_terminos, columns=["termino", "peso"]).sort_values("peso")
    fig = px.bar(
        df_top, x="peso", y="termino", orientation="h",
        title=f"Top terminos TF-IDF -- {categoria_activa}",
        color_discrete_sequence=[VERDE_MEDIO],
    )
    fig.update_layout(paper_bgcolor=BLANCO, plot_bgcolor=BLANCO)

    estilos = []
    for cat in CATEGORIAS:
        activo = cat == categoria_activa
        estilos.append({
            "padding": "10px 22px", "border": f"2px solid {VERDE_MEDIO}",
            "borderRadius": "25px", "cursor": "pointer",
            "backgroundColor": VERDE_MEDIO if activo else BLANCO,
            "color": BLANCO if activo else VERDE_MEDIO,
            "fontFamily": "Arial, sans-serif", "fontSize": "14px", "fontWeight": "bold",
        })
    return [fig] + estilos


@callback(
    Output("grafico-w2v", "figure"),
    Input("input-palabra-w2v", "value"),
)
def _actualizar_w2v(palabra):
    if not palabra:
        return px.bar(title="Escribi una palabra")

    resultado = palabras_similares(modelo_w2v, palabra.strip().lower(), top_n=15)
    if isinstance(resultado, str):
        return px.bar(title=resultado)

    df_resultado = pd.DataFrame(resultado, columns=["palabra", "similitud"]).sort_values("similitud")
    fig = px.bar(
        df_resultado, x="similitud", y="palabra", orientation="h",
        title=f"Palabras mas similares a '{palabra}'",
        color_discrete_sequence=[DORADO],
    )
    fig.update_layout(paper_bgcolor=BLANCO, plot_bgcolor=BLANCO)
    return fig


@callback(
    Output("resultados-busqueda-beto", "children"),
    Input("input-busqueda-beto", "value"),
)
def _buscar_beto(consulta):
    if not BETO_DISPONIBLE:
        return html.P("BETO no disponible.", style={"color": "#999", "fontStyle": "italic"})
    if not consulta:
        return html.P("Escribi una consulta.", style={"color": "#999", "fontStyle": "italic"})

    vector_consulta = embeddings_de_textos([consulta], modo="mean")
    similitudes = cosine_similarity(vector_consulta, embeddings_corpus_beto)[0]
    top_indices = similitudes.argsort()[::-1][:5]

    return html.Div([
        html.Div([
            html.Span(f"[{similitudes[i]:.3f}] ", style={"color": DORADO, "fontWeight": "bold"}),
            html.Span(f"lugar: {corpus.iloc[i]['lugar']} -- ", style={"color": VERDE_MEDIO, "fontWeight": "bold"}),
            html.Span(str(corpus.iloc[i]["comentarios_espanol"])[:200] + "..."),
        ], style={"marginBottom": "12px", "fontSize": "14px", "lineHeight": "1.6"})
        for i in top_indices
    ])


@callback(
    Output("tarjeta-polisemia", "children"),
    Input("tarjeta-polisemia", "id"),
)
def _mostrar_polisemia(_):
    datos = resultados_beto_precalculados.get("polisemia")
    if not datos or isinstance(datos, str):
        return html.P("No disponible.", style={"color": "#999", "fontStyle": "italic"})

    return html.Div([
        html.P(f"Similitud intra-positivas: {datos['similitud_intra_positivas']:.3f}", style={"margin": "4px 0", "fontSize": "14px"}),
        html.P(f"Similitud intra-negativas: {datos['similitud_intra_negativas']:.3f}", style={"margin": "4px 0", "fontSize": "14px"}),
        html.P(f"Similitud entre ambos sentidos: {datos['similitud_entre_positiva_negativa']:.3f}", style={"margin": "4px 0", "fontSize": "14px"}),
    ])


@callback(
    Output("tarjeta-mlm", "children"),
    Input("tarjeta-mlm", "id"),
)
def _mostrar_mlm(_):
    ejemplos = resultados_beto_precalculados.get("mlm_ejemplos", [])
    if not ejemplos:
        return html.P("No disponible.", style={"color": "#999", "fontStyle": "italic"})

    return html.Div([
        html.Div([
            html.P(ejemplo["oracion"], style={"fontStyle": "italic", "margin": "0 0 4px", "fontSize": "14px"}),
            html.P(", ".join(ejemplo["predicciones"]), style={"color": VERDE_MEDIO, "fontWeight": "bold", "margin": "0 0 12px", "fontSize": "14px"}),
        ])
        for ejemplo in ejemplos
    ])