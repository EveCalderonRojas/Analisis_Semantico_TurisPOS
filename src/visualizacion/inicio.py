
import dash
from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import re
import base64
from io import BytesIO
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

dash.register_page(__name__, path='/', name='Inicio')

# PALETA DE COLORES
VERDE_OSCURO = '#1B4332'
VERDE_MEDIO  = '#2D6A4F'
VERDE_CLARO  = '#52B788'
DORADO       = '#D4A017'
DORADO_CLARO = '#F4D03F'
CREMA        = '#F9F5EC'
GRIS_SUAVE   = '#F0EDE6'
BLANCO       = '#FFFFFF'

# CORPUS A UTILIZAR
df = pd.read_csv('../../data/processed/corpus.csv')
muestras = df[df['calificacion'] == 5].dropna(subset=['comentarios_espanol']).sample(6, random_state=42)
stop_words = set(stopwords.words('spanish'))
CATEGORIAS = ['todas'] + sorted(df['categoria'].dropna().unique().tolist())

# COORDENADAS DE LOS LUGARES PARA UBICAR EN EL MAPA
ATRACCIONES = [
    {'lugar': 'Playa Manuel Antonio',        'categoria': 'parque',      'provincia': 'Puntarenas', 'lat':  9.3921, 'lon': -84.1741},
    {'lugar': 'Playa Chiquita',              'categoria': 'parque',      'provincia': 'Limón',      'lat':  9.6333, 'lon': -82.7167},
    {'lugar': 'Catarata Río Fortuna',        'categoria': 'parque',      'provincia': 'Alajuela',   'lat': 10.4631, 'lon': -84.6441},
    {'lugar': 'Corcovado National Park',     'categoria': 'parque',      'provincia': 'Puntarenas', 'lat':  8.5833, 'lon': -83.5833},
    {'lugar': 'Monteverde Country Lodge',    'categoria': 'alojamiento', 'provincia': 'Puntarenas', 'lat': 10.3000, 'lon': -84.8167},
    {'lugar': 'Hotel Los Lagos Spa & Resort','categoria': 'alojamiento', 'provincia': 'Alajuela',   'lat': 10.4700, 'lon': -84.6400},
    {'lugar': 'El Yugo de Mi Tata',          'categoria': 'restaurante', 'provincia': 'Limón',      'lat':  9.9900, 'lon': -83.0300},
    {'lugar': 'Restaurante Mi Finca',        'categoria': 'restaurante', 'provincia': 'Guanacaste', 'lat': 10.3200, 'lon': -85.4400},
    {'lugar': 'Restaurante Fortuneño',       'categoria': 'restaurante', 'provincia': 'Alajuela',   'lat': 10.4650, 'lon': -84.6430},
]

df_atr = pd.DataFrame(ATRACCIONES)
df_atr['reseñas'] = df_atr['lugar'].apply(lambda l: len(df[df['lugar'] == l]))

COLORES_CAT_MAPA = {'parque': VERDE_MEDIO, 'restaurante': DORADO, 'alojamiento': '#5B8DB8'}



# NUBE DE PALABRAS

def limpiar_texto(texto):
    if pd.isna(texto): return ''
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', '', texto)
    tokens = texto.split()
    return ' '.join([t for t in tokens if t not in stop_words and len(t) > 2])

def generar_nube(categoria):
    dff = df if categoria == 'todas' else df[df['categoria'] == categoria]
    corpus_texto = ' '.join(dff['comentarios_espanol'].dropna().apply(limpiar_texto))
    if not corpus_texto.strip(): return None
    wc = WordCloud(
        width=900, height=380, background_color=CREMA,
        colormap='YlGn', max_words=80,
        prefer_horizontal=0.85, collocations=False,
    ).generate(corpus_texto)
    buf = BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')




# LAYOUT
layout = html.Div(children=[


    html.Section(style={
        'background': f'linear-gradient(135deg, {VERDE_OSCURO} 0%, {VERDE_MEDIO} 60%, {VERDE_CLARO} 100%)',
        'padding': '80px 60px 60px', 'textAlign': 'center'
    }, children=[
        html.P('ANÁLISIS SEMÁNTICO', style={
            'color': DORADO_CLARO, 'fontSize': '13px',
            'letterSpacing': '4px', 'marginBottom': '16px', 'fontFamily': 'Arial, sans-serif'
        }),
        html.H1('Turismo Costarricense', style={
            'color': BLANCO, 'fontSize': '52px', 'lineHeight': '1.2',
            'margin': '0 0 24px', 'whiteSpace': 'pre-line'
        }),
        html.P(
            'De la web al análisis: '
            'Análisis de reseñas con modelos de IA',
            style={'color': '#D5E8D4', 'fontSize': '18px', 'maxWidth': '640px',
                   'margin': '0 auto 40px', 'lineHeight': '1.7'}
        ),
        html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '40px', 'flexWrap': 'wrap'}, children=[
            html.Div([
                html.Span(f'{len(df):,}', style={'color': DORADO_CLARO, 'fontSize': '42px', 'fontWeight': 'bold', 'display': 'block'}),
                html.Span('reseñas analizadas', style={'color': '#D5E8D4', 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif'})
            ]),
            html.Div([
                html.Span(f'{df["lugar"].nunique()}', style={'color': DORADO_CLARO, 'fontSize': '42px', 'fontWeight': 'bold', 'display': 'block'}),
                html.Span('destinos turísticos', style={'color': '#D5E8D4', 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif'})
            ]),
            html.Div([
                html.Span(f'{df["categoria"].nunique()}', style={'color': DORADO_CLARO, 'fontSize': '42px', 'fontWeight': 'bold', 'display': 'block'}),
                html.Span('categorías de lugar', style={'color': '#D5E8D4', 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif'})
            ]),
        ])
    ]),

    # DESCRIPCIÓN DEL PROYECTO
    html.Section(style={'padding': '60px', 'maxWidth': '900px', 'margin': '0 auto'}, children=[
        html.H2('Sobre el proyecto', style={
            'color': VERDE_OSCURO, 'fontSize': '28px',
            'borderBottom': f'3px solid {DORADO}', 'paddingBottom': '12px'
        }),
        html.P(
            'Este proyecto aplica técnicas de minería de datos y procesamiento de lenguaje natural (PLN) '
            'sobre reseñas reales extraídas de la web correspondientes a atractivos turísticos costarricenses '
            'El corpus incluye parques nacionales, restaurantes y alojamientos, con reseñas originalmente '
            'escritas en múltiples idiomas y traducidas al español para su análisis',
            style={'color': '#333', 'fontSize': '17px', 'lineHeight': '1.8', 'marginBottom': '20px'}
        ),

    ]),

    # COMENTARIOS SACADOS DEL CORPUS PARA MOSTRARLOS EN EL INICIO
    html.Section(style={'backgroundColor': GRIS_SUAVE, 'padding': '60px'}, children=[
        html.H2('Reseñas destacadas', style={
            'color': VERDE_OSCURO, 'fontSize': '28px', 'textAlign': 'center',
            'borderBottom': f'3px solid {DORADO}', 'paddingBottom': '12px',
            'maxWidth': '900px', 'margin': '0 auto 40px'
        }),
        html.Div(style={
            'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))',
            'gap': '24px', 'maxWidth': '1100px', 'margin': '0 auto'
        }, children=[
            html.Div(style={
                'backgroundColor': BLANCO, 'borderRadius': '12px', 'padding': '24px',
                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)', 'borderTop': f'4px solid {VERDE_CLARO}'
            }, children=[
                html.P('⭐' * int(row['calificacion']), style={'margin': '0 0 8px', 'fontSize': '16px'}),
                html.P(f'📍 {row["lugar"]}', style={
                    'color': VERDE_MEDIO, 'fontSize': '13px', 'margin': '0 0 12px',
                    'fontFamily': 'Arial, sans-serif', 'fontWeight': 'bold'
                }),
                html.P(
                    str(row['comentarios_espanol'])[:200] + '...'
                    if len(str(row['comentarios_espanol'])) > 200
                    else str(row['comentarios_espanol']),
                    style={'color': '#444', 'fontSize': '14px', 'lineHeight': '1.6',
                           'margin': '0', 'fontStyle': 'italic'}
                )
            ]) for _, row in muestras.iterrows()
        ])
    ]),

    # MAPA DE COSTA RICA PARA MOSTAR
    html.Section(style={'padding': '60px', 'maxWidth': '1100px', 'margin': '0 auto'}, children=[
        html.H2('Destinos del corpus', style={
            'color': VERDE_OSCURO, 'fontSize': '28px',
            'borderBottom': f'3px solid {DORADO}', 'paddingBottom': '12px', 'marginBottom': '8px'
        }),
        html.P('Ubicación geográfica de los atractivos turísticos analizados.',
               style={'color': '#666', 'fontFamily': 'Arial, sans-serif', 'fontSize': '14px', 'marginBottom': '28px'}),


        html.Div(style={'display': 'flex', 'gap': '24px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                html.Div(style={'width': '14px', 'height': '14px', 'borderRadius': '50%', 'backgroundColor': VERDE_MEDIO}),
                html.Span('Parque', style={'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', 'color': '#444'})
            ]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                html.Div(style={'width': '14px', 'height': '14px', 'borderRadius': '50%', 'backgroundColor': DORADO}),
                html.Span('Restaurante', style={'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', 'color': '#444'})
            ]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                html.Div(style={'width': '14px', 'height': '14px', 'borderRadius': '50%', 'backgroundColor': '#5B8DB8'}),
                html.Span('Alojamiento', style={'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', 'color': '#444'})
            ]),
        ]),

        dcc.Graph(
            id='mapa-atracciones',
            figure=px.scatter_mapbox(
                df_atr,
                lat='lat', lon='lon',
                color='categoria',
                color_discrete_map=COLORES_CAT_MAPA,
                size='reseñas',
                size_max=20,
                hover_name='lugar',
                hover_data={'provincia': True, 'reseñas': True, 'lat': False, 'lon': False, 'categoria': False},
                zoom=6.5,
                center={'lat': 9.9, 'lon': -84.2},
                mapbox_style='open-street-map',
                height=480,
            ).update_layout(
                paper_bgcolor=BLANCO,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
            ),
            config={'displayModeBar': False}
        )
    ]),

    # NUBE DE PALABRAS A MOSTRAR
    html.Section(style={'backgroundColor': GRIS_SUAVE, 'padding': '60px'}, children=[
        html.H2('Palabras más frecuentes', style={
            'color': VERDE_OSCURO, 'fontSize': '28px',
            'borderBottom': f'3px solid {DORADO}', 'paddingBottom': '12px',
            'maxWidth': '1100px', 'margin': '0 auto 12px'
        }),
        html.P('Palabras con más repetición dentro de los comentarios: ',
               style={'color': '#666', 'fontFamily': 'Arial, sans-serif', 'fontSize': '14px',
                      'marginBottom': '28px', 'maxWidth': '1100px', 'margin': '0 auto 28px'}),

        html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap',
                        'marginBottom': '32px', 'maxWidth': '1100px', 'margin': '0 auto 32px'}, children=[
            html.Button(
                cat.capitalize(), id=f'btn-cat-{cat}', n_clicks=0,
                style={
                    'padding': '10px 22px', 'border': f'2px solid {VERDE_MEDIO}',
                    'borderRadius': '25px', 'cursor': 'pointer',
                    'backgroundColor': VERDE_MEDIO if cat == 'todas' else BLANCO,
                    'color': BLANCO if cat == 'todas' else VERDE_MEDIO,
                    'fontFamily': 'Arial, sans-serif', 'fontSize': '14px', 'fontWeight': 'bold',
                }
            ) for cat in CATEGORIAS
        ]),

        dcc.Store(id='cat-activa', data='todas'),

        html.Div(style={
            'backgroundColor': CREMA, 'borderRadius': '16px', 'padding': '20px',
            'boxShadow': '0 2px 12px rgba(0,0,0,0.08)', 'textAlign': 'center',
            'maxWidth': '1100px', 'margin': '0 auto'
        }, children=[
            html.Img(id='nube-imagen', style={'width': '100%', 'maxWidth': '900px', 'borderRadius': '8px'})
        ])
    ]),
])



@callback(
    Output('cat-activa', 'data'),
    *[Input(f'btn-cat-{cat}', 'n_clicks') for cat in CATEGORIAS],
)
def seleccionar_categoria(*args):
    from dash import ctx
    boton = ctx.triggered_id
    if not boton: return 'todas'
    return boton.replace('btn-cat-', '')


@callback(
    Output('nube-imagen', 'src'),
    *[Output(f'btn-cat-{cat}', 'style') for cat in CATEGORIAS],
    Input('cat-activa', 'data'),
)
def actualizar_nube(cat_activa):
    img = generar_nube(cat_activa)
    estilos = []
    for cat in CATEGORIAS:
        activo = cat == cat_activa
        estilos.append({
            'padding': '10px 22px', 'border': f'2px solid {VERDE_MEDIO}',
            'borderRadius': '25px', 'cursor': 'pointer',
            'backgroundColor': VERDE_MEDIO if activo else BLANCO,
            'color': BLANCO if activo else VERDE_MEDIO,
            'fontFamily': 'Arial, sans-serif', 'fontSize': '14px', 'fontWeight': 'bold',
        })
    return [img] + estilos