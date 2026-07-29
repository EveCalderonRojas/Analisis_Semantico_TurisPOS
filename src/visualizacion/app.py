import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# PALETA DE COLORES
VERDE_OSCURO  = '#1B4332'
VERDE_MEDIO   = '#2D6A4F'
VERDE_CLARO   = '#52B788'
DORADO        = '#D4A017'
DORADO_CLARO  = '#F4D03F'
CREMA         = '#F9F5EC'
BLANCO        = '#FFFFFF'

# INICIO
import os

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=os.path.dirname(os.path.abspath(__file__)),
    suppress_callback_exceptions=True
)
app.title = 'TurisPOS — Reseñas Turísticas'

# BARRA DE NAVEGACIÓN
navbar = html.Nav(style={
    'backgroundColor': VERDE_OSCURO,
    'padding': '0 40px',
    'display': 'flex',
    'alignItems': 'center',
    'justifyContent': 'space-between',
    'position': 'sticky',
    'top': '0',
    'zIndex': '1000',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'
}, children=[
    dcc.Link('🌿 TurisPOS', href='/', style={
        'color': DORADO_CLARO,
        'fontSize': '22px',
        'fontWeight': 'bold',
        'letterSpacing': '1px',
        'textDecoration': 'none'
    }),
    html.Div(style={'display': 'flex', 'gap': '8px'}, children=[
        dcc.Link('Inicio',   href='/',              style={'color': BLANCO, 'textDecoration': 'none', 'padding': '18px 20px', 'fontSize': '15px', 'letterSpacing': '0.5px'}),
        dcc.Link('Modelos',  href='/modelos',       style={'color': BLANCO, 'textDecoration': 'none', 'padding': '18px 20px', 'fontSize': '15px', 'letterSpacing': '0.5px'})
    ])
])

# FOOTER
footer = html.Footer(style={
    'backgroundColor': VERDE_OSCURO,
    'color': '#aaa',
    'textAlign': 'center',
    'padding': '30px',
    'fontSize': '14px',
    'fontFamily': 'Arial, sans-serif'
}, children=[
    html.P('Minería de Datos — Análisis de Reseñas Turísticas Costa Rica', style={'margin': '0 0 6px'}),
    html.P('Análisis Semántico: BoW / TF-IDF, Word2Vec y BETO', style={'margin': '0', 'color': VERDE_CLARO})
])

# LAYOUT ─
app.layout = html.Div(
    style={'fontFamily': 'Georgia, serif', 'backgroundColor': CREMA, 'margin': '0', 'padding': '0'},
    children=[
        navbar,
        dash.page_container,
        footer
    ]
)

# EJECUCIÓN
if __name__ == '__main__':
    app.run(debug=True)