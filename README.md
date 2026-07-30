# Análisis de comentarios 

## TurisPOS

### ✏️ Descripción 

Este es un proyecto de análisis semántico en donde se pone en práctica todo el pipeline de datos requerido:
- Extracción de comentarios 
- Limpieza
- Modelado
- Visualización de resultados 

En donde los comentarios a analizar serán relacionados a lugares turísticos de Costa Rica, pasando por varias categorías para un mayor enriquecimiento de conocimiento y de resultados finales.

El corpus trabajado en este proyecto es el mismo que se trabajó para la primera parte de TurisPOS, con la diferencia de que en esta ocasión, se enriqueció mucho más gracias a la incorporación de más comentarios obtenidos mediante técnicas de Web Scrapping.

### 🛠️ Herramientas 

Python:
- Limpieza de datos.
- Manejo de emojis.
- Guardado de información en .csv de la información limpia para los modelos.
- Uso de modelos BagOfWords, Word2Vec y BETO.
- Visualización de información. 
- Extracción de comentarios mediante Playwrite.


Claude como asistente de IA para entendimiento y optimización de código.


Plotly Dash:
- Parte visual del proyecto 
- Presentación de resultados de métricas y gráficos.
- Vista tipo página web en la que se puede navegar entre secciones.

MongoDB
- Almacenamiento del corpus final.
- Consultas directas sobre la base de datos.

### 📁 Organización

✅ data
- 🗁 processed: Datos a los que se les aplicó limpieza, análisis y traducción.

✅ src
- 🗁 limpieza: ajuste de los nuevos comentarios al formato del primer corpus, limpiando y normalizando los datos.
-   modelos_analisis_semantico: definición y funciones de los modelos a utilizar, cada modelo se encuentra en una clase por aparte.
-   mongodb: establece la conexión y el paso de los datos desde python hasta la base de datos NoSQL.
-   procesamiento_corpus: estandarización del corpus listo para usarse en cada modelo sin necesidad de ajustar modelo por modelo.
-   resultados_analisis: notebooks con los resultados de procesar el corpus en cada uno de los modelos requeridos.
-   web_scrapping: extracción de la información de la página de Booking.com.
- 🗁 visualizaciones: resultados visuales utilizando Plotly Dash.

#### 👩🏻‍💻 Elaborado por:

Evelin Calderón Rojas 

Estudiante de Big Data 

Curso: Minería de Textos



