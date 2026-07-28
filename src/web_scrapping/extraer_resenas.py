"""
Web Scrapping con Playwrite

Sitio web: Booking.com


Más flexible para la extracción pero siempre con cuidado
"""



import csv
import os
import random
import re
import time
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

LUGARES = [
    {
        "lugar": "Hotel Roca Negra del Arenal",
        "categoria": "alojamiento",
        "url": "https://www.booking.com/reviews/cr/hotel/roca-negra-del-arenal.es.html",
    },

]

ARCHIVO_SALIDA = "resenas_booking.csv"
COLUMNAS = [
    "puntuacion", "texto_positivo", "texto_negativo", "texto_resena",
    "lugar", "categoria", "fuente", "fecha_recopilacion", "url_fuente", "idioma_comentario",
]


SELECTORES_TARJETA = [
    'div[data-testid="review-card"]',
    'li[data-testid="review-card"]',
    'div[class*="c-review-block"]',
]
SELECTORES_POSITIVO = [
    'span[data-testid="review-positive-text"]',
    'div[data-testid="review-positive-text"]',
]
SELECTORES_NEGATIVO = [
    'span[data-testid="review-negative-text"]',
    'div[data-testid="review-negative-text"]',
]
SELECTORES_PUNTUACION = [
    'div[data-testid="review-score"]',
    'span[data-testid="review-score"]',
]


def _cerrar_banner_cookies(page) -> bool:
    """Intenta cerrar el banner de cookies/consentimiento si aparece."""
    selectores_posibles = [
        "#onetrust-accept-btn-handler",
        'button[id*="accept"]',
        'button[data-testid="cookie-banner-accept-button"]',
    ]
    for selector in selectores_posibles:
        try:
            page.click(selector, timeout=3000)
            time.sleep(random.uniform(0.5, 1.2))
            return True
        except PlaywrightTimeoutError:
            continue
    return False


def _primer_selector_que_funcione(soup_o_page, selectores, buscar_todos=False):
    """Prueba una lista de selectores CSS y devuelve el primero que encuentre algo."""
    for selector in selectores:
        if buscar_todos:
            resultado = soup_o_page.select(selector)
            if resultado:
                return resultado
        else:
            resultado = soup_o_page.select_one(selector)
            if resultado:
                return resultado
    return [] if buscar_todos else None



"""
------------------------------------------------------------
El sitio muestra los comentarios de un tipo de modal que aparece cuando se da click al botón Leer más comentarios
------------------------------------------------------------
"""

def _abrir_modal_comentarios(page) -> bool:

    import re as _re

    posibles_textos = [
        "Leer todos los comentarios",
        "Read all reviews",
        "Ver todos los comentarios",
    ]
    for texto in posibles_textos:
        try:
            boton = page.get_by_text(_re.compile(_re.escape(texto), _re.IGNORECASE))
            if boton.count() > 0:
                boton.first.scroll_into_view_if_needed()
                time.sleep(random.uniform(0.5, 1.0))
                boton.first.click()
                time.sleep(random.uniform(1.5, 2.5))  # esperar a que abra el modal
                return True
        except Exception:
            continue
    return False


def _guardar_html_diagnostico(page, ruta: str = "debug_booking_modal.html") -> None:

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(page.content())
    print(f"HTML de diagnóstico guardado en: {ruta}")



def _esperar_resenas(page, timeout: int = 15000) -> bool:

    for selector in SELECTORES_TARJETA:
        try:
            page.wait_for_selector(selector, timeout=timeout)
            return True
        except PlaywrightTimeoutError:
            continue
    return False


def _extraer_puntuacion(bloque) -> str:
    elemento = _primer_selector_que_funcione(bloque, SELECTORES_PUNTUACION)
    if elemento:
        texto = elemento.get_text(strip=True)
        match = re.search(r"\d+([.,]\d+)?", texto)
        if match:
            return match.group(0).replace(",", ".")
    return "Sin puntuación"


def _extraer_texto_positivo(bloque) -> str:
    elemento = _primer_selector_que_funcione(bloque, SELECTORES_POSITIVO)
    return elemento.get_text(strip=True) if elemento else ""


def _extraer_texto_negativo(bloque) -> str:
    elemento = _primer_selector_que_funcione(bloque, SELECTORES_NEGATIVO)
    return elemento.get_text(strip=True) if elemento else ""


def _extraer_resenas_de_pagina(soup) -> list:

    bloques = _primer_selector_que_funcione(soup, SELECTORES_TARJETA, buscar_todos=True)

    resenas = []
    for bloque in bloques:
        try:
            positivo = _extraer_texto_positivo(bloque)
            negativo = _extraer_texto_negativo(bloque)
            partes_combinadas = []
            if positivo:
                partes_combinadas.append(f"Positivo: {positivo}")
            if negativo:
                partes_combinadas.append(f"Negativo: {negativo}")

            resenas.append({
                "puntuacion": _extraer_puntuacion(bloque),
                "texto_positivo": positivo,
                "texto_negativo": negativo,
                "texto_resena": " | ".join(partes_combinadas) if partes_combinadas else "Sin contenido",
            })
        except Exception as e:
            print(f"Reseña omitida por error de parseo: {e}")
            continue
    return resenas


def _hacer_clic_siguiente_pagina(page) -> bool:

    selectores_pagina = [
        'button[aria-label="Next page"]',
        'button[aria-label="Página siguiente"]',
        'a[aria-label="Next page"]',
    ]
    for selector in selectores_pagina:
        try:
            elemento = page.query_selector(selector)
            if elemento is None or not elemento.is_enabled():
                continue
            elemento.scroll_into_view_if_needed()
            time.sleep(random.uniform(0.5, 1.0))
            elemento.click()
            return True
        except Exception:
            continue

    # Fallback: botón de "mostrar más" (carga más reseñas en la misma página)
    try:
        boton_mas = page.get_by_text(re.compile("mostrar más|show more|más reseñas", re.IGNORECASE))
        if boton_mas and boton_mas.count() > 0:
            boton_mas.first.scroll_into_view_if_needed()
            time.sleep(random.uniform(0.5, 1.0))
            boton_mas.first.click()
            return True
    except Exception:
        pass

    return False


def extraer_lugar(page, lugar_info: dict, archivo_csv: str,
                   max_paginas: int = 3, escribir_encabezado: bool = False) -> None:

    fecha_recopilacion = datetime.now(timezone.utc).isoformat()

    with open(archivo_csv, mode="a", newline="", encoding="utf-8-sig") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        if escribir_encabezado:
            writer.writeheader()

        page.goto(lugar_info["url"], wait_until="domcontentloaded")
        total_guardadas = 0

        for pagina in range(max_paginas):
            print(f"  === {lugar_info['lugar']} -- página {pagina + 1}/{max_paginas} ===")

            if pagina == 0:
                _cerrar_banner_cookies(page)
                if _abrir_modal_comentarios(page):
                    print("Modal de comentarios abierto.")
                else:
                    print("No se encontró el botón 'Leer todos los comentarios'.")

            if not _esperar_resenas(page):
                print("No se encontraron tarjetas de reseña con ningún selector conocido.")
                _guardar_html_diagnostico(page)
                print("")
                break

            time.sleep(random.uniform(2.0, 4.5))

            soup = BeautifulSoup(page.content(), "lxml")
            resenas_pagina = _extraer_resenas_de_pagina(soup)

            for resena in resenas_pagina:
                writer.writerow({
                    **resena,
                    "lugar": lugar_info["lugar"],
                    "categoria": lugar_info["categoria"],
                    "fuente": "booking",
                    "fecha_recopilacion": fecha_recopilacion,
                    "url_fuente": page.url,
                    "idioma_comentario": None,  # se determina después, al traducir
                })
            archivo.flush()
            total_guardadas += len(resenas_pagina)
            print(f"    -> {len(resenas_pagina)} reseñas guardadas (total acumulado: {total_guardadas}).")

            if pagina < max_paginas - 1:
                if not _hacer_clic_siguiente_pagina(page):
                    print("No hay más páginas/reseñas disponibles para este lugar.")
                    break
                time.sleep(random.uniform(3.0, 5.5))


def ejecutar_scraping(lugares: list = None, archivo_csv: str = ARCHIVO_SALIDA,
                       max_paginas: int = 3, headless: bool = False) -> None:

    lugares = lugares if lugares is not None else LUGARES
    existe_archivo = os.path.exists(archivo_csv)

    with sync_playwright() as p:
        navegador = p.chromium.launch(headless=headless)
        contexto = navegador.new_context(
            locale="es-CR",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = contexto.new_page()

        try:
            for i, lugar_info in enumerate(lugares):
                print(f"\n=== Lugar {i + 1}/{len(lugares)}: {lugar_info['lugar']} ===")
                try:
                    extraer_lugar(
                        page, lugar_info, archivo_csv,
                        max_paginas=max_paginas,
                        escribir_encabezado=(not existe_archivo and i == 0),
                    )
                except Exception as e:
                    print(f"Error scrapeando '{lugar_info['lugar']}': {e}")
                    continue

                if i < len(lugares) - 1:
                    pausa = random.uniform(8.0, 15.0)
                    print(f"Pausa de {pausa:.1f}s antes del siguiente lugar...")
                    time.sleep(pausa)
        finally:
            navegador.close()
            print("\nNavegador cerrado. Proceso finalizado.")


if __name__ == "__main__":

    ejecutar_scraping(headless=False, max_paginas=165)


