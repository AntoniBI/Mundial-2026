"""Fuente: FBref (Sports Reference) — estadísticas avanzadas (xG) de Mundiales.

FBref protege el sitio con Cloudflare y bloquea clientes no-navegador
(HTTP 403). Se intenta primero con `cloudscraper` si está instalado y, si
no, con requests + cabeceras de navegador. Si el bloqueo persiste, el
módulo lo informa y el pipeline continúa: el xG de FBref es un complemento
(solo existe para torneos grandes), no una dependencia del modelo.
"""
import time

import requests

from src.config import RAW, USER_AGENT

# Estadísticas de equipo por edición del Mundial
PAGES = {
    "wc2022": "https://fbref.com/en/comps/1/2022/2022-FIFA-World-Cup-Stats",
    "wc2018": "https://fbref.com/en/comps/1/2018/2018-FIFA-World-Cup-Stats",
}


def _get(url: str) -> bytes | None:
    try:
        import cloudscraper  # type: ignore
        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, timeout=60)
    except ImportError:
        r = requests.get(url, timeout=60, headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://fbref.com/",
        })
    if r.status_code == 200:
        return r.content
    print(f"[fbref] HTTP {r.status_code} en {url} (bloqueo Cloudflare)")
    return None


def download() -> bool:
    ok = True
    for name, url in PAGES.items():
        content = _get(url)
        if content:
            (RAW / f"fbref_{name}.html").write_bytes(content)
            print(f"[fbref] {name}: {len(content):,} bytes")
        else:
            ok = False
        time.sleep(4)  # FBref pide >=3s entre peticiones
    if not ok:
        print("[fbref] AVISO: FBref no accesible desde este entorno; "
              "el modelo funciona sin xG de FBref.")
    return ok


if __name__ == "__main__":
    download()
