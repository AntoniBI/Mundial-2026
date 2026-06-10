"""Fuente: API-Football (vía RapidAPI) — opcional, requiere clave.

Define la variable de entorno API_FOOTBALL_KEY (clave de RapidAPI) para
activarla. Descarga lesiones/alineaciones y cuotas que pueden enriquecer
el modelo. Sin clave, se omite limpiamente.
"""
import os

import requests

from src.config import RAW

HOST = "api-football-v1.p.rapidapi.com"


def download() -> bool:
    key = os.environ.get("API_FOOTBALL_KEY") or os.environ.get("RAPIDAPI_KEY")
    if not key:
        print("[api-football] OMITIDO: define API_FOOTBALL_KEY para activar esta fuente.")
        return False
    headers = {"x-rapidapi-key": key, "x-rapidapi-host": HOST}
    # Ejemplo: equipos del Mundial 2026 (league=1, season=2026)
    r = requests.get(f"https://{HOST}/v3/teams", params={"league": 1, "season": 2026},
                     headers=headers, timeout=60)
    r.raise_for_status()
    (RAW / "api_football_wc26_teams.json").write_bytes(r.content)
    print(f"[api-football] guardado api_football_wc26_teams.json")
    return True


if __name__ == "__main__":
    download()
