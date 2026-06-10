"""Fuente: Sportmonks Football API — opcional, requiere clave.

Define SPORTMONKS_KEY para activarla. Sin clave, se omite limpiamente.
"""
import os

import requests

from src.config import RAW


def download() -> bool:
    key = os.environ.get("SPORTMONKS_KEY")
    if not key:
        print("[sportmonks] OMITIDO: define SPORTMONKS_KEY para activar esta fuente.")
        return False
    r = requests.get("https://api.sportmonks.com/v3/football/teams",
                     params={"api_token": key}, timeout=60)
    r.raise_for_status()
    (RAW / "sportmonks_teams.json").write_bytes(r.content)
    print("[sportmonks] guardado sportmonks_teams.json")
    return True


if __name__ == "__main__":
    download()
