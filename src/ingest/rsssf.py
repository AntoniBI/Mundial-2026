"""Fuente: RSSSF — archivo histórico de Mundiales.

Descarga el índice histórico de la Copa del Mundo (tablesw/worldcup.html),
que lista campeones y podios de todas las ediciones. Se usa como referencia
de validación (sanity-check) de las probabilidades simuladas, no como
feature del modelo (los resultados partido a partido ya vienen de martj42,
cuyo origen primario es precisamente RSSSF/Wikipedia).
"""
import requests

from src.config import RAW, USER_AGENT

URL = "https://www.rsssf.org/tablesw/worldcup.html"


def download() -> str:
    r = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    out = RAW / "rsssf_worldcup.html"
    out.write_bytes(r.content)
    print(f"[rsssf] índice histórico de Mundiales: {len(r.content):,} bytes -> {out.name}")
    return r.text


if __name__ == "__main__":
    download()
