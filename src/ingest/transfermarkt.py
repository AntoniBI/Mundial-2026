"""Fuente: Transfermarkt — valor de mercado de las plantillas por Mundial.

Descarga las páginas de participantes de cada edición (FIWC/saison_id):
2014, 2018, 2022 y 2026. El valor por edición permite usar la variable en
entrenamiento (Mundiales pasados) y en inferencia (2026) sin fuga temporal.
"""
import io
import re
import time

import pandas as pd
import requests

from src.config import RAW, PROCESSED, USER_AGENT
from src.ingest.names import canon

# saison_id de Transfermarkt -> año del Mundial
EDITIONS = {2013: 2014, 2017: 2018, 2021: 2022, 2025: 2026}
URL = "https://www.transfermarkt.com/weltmeisterschaft/teilnehmer/pokalwettbewerb/FIWC/saison_id/{sid}"


def parse_value(txt: str) -> float:
    """'€1.52bn' / '€58.58m' / '€950k' -> millones de EUR."""
    if not isinstance(txt, str):
        return float("nan")
    m = re.search(r"([\d.]+)\s*(bn|m|k)", txt.replace(",", "."), re.I)
    if not m:
        return float("nan")
    v = float(m.group(1))
    unit = m.group(2).lower()
    return v * 1000 if unit == "bn" else (v if unit == "m" else v / 1000)


def parse_participants(html: str, wc_year: int) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    # La tabla de participantes es la que tiene a todos los equipos (>=32 filas)
    t = max(tables, key=len)
    # Columnas desplazadas por la columna de escudos: Club=nombre,
    # Club.1=tamaño plantilla, Squad=edad media, Foreigners=valor total
    out = pd.DataFrame({
        "team": t["Club"].map(canon),
        "squad_size": pd.to_numeric(t["Club.1"], errors="coerce"),
        "avg_age": pd.to_numeric(t["Squad"], errors="coerce"),
        "market_value_meur": t["Foreigners"].map(parse_value),
    })
    out["wc_year"] = wc_year
    return out


def download(sleep_s: float = 3.0) -> pd.DataFrame:
    frames = []
    for sid, year in EDITIONS.items():
        cache = RAW / f"tm_participants_{sid}.html"
        if not cache.exists():
            r = requests.get(URL.format(sid=sid), timeout=60,
                             headers={"User-Agent": USER_AGENT,
                                      "Accept-Language": "en-US,en;q=0.9"})
            r.raise_for_status()
            cache.write_bytes(r.content)
            time.sleep(sleep_s)
        df = parse_participants(cache.read_text(encoding="utf-8", errors="replace"), year)
        print(f"[transfermarkt] WC{year}: {len(df)} equipos, "
              f"top={df.sort_values('market_value_meur').iloc[-1]['team']}")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(PROCESSED / "market_values.csv", index=False, encoding="utf-8")
    return all_df


if __name__ == "__main__":
    print(download())
