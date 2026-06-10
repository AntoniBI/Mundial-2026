"""Fuente: Wikipedia — tablas del torneo.

Extrae de la página del Mundial 2026 la composición oficial de los 12
grupos (A-L) para validar/derivar el cuadro, y de la página histórica la
lista de campeones (referencia para sanity-check de la simulación).
"""
import io

import pandas as pd
import requests

from src.config import RAW, PROCESSED, USER_AGENT
from src.ingest.names import canon

WC26 = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"


def download_groups() -> pd.DataFrame:
    r = requests.get(WC26, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    (RAW / "wikipedia_wc26.html").write_bytes(r.content)
    tables = pd.read_html(io.StringIO(r.text))
    rows = []
    for t in tables:
        cols = [str(c) for c in t.columns]
        # Las tablas de clasificación de grupo tienen columnas Pos/Team/Pld
        if any("Team" in c for c in cols) and any(c.startswith("Pld") for c in cols) and len(t) == 4:
            team_col = [c for c in t.columns if "Team" in str(c)][0]
            for pos, raw in enumerate(t[team_col], start=1):
                name = str(raw).split("(")[0].strip()
                rows.append({"group_table_idx": len(rows) // 4, "pos": pos, "team": canon(name)})
    df = pd.DataFrame(rows)
    if not df.empty:
        letters = "ABCDEFGHIJKL"
        df["group"] = df["group_table_idx"].map(lambda i: letters[i] if i < 12 else "?")
        df = df[["group", "team"]]
        df.to_csv(PROCESSED / "wc26_groups_wikipedia.csv", index=False, encoding="utf-8")
        print(f"[wikipedia] {len(df)} equipos en {df['group'].nunique()} grupos")
    else:
        print("[wikipedia] AVISO: no se encontraron tablas de grupos (formato cambiado)")
    return df


if __name__ == "__main__":
    print(download_groups())
