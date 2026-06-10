"""Fuente: Ranking FIFA oficial (API de inside.fifa.com).

Extrae los dateId disponibles de la página del ranking y descarga el
histórico de rankings (equivalente actualizado del dataset de Kaggle
"FIFA World Ranking"). Guarda data/raw/fifa_rankings.csv con
(date, rank, team, points).
"""
import re
import time

import pandas as pd
import requests

from src.config import RAW, USER_AGENT
from src.ingest.names import canon

PAGE = "https://inside.fifa.com/fifa-world-ranking/men"
API = "https://inside.fifa.com/api/ranking-overview?locale=en&dateId={did}"


def get_date_ids(session: requests.Session) -> list[str]:
    html = session.get(PAGE, timeout=60).text
    ids = sorted(set(re.findall(r"id\d{4,6}", html)), key=lambda s: int(s[2:]))
    return ids


def download(since_id: int = 12000, sleep_s: float = 0.6) -> pd.DataFrame:
    """since_id≈12000 cubre desde ~2017; sube/baja según necesidad."""
    out_path = RAW / "fifa_rankings.csv"
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    rows = []
    ids = [d for d in get_date_ids(s) if int(d[2:]) >= since_id]
    print(f"[fifa] {len(ids)} fechas de ranking a descargar")
    for i, did in enumerate(ids):
        try:
            data = s.get(API.format(did=did), timeout=30).json()
        except Exception as e:
            print(f"[fifa] {did} fallo: {e}")
            continue
        for item in data.get("rankings", []):
            ri = item.get("rankingItem", {})
            rows.append({
                "date_id": did,
                "date": item.get("lastUpdateDate", "")[:10],
                "rank": ri.get("rank"),
                "team": canon(ri.get("name", "")),
                "points": ri.get("totalPoints"),
            })
        if i % 20 == 0:
            print(f"[fifa] {i}/{len(ids)}")
        time.sleep(sleep_s)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[fifa] guardado {len(df):,} filas en {out_path}")
    return df


if __name__ == "__main__":
    download()
