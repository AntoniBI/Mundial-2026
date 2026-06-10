"""Fuente: dataset Kaggle "International football results from 1872" (Mart Jürisoo).

Se descarga desde su repositorio GitHub original (la fuente del dataset de
Kaggle), que está más actualizado e incluye el calendario del Mundial 2026
con marcadores NA.
"""
import io

import pandas as pd
import requests

from src.config import RAW, USER_AGENT

BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "shootouts.csv", "former_names.csv"]


def download() -> pd.DataFrame:
    for fname in FILES:
        out = RAW / fname
        r = requests.get(f"{BASE}/{fname}", headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"[martj42] {fname}: {len(r.content):,} bytes")
    return load()


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    return df


if __name__ == "__main__":
    df = download()
    print(df.tail())
