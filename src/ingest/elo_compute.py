"""Cálculo de rating Elo por partido siguiendo la metodología de eloratings.net.

A diferencia del repo original (Elo fijo hardcodeado por equipo), aquí el Elo
se recalcula partido a partido sobre todo el histórico 1872-2026, de modo que
cada fila del dataset lleva el Elo REAL de ambos equipos en la fecha del
partido. Incluye:
  - Factor K según importancia del torneo (Mundial 60 ... amistoso 20).
  - Multiplicador por diferencia de goles.
  - Ventaja de campo (+100) cuando no es terreno neutral.

También descarga el snapshot actual de eloratings.net (World.tsv) para
validar la correlación con nuestro Elo calculado.
"""
import numpy as np
import pandas as pd
import requests

from src.config import DEFAULT_ELO_K, ELO_HOME_ADV, ELO_K, RAW, PROCESSED, USER_AGENT

START_ELO = 1500.0


def goal_diff_multiplier(diff: int) -> float:
    diff = abs(diff)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return 1.75 + (diff - 3) / 8.0


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas home_elo_pre / away_elo_pre y devuelve ratings finales."""
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    home_pre = np.empty(len(df))
    away_pre = np.empty(len(df))

    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    neutral = df["neutral"].to_numpy()
    ks = df["tournament"].map(lambda t: ELO_K.get(t, DEFAULT_ELO_K)).to_numpy()

    for i in range(len(df)):
        rh = ratings.get(homes[i], START_ELO)
        ra = ratings.get(aways[i], START_ELO)
        home_pre[i] = rh
        away_pre[i] = ra

        if np.isnan(hs[i]) or np.isnan(as_[i]):
            continue  # partido futuro sin resultado

        rh_adj = rh + (0 if neutral[i] else ELO_HOME_ADV)
        we_home = 1.0 / (1.0 + 10 ** ((ra - rh_adj) / 400.0))
        w_home = 1.0 if hs[i] > as_[i] else (0.5 if hs[i] == as_[i] else 0.0)
        k = ks[i] * goal_diff_multiplier(int(hs[i] - as_[i]))
        delta = k * (w_home - we_home)
        ratings[homes[i]] = rh + delta
        ratings[aways[i]] = ra - delta

    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    pd.Series(ratings, name="elo").rename_axis("team").to_csv(PROCESSED / "elo_final.csv")
    return df


def download_eloratings_snapshot() -> None:
    """Snapshot actual de eloratings.net para validación."""
    try:
        r = requests.get("https://www.eloratings.net/World.tsv",
                         headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        (RAW / "eloratings_world.tsv").write_bytes(r.content)
        print(f"[eloratings.net] snapshot: {len(r.content):,} bytes")
    except Exception as e:  # validación opcional, no bloquea el pipeline
        print(f"[eloratings.net] AVISO: no disponible ({e})")
