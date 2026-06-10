"""Feature engineering.

Mejoras sobre el repo original:
  - Elo calculado EN LA FECHA de cada partido (no hardcodeado).
  - Formas recientes con DECAIMIENTO EXPONENCIAL (half-life 3 años) y
    ponderadas por la importancia del torneo, en lugar de medias móviles
    planas de 5/15 partidos.
  - Contexto real local/visitante/neutral (el repo solo marcaba "host").
  - Valor de mercado de la plantilla (Transfermarkt) por edición de
    Mundial, sin fuga temporal (cada Mundial usa el valor de SU año).
  - Columnas de peso de muestra (decay × torneo) calculadas respecto a
    una fecha de referencia parametrizable (para poder hacer backtest).
"""
import numpy as np
import pandas as pd

from src.config import (DEFAULT_TOURNAMENT_WEIGHT, HALF_LIFE_DAYS, PROCESSED,
                        RAW, TOURNAMENT_WEIGHTS)
from src.ingest.elo_compute import compute_elo

CONFED = {
    "CONMEBOL": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
                 "Paraguay", "Peru", "Uruguay", "Venezuela"],
}
CONFED_NUM = {"UEFA": 1, "CONMEBOL": 2, "CONCACAF": 3, "CAF": 4, "AFC": 5, "OFC": 6, "OTHER": 0}


def tournament_weight(t: str) -> float:
    return TOURNAMENT_WEIGHTS.get(t, DEFAULT_TOURNAMENT_WEIGHT)


def load_confed_map() -> dict:
    """Mapa equipo->confederación reutilizando el diccionario del repo original."""
    import importlib.util, sys
    from pathlib import Path
    # Diccionario embebido (copiado del repo de muestra, fuente: FIFA)
    d = {
        'CONMEBOL': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela'],
        'CONCACAF': ['United States', 'Mexico', 'Canada', 'Costa Rica', 'Jamaica', 'Panama', 'Honduras', 'El Salvador', 'Haiti', 'Trinidad and Tobago', 'Guatemala', 'Cuba', 'Curaçao', 'Martinique', 'Guadeloupe', 'Suriname'],
        'UEFA': ['Spain', 'France', 'Germany', 'England', 'Italy', 'Netherlands', 'Portugal', 'Belgium', 'Croatia', 'Denmark', 'Switzerland', 'Serbia', 'Poland', 'Sweden', 'Wales', 'Scotland', 'Czech Republic', 'Austria', 'Hungary', 'Ukraine', 'Turkey', 'Russia', 'Norway', 'Republic of Ireland', 'Northern Ireland', 'Slovakia', 'Romania', 'Greece', 'Bosnia and Herzegovina', 'Finland', 'Iceland', 'Albania', 'Slovenia', 'Montenegro', 'North Macedonia', 'Georgia', 'Israel', 'Bulgaria', 'Armenia', 'Luxembourg', 'Cyprus', 'Kosovo', 'Estonia', 'Lithuania', 'Latvia', 'Moldova', 'Belarus', 'Malta', 'Liechtenstein', 'Andorra', 'San Marino', 'Gibraltar', 'Kazakhstan', 'Faroe Islands', 'Azerbaijan'],
        'CAF': ['Senegal', 'Morocco', 'Nigeria', 'Egypt', 'Ivory Coast', 'Cameroon', 'Ghana', 'Algeria', 'Mali', 'Tunisia', 'Burkina Faso', 'South Africa', 'DR Congo', 'Guinea', 'Cape Verde', 'Zambia', 'Gabon', 'Uganda', 'Equatorial Guinea', 'Gambia', 'Angola', 'Mauritania', 'Namibia', 'Benin', 'Mozambique', 'Togo', 'Tanzania', 'Zimbabwe', 'Malawi', 'Kenya', 'Congo', 'Rwanda', 'Madagascar', 'Central African Republic', 'Sudan', 'Sierra Leone'],
        'AFC': ['Japan', 'Iran', 'South Korea', 'Australia', 'Saudi Arabia', 'Qatar', 'Iraq', 'United Arab Emirates', 'Oman', 'Uzbekistan', 'China', 'Jordan', 'Bahrain', 'Syria', 'Vietnam', 'Palestine', 'Kyrgyzstan', 'India', 'Lebanon', 'Tajikistan', 'Thailand', 'North Korea', 'Philippines', 'Malaysia', 'Kuwait', 'Turkmenistan', 'Hong Kong', 'Indonesia', 'Yemen', 'Afghanistan', 'Singapore', 'Myanmar', 'Maldives', 'Nepal', 'Cambodia'],
        'OFC': ['New Zealand', 'Solomon Islands', 'Fiji', 'New Caledonia', 'Tahiti', 'Vanuatu', 'Papua New Guinea', 'Samoa', 'Tonga', 'Cook Islands'],
    }
    return {team: conf for conf, teams in d.items() for team in teams}


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (partido, equipo): perspectiva propia + del rival."""
    base = {
        "date": df["date"], "tournament": df["tournament"],
        "neutral": df["neutral"].astype(bool), "country": df["country"],
        "match_id": df.index,
    }
    home = pd.DataFrame({**base,
        "team": df["home_team"], "opponent": df["away_team"],
        "goals": df["home_score"], "goals_against": df["away_score"],
        "elo": df["home_elo_pre"], "opp_elo": df["away_elo_pre"],
        "is_home": ~df["neutral"].astype(bool),
        "is_away": False,
    })
    away = pd.DataFrame({**base,
        "team": df["away_team"], "opponent": df["home_team"],
        "goals": df["away_score"], "goals_against": df["home_score"],
        "elo": df["away_elo_pre"], "opp_elo": df["home_elo_pre"],
        "is_home": False,
        "is_away": ~df["neutral"].astype(bool),
    })
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["date", "match_id"]).reset_index(drop=True)
    return long


def add_xg(long: pd.DataFrame) -> pd.DataFrame:
    """Cruza el xG de FotMob (si existe) con las filas equipo-partido.

    El join es por (fecha, equipo, rival) con tolerancia de ±1 día (la fecha
    UTC de FotMob puede diferir de la fecha local del dataset). Las columnas
    xg_blend_for/against usan el xG cuando hay cobertura y el marcador real
    como fallback: una única serie coherente con todo el histórico.
    """
    long = long.copy()
    xg_path = PROCESSED / "fotmob_xg.csv"
    long["xg_for"] = np.nan
    long["xg_against"] = np.nan
    if xg_path.exists():
        xg = pd.read_csv(xg_path, parse_dates=["date"])
        lut = {}
        for _, r in xg.iterrows():
            for off in (0, 1, -1):
                d = (r["date"] + pd.Timedelta(days=off)).strftime("%Y-%m-%d")
                lut.setdefault((d, r["team_home"], r["team_away"]),
                               (r["xg_home"], r["xg_away"]))
                lut.setdefault((d, r["team_away"], r["team_home"]),
                               (r["xg_away"], r["xg_home"]))
        keys = list(zip(long["date"].dt.strftime("%Y-%m-%d"),
                        long["team"], long["opponent"]))
        vals = [lut.get(k, (np.nan, np.nan)) for k in keys]
        long["xg_for"] = [v[0] for v in vals]
        long["xg_against"] = [v[1] for v in vals]
        n = long["xg_for"].notna().sum()
        print(f"[features] xG FotMob cruzado en {n:,} filas "
              f"({100 * n / len(long):.1f}%)")
    else:
        print("[features] AVISO: sin fotmob_xg.csv; formas xG = goles")
    long["xg_blend_for"] = long["xg_for"].fillna(long["goals"])
    long["xg_blend_against"] = long["xg_against"].fillna(long["goals_against"])
    long["has_xg"] = long["xg_for"].notna().astype(float)
    return long


def add_minutes(long: pd.DataFrame) -> pd.DataFrame:
    """Duración real del partido para el ajuste de exposición Poisson.

    Los marcadores del dataset incluyen la prórroga. Los partidos que
    acabaron en penaltis (shootouts.csv) duraron seguro 120'; el resto se
    asume 90' (las prórrogas decididas sin penaltis no son identificables,
    sesgo residual pequeño). El modelo entrena con offset log(minutos/90)
    y por tanto predice λ por 90 minutos.
    """
    shoot = pd.read_csv(RAW / "shootouts.csv", parse_dates=["date"])
    keys = set(zip(shoot["date"].dt.strftime("%Y-%m-%d"),
                   shoot["home_team"], shoot["away_team"]))
    long = long.copy()
    d = long["date"].dt.strftime("%Y-%m-%d")
    is_pen = [
        (dd, t, o) in keys or (dd, o, t) in keys
        for dd, t, o in zip(d, long["team"], long["opponent"])
    ]
    long["minutes"] = np.where(is_pen, 120.0, 90.0)
    print(f"[features] {int(sum(is_pen)) // 2} partidos con prórroga+penaltis "
          f"normalizados a 90' (exposure)")
    return long


def decayed_form(long: pd.DataFrame, n_last: int = 30) -> pd.DataFrame:
    """Formas previas al partido con decaimiento temporal y peso de torneo.

    Para cada (equipo, partido) calcula sobre sus últimos `n_last` partidos
    ANTERIORES: media ponderada de goles/xG a favor y en contra, % victorias
    y Elo medio de los rivales. Peso = 0.5^(días/half-life) × peso_torneo.
    """
    lam = np.log(2) / HALF_LIFE_DAYS
    long = long.copy()
    long["t_weight"] = long["tournament"].map(tournament_weight)

    out_cols = ["form_gf", "form_ga", "form_xgf", "form_xga", "form_xg_cov",
                "form_winrate", "form_opp_elo", "n_prev", "days_since_last"]
    results = {c: np.full(len(long), np.nan) for c in out_cols}

    for _, idx in long.groupby("team", sort=False).indices.items():
        idx = np.sort(idx)
        dates = long["date"].to_numpy()[idx].astype("datetime64[D]").astype(float)
        gf = long["goals"].to_numpy()[idx]
        ga = long["goals_against"].to_numpy()[idx]
        xgf = long["xg_blend_for"].to_numpy()[idx]
        xga = long["xg_blend_against"].to_numpy()[idx]
        hxg = long["has_xg"].to_numpy()[idx]
        oe = long["opp_elo"].to_numpy()[idx]
        tw = long["t_weight"].to_numpy()[idx]
        win = (gf > ga).astype(float) + 0.5 * (gf == ga)

        for j in range(len(idx)):
            lo = max(0, j - n_last)
            sl = slice(lo, j)
            g = gf[sl]
            valid = ~np.isnan(g)
            if valid.sum() == 0:
                continue
            w = np.exp(-lam * (dates[j] - dates[sl])) * tw[sl]
            w = np.where(valid, w, 0.0)
            sw = w.sum()
            if sw <= 0:
                continue
            i0 = idx[j]
            results["form_gf"][i0] = np.nansum(w * g) / sw
            results["form_ga"][i0] = np.nansum(w * ga[sl]) / sw
            results["form_xgf"][i0] = np.nansum(w * xgf[sl]) / sw
            results["form_xga"][i0] = np.nansum(w * xga[sl]) / sw
            results["form_xg_cov"][i0] = np.nansum(w * hxg[sl]) / sw
            results["form_winrate"][i0] = np.nansum(w * win[sl]) / sw
            results["form_opp_elo"][i0] = np.nansum(w * oe[sl]) / sw
            results["n_prev"][i0] = valid.sum()
            results["days_since_last"][i0] = dates[j] - dates[sl][valid][-1] if valid.any() else np.nan

    for c, v in results.items():
        long[c] = v
    return long


def add_market_values(long: pd.DataFrame) -> pd.DataFrame:
    """Valor de mercado Transfermarkt: solo para partidos de Mundial,
    con el valor de la edición correspondiente (sin fuga temporal)."""
    mv = pd.read_csv(PROCESSED / "market_values.csv", encoding="utf-8")
    long = long.copy()
    long["wc_year"] = np.where(long["tournament"] == "FIFA World Cup",
                               long["date"].dt.year, -1)
    # Mundiales con datos: 2014, 2018, 2022, 2026
    mv_team = mv.rename(columns={"team": "team", "market_value_meur": "mv_team",
                                 "avg_age": "age_team"})[["team", "wc_year", "mv_team", "age_team"]]
    mv_opp = mv.rename(columns={"team": "opponent", "market_value_meur": "mv_opp",
                                "avg_age": "age_opp"})[["opponent", "wc_year", "mv_opp", "age_opp"]]
    long = long.merge(mv_team, on=["team", "wc_year"], how="left")
    long = long.merge(mv_opp, on=["opponent", "wc_year"], how="left")
    long["log_mv_ratio"] = np.log(long["mv_team"] / long["mv_opp"])
    return long


def add_opponent_form(long: pd.DataFrame) -> pd.DataFrame:
    """Cruza las formas del rival en el mismo partido."""
    opp = long[["match_id", "team", "form_gf", "form_ga", "form_xgf", "form_xga",
                "form_xg_cov", "form_winrate", "form_opp_elo", "n_prev",
                "days_since_last"]].copy()
    opp.columns = ["match_id", "opponent", "opp_form_gf", "opp_form_ga",
                   "opp_form_xgf", "opp_form_xga", "opp_form_xg_cov",
                   "opp_form_winrate", "opp_form_opp_elo", "opp_n_prev",
                   "opp_days_since_last"]
    return long.merge(opp, on=["match_id", "opponent"], how="left")


def build(min_date: str = "1990-01-01") -> pd.DataFrame:
    """Construye el dataset largo de features y lo guarda en processed/."""
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    df = compute_elo(df)  # Elo en fecha sobre TODO el histórico (desde 1872)

    long = to_long(df)
    long = add_xg(long)
    long = add_minutes(long)
    long = decayed_form(long)
    long = add_opponent_form(long)
    long = add_market_values(long)

    confed_map = load_confed_map()
    long["confed"] = long["team"].map(confed_map).fillna("OTHER").map(CONFED_NUM)
    long["opp_confed"] = long["opponent"].map(confed_map).fillna("OTHER").map(CONFED_NUM)

    long["elo_diff"] = long["elo"] - long["opp_elo"]
    long["match_t_weight"] = long["tournament"].map(tournament_weight)

    # Recorta el histórico muy antiguo SOLO para entrenar (el Elo ya absorbió
    # todo el pasado); los partidos previos a min_date no aportan al modelo.
    long = long[long["date"] >= min_date].reset_index(drop=True)

    long.to_parquet(PROCESSED / "matches_long.parquet", index=False)
    print(f"[features] {len(long):,} filas equipo-partido desde {min_date} "
          f"-> matches_long.parquet")
    return long


# Cambios validados contra el holdout WC22 (ver README):
#  + days_since_last / n_prev: descanso y fiabilidad de la forma
#    (logloss 1.0319 -> 1.0292).
#  Probados y DESCARTADOS por empeorar la validación: form_xg_cov (1.0414),
#  quitar confederaciones en combinación con lo anterior (1.0320), y la
#  corrección de prórrogas vía base_margin/target escalado.
FEATURES = [
    "elo", "opp_elo", "elo_diff",
    "form_gf", "form_ga", "form_xgf", "form_xga", "form_winrate", "form_opp_elo",
    "opp_form_gf", "opp_form_ga", "opp_form_xgf", "opp_form_xga",
    "opp_form_winrate", "opp_form_opp_elo",
    "is_home", "is_away", "neutral", "match_t_weight",
    "confed", "opp_confed",
    "log_mv_ratio", "age_team", "age_opp",
    "days_since_last", "n_prev",
]
TARGET = "goals"


def sample_weights(d: pd.DataFrame, reference_date: str) -> pd.Series:
    """Peso de muestra = decaimiento temporal × importancia del torneo."""
    from src.config import WEIGHT_HALF_LIFE_DAYS
    ref = pd.Timestamp(reference_date)
    days = (ref - d["date"]).dt.days.clip(lower=0)
    decay = 0.5 ** (days / WEIGHT_HALF_LIFE_DAYS)
    return decay * d["match_t_weight"]


if __name__ == "__main__":
    build()
