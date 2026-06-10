"""Modelo de goles esperados: XGBoost con objetivo Poisson.

Mejoras sobre el repo original:
  - Objetivo count:poisson (los goles son un conteo; el repo usaba Tweedie
    y luego NO usaba la distribución: convertía el λ en "prob. de gol por
    minuto" con multiplicadores arbitrarios).
  - Peso de muestra = decaimiento temporal × importancia del torneo,
    referenciado a la fecha de corte (parametrizable → backtest honesto).
  - Sin fuga temporal: nada posterior al corte entra en entrenamiento;
    los NaN los maneja XGBoost de forma nativa (no se imputa con medias
    calculadas sobre el futuro, como hacía el repo).
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import PROCESSED, RANDOM_SEED
from src.features.build_features import FEATURES, TARGET, sample_weights

PARAMS = dict(
    objective="count:poisson",
    n_estimators=600,
    learning_rate=0.03,
    # depth 5 + half-life de pesos de 4 años: única combinación que mejoró
    # el log-loss en las 4 ventanas del banco multi-torneo.
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


def load_long() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "matches_long.parquet")
    for col in ("is_home", "is_away", "neutral"):
        df[col] = df[col].astype(int)
    return df


def train(cutoff_date: str, min_prev_matches: int = 5) -> xgb.XGBRegressor:
    """Entrena SOLO con partidos anteriores a cutoff_date."""
    df = load_long()
    mask = (
        (df["date"] < cutoff_date)
        & df[TARGET].notna()
        & (df["n_prev"] >= min_prev_matches)
    )
    d = df[mask]
    w = sample_weights(d, cutoff_date)
    model = xgb.XGBRegressor(**PARAMS)
    # Nota: se probó corregir los marcadores de 120' (prórroga) vía
    # base_margin y vía target escalado; ambas empeoraron el holdout WC22,
    # así que el target queda como está (consistente con la evaluación).
    model.fit(d[FEATURES], d[TARGET], sample_weight=w)
    print(f"[train] {len(d):,} filas < {cutoff_date} "
          f"(peso medio={w.mean():.3f})")
    return model


def predict_lambda(model: xgb.XGBRegressor, rows: pd.DataFrame) -> np.ndarray:
    lam = model.predict(rows[FEATURES])
    return np.clip(lam, 0.05, 6.0)


MAX_GOALS = 12


def score_grid(lam1: np.ndarray, lam2: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """Malla conjunta P(g1, g2) Poisson con corrección Dixon-Coles.

    La Poisson independiente infra-predice los empates (23.8% predicho vs
    25.9% real en 290 partidos de 4 torneos grandes). Con ρ<0, Dixon-Coles
    sube la probabilidad de 0-0 y 1-1 y baja la de 0-1/1-0, corrigiendo la
    dependencia entre marcadores bajos. La malla se renormaliza.
    """
    from scipy.stats import poisson  # type: ignore
    lam1 = np.atleast_1d(lam1)[:, None]
    lam2 = np.atleast_1d(lam2)[:, None]
    g = np.arange(MAX_GOALS + 1)
    joint = poisson.pmf(g, lam1)[:, :, None] * poisson.pmf(g, lam2)[:, None, :]
    if rho:
        l1, l2 = lam1[:, 0], lam2[:, 0]
        joint[:, 0, 0] *= np.clip(1 - l1 * l2 * rho, 0, None)
        joint[:, 0, 1] *= np.clip(1 + l1 * rho, 0, None)
        joint[:, 1, 0] *= np.clip(1 + l2 * rho, 0, None)
        joint[:, 1, 1] *= np.clip(1 - rho, 0, None)
        joint /= joint.sum(axis=(1, 2), keepdims=True)
    return joint


def outcome_probs(lam1: np.ndarray, lam2: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """P(victoria, empate, derrota) del equipo 1 vía malla de Poisson(+DC)."""
    joint = score_grid(lam1, lam2, rho)
    win = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1)
    p_win = (joint * win[None]).sum(axis=(1, 2))
    p_draw = joint.diagonal(axis1=1, axis2=2).sum(axis=1)
    p_loss = 1.0 - p_win - p_draw
    return np.stack([p_win, p_draw, p_loss], axis=1)


def fit_rho(model, df: pd.DataFrame, cutoff_date: str,
            since_years: int = 12) -> float:
    """Ajusta ρ de Dixon-Coles por máxima verosimilitud ponderada sobre el
    propio entrenamiento (sin tocar nada posterior al corte)."""
    since = pd.Timestamp(cutoff_date) - pd.Timedelta(days=365 * since_years)
    m = df[(df["date"] < cutoff_date) & (df["date"] >= since)
           & df["goals"].notna() & (df["n_prev"] >= 5)]
    home = m.drop_duplicates("match_id", keep="first")
    away = m.drop_duplicates("match_id", keep="last")
    away = away.set_index("match_id").loc[home["match_id"]].reset_index()
    l1 = predict_lambda(model, home)
    l2 = predict_lambda(model, away)
    g1 = home["goals"].clip(upper=MAX_GOALS).astype(int).to_numpy()
    g2 = away["goals"].clip(upper=MAX_GOALS).astype(int).to_numpy()
    w = sample_weights(home, cutoff_date).to_numpy()

    best_rho, best_ll = 0.0, -np.inf
    idx = np.arange(len(g1))
    for rho in np.linspace(-0.20, 0.05, 26):
        joint = score_grid(l1, l2, float(rho))
        ll = float(np.sum(w * np.log(np.clip(joint[idx, g1, g2], 1e-12, None))))
        if ll > best_ll:
            best_rho, best_ll = float(rho), ll
    print(f"[dixon-coles] rho={best_rho:+.2f} "
          f"(ajustado sobre {len(g1):,} partidos < {cutoff_date})")
    return best_rho
