"""Backtest: Mundial 2022 como holdout aislado.

Protocolo:
  1. Se entrena el modelo SOLO con partidos anteriores al 2022-11-20
     (primer día del Mundial 2022). El decaimiento temporal se referencia
     a esa fecha, como si estuviéramos en la víspera del torneo.
  2. Métricas partido a partido sobre los 64 partidos reales del Mundial:
     log-loss, Brier y RPS sobre 1X2, y MAE de goles. Se compara contra
     un baseline Poisson solo-Elo y contra el azar uniforme.
  3. Calibración: probabilidad predicha vs frecuencia observada por bins.
  4. Backtest a nivel TORNEO: se simula el Mundial 2022 completo 10.000
     veces y se comparan las probabilidades con lo que de verdad pasó
     (Argentina campeona, Francia finalista, Marruecos y Croacia en SF).

Uso: python -m src.model.backtest
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from src.config import N_SIMULATIONS, OUTPUTS, WC22_END, WC22_START
from src.features.build_features import FEATURES, sample_weights
from src.model.train import (fit_rho, load_long, outcome_probs,
                             predict_lambda, train)
from src.sim.engine import (fit_penalty_model, lambda_table, penalty_table,
                            team_snapshots)
from src.sim.formats import WC22_GROUPS, simulate_wc22


def wc22_matches(df: pd.DataFrame) -> pd.DataFrame:
    m = df[(df["tournament"] == "FIFA World Cup")
           & (df["date"] >= WC22_START) & (df["date"] <= WC22_END)
           & df["goals"].notna()]
    return m.sort_values(["date", "match_id"])


def rps(probs: np.ndarray, outcome_idx: np.ndarray) -> float:
    """Ranked Probability Score (ordenado W>D>L), menor = mejor."""
    n = len(probs)
    y = np.zeros_like(probs)
    y[np.arange(n), outcome_idx] = 1.0
    cp, cy = np.cumsum(probs, axis=1), np.cumsum(y, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / (probs.shape[1] - 1)))


def evaluate(probs: np.ndarray, outcome_idx: np.ndarray, label: str) -> dict:
    n = len(outcome_idx)
    p_true = np.clip(probs[np.arange(n), outcome_idx], 1e-12, 1)
    y = np.zeros_like(probs)
    y[np.arange(n), outcome_idx] = 1.0
    res = {
        "modelo": label,
        "log_loss": float(-np.mean(np.log(p_true))),
        "brier": float(np.mean(np.sum((probs - y) ** 2, axis=1))),
        "rps": rps(probs, outcome_idx),
        "acierto_%": float(100 * np.mean(probs.argmax(1) == outcome_idx)),
    }
    return res


def calibration_table(probs: np.ndarray, outcome_idx: np.ndarray, bins=5) -> pd.DataFrame:
    """Calibración sobre la prob. de victoria del equipo 1."""
    p_win = probs[:, 0]
    won = (outcome_idx == 0).astype(float)
    edges = np.quantile(p_win, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        m = (p_win >= edges[i]) & (p_win <= edges[i + 1] if i == bins - 1 else p_win < edges[i + 1])
        if m.sum():
            rows.append({"bin": f"{edges[i]:.2f}-{edges[i+1]:.2f}",
                         "n": int(m.sum()),
                         "prob_media": round(float(p_win[m].mean()), 3),
                         "freq_real": round(float(won[m].mean()), 3)})
    return pd.DataFrame(rows)


def main():
    df = load_long()
    print(f"=== BACKTEST: corte temporal estricto en {WC22_START} ===")
    model = train(WC22_START)

    # ---- 1. Partido a partido (64 partidos del Mundial 2022)
    matches = wc22_matches(df)
    home_rows = matches.drop_duplicates("match_id", keep="first")
    away_rows = matches.drop_duplicates("match_id", keep="last")
    away_rows = away_rows.set_index("match_id").loc[home_rows["match_id"]].reset_index()
    print(f"Partidos de holdout: {len(home_rows)}")

    rho = fit_rho(model, df, WC22_START)
    lam1 = predict_lambda(model, home_rows)
    lam2 = predict_lambda(model, away_rows)
    probs = outcome_probs(lam1, lam2, rho=rho)

    g1 = home_rows["goals"].to_numpy()
    g2 = away_rows["goals"].to_numpy()
    outcome = np.where(g1 > g2, 0, np.where(g1 == g2, 1, 2))

    # Baseline 1: Poisson solo con Elo (mismo corte, mismos pesos)
    tr = df[(df["date"] < WC22_START) & df["goals"].notna() & (df["n_prev"] >= 5)]
    base_feats = ["elo_diff", "is_home", "is_away"]
    base = PoissonRegressor(alpha=1e-6, max_iter=300)
    base.fit(tr[base_feats], tr["goals"], sample_weight=sample_weights(tr, WC22_START))
    bl1 = np.clip(base.predict(home_rows[base_feats]), 0.05, 6)
    bl2 = np.clip(base.predict(away_rows[base_feats]), 0.05, 6)
    probs_base = outcome_probs(bl1, bl2)
    probs_unif = np.full_like(probs, 1 / 3)

    res = pd.DataFrame([
        evaluate(probs, outcome, "XGBoost Poisson (completo)"),
        evaluate(probs_base, outcome, "Baseline Poisson solo-Elo"),
        evaluate(probs_unif, outcome, "Uniforme (azar)"),
    ])
    mae = float(np.mean(np.abs(np.concatenate([lam1 - g1, lam2 - g2]))))
    print("\n--- Métricas 1X2 sobre Mundial 2022 (menor = mejor) ---")
    print(res.to_string(index=False))
    print(f"MAE de goles del modelo: {mae:.3f}")
    res.to_csv(OUTPUTS / "backtest_wc22_metrics.csv", index=False)

    print("\n--- Calibración (prob. victoria equipo 1) ---")
    cal = calibration_table(probs, outcome)
    print(cal.to_string(index=False))
    cal.to_csv(OUTPUTS / "backtest_wc22_calibration.csv", index=False)

    # ---- 2. Torneo completo: simulación Monte Carlo del Mundial 2022
    print(f"\n--- Simulación del torneo ({N_SIMULATIONS:,} iteraciones) ---")
    teams = [t for g in WC22_GROUPS.values() for t in g]
    snaps = team_snapshots(teams, WC22_START, wc_year=2022)
    lam_tab = lambda_table(model, snaps, hosts={"Qatar"})
    pen_tab = penalty_table(snaps, fit_penalty_model(WC22_START))
    table = simulate_wc22(lam_tab, N_SIMULATIONS, pen_tab=pen_tab, rho=rho)
    print(table.head(12).to_string())
    table.to_csv(OUTPUTS / "backtest_wc22_tournament_probs.csv", encoding="utf-8")

    real = {"Champion": "Argentina", "Final": "France", "SF": "Morocco/Croatia"}
    print(f"\nRealidad: campeón {real['Champion']} "
          f"(prob. simulada: {table.loc['Argentina', 'Champion']}%), "
          f"finalista {real['Final']} ({table.loc['France', 'Final']}%), "
          f"semifinalistas Marruecos ({table.loc['Morocco', 'SF']}%) "
          f"y Croacia ({table.loc['Croatia', 'SF']}%)")


if __name__ == "__main__":
    main()
