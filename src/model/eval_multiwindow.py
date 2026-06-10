"""Banco de validación multi-torneo (solo análisis, no toca el pipeline).

Evalúa configuraciones del modelo contra 4 torneos grandes dejados fuera,
cada uno con corte temporal estricto en su fecha de inicio. Un parámetro
solo merece cambiarse si mejora de forma CONSISTENTE en las 4 ventanas.

Uso: python -m src.model.eval_multiwindow
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.build_features import FEATURES
from src.model.backtest import evaluate
from src.model.train import PARAMS, load_long, outcome_probs

WINDOWS = {
    "WC2018": ("2018-06-14", "2018-07-16", ["FIFA World Cup"]),
    "EU/CA21": ("2021-06-11", "2021-07-12", ["UEFA Euro", "Copa América"]),
    "WC2022": ("2022-11-20", "2022-12-19", ["FIFA World Cup"]),
    "EU/CA24": ("2024-06-14", "2024-07-15", ["UEFA Euro", "Copa América"]),
}


def window_matches(df, start, end, tournaments):
    m = df[df["tournament"].isin(tournaments)
           & (df["date"] >= start) & (df["date"] <= end)
           & df["goals"].notna()].sort_values(["date", "match_id"])
    home = m.drop_duplicates("match_id", keep="first")
    away = m.drop_duplicates("match_id", keep="last")
    away = away.set_index("match_id").loc[home["match_id"]].reset_index()
    return home, away


def eval_config(df, label, half_life=1095, params_extra=None, friendly_w=None):
    rows = {}
    for wname, (start, end, tours) in WINDOWS.items():
        tr = df[(df["date"] < start) & df["goals"].notna() & (df["n_prev"] >= 5)]
        days = (pd.Timestamp(start) - tr["date"]).dt.days.clip(lower=0)
        tw = tr["match_t_weight"]
        if friendly_w is not None:
            tw = tw.where(tr["tournament"] != "Friendly", friendly_w)
        w = (0.5 ** (days / half_life)) * tw

        p = dict(PARAMS)
        p.update(params_extra or {})
        mod = xgb.XGBRegressor(**p)
        mod.fit(tr[FEATURES], tr["goals"], sample_weight=w)

        home, away = window_matches(df, start, end, tours)
        l1 = np.clip(mod.predict(home[FEATURES]), .05, 6)
        l2 = np.clip(mod.predict(away[FEATURES]), .05, 6)
        g1, g2 = home["goals"].to_numpy(), away["goals"].to_numpy()
        outcome = np.where(g1 > g2, 0, np.where(g1 == g2, 1, 2))
        r = evaluate(outcome_probs(l1, l2), outcome, label)
        rows[wname] = r["log_loss"]
    rows["MEDIA"] = float(np.mean(list(rows.values())))
    print(f"{label:34s} " + "  ".join(f"{k}={v:.4f}" for k, v in rows.items()))
    return rows


def main():
    df = load_long()
    n = {w: len(window_matches(df, *cfg)[0]) for w, cfg in WINDOWS.items()}
    print(f"Partidos por ventana: {n} (total {sum(n.values())})\n")

    eval_config(df, "BASE (actual)")
    eval_config(df, "max_depth=6", params_extra={"max_depth": 6})
    eval_config(df, "max_depth=5", params_extra={"max_depth": 5})
    eval_config(df, "half-life 730d", half_life=730)
    eval_config(df, "half-life 1460d", half_life=1460)
    eval_config(df, "half-life 2190d", half_life=2190)
    eval_config(df, "depth=6 + hl=1460", half_life=1460, params_extra={"max_depth": 6})
    eval_config(df, "lr=0.05, n=400", params_extra={"learning_rate": .05, "n_estimators": 400})
    eval_config(df, "n=1000", params_extra={"n_estimators": 1000})
    eval_config(df, "amistosos peso 0.5", friendly_w=0.5)
    eval_config(df, "min_child_weight=20", params_extra={"min_child_weight": 20})

    # Calibración de empates agregada (config base, 4 ventanas)
    print("\n--- Calibración de empates agregada (BASE) ---")
    draws_p, draws_r = [], []
    for wname, (start, end, tours) in WINDOWS.items():
        tr = df[(df["date"] < start) & df["goals"].notna() & (df["n_prev"] >= 5)]
        days = (pd.Timestamp(start) - tr["date"]).dt.days.clip(lower=0)
        w = (0.5 ** (days / 1095)) * tr["match_t_weight"]
        mod = xgb.XGBRegressor(**PARAMS)
        mod.fit(tr[FEATURES], tr["goals"], sample_weight=w)
        home, away = window_matches(df, start, end, tours)
        l1 = np.clip(mod.predict(home[FEATURES]), .05, 6)
        l2 = np.clip(mod.predict(away[FEATURES]), .05, 6)
        p = outcome_probs(l1, l2)
        draws_p.append(p[:, 1])
        draws_r.append((home["goals"].to_numpy() == away["goals"].to_numpy()))
    dp, dr = np.concatenate(draws_p), np.concatenate(draws_r)
    print(f"prob. empate media predicha: {dp.mean():.3f} | real: {dr.mean():.3f} "
          f"(n={len(dp)})")


if __name__ == "__main__":
    main()
