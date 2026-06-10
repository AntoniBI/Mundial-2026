"""Auditoría profunda del cálculo de λ (goles esperados). Solo análisis.

Contrasta sobre las 4 ventanas de validación (WC18, EU/CA21, WC22, EU/CA24):
  1. Sesgo de λ por fase: grupos vs eliminatorias.
  2. Sobredispersión (¿Poisson o binomial negativa?).
  3. Feature is_ko (el modelo hoy no distingue grupos de KO dentro de un torneo).
  4. Ensemble XGBoost + GLM Poisson.
  5. Saturación del clip de λ [0.05, 6].

Uso: python -m src.model.eval_goals_deep
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import PoissonRegressor

from src.features.build_features import FEATURES, sample_weights
from src.model.backtest import evaluate
from src.model.eval_multiwindow import WINDOWS, window_matches
from src.model.train import PARAMS, load_long, outcome_probs

# Inicio de eliminatorias de cada ventana (entre jornada 3 y primera KO)
KO_START = {"WC2018": "2018-06-30", "EU/CA21": "2021-06-26",
            "WC2022": "2022-12-03", "EU/CA24": "2024-06-29"}

BIG = ["FIFA World Cup", "UEFA Euro", "Copa América",
       "African Cup of Nations", "AFC Asian Cup", "Gold Cup"]


def add_is_ko(df: pd.DataFrame) -> pd.DataFrame:
    """Aproxima la fase KO: 4º+ partido del equipo en esa edición de un
    torneo grande (las fases de grupos son de 3 partidos)."""
    df = df.copy()
    df["edition"] = df["tournament"] + "_" + df["date"].dt.year.astype(str)
    seq = df[df["tournament"].isin(BIG)].sort_values("date") \
            .groupby(["edition", "team"]).cumcount()
    df["is_ko"] = 0
    df.loc[seq.index, "is_ko"] = (seq >= 3).astype(int)
    return df


def fit_predict(df, start, feats, params=None):
    tr = df[(df["date"] < start) & df["goals"].notna() & (df["n_prev"] >= 5)]
    w = sample_weights(tr, start)
    mod = xgb.XGBRegressor(**(params or PARAMS))
    mod.fit(tr[feats], tr["goals"], sample_weight=w)
    return mod, tr, w


def main():
    df = add_is_ko(load_long())

    # ---------- 1+2+5: sesgo por fase, dispersión y clip (modelo actual)
    print("=== 1. Sesgo de λ por fase (modelo actual) ===")
    all_l, all_g, all_ko = [], [], []
    for wname, (start, end, tours) in WINDOWS.items():
        mod, _, _ = fit_predict(df, start, FEATURES)
        home, away = window_matches(df, start, end, tours)
        raw_l = np.concatenate([mod.predict(home[FEATURES]),
                                mod.predict(away[FEATURES])])
        l = np.clip(raw_l, .05, 6)
        g = np.concatenate([home["goals"].to_numpy(), away["goals"].to_numpy()])
        ko = np.concatenate([(home["date"] >= KO_START[wname]).to_numpy()] * 2)
        all_l.append(l); all_g.append(g); all_ko.append(ko)
        print(f"  {wname}: clip activo en {np.mean((raw_l <= .05) | (raw_l >= 6)) * 100:.1f}% de predicciones")
    l, g, ko = map(np.concatenate, (all_l, all_g, all_ko))
    for name, m in (("GRUPOS", ~ko), ("ELIMINATORIAS", ko)):
        bias = l[m].mean() - g[m].mean()
        print(f"  {name:14s} n={m.sum():4d}  λ medio={l[m].mean():.3f}  goles reales={g[m].mean():.3f}  sesgo={bias:+.3f}")
    disp = np.sum((g - l) ** 2 / np.clip(l, .05, None)) / len(g)
    print(f"\n=== 2. Dispersión de Pearson (1.0 = Poisson perfecta): {disp:.3f} ===")

    # ---------- 3: feature is_ko
    print("\n=== 3. ¿Añadir feature is_ko? (log-loss por ventana) ===")
    for label, feats in (("BASE", FEATURES), ("+ is_ko", FEATURES + ["is_ko"])):
        lls = {}
        for wname, (start, end, tours) in WINDOWS.items():
            mod, _, _ = fit_predict(df, start, feats)
            home, away = window_matches(df, start, end, tours)
            # en el eval, los partidos KO de la ventana llevan is_ko=1
            l1 = np.clip(mod.predict(home[feats]), .05, 6)
            l2 = np.clip(mod.predict(away[feats]), .05, 6)
            g1, g2 = home["goals"].to_numpy(), away["goals"].to_numpy()
            out = np.where(g1 > g2, 0, np.where(g1 == g2, 1, 2))
            r = evaluate(outcome_probs(l1, l2), out, label)
            lls[wname] = r["log_loss"]
        print(f"  {label:10s} " + "  ".join(f"{k}={v:.4f}" for k, v in lls.items())
              + f"  MEDIA={np.mean(list(lls.values())):.4f}")

    # ---------- 4: ensemble XGB + GLM Poisson
    print("\n=== 4. Ensemble XGBoost + GLM Poisson ===")
    glm_feats = ["elo", "opp_elo", "elo_diff", "form_gf", "form_ga", "form_xgf",
                 "form_xga", "opp_form_gf", "opp_form_ga", "opp_form_xgf",
                 "opp_form_xga", "is_home", "is_away", "match_t_weight"]
    for label, mix in (("XGB solo", 1.0), ("50/50 XGB+GLM", 0.5), ("70/30 XGB+GLM", 0.7)):
        lls = {}
        for wname, (start, end, tours) in WINDOWS.items():
            mod, tr, w = fit_predict(df, start, FEATURES)
            mu = tr[glm_feats].mean()
            glm = PoissonRegressor(alpha=1e-4, max_iter=300)
            glm.fit(tr[glm_feats].fillna(mu), tr["goals"], sample_weight=w)
            home, away = window_matches(df, start, end, tours)

            def lam(rows):
                lx = np.clip(mod.predict(rows[FEATURES]), .05, 6)
                lg = np.clip(glm.predict(rows[glm_feats].fillna(mu)), .05, 6)
                return mix * lx + (1 - mix) * lg

            g1, g2 = home["goals"].to_numpy(), away["goals"].to_numpy()
            out = np.where(g1 > g2, 0, np.where(g1 == g2, 1, 2))
            r = evaluate(outcome_probs(lam(home), lam(away)), out, label)
            lls[wname] = r["log_loss"]
        print(f"  {label:14s} " + "  ".join(f"{k}={v:.4f}" for k, v in lls.items())
              + f"  MEDIA={np.mean(list(lls.values())):.4f}")


if __name__ == "__main__":
    main()
