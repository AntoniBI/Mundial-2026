"""Motor de simulación Monte Carlo de torneos.

Mejoras sobre el repo original:
  - El resultado de cada partido se MUESTREA de una Poisson(λ) (el repo
    simulaba minuto a minuto con multiplicadores arbitrarios que además
    quedaban sobreescritos por un bug, y se quedaba con la MODA de 30
    iteraciones, propagando un único cuadro determinista).
  - El torneo COMPLETO se simula N veces de principio a fin: la salida son
    probabilidades de alcanzar cada fase, no un único resultado.
  - λ precalculadas para todos los emparejamientos posibles -> simulación
    vectorizable y rápida.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, HALF_LIFE_DAYS
from src.features.build_features import (FEATURES, load_confed_map,
                                         tournament_weight, CONFED_NUM)
from src.ingest.elo_compute import (DEFAULT_ELO_K, ELO_HOME_ADV, ELO_K,
                                    START_ELO, goal_diff_multiplier)
from src.model.train import predict_lambda


# ---------------------------------------------------------------- snapshots
def elo_dict_at(cutoff: str) -> dict[str, float]:
    """Elo de todas las selecciones justo antes de `cutoff`."""
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    df = df[(df["date"] < cutoff) & df["home_score"].notna()].sort_values("date")
    ratings: dict[str, float] = {}
    for h, a, hs, as_, neu, t in zip(df["home_team"], df["away_team"],
                                     df["home_score"], df["away_score"],
                                     df["neutral"], df["tournament"]):
        rh = ratings.get(h, START_ELO)
        ra = ratings.get(a, START_ELO)
        rh_adj = rh + (0 if neu else ELO_HOME_ADV)
        we = 1.0 / (1.0 + 10 ** ((ra - rh_adj) / 400.0))
        w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = ELO_K.get(t, DEFAULT_ELO_K) * goal_diff_multiplier(int(hs - as_))
        delta = k * (w - we)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return ratings


def team_snapshots(teams: list[str], cutoff: str, wc_year: int,
                   n_last: int = 30) -> pd.DataFrame:
    """Foto fija pre-torneo: Elo en fecha, formas con decaimiento
    (referenciadas al cutoff) y valor de mercado de la edición."""
    long = pd.read_parquet(PROCESSED / "matches_long.parquet")
    long = long[(long["date"] < cutoff) & long["goals"].notna()]
    elo = elo_dict_at(cutoff)
    mv = pd.read_csv(PROCESSED / "market_values.csv", encoding="utf-8")
    mv = mv[mv["wc_year"] == wc_year].set_index("team")
    confed_map = load_confed_map()
    lam_decay = np.log(2) / HALF_LIFE_DAYS
    ref = pd.Timestamp(cutoff)

    rows = []
    for t in teams:
        h = long[long["team"] == t].sort_values("date").tail(n_last)
        if len(h) == 0:
            raise ValueError(f"Sin histórico para {t}")
        days = (ref - h["date"]).dt.days.to_numpy()
        w = np.exp(-lam_decay * days) * h["tournament"].map(tournament_weight).to_numpy()
        win = ((h["goals"] > h["goals_against"]).astype(float)
               + 0.5 * (h["goals"] == h["goals_against"])).to_numpy()
        rows.append({
            "team": t,
            "elo": elo.get(t, START_ELO),
            "form_gf": np.average(h["goals"], weights=w),
            "form_ga": np.average(h["goals_against"], weights=w),
            "form_xgf": np.average(h["xg_blend_for"], weights=w),
            "form_xga": np.average(h["xg_blend_against"], weights=w),
            "form_xg_cov": np.average(h["has_xg"], weights=w),
            "form_winrate": np.average(win, weights=w),
            "form_opp_elo": np.average(h["opp_elo"], weights=w),
            "n_prev": float(len(h)),
            "days_since_last": float((ref - h["date"].max()).days),
            "mv": mv["market_value_meur"].get(t, np.nan),
            "age": mv["avg_age"].get(t, np.nan),
            "confed": CONFED_NUM.get(confed_map.get(t, "OTHER"), 0),
        })
    return pd.DataFrame(rows).set_index("team")


# ------------------------------------------------------------- λ por pareja
def lambda_table(model, snaps: pd.DataFrame, hosts: set[str]) -> dict:
    """λ de goles para TODOS los pares ordenados (t1, t2).

    Los anfitriones juegan como locales (no neutral); el resto, neutral.
    Torneo = Mundial (peso 3.0).
    """
    teams = list(snaps.index)
    rows, keys = [], []
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            s1, s2 = snaps.loc[t1], snaps.loc[t2]
            home1 = t1 in hosts and t2 not in hosts
            home2 = t2 in hosts and t1 not in hosts
            rows.append({
                "elo": s1["elo"], "opp_elo": s2["elo"],
                "elo_diff": s1["elo"] - s2["elo"],
                "form_gf": s1["form_gf"], "form_ga": s1["form_ga"],
                "form_xgf": s1["form_xgf"], "form_xga": s1["form_xga"],
                "form_xg_cov": s1["form_xg_cov"],
                "days_since_last": s1["days_since_last"], "n_prev": s1["n_prev"],
                "form_winrate": s1["form_winrate"], "form_opp_elo": s1["form_opp_elo"],
                "opp_form_gf": s2["form_gf"], "opp_form_ga": s2["form_ga"],
                "opp_form_xgf": s2["form_xgf"], "opp_form_xga": s2["form_xga"],
                "opp_form_xg_cov": s2["form_xg_cov"],
                "opp_form_winrate": s2["form_winrate"], "opp_form_opp_elo": s2["form_opp_elo"],
                "is_home": int(home1), "is_away": int(home2),
                "neutral": int(not (home1 or home2)),
                "match_t_weight": 3.0,
                "confed": s1["confed"], "opp_confed": s2["confed"],
                "log_mv_ratio": np.log(s1["mv"] / s2["mv"]) if s1["mv"] > 0 and s2["mv"] > 0 else np.nan,
                "age_team": s1["age"], "age_opp": s2["age"],
            })
            keys.append((t1, t2))
    lam = predict_lambda(model, pd.DataFrame(rows))
    return {k: l for k, l in zip(keys, lam)}


def lambda_tables_wc26(model, snaps: pd.DataFrame) -> tuple[dict, dict]:
    """Tablas de λ para el Mundial 2026 con ventaja de anfitrión por fase.

    - `early` (grupos, R32 y octavos): EE.UU., México y Canadá juegan en su
      país. `late` (cuartos en adelante): todas las sedes son de EE.UU.
    - El efecto local se amortigua con HOST_ADVANTAGE_DAMP: en un Mundial
      no hay viajes largos del rival ni estadio 100% hostil, así que solo
      se aplica una fracción del efecto aprendido del histórico.
    """
    from src.config import (HOST_ADVANTAGE_DAMP, WC26_HOSTS_FROM_QF,
                            WC26_HOSTS_THROUGH_R16)
    d = HOST_ADVANTAGE_DAMP
    neutral = lambda_table(model, snaps, hosts=set())
    early_full = lambda_table(model, snaps, hosts=WC26_HOSTS_THROUGH_R16)
    late_full = lambda_table(model, snaps, hosts=WC26_HOSTS_FROM_QF)
    early = {k: d * early_full[k] + (1 - d) * neutral[k] for k in neutral}
    late = {k: d * late_full[k] + (1 - d) * neutral[k] for k in neutral}
    return early, late


# ------------------------------------------------------------- penaltis
def fit_penalty_model(cutoff: str) -> tuple[float, float]:
    """Logística P(gana la tanda el equipo 1) ~ diferencia de Elo, ajustada
    sobre las tandas históricas de shootouts.csv anteriores al corte."""
    from sklearn.linear_model import LogisticRegression

    shoot = pd.read_csv(RAW / "shootouts.csv", parse_dates=["date"])
    shoot = shoot[shoot["date"] < cutoff]
    long = pd.read_parquet(PROCESSED / "matches_long.parquet",
                           columns=["date", "team", "opponent", "elo", "opp_elo"])
    m = shoot.merge(long, left_on=["date", "home_team", "away_team"],
                    right_on=["date", "team", "opponent"], how="inner")
    if len(m) < 50:
        return 0.0, 0.0  # sin datos suficientes -> 50/50
    x = (m["elo"] - m["opp_elo"]).to_numpy().reshape(-1, 1)
    y = (m["winner"] == m["home_team"]).astype(int).to_numpy()
    lr = LogisticRegression(C=1.0).fit(x, y)
    b0, b1 = float(lr.intercept_[0]), float(lr.coef_[0][0])
    print(f"[penaltis] {len(m)} tandas < {cutoff}: "
          f"P(gana mejor Elo +200) = {1 / (1 + np.exp(-(b0 + b1 * 200))):.3f}")
    return b0, b1


def penalty_table(snaps: pd.DataFrame, pen_coefs: tuple[float, float]) -> dict:
    """P(t1 gana la tanda) para todos los pares ordenados."""
    b0, b1 = pen_coefs
    out = {}
    for t1 in snaps.index:
        for t2 in snaps.index:
            if t1 != t2:
                d = snaps.loc[t1, "elo"] - snaps.loc[t2, "elo"]
                out[(t1, t2)] = 1.0 / (1.0 + np.exp(-(b0 + b1 * d)))
    return out


# ---------------------------------------------------------------- simulación
_GRID_CACHE: dict = {}


def _sample_score(l1: float, l2: float, rho: float, rng) -> tuple[int, int]:
    """Muestrea (g1, g2) de la malla Poisson con corrección Dixon-Coles.

    Las mallas acumuladas se cachean por (λ1, λ2, ρ): en una simulación las
    parejas se repiten miles de veces, así que el coste es de una sola vez.
    """
    if rho == 0.0:
        return int(rng.poisson(l1)), int(rng.poisson(l2))
    key = (l1, l2, rho)
    cum = _GRID_CACHE.get(key)
    if cum is None:
        from src.model.train import score_grid
        cum = np.cumsum(score_grid(np.array([l1]), np.array([l2]), rho)[0].ravel())
        _GRID_CACHE[key] = cum
    flat = int(np.searchsorted(cum, rng.random()))
    n = int(np.sqrt(len(cum)))
    return flat // n, flat % n


def play_match(lam_tab, t1, t2, rng, ko=False, pen_tab=None, rho=0.0):
    """Devuelve (g1, g2, ganador) muestreando la malla Poisson+Dixon-Coles;
    prórroga λ/3 y, si es eliminatoria, penaltis según el modelo
    Elo-logístico (50/50 si no se pasa pen_tab)."""
    l1, l2 = lam_tab[(t1, t2)], lam_tab[(t2, t1)]
    g1, g2 = _sample_score(l1, l2, rho, rng)
    if not ko:
        return g1, g2, (t1 if g1 > g2 else t2 if g2 > g1 else None)
    if g1 == g2:
        e1, e2 = _sample_score(l1 / 3, l2 / 3, rho, rng)
        g1, g2 = g1 + e1, g2 + e2
        if g1 == g2:
            p1 = pen_tab[(t1, t2)] if pen_tab else 0.5
            return g1, g2, (t1 if rng.random() < p1 else t2)
    return g1, g2, (t1 if g1 > g2 else t2)


def group_standings(group_teams, lam_tab, rng, played=None, rho=0.0):
    """Simula los partidos del grupo y devuelve el orden final.

    `played`: resultados reales ya disputados, dict {(t1, t2): (g1, g2)}.
    Esos partidos se fijan y solo se simulan los restantes.

    Desempates FIFA: puntos, diferencia de goles, goles a favor,
    head-to-head entre empatados y, en última instancia, sorteo.
    """
    played = played or {}
    stats = {t: [0, 0, 0] for t in group_teams}  # pts, gd, gf
    h2h = {}
    for i, t1 in enumerate(group_teams):
        for t2 in group_teams[i + 1:]:
            if (t1, t2) in played:
                g1, g2 = played[(t1, t2)]
            elif (t2, t1) in played:
                g2, g1 = played[(t2, t1)]
            else:
                g1, g2, _ = play_match(lam_tab, t1, t2, rng, rho=rho)
            stats[t1][1] += g1 - g2; stats[t1][2] += g1
            stats[t2][1] += g2 - g1; stats[t2][2] += g2
            if g1 > g2:
                stats[t1][0] += 3; h2h[(t1, t2)] = t1
            elif g2 > g1:
                stats[t2][0] += 3; h2h[(t1, t2)] = t2
            else:
                stats[t1][0] += 1; stats[t2][0] += 1; h2h[(t1, t2)] = None
    order = sorted(group_teams,
                   key=lambda t: (stats[t][0], stats[t][1], stats[t][2], rng.random()),
                   reverse=False)
    order.reverse()  # de mejor a peor (sorted asc + reverse para ruido estable)
    # head-to-head para pares totalmente empatados
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        if stats[a] == stats[b]:
            winner = h2h.get((a, b), h2h.get((b, a)))
            if winner == b:
                order[i], order[i + 1] = b, a
    return order, stats
