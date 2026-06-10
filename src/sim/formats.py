"""Formatos de torneo: Mundial 2022 (32 equipos) y Mundial 2026 (48).

El cuadro de 2026 (R32 con mejores terceros) replica el del repo original,
validado contra el sorteo oficial; la asignación de terceros usa la tabla
oficial de 495 combinaciones (data/raw/mejores_terceros.csv).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import RAW
from src.sim.engine import group_standings, play_match

WC22_GROUPS = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

STAGES_32 = ["R16", "QF", "SF", "Final", "Champion"]
STAGES_48 = ["R32", "R16", "QF", "SF", "Final", "Third", "Champion"]


def simulate_wc22(lam_tab, n_sims: int, seed: int = 42, pen_tab=None, rho=0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = {t: dict.fromkeys(STAGES_32, 0)
              for teams in WC22_GROUPS.values() for t in teams}

    for _ in range(n_sims):
        firsts, seconds = {}, {}
        for letter, teams in WC22_GROUPS.items():
            order, _ = group_standings(teams, lam_tab, rng, rho=rho)
            firsts[letter], seconds[letter] = order[0], order[1]
        # Octavos (cuadro oficial 2022)
        r16 = [
            (firsts["A"], seconds["B"]), (firsts["C"], seconds["D"]),
            (firsts["D"], seconds["C"]), (firsts["B"], seconds["A"]),
            (firsts["E"], seconds["F"]), (firsts["G"], seconds["H"]),
            (firsts["F"], seconds["E"]), (firsts["H"], seconds["G"]),
        ]
        for t1, t2 in r16:
            counts[t1]["R16"] += 1
            counts[t2]["R16"] += 1
        w16 = [play_match(lam_tab, t1, t2, rng, ko=True, pen_tab=pen_tab, rho=rho)[2] for t1, t2 in r16]
        qf = [(w16[0], w16[1]), (w16[4], w16[5]), (w16[2], w16[3]), (w16[6], w16[7])]
        for t1, t2 in qf:
            counts[t1]["QF"] += 1
            counts[t2]["QF"] += 1
        wqf = [play_match(lam_tab, t1, t2, rng, ko=True, pen_tab=pen_tab, rho=rho)[2] for t1, t2 in qf]
        sf = [(wqf[0], wqf[1]), (wqf[2], wqf[3])]
        for t1, t2 in sf:
            counts[t1]["SF"] += 1
            counts[t2]["SF"] += 1
        wsf = [play_match(lam_tab, t1, t2, rng, ko=True, pen_tab=pen_tab, rho=rho)[2] for t1, t2 in sf]
        for t in wsf:
            counts[t]["Final"] += 1
        champ = play_match(lam_tab, wsf[0], wsf[1], rng, ko=True, pen_tab=pen_tab, rho=rho)[2]
        counts[champ]["Champion"] += 1

    return _to_table(counts, STAGES_32, n_sims)


# --------------------------------------------------------------- WC26 (48)
def load_thirds_map() -> dict[str, dict[str, str]]:
    df = pd.read_csv(RAW / "mejores_terceros.csv", sep=";", encoding="utf-8-sig")
    out = {}
    for _, row in df.iterrows():
        combo = row["Combinación"]
        out[combo] = {col: row[col][1] for col in df.columns[1:]}  # '3E' -> 'E'
    return out


def simulate_wc26(lam_tab, groups: dict[str, list[str]], n_sims: int,
                  seed: int = 42, played_group=None, ko_winners=None,
                  collect: dict | None = None, lam_tab_late=None, pen_tab=None, rho=0.0) -> pd.DataFrame:
    """Simula el torneo. Acepta el estado REAL ya disputado:

    - `played_group`: dict {(t1, t2): (g1, g2)} de partidos de grupo jugados.
    - `ko_winners`: dict {frozenset({t1, t2}): ganador} de eliminatorias
      ya jugadas (con penaltis ya resueltos vía shootouts.csv).
    - `collect`: dict opcional donde registrar detalle por simulación
      (posiciones de grupo, puntos y emparejamientos/ganadores por ronda)
      para el visor. Se rellena in-place.
    - `lam_tab_late`: tabla de λ para cuartos en adelante (sedes solo de
      EE.UU.); si no se pasa, se usa `lam_tab` para todo el torneo.
    """
    played_group = played_group or {}
    ko_winners = ko_winners or {}
    lam_late = lam_tab_late if lam_tab_late is not None else lam_tab
    rng = np.random.default_rng(seed)
    thirds_map = load_thirds_map()
    counts = {t: dict.fromkeys(STAGES_48, 0)
              for teams in groups.values() for t in teams}

    def ko(t1, t2, lam=None):
        real = ko_winners.get(frozenset((t1, t2)))
        if real is not None:
            return real
        return play_match(lam if lam is not None else lam_tab, t1, t2, rng,
                          ko=True, pen_tab=pen_tab, rho=rho)[2]

    group_of = {t: g for g, ts in groups.items() for t in ts}
    by_group = {g: {} for g in groups}
    for (t1, t2), score in played_group.items():
        by_group[group_of[t1]][(t1, t2)] = score

    if collect is not None:
        collect["group_pos"] = {g: {t: [0, 0, 0, 0] for t in ts}
                                for g, ts in groups.items()}
        collect["group_pts"] = {g: {t: 0.0 for t in ts} for g, ts in groups.items()}
        collect["ko"] = {rnd: [defaultdict(int) for _ in range(n)]
                         for rnd, n in (("R32", 16), ("R16", 8), ("QF", 4),
                                        ("SF", 2), ("Final", 1), ("Third", 1))}
        collect["n_sims"] = n_sims

    def record_ko(rnd, fixtures, winners):
        if collect is not None:
            for i, ((t1, t2), w) in enumerate(zip(fixtures, winners)):
                collect["ko"][rnd][i][(t1, t2, w)] += 1

    for _ in range(n_sims):
        pos = {}     # 'A1' -> team, 'A2' -> team
        thirds = {}  # letra -> (team, pts, gd, gf)
        for letter, teams in groups.items():
            order, stats = group_standings(teams, lam_tab, rng,
                                           played=by_group[letter], rho=rho)
            if collect is not None:
                for p, t in enumerate(order):
                    collect["group_pos"][letter][t][p] += 1
                    collect["group_pts"][letter][t] += stats[t][0]
            pos[f"{letter}1"], pos[f"{letter}2"] = order[0], order[1]
            t3 = order[2]
            thirds[letter] = (t3, *stats[t3])

        best8 = sorted(thirds.items(),
                       key=lambda kv: (kv[1][1], kv[1][2], kv[1][3], rng.random()),
                       reverse=True)[:8]
        combo = "".join(sorted(k for k, _ in best8))
        assign = thirds_map[combo]            # '1A' -> letra del grupo del tercero
        third_of = {letter: team for letter, (team, *_) in thirds.items()}
        rival = {slot[1]: third_of[g] for slot, g in assign.items()}  # 'A' -> tercero

        # Dieciseisavos (cuadro oficial, orden M1..M16 del repo validado)
        r32 = [
            (pos["A2"], pos["B2"]),
            (pos["C1"], pos["F2"]),
            (pos["E1"], rival["E"]),
            (pos["F1"], pos["C2"]),
            (pos["E2"], pos["I2"]),
            (pos["I1"], rival["I"]),
            (pos["A1"], rival["A"]),
            (pos["L1"], rival["L"]),
            (pos["G1"], rival["G"]),
            (pos["D1"], rival["D"]),
            (pos["H1"], pos["J2"]),
            (pos["K2"], pos["L2"]),
            (pos["B1"], rival["B"]),
            (pos["D2"], pos["G2"]),
            (pos["J1"], pos["H2"]),
            (pos["K1"], rival["K"]),
        ]
        for t1, t2 in r32:
            counts[t1]["R32"] += 1
            counts[t2]["R32"] += 1
        w32 = [ko(t1, t2) for t1, t2 in r32]
        record_ko("R32", r32, w32)
        # Octavos (índices M1..M16 -> 0..15)
        r16 = [(w32[0], w32[3]), (w32[2], w32[5]), (w32[1], w32[4]), (w32[6], w32[7]),
               (w32[10], w32[11]), (w32[9], w32[8]), (w32[14], w32[13]), (w32[12], w32[15])]
        for t1, t2 in r16:
            counts[t1]["R16"] += 1
            counts[t2]["R16"] += 1
        w16 = [ko(t1, t2) for t1, t2 in r16]
        record_ko("R16", r16, w16)
        qf = [(w16[0], w16[1]), (w16[4], w16[5]), (w16[2], w16[3]), (w16[6], w16[7])]
        for t1, t2 in qf:
            counts[t1]["QF"] += 1
            counts[t2]["QF"] += 1
        wqf = [ko(t1, t2, lam_late) for t1, t2 in qf]
        record_ko("QF", qf, wqf)
        sf = [(wqf[0], wqf[1]), (wqf[2], wqf[3])]
        for t1, t2 in sf:
            counts[t1]["SF"] += 1
            counts[t2]["SF"] += 1
        wsf = [ko(t1, t2, lam_late) for t1, t2 in sf]
        record_ko("SF", sf, wsf)
        for t in wsf:
            counts[t]["Final"] += 1
        champ = ko(wsf[0], wsf[1], lam_late)
        record_ko("Final", [(wsf[0], wsf[1])], [champ])
        counts[champ]["Champion"] += 1
        # partido por el 3er y 4o puesto (perdedores de semifinales)
        losers_sf = [t for (a, b), w in zip(sf, wsf) for t in (a, b) if t != w]
        third = ko(losers_sf[0], losers_sf[1], lam_late)
        record_ko("Third", [(losers_sf[0], losers_sf[1])], [third])
        counts[third]["Third"] += 1

    return _to_table(counts, STAGES_48, n_sims)


def _to_table(counts, stages, n_sims) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(counts, orient="index")[stages]
    df = (100.0 * df / n_sims).round(2)
    return df.sort_values(stages[-1], ascending=False).rename_axis("team")
