"""Genera el visor HTML interactivo: outputs/wc26_viewer.html.

Muestra, según 10.000 simulaciones Monte Carlo:
  - Cada partido de grupos con % victoria/empate/derrota y marcador más probable.
  - La clasificación simulada de cada grupo (% 1º/2º/3º, puntos esperados,
    % de clasificarse a R32).
  - El cuadro completo R32 -> Final: emparejamiento más probable de cada
    cruce y % de victoria condicionado a ese cruce.
  - El camino al título: % de cada selección de alcanzar cada fase.

Tiene en cuenta el estado real del torneo (partidos ya jugados quedan
fijados), igual que run_update. Uso:

    python -m src.run_viewer [--sims 10000]
"""
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import (N_SIMULATIONS, OUTPUTS, PROCESSED, RAW, ROOT,
                        WC26_HOSTS_FROM_QF, WC26_HOSTS_THROUGH_R16, WC26_START)
from src.model.train import fit_rho, load_long, outcome_probs, score_grid, train
from src.run_predict import load_groups
from src.run_update import load_tournament_state
from src.sim.engine import (fit_penalty_model, lambda_tables_wc26,
                            penalty_table, team_snapshots)
from src.sim.formats import simulate_wc26

R32_LABELS = ["A2-B2", "C1-F2", "E1-3º", "F1-C2", "E2-I2", "I1-3º", "A1-3º",
              "L1-3º", "G1-3º", "D1-3º", "H1-J2", "K2-L2", "B1-3º", "D2-G2",
              "J1-H2", "K1-3º"]

# Nombres para mostrar en la UI (las claves internas siguen en inglés)
NAMES_ES = {
    "Mexico": "México", "South Africa": "Sudáfrica", "South Korea": "Corea del Sur",
    "Czech Republic": "Chequia", "Canada": "Canadá",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina", "Qatar": "Catar",
    "Switzerland": "Suiza", "Brazil": "Brasil", "Morocco": "Marruecos",
    "Haiti": "Haití", "Scotland": "Escocia", "United States": "Estados Unidos",
    "Turkey": "Turquía", "Germany": "Alemania", "Ivory Coast": "Costa de Marfil",
    "Netherlands": "Países Bajos", "Japan": "Japón", "Sweden": "Suecia",
    "Tunisia": "Túnez", "Belgium": "Bélgica", "Egypt": "Egipto", "Iran": "Irán",
    "New Zealand": "Nueva Zelanda", "Spain": "España", "Cape Verde": "Cabo Verde",
    "Saudi Arabia": "Arabia Saudí", "France": "Francia", "Iraq": "Irak",
    "Norway": "Noruega", "Algeria": "Argelia", "Jordan": "Jordania",
    "DR Congo": "RD del Congo", "Uzbekistan": "Uzbekistán",
    "England": "Inglaterra", "Croatia": "Croacia", "Panama": "Panamá",
}

# Códigos ISO para las banderas (flagcdn.com)
FLAGS = {
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr",
    "Czech Republic": "cz", "Canada": "ca", "Bosnia and Herzegovina": "ba",
    "Qatar": "qa", "Switzerland": "ch", "Brazil": "br", "Morocco": "ma",
    "Haiti": "ht", "Scotland": "gb-sct", "United States": "us",
    "Paraguay": "py", "Australia": "au", "Turkey": "tr", "Germany": "de",
    "Curaçao": "cw", "Ivory Coast": "ci", "Ecuador": "ec",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Spain": "es", "Cape Verde": "cv", "Saudi Arabia": "sa", "Uruguay": "uy",
    "France": "fr", "Senegal": "sn", "Iraq": "iq", "Norway": "no",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "DR Congo": "cd", "Uzbekistan": "uz", "Colombia": "co",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}
# Cómo se alimenta cada ronda desde la anterior (índices 0-based)
FEEDS = {"R16": [(0, 3), (2, 5), (1, 4), (6, 7), (10, 11), (9, 8), (14, 13), (12, 15)],
         "QF": [(0, 1), (4, 5), (2, 3), (6, 7)],
         "SF": [(0, 1), (2, 3)],
         "Final": [(0, 1)]}


def logo_uris() -> tuple[str, str]:
    """Logo del visor (Mundial-2026.jpg) como data-URIs embebidos:
    favicon = recorte cuadrado centrado en el trofeo; hero = imagen
    completa reducida. Si falta la imagen, devuelve URIs vacíos."""
    import base64
    import io

    path = ROOT / "Mundial-2026.jpg"
    if not path.exists():
        print("[viewer] AVISO: falta Mundial-2026.jpg, sin logo/favicon")
        return "", ""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    # recorte cuadrado centrado en el trofeo (centro de la imagen)
    side = min(w, h)
    left = (w - side) // 2
    icon = img.crop((left, 0, left + side, side)).resize((96, 96), Image.LANCZOS)
    buf = io.BytesIO()
    icon.save(buf, "PNG")
    icon_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    hero = img.resize((int(w * 260 / h), 260), Image.LANCZOS)
    buf = io.BytesIO()
    hero.save(buf, "JPEG", quality=82)
    hero_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return icon_uri, hero_uri


def modal_score(l1: float, l2: float, rho: float = 0.0) -> str:
    grid = score_grid(np.array([l1]), np.array([l2]), rho)[0][:9, :9]
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    return f"{i}-{j}"


def world_ranking() -> pd.Series:
    """Ranking mundial (1 = mejor) según el Elo propio de ~240 selecciones."""
    elo = pd.read_csv(PROCESSED / "elo_final.csv", index_col="team")["elo"]
    return elo.rank(ascending=False).astype(int)


def _flag_img(team: str, cls: str = "flag sm") -> str:
    c = FLAGS.get(team)
    return (f'<img class="{cls}" src="https://flagcdn.com/w40/{c}.png" alt="">'
            if c else "")


def match_explanation(t1: str, t2: str, p1: float, px, p2: float,
                      snaps: pd.DataFrame, world_rank: pd.Series,
                      host1: bool = False, host2: bool = False,
                      ko: bool = False) -> str:
    """HTML que explica, en lenguaje sencillo, de dónde salen las
    probabilidades de un partido: fuerza Elo, forma reciente (xG), valor de
    plantilla y localía — los mismos datos que alimentan el modelo.

    En grupos `p1/px/p2` son victoria/empate/derrota del primer equipo; en
    eliminatoria (`ko=True`) `p1/p2` son la probabilidad de PASAR el cruce
    (prórroga y penaltis incluidos) y `px` se ignora."""
    s1, s2 = snaps.loc[t1], snaps.loc[t2]
    n1, n2 = NAMES_ES.get(t1, t1), NAMES_ES.get(t2, t2)
    elo1, elo2 = float(s1["elo"]), float(s2["elo"])
    r1, r2 = int(world_rank.get(t1, 0)), int(world_rank.get(t2, 0))
    diff = elo1 - elo2
    strongn = n1 if diff >= 0 else n2
    exp = 1.0 / (1.0 + 10 ** (-abs(diff) / 400.0))
    rk = lambda r: f"la nº {r} del mundo" if r else "sin ranking"

    facts = []
    # --- Fuerza (Elo) ---
    if abs(diff) < 30:
        fuerza = (f"{n1} ({rk(r1)}, Elo {elo1:.0f}) y {n2} ({rk(r2)}, Elo "
                  f"{elo2:.0f}) están prácticamente igualadas en fuerza, "
                  f"así que ninguna parte como clara favorita.")
    else:
        fuerza = (f"{n1} es {rk(r1)} (Elo {elo1:.0f}) y {n2} {rk(r2)} (Elo "
                  f"{elo2:.0f}). Esa diferencia de {abs(diff):.0f} puntos hace "
                  f"que, en igualdad de condiciones, {strongn} se imponga unas "
                  f"{round(exp * 10)} de cada 10 veces. Es el factor que más "
                  f"pesa en el modelo.")
    facts.append(("Fuerza (ranking Elo)", fuerza))

    # --- Forma reciente (xG) ---
    net1 = float(s1["form_xgf"]) - float(s1["form_xga"])
    net2 = float(s2["form_xgf"]) - float(s2["form_xga"])
    formn = n1 if net1 >= net2 else n2
    forma = (f"Por su juego reciente, {n1} genera {float(s1['form_xgf']):.1f} "
             f"goles esperados por partido y encaja {float(s1['form_xga']):.1f}; "
             f"{n2}, {float(s2['form_xgf']):.1f} y {float(s2['form_xga']):.1f}. "
             f"El balance reciente favorece a {formn}.")
    facts.append(("Forma reciente (xG)", forma))

    # --- Valor de plantilla ---
    mv1, mv2 = s1["mv"], s2["mv"]
    if pd.notna(mv1) and pd.notna(mv2) and mv1 > 0 and mv2 > 0:
        richn = n1 if mv1 >= mv2 else n2
        ratio = max(mv1, mv2) / min(mv1, mv2)
        comp = (f"la de {richn} es unas {ratio:.1f} veces más cara"
                if ratio >= 1.3 else "son de valor parecido")
        facts.append(("Valor de plantilla",
                      f"La plantilla de {n1} está tasada en {mv1:.0f} M€ y la de "
                      f"{n2} en {mv2:.0f} M€ — {comp}. Más valor suele reflejar "
                      f"más calidad individual."))

    # --- Localía ---
    if host1 and not host2:
        facts.append(("Localía", f"{n1} juega como anfitriona en su país: el "
                      "apoyo del público y no viajar le dan una ventaja extra."))
    elif host2 and not host1:
        facts.append(("Localía", f"{n2} juega como anfitriona en su país: el "
                      "apoyo del público y no viajar le dan una ventaja extra."))

    # --- Barra y conclusión ---
    if ko:
        bar = (f'<div class="why-bar"><div class="w" style="width:{p1}%"></div>'
               f'<div class="l" style="width:{p2}%"></div></div>'
               f'<div class="why-leg"><span>{_flag_img(t1)} {n1} pasa <b>{p1}%</b></span>'
               f'<span><b>{p2}%</b> pasa {n2} {_flag_img(t2)}</span></div>')
        concl = (f"En resumen, el modelo da un <b>{p1}%</b> a que {n1} avance y un "
                 f"<b>{p2}%</b> a {n2}. En eliminatoria no hay empate: si acaban "
                 f"iguales a los 90', deciden la prórroga y, en su caso, los "
                 f"penaltis (donde la favorita por Elo tiene una ligera ventaja).")
    else:
        bar = (f'<div class="why-bar"><div class="w" style="width:{p1}%"></div>'
               f'<div class="d" style="width:{px}%"></div>'
               f'<div class="l" style="width:{p2}%"></div></div>'
               f'<div class="why-leg"><span>{_flag_img(t1)} {n1} <b>{p1}%</b></span>'
               f'<span>Empate <b>{px}%</b></span>'
               f'<span><b>{p2}%</b> {n2} {_flag_img(t2)}</span></div>')
        concl = (f"En resumen, el modelo da un <b>{p1}%</b> a la victoria de {n1}, "
                 f"un <b>{px}%</b> al empate y un <b>{p2}%</b> a la de {n2}. El "
                 f"empate y la opción de la menos favorita recogen el azar de un "
                 f"solo partido: una expulsión, un penalti o una buena tarde del "
                 f"portero pueden cambiarlo todo.")

    head = (f'<div class="mp-head">{_flag_img(t1, "flag")}'
            f'<div><h2 style="font-size:17px">{n1} '
            f'<span style="color:var(--dim);font-weight:600">vs</span> {n2}</h2>'
            f'<div class="grp">¿De dónde salen estas probabilidades?</div></div>'
            f'{_flag_img(t2, "flag")}</div>')
    flist = "".join(f'<div class="why-f"><span class="why-k">{k}</span>{v}</div>'
                    for k, v in facts)
    return (f'{head}{bar}<div class="mp-sec">Factores que mueven el pronóstico'
            f'</div>{flist}<div class="why-concl">{concl}</div>')


def fixtures_payload(lam_tab, groups, played_group, snaps, world_rank,
                     rho=0.0) -> list[dict]:
    """Los 72 partidos reales de grupos con sus probabilidades 1X2."""
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= WC26_START)]
    group_of = {t: g for g, ts in groups.items() for t in ts}
    out = []
    for _, r in wc.iterrows():
        t1, t2 = r["home_team"], r["away_team"]
        if group_of.get(t1) is None or group_of.get(t1) != group_of.get(t2):
            continue  # eliminatoria: se pinta en el cuadro
        l1, l2 = lam_tab[(t1, t2)], lam_tab[(t2, t1)]
        p = outcome_probs(np.array([l1]), np.array([l2]), rho=rho)[0]
        played = (t1, t2) in played_group
        p1, px, p2 = (round(100 * p[0], 1), round(100 * p[1], 1),
                      round(100 * p[2], 1))
        out.append({
            "group": group_of[t1], "date": r["date"].strftime("%d %b"),
            "t1": t1, "t2": t2,
            "p1": p1, "px": px, "p2": p2,
            "score": (f"{int(r['home_score'])}-{int(r['away_score'])}"
                      if played else modal_score(l1, l2, rho)),
            "played": played,
            "why": match_explanation(
                t1, t2, p1, px, p2, snaps, world_rank,
                host1=t1 in WC26_HOSTS_THROUGH_R16 and t2 not in WC26_HOSTS_THROUGH_R16,
                host2=t2 in WC26_HOSTS_THROUGH_R16 and t1 not in WC26_HOSTS_THROUGH_R16),
        })
    return out


def ko_win_prob(l1: float, l2: float, pen_p1: float = 0.5, rho: float = 0.0, max_g: int = 12) -> float:
    """P(equipo 1 pasa la eliminatoria): 90' Poisson + prórroga λ/3 +
    penaltis según el modelo Elo-logístico."""
    def wdl(a, b):
        joint = score_grid(np.array([a]), np.array([b]), rho)[0]
        win = np.tril(joint, -1).sum()
        draw = np.trace(joint)
        return win, draw, 1 - win - draw

    p1, pd, _ = wdl(l1, l2)
    e1, ed, _ = wdl(l1 / 3, l2 / 3)
    return p1 + pd * (e1 + pen_p1 * ed)


PREV_ROUND = {"R16": "R32", "QF": "R16", "SF": "QF", "Final": "SF"}


def coherent_groups(collect, groups) -> tuple[dict, list]:
    """Clasificación de grupo coherente y cuadro R32 derivado de ella.

    1º = mayor % de acabar 1º; 2º = mayor % de acabar 2º entre los
    restantes; 3º = ídem. Los 8 mejores terceros se eligen por puntos
    esperados y se asignan a su cruce con la tabla oficial de 495
    combinaciones. Así la tabla de cada grupo y el cuadro cuentan la
    misma historia (antes cada cruce elegía su emparejamiento modal de
    forma independiente y un equipo podía aparecer en dos sitios).
    """
    from src.sim.formats import load_thirds_map

    order = {}
    thirds = {}
    for g, ts in groups.items():
        remaining = set(ts)
        ranked = []
        for pidx in range(3):
            pick = max(remaining,
                       key=lambda t: collect["group_pos"][g][t][pidx])
            ranked.append(pick)
            remaining.remove(pick)
        ranked.append(remaining.pop())
        order[g] = ranked
        thirds[g] = ranked[2]

    best8 = sorted(thirds, key=lambda g: -collect["group_pts"][g][thirds[g]])[:8]
    combo = "".join(sorted(best8))
    assign = load_thirds_map()[combo]              # '1A' -> letra del tercero
    rival = {slot[1]: thirds[gl] for slot, gl in assign.items()}

    pos = {f"{g}{i + 1}": t for g, ts in order.items() for i, t in enumerate(ts)}
    r32 = [
        (pos["A2"], pos["B2"]), (pos["C1"], pos["F2"]), (pos["E1"], rival["E"]),
        (pos["F1"], pos["C2"]), (pos["E2"], pos["I2"]), (pos["I1"], rival["I"]),
        (pos["A1"], rival["A"]), (pos["L1"], rival["L"]), (pos["G1"], rival["G"]),
        (pos["D1"], rival["D"]), (pos["H1"], pos["J2"]), (pos["K2"], pos["L2"]),
        (pos["B1"], rival["B"]), (pos["D2"], pos["G2"]), (pos["J1"], pos["H2"]),
        (pos["K1"], rival["K"]),
    ]
    return order, r32


LATE_ROUNDS = {"QF", "SF", "Final", "Third"}  # sedes solo de EE.UU.
QF_FROM = "2026-07-09"               # de cuartos en adelante -> tabla "late"


# ------------------------------------------------------------ nuevos paneles
def today_matches(lam_tab, lam_late, groups, rho, today: str) -> list[dict]:
    """Partidos del Mundial que se juegan HOY, con probabilidades y, si ya
    se jugaron, el resultado real."""
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup")
            & (df["date"] == today)]
    group_of = {t: g for g, ts in groups.items() for t in ts}
    out = []
    for _, r in wc.iterrows():
        t1, t2 = r["home_team"], r["away_team"]
        if t1 not in group_of or t2 not in group_of:
            continue
        same_group = group_of[t1] == group_of[t2] and today <= "2026-06-27"
        lam = lam_late if today >= QF_FROM else lam_tab
        l1, l2 = lam[(t1, t2)], lam[(t2, t1)]
        p = outcome_probs(np.array([l1]), np.array([l2]), rho=rho)[0]
        played = pd.notna(r["home_score"])
        if same_group:
            label = f"Grupo {group_of[t1]}"
        elif today <= "2026-07-03":
            label = "Dieciseisavos"
        elif today <= "2026-07-08":
            label = "Octavos"
        elif today <= "2026-07-12":
            label = "Cuartos de final"
        elif today <= "2026-07-16":
            label = "Semifinal"
        elif today <= "2026-07-18":
            label = "3er puesto"
        else:
            label = "FINAL"
        out.append({
            "t1": t1, "t2": t2,
            "label": label,
            "city": r["city"],
            "p1": round(100 * p[0], 1), "px": round(100 * p[1], 1),
            "p2": round(100 * p[2], 1),
            "score": (f"{int(r['home_score'])}-{int(r['away_score'])}"
                      if played else modal_score(l1, l2, rho)),
            "played": bool(played),
        })
    return out


def update_prediction_log(lam_tab, lam_late, groups, rho, today: str) -> pd.DataFrame:
    """Registra las predicciones de los partidos AÚN NO JUGADOS para poder
    evaluar después al modelo de forma honesta (out-of-sample)."""
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= WC26_START)
            & df["home_score"].isna()]
    teams = {t for ts in groups.values() for t in ts}
    rows = []
    for _, r in wc.iterrows():
        t1, t2 = r["home_team"], r["away_team"]
        if t1 not in teams or t2 not in teams:
            continue
        lam = lam_late if r["date"].strftime("%Y-%m-%d") >= QF_FROM else lam_tab
        l1, l2 = lam[(t1, t2)], lam[(t2, t1)]
        p = outcome_probs(np.array([l1]), np.array([l2]), rho=rho)[0]
        rows.append({"gen_date": today, "match_date": r["date"].strftime("%Y-%m-%d"),
                     "t1": t1, "t2": t2,
                     "p1": round(p[0], 4), "px": round(p[1], 4),
                     "p2": round(p[2], 4),
                     "gpred": modal_score(l1, l2, rho)})
    log_path = OUTPUTS / "match_predictions_log.csv"
    log = pd.DataFrame(rows)
    if log_path.exists():
        log = pd.concat([pd.read_csv(log_path), log], ignore_index=True)
    log = log.drop_duplicates(["gen_date", "t1", "t2"], keep="last")
    # El log publicado lo escribe SOLO GitHub Actions: el entrenamiento
    # XGBoost no es reproducible bit a bit entre máquinas, y un run local
    # pisaría las predicciones ya publicadas en la web.
    if os.environ.get("GITHUB_ACTIONS"):
        log.to_csv(log_path, index=False, encoding="utf-8")
    return log


def scoreboard_payload(log: pd.DataFrame, names: dict) -> dict | None:
    """Métricas del modelo sobre los partidos YA jugados, usando la última
    predicción registrada ANTES de cada partido."""
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    played = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= WC26_START)
                & df["home_score"].notna()]
    if played.empty or log.empty:
        return None
    rows = []
    for _, r in played.iterrows():
        md = r["date"].strftime("%Y-%m-%d")
        cand = log[(log["t1"] == r["home_team"]) & (log["t2"] == r["away_team"])
                   & (log["match_date"] == md) & (log["gen_date"] <= md)]
        if cand.empty:
            continue
        pr = cand.sort_values("gen_date").iloc[-1]
        g1, g2 = int(r["home_score"]), int(r["away_score"])
        oc = 0 if g1 > g2 else (1 if g1 == g2 else 2)
        probs = np.array([pr["p1"], pr["px"], pr["p2"]])
        y = np.zeros(3); y[oc] = 1
        gp = pr["gpred"] if "gpred" in pr else None
        real = f"{g1}-{g2}"
        rows.append({"t1": r["home_team"], "t2": r["away_team"],
                     "score": real,
                     "p_real": float(probs[oc]),
                     "hit": bool(probs.argmax() == oc),
                     "brier": float(np.sum((probs - y) ** 2)),
                     "gpred": gp if pd.notna(gp) else None,
                     "exact": bool(pd.notna(gp) and gp == real)})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    best = d.loc[d["p_real"].idxmax()]
    worst = d.loc[d["p_real"].idxmin()]
    fmt = lambda x: (f"{names.get(x['t1'], x['t1'])} {x['score']} "
                     f"{names.get(x['t2'], x['t2'])} ({100 * x['p_real']:.0f}%)")
    exact = d[d["gpred"].notna()]
    out = {"n": len(d), "acc": round(100 * d["hit"].mean(), 1),
           "brier": round(d["brier"].mean(), 3),
           "logloss": round(float(-np.mean(np.log(np.clip(d["p_real"], 1e-9, 1)))), 3),
           "best": fmt(best), "worst": fmt(worst),
           "exact_n": int(len(exact)),
           "exact_acc": round(100 * exact["exact"].mean(), 1) if len(exact) else None}
    return out


def predictions_payload(log: pd.DataFrame, groups) -> list[dict]:
    """Pronóstico vs resultado, partido a partido: para cada partido se
    toma la ÚLTIMA predicción registrada antes de jugarse (out-of-sample,
    igual que el marcador) y, si ya se jugó, el resultado real."""
    if log.empty:
        return []
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= WC26_START)]
    teams = {t for ts in groups.values() for t in ts}
    out = []
    for _, r in wc.iterrows():
        t1, t2 = r["home_team"], r["away_team"]
        if t1 not in teams or t2 not in teams:
            continue
        md = r["date"].strftime("%Y-%m-%d")
        cand = log[(log["t1"] == t1) & (log["t2"] == t2)
                   & (log["match_date"] == md) & (log["gen_date"] <= md)]
        if cand.empty:
            continue
        pr = cand.sort_values("gen_date").iloc[-1]
        probs = np.array([pr["p1"], pr["px"], pr["p2"]], dtype=float)
        played = pd.notna(r["home_score"])
        gp = pr["gpred"] if "gpred" in pr else None
        gp = gp if pd.notna(gp) else None
        row = {
            "date": md, "date_str": r["date"].strftime("%d %b"),
            "t1": t1, "t2": t2,
            "p1": round(100 * probs[0], 1), "px": round(100 * probs[1], 1),
            "p2": round(100 * probs[2], 1),
            "pred": int(probs.argmax()),
            "pred_score": gp,
            "played": bool(played),
        }
        if played:
            g1, g2 = int(r["home_score"]), int(r["away_score"])
            oc = 0 if g1 > g2 else (1 if g1 == g2 else 2)
            row.update({"score": f"{g1}-{g2}",
                        "hit": bool(probs.argmax() == oc),
                        "exact_hit": bool(gp is not None and gp == f"{g1}-{g2}")})
        out.append(row)
    out.sort(key=lambda x: x["date"])
    return out


def trends_payload(groups) -> dict:
    """Evolución diaria de la probabilidad de campeón (CSV fechados)."""
    entries = []
    base = OUTPUTS / "wc26_stage_probabilities.csv"
    if base.exists():
        entries.append(("Pre", pd.read_csv(base, index_col="team")["Champion"]))
    for f in sorted(OUTPUTS.glob("wc26_stage_probabilities_live_*.csv")):
        d = f.stem.replace("wc26_stage_probabilities_live_", "")
        entries.append((d[5:], pd.read_csv(f, index_col="team")["Champion"]))
    teams = [t for ts in groups.values() for t in ts]
    return {"dates": [e[0] for e in entries],
            "series": {t: [round(float(s.get(t, 0)), 2) for _, s in entries]
                       for t in teams}}


def profiles_payload(snaps, table, collect, groups) -> dict:
    """Ficha por selección: stats, posiciones de grupo y rivales más
    probables en cada ronda del cuadro."""
    n = collect["n_sims"]
    group_of = {t: g for g, ts in groups.items() for t in ts}
    # posición en el ranking mundial de fuerza (Elo propio, ~240 selecciones)
    world_elo = pd.read_csv(PROCESSED / "elo_final.csv", index_col="team")["elo"]
    world_rank = world_elo.rank(ascending=False).astype(int)
    out = {}
    for t in snaps.index:
        g = group_of[t]
        rivals = {}
        for rnd, slots in collect["ko"].items():
            agg: dict = {}
            for counter in slots:
                for (a, b, _), c in counter.items():
                    if a == t:
                        agg[b] = agg.get(b, 0) + c
                    elif b == t:
                        agg[a] = agg.get(a, 0) + c
            top = sorted(agg.items(), key=lambda kv: -kv[1])[:3]
            rivals[rnd] = [[opp, round(100 * c / n, 1)] for opp, c in top]
        s = snaps.loc[t]
        out[t] = {
            "group": g,
            "elo": round(float(s["elo"])),
            "rank": int(world_rank.get(t, 0)),
            "xgf": round(float(s["form_xgf"]), 2),
            "xga": round(float(s["form_xga"]), 2),
            "win": round(100 * float(s["form_winrate"]), 1),
            "mv": None if pd.isna(s["mv"]) else round(float(s["mv"])),
            "age": None if pd.isna(s["age"]) else round(float(s["age"]), 1),
            "pos": [round(100 * collect["group_pos"][g][t][i] / n, 1) for i in range(4)],
            "stages": {k: float(v) for k, v in table.loc[t].items()},
            "rivals": rivals,
        }
    return out


def ko_payload(collect, lam_tab, lam_late, r32_pairs, snaps, world_rank,
               pen_tab=None, rho=0.0) -> dict:
    """Cuadro COHERENTE por propagación.

    - R32: emparejamiento más frecuente de cada cruce en las simulaciones.
    - Rondas siguientes: el cruce se construye con los GANADORES marcados de
      los cruces anteriores (el favorito de cada uno), de modo que el cuadro
      nunca se contradice entre rondas. El % de victoria es analítico
      (Poisson + prórroga + penaltis) para ese emparejamiento exacto, y
      'pair_pct' indica en qué % de simulaciones ocurre realmente el cruce.
    """
    n = collect["n_sims"]
    out = {}
    picked: dict[tuple[str, int], str] = {}  # (ronda, slot) -> ganador marcado
    for rnd, slots in collect["ko"].items():
        rows = []
        for i, counter in enumerate(slots):
            if rnd == "R32":
                t1, t2 = r32_pairs[i]
                freq = sum(c for (x, y, _), c in counter.items()
                           if {x, y} == {t1, t2})
            elif rnd == "Third":
                # perdedores de las semifinales marcadas
                sf_rows = out["SF"]
                t1 = (sf_rows[0]["t2"] if picked[("SF", 0)] == sf_rows[0]["t1"]
                      else sf_rows[0]["t1"])
                t2 = (sf_rows[1]["t2"] if picked[("SF", 1)] == sf_rows[1]["t1"]
                      else sf_rows[1]["t1"])
                freq = sum(c for (x, y, _), c in counter.items()
                           if {x, y} == {t1, t2})
            else:
                a, b = FEEDS[rnd][i]
                t1 = picked[(PREV_ROUND[rnd], a)]
                t2 = picked[(PREV_ROUND[rnd], b)]
                freq = sum(c for (x, y, _), c in counter.items()
                           if {x, y} == {t1, t2})
            lam = lam_late if rnd in LATE_ROUNDS else lam_tab
            pen = pen_tab[(t1, t2)] if pen_tab else 0.5
            w1 = ko_win_prob(lam[(t1, t2)], lam[(t2, t1)], pen_p1=pen, rho=rho)
            picked[(rnd, i)] = t1 if w1 >= 0.5 else t2
            # Marcador más probable a 90' con su probabilidad; si es
            # empate, se decidiría en prórroga/penaltis (se marca con +)
            grid = score_grid(np.array([lam[(t1, t2)]]),
                              np.array([lam[(t2, t1)]]), rho)[0][:9, :9]
            gi, gj = np.unravel_index(grid.argmax(), grid.shape)
            w1r, w2r = round(100 * w1, 1), round(100 * (1 - w1), 1)
            hosts = WC26_HOSTS_FROM_QF if rnd in LATE_ROUNDS else WC26_HOSTS_THROUGH_R16
            rows.append({
                "t1": t1, "t2": t2,
                "pair_pct": round(100 * freq / n, 1),
                "w1": w1r,
                "w2": w2r,
                "score": f"{gi}-{gj}" + ("+" if gi == gj else ""),
                "score_pct": round(100 * float(grid[gi, gj])),
                "why": match_explanation(
                    t1, t2, w1r, None, w2r, snaps, world_rank, ko=True,
                    host1=t1 in hosts and t2 not in hosts,
                    host2=t2 in hosts and t1 not in hosts),
            })
        out[rnd] = rows
    return out


def main():
    n_sims = N_SIMULATIONS
    if "--sims" in sys.argv:
        n_sims = int(sys.argv[sys.argv.index("--sims") + 1])
    today = date.today().isoformat()
    if "--today" in sys.argv:  # útil para previsualizar la franja de partidos
        today = sys.argv[sys.argv.index("--today") + 1]

    cutoff = (date.today() + timedelta(days=1)).isoformat()
    model = train(cutoff)
    groups = load_groups()
    played_group, ko_winners = load_tournament_state(groups)
    teams = [t for ts in groups.values() for t in ts]
    snaps = team_snapshots(teams, cutoff, wc_year=2026)
    lam_tab, lam_late = lambda_tables_wc26(model, snaps)
    pen_tab = penalty_table(snaps, fit_penalty_model(cutoff))
    rho = fit_rho(model, load_long(), cutoff)

    collect: dict = {}
    print(f"Simulando {n_sims:,} torneos...")
    table = simulate_wc26(lam_tab, groups, n_sims, lam_tab_late=lam_late,
                          pen_tab=pen_tab, rho=rho,
                          played_group=played_group, ko_winners=ko_winners,
                          collect=collect)
    order_map, r32_pairs = coherent_groups(collect, groups)
    world_rank = world_ranking()

    log = update_prediction_log(lam_tab, lam_late, groups, rho, today)

    data = {
        "today": today_matches(lam_tab, lam_late, groups, rho, today),
        "today_date": today,
        "scoreboard": scoreboard_payload(log, NAMES_ES),
        "predictions": predictions_payload(log, groups),
        "trends": trends_payload(groups),
        "profiles": profiles_payload(snaps, table, collect, groups),
        "generated": date.today().isoformat(),
        "n_sims": n_sims,
        "played_matches": len(played_group) + len(ko_winners),
        "groups": {
            g: [{
                "team": t,
                "p1": round(100 * collect["group_pos"][g][t][0] / n_sims, 1),
                "p2": round(100 * collect["group_pos"][g][t][1] / n_sims, 1),
                "p3": round(100 * collect["group_pos"][g][t][2] / n_sims, 1),
                "exp_pts": round(collect["group_pts"][g][t] / n_sims, 1),
                "qual": float(table.loc[t, "R32"]),
            } for t in order_map[g]]
            for g in groups
        },
        "fixtures": fixtures_payload(lam_tab, groups, played_group, snaps,
                                     world_rank, rho=rho),
        "ko": ko_payload(collect, lam_tab, lam_late, r32_pairs, snaps,
                         world_rank, pen_tab=pen_tab, rho=rho),
        "ko_labels": R32_LABELS,
        "feeds": FEEDS,
        "stages": [{"team": t, **{k: float(v) for k, v in row.items()}}
                   for t, row in table.iterrows()],
        "flags": FLAGS,
        "names": NAMES_ES,
    }

    template = (ROOT / "src" / "viewer_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    icon_uri, hero_uri = logo_uris()
    html = html.replace("__FAVICON_URI__", icon_uri).replace("__HERO_URI__", hero_uri)
    out = OUTPUTS / "wc26_viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"Visor generado: {out}")
    print("Ábrelo con doble clic (no necesita servidor).")


if __name__ == "__main__":
    main()
