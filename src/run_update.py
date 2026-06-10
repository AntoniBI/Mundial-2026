"""Actualización EN VIVO durante el Mundial 2026.

Ejecútalo después de cada jornada (o cada mañana). Hace todo el ciclo:

  1. Re-descarga los resultados (el dataset martj42 se actualiza a diario
     con los partidos ya jugados del Mundial).
  2. Reconstruye las features (el Elo y las formas absorben los partidos
     reales del torneo ya disputados).
  3. Re-entrena el modelo con corte = hoy.
  4. Simula 10.000 veces SOLO lo que queda de torneo: los partidos ya
     jugados (grupos y eliminatorias, penaltis incluidos) quedan fijados
     con su resultado real.

Salida: outputs/wc26_stage_probabilities_live_<fecha>.csv (+ copia "latest")
y comparación con la predicción pre-torneo.

Uso: python -m src.run_update [--no-download] [--sims 10000]
"""
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import N_SIMULATIONS, OUTPUTS, RAW, WC26_START
from src.features import build_features
from src.ingest import results_martj42
from src.model.train import fit_rho, load_long, train
from src.run_predict import load_groups
from src.sim.engine import (fit_penalty_model, lambda_tables_wc26,
                            penalty_table, team_snapshots)
from src.sim.formats import simulate_wc26


def load_tournament_state(groups: dict[str, list[str]]):
    """Lee del dataset los partidos del Mundial ya jugados y los separa en
    fase de grupos y eliminatorias (con el ganador de penaltis si lo hubo)."""
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= WC26_START)]
    played = wc[wc["home_score"].notna()]

    group_of = {t: g for g, ts in groups.items() for t in ts}
    shoot = pd.read_csv(RAW / "shootouts.csv", parse_dates=["date"])

    played_group: dict[tuple, tuple] = {}
    ko_winners: dict[frozenset, str] = {}
    for _, r in played.iterrows():
        t1, t2 = r["home_team"], r["away_team"]
        g1, g2 = int(r["home_score"]), int(r["away_score"])
        same_group = group_of.get(t1) is not None and group_of.get(t1) == group_of.get(t2)
        # Heurística robusta: hasta el fin de grupos (27-jun) todo partido
        # entre equipos del mismo grupo es de la fase de grupos.
        if same_group and r["date"] <= pd.Timestamp("2026-06-27"):
            played_group[(t1, t2)] = (g1, g2)
        else:
            if g1 != g2:
                winner = t1 if g1 > g2 else t2
            else:  # decidido en penaltis -> shootouts.csv
                m = shoot[(shoot["date"] == r["date"])
                          & (shoot["home_team"] == t1) & (shoot["away_team"] == t2)]
                if m.empty:
                    # shootouts.csv puede actualizarse con retraso respecto a
                    # results.csv: no bloqueamos la actualización diaria, ese
                    # cruce se simula hasta que llegue el dato real
                    print(f"AVISO: empate {t1}-{t2} ({r['date']:%Y-%m-%d}) sin "
                          f"penaltis registrados aún; el cruce se simulará")
                    continue
                winner = m.iloc[0]["winner"]
            ko_winners[frozenset((t1, t2))] = winner
    return played_group, ko_winners


def main():
    n_sims = N_SIMULATIONS
    if "--sims" in sys.argv:
        n_sims = int(sys.argv[sys.argv.index("--sims") + 1])

    if "--no-download" not in sys.argv:
        print("=== 1/4 Descargando resultados actualizados ===")
        results_martj42.download()
        # xG de los últimos 30 días (incremental: usa la caché de disco)
        from src.ingest import fotmob
        fotmob.download(start=(date.today() - timedelta(days=30)).isoformat())

    print("=== 2/4 Reconstruyendo features ===")
    build_features.build()

    cutoff = (date.today() + timedelta(days=1)).isoformat()
    print(f"=== 3/4 Re-entrenando (corte {cutoff}) ===")
    model = train(cutoff)

    print("=== 4/4 Simulando lo que queda de torneo ===")
    groups = load_groups()
    played_group, ko_winners = load_tournament_state(groups)
    print(f"Estado real: {len(played_group)} partidos de grupo jugados, "
          f"{len(ko_winners)} eliminatorias resueltas")

    teams = [t for ts in groups.values() for t in ts]
    snaps = team_snapshots(teams, cutoff, wc_year=2026)
    lam_early, lam_late = lambda_tables_wc26(model, snaps)
    pen_tab = penalty_table(snaps, fit_penalty_model(cutoff))
    rho = fit_rho(model, load_long(), cutoff)
    table = simulate_wc26(lam_early, groups, n_sims, lam_tab_late=lam_late,
                          pen_tab=pen_tab, rho=rho,
                          played_group=played_group, ko_winners=ko_winners)

    stamp = date.today().isoformat()
    table.to_csv(OUTPUTS / f"wc26_stage_probabilities_live_{stamp}.csv", encoding="utf-8")
    table.to_csv(OUTPUTS / "wc26_stage_probabilities_latest.csv", encoding="utf-8")

    print(f"\n%% de alcanzar cada fase a fecha {stamp} (top 15):")
    print(table.head(15).to_string())

    # Comparación con la predicción pre-torneo
    pre_path = OUTPUTS / "wc26_stage_probabilities.csv"
    if pre_path.exists():
        pre = pd.read_csv(pre_path, index_col="team")
        cmp = pd.DataFrame({
            "champion_pre": pre["Champion"],
            "champion_hoy": table["Champion"],
        })
        cmp["delta"] = (cmp["champion_hoy"] - cmp["champion_pre"]).round(2)
        movers = cmp.reindex(cmp["delta"].abs().sort_values(ascending=False).index).head(8)
        print("\nMayores movimientos en prob. de título vs pre-torneo:")
        print(movers.to_string())


if __name__ == "__main__":
    main()
