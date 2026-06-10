"""Predicción final: Mundial 2026 con 10.000 simulaciones Monte Carlo.

Entrena con TODO el histórico hasta la víspera del torneo (2026-06-11) y
simula el torneo completo (12 grupos, mejores terceros, cuadro oficial)
N veces. Salida: probabilidad de cada selección de alcanzar cada fase.

Uso: python -m src.run_predict
"""
import pandas as pd

from src.config import N_SIMULATIONS, OUTPUTS, PROCESSED, WC26_START
from src.model.train import fit_rho, load_long, train
from src.sim.engine import (fit_penalty_model, lambda_tables_wc26,
                            penalty_table, team_snapshots)
from src.sim.formats import simulate_wc26

HOSTS = {"United States", "Mexico", "Canada"}


def load_groups() -> dict[str, list[str]]:
    g = pd.read_csv(PROCESSED / "wc26_groups_wikipedia.csv", encoding="utf-8")
    return {letter: list(sub["team"]) for letter, sub in g.groupby("group")}


def main():
    print(f"=== PREDICCIÓN MUNDIAL 2026 ({N_SIMULATIONS:,} simulaciones) ===")
    model = train(WC26_START)

    groups = load_groups()
    teams = [t for ts in groups.values() for t in ts]
    assert len(teams) == 48, f"se esperaban 48 equipos, hay {len(teams)}"

    snaps = team_snapshots(teams, WC26_START, wc_year=2026)
    lam_early, lam_late = lambda_tables_wc26(model, snaps)
    pen_tab = penalty_table(snaps, fit_penalty_model(WC26_START))
    rho = fit_rho(model, load_long(), WC26_START)

    table = simulate_wc26(lam_early, groups, N_SIMULATIONS,
                          lam_tab_late=lam_late, pen_tab=pen_tab, rho=rho)
    table.to_csv(OUTPUTS / "wc26_stage_probabilities.csv", encoding="utf-8")

    print("\n%% de alcanzar cada fase (top 20 por prob. de título):")
    print(table.head(20).to_string())
    print(f"\nTabla completa: outputs/wc26_stage_probabilities.csv")

    snaps.round(2).to_csv(OUTPUTS / "wc26_team_snapshots.csv", encoding="utf-8")


if __name__ == "__main__":
    main()
