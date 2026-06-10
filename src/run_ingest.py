"""Orquestador de ingesta: ejecuta todas las fuentes de datos.

Uso:  python -m src.run_ingest [--skip-fifa]
"""
import sys

from src.ingest import results_martj42, transfermarkt, wikipedia, rsssf, fbref
from src.ingest import api_football, sportmonks, elo_compute


def main():
    print("=== 1/7 martj42 (resultados 1872-2026, Kaggle/GitHub) ===")
    df = results_martj42.download()
    print(f"    {len(df):,} partidos hasta {df['date'].max().date()}")

    print("=== 2/7 eloratings.net (snapshot de validación) ===")
    elo_compute.download_eloratings_snapshot()

    print("=== 3/7 Transfermarkt (valores de mercado WC14/18/22/26) ===")
    transfermarkt.download()

    print("=== 4/7 Wikipedia (grupos oficiales WC26) ===")
    wikipedia.download_groups()

    print("=== 5/7 RSSSF (histórico de Mundiales) ===")
    rsssf.download()

    print("=== 6/7 FBref (xG, mejor esfuerzo) ===")
    fbref.download()

    print("=== 7/7 APIs opcionales (requieren clave) ===")
    api_football.download()
    sportmonks.download()

    if "--skip-fifa" not in sys.argv:
        print("=== extra: ranking FIFA (histórico, lento) ===")
        from src.ingest import fifa_ranking
        fifa_ranking.download()


if __name__ == "__main__":
    main()
