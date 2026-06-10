@echo off
REM La web la genera y publica SOLO GitHub Actions (fuente de verdad):
REM el entrenamiento XGBoost no es reproducible bit a bit entre maquinas,
REM asi que publicar desde aqui pisaria los valores ya publicados.
REM Este script lanza esa ejecucion en remoto y no toca nada local.
cd /d "%~dp0"
gh workflow run update.yml
echo Workflow lanzado. Sigue el progreso con: gh run watch
echo Web: https://antonibi.github.io/Mundial-2026/
