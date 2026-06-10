@echo off
REM Actualizacion diaria: datos -> modelo -> visor -> web
cd /d "%~dp0"
python -m src.run_update
python -m src.run_viewer
copy /Y outputs\wc26_viewer.html docs\index.html
git add -A
git commit -m "Actualizacion diaria"
git push
echo Web actualizada: https://antonibi.github.io/Mundial-2026/
