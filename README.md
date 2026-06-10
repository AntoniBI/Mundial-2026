# wc26-predictor — Predicción del Mundial 2026

Pipeline completo: ingesta multi-fuente → feature engineering → modelo de
goles → backtest sobre el Mundial 2022 → 10.000 simulaciones Monte Carlo
del Mundial 2026.

## Resultado (10.000 simulaciones, datos hasta 2026-06-10)

| Equipo | Campeón | Final | SF |
|---|---|---|---|
| Argentina | 14,4 % | 23,1 % | 36,9 % |
| España | 13,9 % | 22,5 % | 37,1 % |
| Francia | 13,5 % | 21,7 % | 33,7 % |
| Inglaterra | 8,9 % | 15,8 % | 26,5 % |
| Brasil | 7,6 % | 14,6 % | 25,1 % |
| Portugal | 4,7 % | 9,8 % | 20,2 % |

**Ventaja de anfitrión por fase y amortiguada** (`config.py`):
México/Canadá juegan como locales solo hasta octavos (de cuartos en
adelante todas las sedes son de EE.UU.) y, al ser un Mundial y no una
eliminatoria con estadio 100% hostil, solo se aplica el 50% del efecto
local aprendido del histórico (`HOST_ADVANTAGE_DAMP = 0.5`). Con ello
México queda en 2,3% de título (~15º), Canadá 1,3% y EE.UU. 0,7%, en
línea con su fuerza real más un plus razonable de anfitrión.

Tabla completa por fase (R32→Campeón) en `outputs/wc26_stage_probabilities.csv`.

## Revisión crítica del repo original (`Mundial2026/`)

Lo que estaba bien planteado y se ha conservado:
- XGBoost sobre formato largo (una fila por equipo-partido) prediciendo goles.
- Idea de time decay con semivida y de ponderar torneos.
- Cuadro oficial del torneo y tabla de 495 combinaciones de mejores terceros
  (`mejores_terceros.csv`, reutilizada aquí).

Problemas detectados y cómo se han corregido:

1. **Bug en `get_multiplier` (clases_simulacion.py:106-108)**: tras calcular
   los multiplicadores tácticos, `mult1 = 1.3; mult2 = 1.3` los
   **sobreescribía siempre**, dejando muerto todo el bloque anterior.
   *Aquí*: el resultado se muestrea de una Poisson(λ) — sin heurísticas
   minuto a minuto con constantes arbitrarias.
2. **No era un Monte Carlo real**: cada partido se simulaba 30 veces y se
   tomaba la **moda**, propagando un único cuadro determinista. La salida
   era *un* resultado, no probabilidades. *Aquí*: el torneo completo se
   simula 10.000 veces y la salida es el % de alcanzar cada fase.
3. **Elo hardcodeado** por equipo en el código y constante durante todo el
   torneo. *Aquí*: Elo recalculado partido a partido sobre 49.450 partidos
   (1872-2026) con metodología eloratings.net (K por torneo, multiplicador
   por diferencia de goles, +100 al local), de modo que cada fila de
   entrenamiento lleva el Elo **en la fecha del partido**.
4. **Sin backtest**: el modelo nunca se validó contra un torneo real.
   *Aquí*: holdout estricto del Mundial 2022 (ver abajo).
5. **Fugas temporales**: PCA ajustado sobre todo el histórico, imputación
   con medias globales, decay referenciado a una fecha fija (2024-06-01)
   ya pasada. *Aquí*: todo lo que entra al modelo se calcula solo con
   datos anteriores al corte; XGBoost trata los NaN de forma nativa.
6. **Ponderación de torneos solo como feature** (`tournament_num`), no como
   peso de entrenamiento. *Aquí*: peso de muestra = decay × peso del torneo,
   además de mantener la importancia del partido como feature.
7. **Ventanas móviles planas** (medias de 5/15 partidos). *Aquí*: formas
   ponderadas por decaimiento exponencial (semivida 3 años) e importancia
   del torneo sobre los últimos 30 partidos.
8. **Sin valor de mercado** ni variables de plantilla. *Aquí*: valor de
   mercado y edad media (Transfermarkt) por edición de Mundial (2014, 2018,
   2022, 2026), sin fuga temporal (cada Mundial usa los valores de su año).
9. **Tweedie + reloj por minutos**: el λ predicho se convertía en "prob. de
   gol por minuto" (xg/90) y multiplicadores ad hoc. *Aquí*: objetivo
   `count:poisson` y muestreo directo de la distribución; prórroga = λ/3 y
   penaltis 50/50.
10. **Empates del grupo**: el orden de criterios no era el de FIFA. *Aquí*:
    puntos → dif. goles → goles → head-to-head → sorteo.

## Fuentes de datos (`src/ingest/`)

| Fuente | Módulo | Estado | Uso |
|---|---|---|---|
| Kaggle "International football results 1872-2026" (martj42) | `results_martj42.py` | ✅ (vía GitHub del autor, más actualizado: incluye el calendario WC26) | Resultados, base del Elo y las formas |
| eloratings.net | `elo_compute.py` | ✅ snapshot de validación; el Elo en fecha se calcula localmente | Rating en la fecha del partido |
| Ranking FIFA oficial (≈ Kaggle "FIFA World Ranking") | `fifa_ranking.py` | ✅ API inside.fifa.com, 62 fechas 2017-2025 | Referencia/validación |
| Transfermarkt | `transfermarkt.py` | ✅ participantes WC 2014/18/22/26 | Valor de mercado y edad media |
| Wikipedia | `wikipedia.py` | ✅ | Grupos oficiales WC26 (validan el cuadro) |
| RSSSF | `rsssf.py` | ✅ índice histórico de Mundiales | Sanity-check |
| FotMob (API JSON no oficial) | `fotmob.py` | ✅ xG por partido desde 2021, caché incremental en `data/raw/fotmob/` | Formas xG (form_xgf/form_xga) |
| FBref | `fbref.py` | ⚠️ bloqueado por Cloudflare (403) desde este entorno | xG (cubierto vía FotMob) |
| API-Football (RapidAPI) | `api_football.py` | ⏸ requiere `API_FOOTBALL_KEY` | Lesiones/alineaciones (opcional) |
| Sportmonks | `sportmonks.py` | ⏸ requiere `SPORTMONKS_KEY` | (opcional) |

## Metodología

1. **Features** (`src/features/build_features.py`): formato largo
   (64.664 filas equipo-partido desde 1990), Elo en fecha, formas con
   decaimiento, contexto local/visitante/neutral real, confederaciones,
   valor de mercado por edición, peso del torneo. Las formas se calculan
   por duplicado sobre goles y sobre **xG de FotMob** (con el marcador
   como fallback donde no hay cobertura xG): el ataque/defensa "real" de
   un equipo se mide mucho mejor con xG que con goles.
2. **Modelo** (`src/model/train.py`): XGBoost `count:poisson` que predice
   los goles de cada equipo; probabilidades 1X2 vía malla de Poisson.
   Peso de muestra = `0.5^(días/1095) × peso_torneo` referenciado a la
   fecha de corte.
3. **Backtest** (`src/model/backtest.py`): Mundial 2022 aislado
   (entrenamiento solo con datos < 2022-11-20):

   | Modelo | log-loss | Brier | RPS |
   |---|---|---|---|
   | + depth=5, pesos hl=4 años, Dixon-Coles (actual) | **1.028** | **0.595** | **0.2120** |
   | + days_since_last / n_prev | 1.029 | 0.601 | 0.2141 |
   | XGBoost Poisson + formas xG | 1.032 | 0.603 | 0.2156 |
   | XGBoost Poisson sin xG | 1.038 | 0.605 | 0.2163 |
   | Baseline Poisson solo-Elo | 1.077 | 0.653 | 0.2364 |
   | Uniforme (azar) | 1.099 | 0.667 | 0.2387 |

   `max_depth=5` y half-life de pesos de 4 años fueron validados en un
   banco multi-torneo (WC18, EU/CA21, WC22, EU/CA24, 290 partidos): única
   combinación que mejora el log-loss en las 4 ventanas
   (`src/model/eval_multiwindow.py`). La corrección **Dixon-Coles**
   (ρ ajustado por MLE sobre el entrenamiento, ≈−0.05) corrige la
   infra-predicción de empates de la Poisson independiente (23,8%
   predicho vs 25,9% real agregado).

   Probados contra el holdout y **descartados por empeorar**: flag de
   cobertura xG de la forma (1.041), quitar confederaciones en combinación
   con las features nuevas (1.032), corrección de prórrogas vía
   `base_margin` (1.054, rompe el intercepto de XGBoost) o vía target
   escalado (1.048), y monotonía en elo_diff (1.039).

   **Penaltis**: ya no son 50/50 — logística P(ganar la tanda) ~ diferencia
   de Elo ajustada sobre 541 tandas históricas (`shootouts.csv`): el equipo
   con +200 Elo gana el 61,9% de las tandas.

   A nivel torneo (10.000 sims del WC22): Brasil 19,7 % y Argentina 17,4 %
   favoritos (en línea con las cuotas reales de noviembre 2022; Argentina
   fue campeona), Francia 14,8 % de llegar a la final (llegó).
4. **Simulación** (`src/sim/`): λ precalculadas para los 2.256 pares
   posibles; 10.000 torneos completos (grupos con desempates FIFA, mejores
   terceros con la tabla oficial de 495 combinaciones, cuadro oficial,
   prórroga y penaltis). Anfitriones (USA/MEX/CAN) juegan como locales.

## Uso

```bash
pip install -r requirements.txt
python -m src.run_ingest          # descarga todas las fuentes
python -m src.features.build_features
python -m src.model.backtest      # validación Mundial 2022
python -m src.run_predict         # predicción Mundial 2026
```

### Visor HTML

```bash
python -m src.run_viewer            # genera outputs/wc26_viewer.html
```

Visor autocontenido (doble clic, sin servidor) con estética FIFA
(paleta de fifa.com/canadamexicousa2026 y colores de la marca 26),
banderas de las 48 selecciones (flagcdn.com) y tres pestañas:
**Fase de grupos** (los 72 partidos con % 1X2, marcador más probable y
clasificación simulada de cada grupo), **Cuadro final** (en espejo:
mitad izquierda y derecha convergen en la final; cada cruce con su
emparejamiento más probable y % de victoria condicionado) y
**Camino al título** (% de cada selección de alcanzar cada fase).
Enlaces directos: `wc26_viewer.html#groups`, `#bracket`, `#stages`.
Fija los partidos reales ya jugados, igual que `run_update`: regenerarlo
cada día durante el torneo lo mantiene al día.

### Actualización en vivo durante el torneo

```bash
python -m src.run_update                  # ciclo completo (descarga+features+modelo+sim)
python -m src.run_update --sims 20000     # más iteraciones
python -m src.run_update --no-download    # sin re-descargar datos
```

Ejecútalo cada mañana durante el Mundial: fija los partidos ya jugados con
su resultado real (grupos, eliminatorias y penaltis vía `shootouts.csv`),
reentrena con el histórico hasta hoy y simula 10.000 veces **solo lo que
queda de torneo**. Guarda `outputs/wc26_stage_probabilities_live_<fecha>.csv`
(histórico diario) y `..._latest.csv`, y muestra los mayores movimientos en
probabilidad de título respecto a la predicción pre-torneo.

## Limitaciones conocidas / siguientes pasos

- Las features quedan **congeladas** al inicio del torneo (no se actualiza
  el Elo dentro de la simulación). Es la práctica habitual, pero infraestima
  ligeramente la varianza.
- Poisson independiente (sin corrección Dixon-Coles para marcadores bajos
  ni correlación entre goles de ambos equipos).
- Penaltis al 50/50 (los datos de `shootouts.csv` permitirían modelarlos).
- Los marcadores de eliminatorias del dataset incluyen la prórroga (sesgo
  pequeño en el entrenamiento).
- FBref bloqueado por Cloudflare: el xG de torneos grandes queda como
  mejora si se consigue acceso (o vía API-Football con clave).
- **Marcadores a 90' en eliminatorias (línea de trabajo)**: el dataset
  registra los marcadores de KO con prórroga incluida, lo que (a) impide
  añadir una feature `is_ko` sin que la simulación cuente la prórroga dos
  veces (mejoraba el banco multi-torneo 0.9591→0.9556 pero corrompe la λ
  a 90'), y (b) explica el aparente sesgo de −0.10 goles en eliminatorias
  (es un artefacto de medición, no error del modelo). La caché de FotMob
  (`data/raw/fotmob/matches/`) contiene los marcadores por periodos desde
  2021: cuando haya suficiente histórico, extraer los 90' reales y
  reentrenar λ limpias + `is_ko`. Revisar a mitad del Mundial 2026.
- Auditoría de la cadena de λ (jun-2026): dispersión de Pearson 0.912
  (Poisson correcta; binomial negativa iría en dirección equivocada),
  clip de λ nunca activo, ensemble XGB+GLM probado y descartado (peor en
  las 4 ventanas), calibración global +0.01 goles.
