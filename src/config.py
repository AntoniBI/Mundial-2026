"""Configuración global del proyecto de predicción del Mundial 2026."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

for p in (RAW, PROCESSED, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# --- Fechas clave ---
WC26_START = "2026-06-11"          # primer partido del Mundial 2026
WC22_START = "2022-11-20"          # primer partido del Mundial 2022 (corte de backtest)
WC22_END = "2022-12-18"

# --- Decaimiento temporal ---
# Peso = 0.5 ** (días_atrás / half_life).
# HALF_LIFE_DAYS gobierna las VENTANAS DE FORMA (features).
# WEIGHT_HALF_LIFE_DAYS gobierna los PESOS DE ENTRENAMIENTO: el banco de
# validación multi-torneo (WC18, EU/CA21, WC22, EU/CA24) mostró que una
# memoria de 4 años mejora el log-loss en las 4 ventanas.
HALF_LIFE_DAYS = 1095
WEIGHT_HALF_LIFE_DAYS = 1460

# --- Ponderación de torneos (multiplicador de peso de muestra) ---
# Escala inspirada en los factores K de eloratings.net (K/20).
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 3.0,
    "Copa América": 2.5,
    "UEFA Euro": 2.5,
    "African Cup of Nations": 2.5,
    "AFC Asian Cup": 2.5,
    "Gold Cup": 2.5,
    "Confederations Cup": 2.5,
    "CONMEBOL-UEFA Cup of Champions": 2.5,
    "FIFA World Cup qualification": 2.0,
    "UEFA Nations League": 2.0,
    "UEFA Euro qualification": 1.5,
    "Copa América qualification": 1.5,
    "African Cup of Nations qualification": 1.5,
    "AFC Asian Cup qualification": 1.5,
    "Gold Cup qualification": 1.5,
    "CONCACAF Nations League": 1.5,
    "Friendly": 1.0,
}
DEFAULT_TOURNAMENT_WEIGHT = 1.25   # resto de torneos menores

# --- Factores K del Elo (metodología eloratings.net) ---
ELO_K = {
    "FIFA World Cup": 60,
    "Copa América": 50,
    "UEFA Euro": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "Confederations Cup": 50,
    "CONMEBOL-UEFA Cup of Champions": 50,
    "FIFA World Cup qualification": 40,
    "UEFA Nations League": 40,
    "UEFA Euro qualification": 40,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup qualification": 40,
    "Gold Cup qualification": 40,
    "CONCACAF Nations League": 40,
    "Friendly": 20,
}
DEFAULT_ELO_K = 30
ELO_HOME_ADV = 100  # puntos sumados al local (si no es campo neutral)

N_SIMULATIONS = 10_000
RANDOM_SEED = 42

# --- Ventaja de anfitrión en el Mundial 2026 ---
# El efecto local que aprende el modelo viene sobre todo de eliminatorias y
# amistosos (viajes, altitud, estadio hostil). En un Mundial el contexto es
# más neutro, así que solo se aplica una fracción del efecto:
HOST_ADVANTAGE_DAMP = 0.5   # 1.0 = efecto completo, 0.0 = campo neutral
# México y Canadá solo juegan en su país hasta octavos; de cuartos en
# adelante todas las sedes son estadounidenses.
WC26_HOSTS_THROUGH_R16 = {"United States", "Mexico", "Canada"}
WC26_HOSTS_FROM_QF = {"United States"}
