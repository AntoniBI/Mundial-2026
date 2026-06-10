"""Fuente: FotMob (API JSON no oficial) — xG por partido de selecciones.

Estrategia:
  1. results.csv nos dice qué fechas tienen partidos de los equipos que nos
     interesan (los 48 del WC26 y los 32 del WC22 para el backtest).
  2. /api/data/matches?date=YYYYMMDD lista los partidos del día; filtramos
     los emparejamientos que coinciden con nuestros equipos.
  3. /api/data/matchDetails?matchId=N trae el xG de ambos equipos.

Todo se cachea en data/raw/fotmob/ (días y partidos), de modo que las
ejecuciones son incrementales: durante el Mundial solo descargará los
partidos nuevos del día. Salida: data/processed/fotmob_xg.csv con
(date, team_home, team_away, xg_home, xg_away, goals_home, goals_away).

Uso: python -m src.ingest.fotmob [--start 2021-01-01] [--end hoy]
"""
import json
import sys
import time

import pandas as pd
import requests

from src.config import PROCESSED, RAW, USER_AGENT
from src.ingest.names import canon

DAYS_DIR = RAW / "fotmob" / "days"
MATCH_DIR = RAW / "fotmob" / "matches"
DAYS_DIR.mkdir(parents=True, exist_ok=True)
MATCH_DIR.mkdir(parents=True, exist_ok=True)

API_DAY = "https://www.fotmob.com/api/data/matches?date={d}"
API_MATCH = "https://www.fotmob.com/api/data/matchDetails?matchId={mid}"

# Alias FotMob -> canon martj42 (se amplía si aparecen misses)
FOTMOB_ALIASES = {
    "USA": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Turkiye": "Turkey",
    "Ireland": "Republic of Ireland",
    "Congo DR": "DR Congo",
}


def fm_canon(name: str) -> str:
    return canon(FOTMOB_ALIASES.get(name, name))


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT,
                      "Accept": "application/json",
                      "Referer": "https://www.fotmob.com/"})
    return s


def relevant_dates(teams: set[str], start: str, end: str) -> list[str]:
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    m = ((df["date"] >= start) & (df["date"] <= end)
         & df["home_score"].notna()
         & (df["home_team"].isin(teams) | df["away_team"].isin(teams)))
    return sorted(df.loc[m, "date"].dt.strftime("%Y%m%d").unique())


def day_matches(s: requests.Session, day: str, sleep_s: float) -> list[dict]:
    cache = DAYS_DIR / f"{day}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        r = s.get(API_DAY.format(d=day), timeout=30)
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(sleep_s)
    out = []
    for lg in data.get("leagues", []):
        for m in lg.get("matches", []):
            out.append({"id": m["id"],
                        "home": m["home"]["name"], "away": m["away"]["name"],
                        "finished": m.get("status", {}).get("finished", False)})
    return out


def extract_xg(detail: dict) -> dict | None:
    g = detail.get("general", {})
    home = g.get("homeTeam", {}).get("name")
    away = g.get("awayTeam", {}).get("name")
    date = (g.get("matchTimeUTCDate") or "")[:10]
    xg = None
    try:
        periods = detail["content"]["stats"]["Periods"]["All"]["stats"]
        for section in periods:
            for st in section.get("stats", []):
                if st.get("key") == "expected_goals" and st.get("type") == "text":
                    vals = st.get("stats", [None, None])
                    if vals and vals[0] is not None:
                        xg = (float(vals[0]), float(vals[1]))
                        break
            if xg:
                break
    except (KeyError, TypeError, ValueError):
        return None
    if not (home and away and date and xg):
        return None
    score = detail.get("header", {}).get("status", {}).get("scoreStr", "")
    return {"date": date, "home_fm": home, "away_fm": away,
            "xg_home": xg[0], "xg_away": xg[1], "score": score}


def match_detail(s: requests.Session, mid: int, sleep_s: float) -> dict | None:
    cache = MATCH_DIR / f"{mid}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        r = s.get(API_MATCH.format(mid=mid), timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    cache.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(sleep_s)
    return data


def download(start: str = "2021-01-01", end: str | None = None,
             sleep_s: float = 0.5) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    # Equipos objetivo: los 48 del WC26 + los 32 del WC22 (backtest)
    from src.sim.formats import WC22_GROUPS
    g26 = pd.read_csv(PROCESSED / "wc26_groups_wikipedia.csv", encoding="utf-8")
    teams = set(g26["team"]) | {t for ts in WC22_GROUPS.values() for t in ts}

    s = _session()
    days = relevant_dates(teams, start, end)
    print(f"[fotmob] {len(days)} fechas con partidos de equipos objetivo "
          f"({start} -> {end})")

    rows, miss_names = [], set()
    for i, day in enumerate(days):
        try:
            matches = day_matches(s, day, sleep_s)
        except Exception as e:
            print(f"[fotmob] día {day} fallo: {e}")
            continue
        for m in matches:
            h, a = fm_canon(m["home"]), fm_canon(m["away"])
            if not m["finished"] or (h not in teams and a not in teams):
                continue
            detail = match_detail(s, m["id"], sleep_s)
            if detail is None:
                continue
            row = extract_xg(detail)
            if row is None:
                continue  # partido sin cobertura xG
            row["team_home"] = fm_canon(row["home_fm"])
            row["team_away"] = fm_canon(row["away_fm"])
            if row["team_home"] != h or row["team_away"] != a:
                miss_names.add((row["home_fm"], row["away_fm"]))
            row["match_id"] = m["id"]
            rows.append(row)
        if i % 25 == 0:
            print(f"[fotmob] {i}/{len(days)} días, {len(rows)} partidos con xG")

    df = pd.DataFrame(rows)
    out = PROCESSED / "fotmob_xg.csv"
    if out.exists() and len(df):
        prev = pd.read_csv(out)
        df = pd.concat([prev, df], ignore_index=True)
    if len(df):
        df = df.drop_duplicates("match_id").sort_values("date")
        df.to_csv(out, index=False, encoding="utf-8")
    print(f"[fotmob] {len(df)} partidos con xG acumulados -> {out}")
    if miss_names:
        print(f"[fotmob] nombres con alias dudoso: {sorted(miss_names)[:10]}")
    return df


if __name__ == "__main__":
    kw = {}
    if "--start" in sys.argv:
        kw["start"] = sys.argv[sys.argv.index("--start") + 1]
    if "--end" in sys.argv:
        kw["end"] = sys.argv[sys.argv.index("--end") + 1]
    download(**kw)
