"""Every unblocked shot attempt since 2010, from MoneyPuck's free shot data
(moneypuck.com, non-commercial use).

Each season is one ~20MB zip, cached in data/moneypuck/. The current season's
file is updated by MoneyPuck every night, so it's re-downloaded once a day.
Only where and how each shot was taken is used - the expected goals model in
tims_xg.py is our own, so the backtest doesn't rely on a model that has seen
the seasons it's tested on.
"""
from datetime import date

import pandas as pd
import requests

import tims_data

MP_URL = "https://peter-tanner.com/moneypuck/downloads/shots_{year}.zip"
MP_DIR = tims_data.DATA_DIR / "moneypuck"

COLUMNS = ["game_id", "season", "period", "time", "event", "goal", "shooterPlayerId", "goalieIdForShot",
           "isHomeTeam", "shotDistance", "shotAngleAdjusted", "shotType", "shotRebound", "shotRush",
           "timeSinceLastEvent", "lastEventCategory", "homeSkatersOnIce", "awaySkatersOnIce", "shotOnEmptyNet"]
SHOT_TYPES = {"WRIST": "wrist", "SNAP": "snap", "SLAP": "slap", "BACK": "backhand", "TIP": "tip-in",
              "DEFL": "deflected", "WRAP": "wrap-around"}
EVENTS = {"GOAL": "goal", "SHOT": "shot-on-goal", "MISS": "missed-shot"}


def _zip_path(season):
    year = int(str(season)[:4])
    path = MP_DIR / f"shots_{year}.zip"
    stale = season == tims_data.current_season() and path.exists() and \
        date.fromtimestamp(path.stat().st_mtime) < date.today()
    if not path.exists() or stale:
        MP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading MoneyPuck shots {year}...", flush=True)
        r = requests.get(MP_URL.format(year=year), timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def load_season(season):
    mp = pd.read_csv(_zip_path(season), usecols=COLUMNS)
    home = mp["isHomeTeam"] == 1
    return pd.DataFrame({
        "gameId": mp["season"] * 1_000_000 + mp["game_id"],
        "season": season,
        "period": mp["period"],
        "t": mp["time"],
        "type": mp["event"].map(EVENTS),
        "shooter": mp["shooterPlayerId"],
        "goalie": mp["goalieIdForShot"].where(mp["goalieIdForShot"] > 0),
        "is_home": mp["isHomeTeam"].astype(int),
        "dist": mp["shotDistance"],
        "angle": mp["shotAngleAdjusted"].abs(),
        "shotType": mp["shotType"].map(SHOT_TYPES),
        "rebound": mp["shotRebound"],
        "rush": mp["shotRush"],
        "dt_prev": mp["timeSinceLastEvent"],
        "prev_shot": mp["lastEventCategory"].isin(["SHOT", "MISS", "BLOCK", "GOAL"]).astype(int),
        "skaters_for": mp["homeSkatersOnIce"].where(home, mp["awaySkatersOnIce"]),
        "skaters_against": mp["awaySkatersOnIce"].where(home, mp["homeSkatersOnIce"]),
        "empty_net": mp["shotOnEmptyNet"],
        "goal": mp["goal"],
    })


def load_shots(first_season, last_season=None):
    """Unblocked shot attempts for a span of seasons."""
    seasons = tims_data.season_range(first_season, last_season or tims_data.current_season())
    return pd.concat([load_season(s) for s in seasons], ignore_index=True)
