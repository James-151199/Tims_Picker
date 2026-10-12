"""Download and cache historical NHL game-by-game data from the NHL stats API.

Each season is saved to data/ as two files:
    skaters_<season>.csv.gz  - one row per skater per game (goals, shots, TOI, PP TOI...)
    teams_<season>.csv.gz    - one row per team per game (GF, GA, shots, PP/PK counts)

Completed seasons are only downloaded once. The current season is re-downloaded
whenever it's requested with refresh=True so the latest games are included.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()

STATS_API = "https://api.nhle.com/stats/rest/en/"
WEB_API   = "https://api-web.nhle.com/v1/"
DATA_DIR  = Path(__file__).parent / "data"

#Regular season (2) and playoffs (3)
GAME_TYPES = "gameTypeId>=2 and gameTypeId<=3"

session = requests.Session()
session.verify = False


def current_season(today=None):
    """Season key (e.g. 20262027) that is being played on a given day."""
    today = today or date.today()
    start = today.year if today.month >= 8 else today.year - 1
    return int(f"{start}{start + 1}")


def season_range(first, last):
    """All season keys from first to last inclusive, e.g. 20202021..20222023."""
    return [int(f"{y}{y + 1}") for y in range(int(str(first)[:4]), int(str(last)[:4]) + 1)]


def get_json(url, params=None):
    for attempt in range(9):
        try:
            r = session.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 8:
                raise
            #Back off if the API is rate limiting us
            time.sleep(min(2 ** attempt * 2, 60))


def fetch_report(report, season, start, end):
    """Every per-game row of a stats report between two dates (inclusive).

    The API stops returning rows after 10,000, so if a window is too big it
    gets split in half and each half is fetched separately.
    """
    params = {
        "isAggregate": "false",
        "isGame": "true",
        "start": 0,
        "limit": -1,
        "cayenneExp": f'seasonId={season} and {GAME_TYPES} and gameDate>="{start}" and gameDate<="{end}"',
    }
    j = get_json(STATS_API + report, params)
    rows = j.get("data", [])
    if len(rows) < j.get("total", 0) and start < end:
        mid = start + (end - start) / 2
        return fetch_report(report, season, start, mid) + fetch_report(report, season, mid + timedelta(days=1), end)
    return rows


def fetch_season_report(report, season, since=None):
    """A season of a report (or just the games since a date), fetched in month long windows."""
    year = int(str(season)[:4])
    #Wide enough to cover odd seasons (2020 bubble playoffs ended in late Sept)
    start, end = date(year, 9, 1), min(date(year + 1, 10, 31), date.today())
    if since:
        start = max(start, since)
    windows = []
    while start <= end:
        windows.append((start, min(start + timedelta(days=29), end)))
        start += timedelta(days=30)

    #Only a couple at a time, the API rate limits fairly quickly
    with ThreadPoolExecutor(max_workers=2) as pool:
        chunks = pool.map(lambda w: fetch_report(report, season, *w), windows)
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def download_skaters(season, since=None):
    summary = fetch_season_report("skater/summary", season, since)
    if summary.empty:
        return summary
    toi = fetch_season_report("skater/timeonice", season, since)

    summary = summary[["playerId", "skaterFullName", "positionCode", "gameId", "gameDate",
                       "teamAbbrev", "opponentTeamAbbrev", "homeRoad",
                       "goals", "assists", "ppGoals", "shots", "timeOnIcePerGame"]]
    toi = toi[["playerId", "gameId", "ppTimeOnIce", "evTimeOnIce", "shifts"]]

    df = summary.merge(toi, on=["playerId", "gameId"], how="left")
    df = df.rename(columns={"skaterFullName": "name", "teamAbbrev": "team",
                            "opponentTeamAbbrev": "opp", "timeOnIcePerGame": "toi",
                            "ppTimeOnIce": "pp_toi", "evTimeOnIce": "ev_toi"})
    return df.drop_duplicates(["playerId", "gameId"])


def download_teams(season, since=None):
    summary = fetch_season_report("team/summary", season, since)
    if summary.empty:
        return summary
    pp = fetch_season_report("team/powerplay", season, since)
    pk = fetch_season_report("team/penaltykill", season, since)

    df = summary[["teamId", "teamFullName", "gameId", "gameDate", "opponentTeamAbbrev", "homeRoad",
                  "goalsFor", "goalsAgainst", "shotsForPerGame", "shotsAgainstPerGame", "wins"]]
    df = df.merge(pp[["teamId", "gameId", "ppOpportunities", "powerPlayGoalsFor"]], on=["teamId", "gameId"], how="left")
    df = df.merge(pk[["teamId", "gameId", "timesShorthanded", "ppGoalsAgainst"]], on=["teamId", "gameId"], how="left")
    df = df.drop_duplicates(["teamId", "gameId"])

    #Team rows don't include the team's own abbreviation, but the other team's
    #row in the same game lists it as the opponent
    other = df[["gameId", "teamId", "opponentTeamAbbrev"]].rename(
        columns={"teamId": "oppId", "opponentTeamAbbrev": "team"})
    df = df.merge(other, on="gameId")
    df = df[df["teamId"] != df["oppId"]].drop(columns="oppId")

    return df.rename(columns={"opponentTeamAbbrev": "opp",
                              "shotsForPerGame": "shotsFor", "shotsAgainstPerGame": "shotsAgainst",
                              "ppOpportunities": "ppOpps", "powerPlayGoalsFor": "ppGoals",
                              "timesShorthanded": "timesSH"})


#Recent days are fetched again when topping up the cache, in case of stat corrections
REFETCH_DAYS = 2


def _read(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def cached_season(path, download, season, refresh, keys):
    """A season's game logs from the cache.

    The first time, the whole season is downloaded. After that, with refresh,
    only games from the last few cached days onwards are downloaded and added -
    at most once a day, since finished games don't change.
    """
    DATA_DIR.mkdir(exist_ok=True)
    old = _read(path)
    if path.exists() and not old.empty and not refresh:
        return old
    if path.exists() and refresh and date.fromtimestamp(path.stat().st_mtime) == date.today():
        return old

    if old.empty:
        print(f"  Downloading {path.name}...")
        new, since = download(season), None
    else:
        since = pd.to_datetime(old["gameDate"]).max().date() - timedelta(days=REFETCH_DAYS)
        print(f"  Updating {path.name} from {since}...")
        new = download(season, since)
        if new.empty:
            path.touch()    #Nothing new, but checked today
            return old
        old = old[pd.to_datetime(old["gameDate"]).dt.date < since]
    df = pd.concat([old, new], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates(keys, keep="last")
    df.to_csv(path, index=False)
    return _read(path)


def load_season(season, refresh=False):
    """(skaters, teams) DataFrames for one season, from the cache when possible."""
    skaters = cached_season(DATA_DIR / f"skaters_{season}.csv.gz", download_skaters, season, refresh,
                            ["playerId", "gameId"])
    teams = cached_season(DATA_DIR / f"teams_{season}.csv.gz", download_teams, season, refresh,
                          ["teamId", "gameId"])
    return skaters, teams


def download_goalies(season, since=None):
    df = fetch_season_report("goalie/summary", season, since)
    if df.empty:
        return df
    df = df[["playerId", "goalieFullName", "gameId", "gameDate", "teamAbbrev", "opponentTeamAbbrev",
             "gamesStarted", "shotsAgainst", "goalsAgainst", "timeOnIce"]]
    return df.rename(columns={"goalieFullName": "name", "teamAbbrev": "team", "opponentTeamAbbrev": "opp",
                              "gamesStarted": "started", "timeOnIce": "toi"}).drop_duplicates(["playerId", "gameId"])


def load_goalies(first_season, last_season=None, refresh_current=True):
    """Goalie game logs (who started, shots and goals against) for a span of seasons."""
    now = current_season()
    out = []
    for season in season_range(first_season, last_season or now):
        g = cached_season(DATA_DIR / f"goalies_{season}.csv.gz", download_goalies, season,
                          refresh_current and season == now, ["playerId", "gameId"])
        if not g.empty:
            out.append(g.assign(season=season))
    df = pd.concat(out, ignore_index=True)
    df["gameDate"] = pd.to_datetime(df["gameDate"])
    return df


def load_history(first_season, last_season=None, refresh_current=True):
    """Concatenated skater and team game logs for a span of seasons."""
    now = current_season()
    last_season = last_season or now

    skaters, teams = [], []
    for season in season_range(first_season, last_season):
        s, t = load_season(season, refresh=refresh_current and season == now)
        if not s.empty:
            skaters.append(s.assign(season=season))
            teams.append(t.assign(season=season))

    skaters = pd.concat(skaters, ignore_index=True)
    teams   = pd.concat(teams, ignore_index=True)
    for df in (skaters, teams):
        df["gameDate"] = pd.to_datetime(df["gameDate"])
        df["gameType"] = df["gameId"] // 10000 % 100
    return skaters, teams


def todays_games(day=None):
    """List of (home, away) team abbreviations playing on a given day."""
    day = (day or date.today()).isoformat()
    week = get_json(WEB_API + f"schedule/{day}")["gameWeek"]
    games = next((d["games"] for d in week if d["date"] == day), [])
    return [(g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"], g["gameType"]) for g in games
            if g["gameType"] in (2, 3)]


def current_roster(team):
    """{playerId: (name, positionCode)} for every skater on a team's current roster."""
    roster = get_json(WEB_API + f"roster/{team}/current")
    players = {}
    for p in roster.get("forwards", []) + roster.get("defensemen", []):
        players[p["id"]] = (f'{p["firstName"]["default"]} {p["lastName"]["default"]}', p["positionCode"])
    return players


TIMS_API = "https://api.hockeychallengehelper.com/api/history"
TIMS_DIR = DATA_DIR / "tims_history"
#The site's API only answers requests that look like they come from its own page
TIMS_HEADERS = {"Origin": "https://hockeychallengehelper.com", "Referer": "https://hockeychallengehelper.com/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}


def tims_history(days):
    """The real Tim's lists for past days and who scored, from hockeychallengehelper.com.

    Returns one row per player per list: gameDate, list (1-3), playerId, name, scored.
    Each finished day is cached in data/tims_history/ so it's only downloaded once.
    """
    TIMS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in sorted(set(pd.to_datetime(days).date)):
        if day >= date.today():
            continue    #Results aren't final yet
        path = TIMS_DIR / f"{day}.json"
        if not path.exists():
            try:
                r = requests.get(TIMS_API, params={"datetime": day.isoformat()}, headers=TIMS_HEADERS, timeout=30)
                r.raise_for_status()
                path.write_text(r.text, encoding="utf-8")
            except requests.RequestException as e:
                print(f"  Couldn't get Tim's lists for {day}: {e}")
                continue
            #Go easy on a small site
            time.sleep(1.5)
        for lst in json.loads(path.read_text(encoding="utf-8")).get("playerLists", []):
            for p in lst["players"]:
                rows.append({"gameDate": pd.Timestamp(day), "list": lst["id"], "playerId": p["nhlPlayerId"],
                             "name": p["fullName"], "scored": int(p["scored"])})
    return pd.DataFrame(rows, columns=["gameDate", "list", "playerId", "name", "scored"])
