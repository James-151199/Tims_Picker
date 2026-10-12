"""Tim's Picks helper - ranks today's Tim's Hockey Challenge players, list by
list, using the method that won the backtest in train_model.py.

    python Tims_Picker.py                  #today's Tim's lists
    python Tims_Picker.py --all            #everyone playing today
    python Tims_Picker.py --date 2026-10-14

Uses tonight's projected lines and power play units for the Tim's players
(from hockeychallengehelper.com) and the projected starting goalies (from
dailyfaceoff.com) when they're available.
"""
import argparse
import json
import re
import sys
import unicodedata
from datetime import date

import joblib
import pandas as pd
import requests

import tims_data
import tims_features
import tims_rank
from train_model import METHODS, MODEL_FILE

BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}


def get_tims_players():
    """Today's Tim's lists, with each player's projected line and PP unit."""
    j = requests.get("https://api.hockeychallengehelper.com/api/picks", headers=tims_data.TIMS_HEADERS,
                     timeout=30).json()
    rows = []
    for lst in j.get("playerLists", []):
        for p in lst["players"]:
            line = re.search(r"\d", p.get("line") or "")
            pp = re.search(r"\d", p.get("ppLine") or "")
            rows.append({"list": lst["id"], "playerId": p["nhlPlayerId"], "name": p["fullName"],
                         "line_override": int(line.group()) if line else None,
                         "pp_override": int(pp.group()) if pp else 0})
    return pd.DataFrame(rows)


def todays_skaters(day):
    """Every skater on a roster of a team playing on the day, with their matchup."""
    rows = []
    for home, away, game_type in tims_data.todays_games(day):
        for team, opp, home_road in [(home, away, "H"), (away, home, "R")]:
            for pid, (name, pos) in tims_data.current_roster(team).items():
                rows.append({"playerId": pid, "rosterName": name, "positionCode": pos, "team": team,
                             "opp": opp, "homeRoad": home_road, "gameType": game_type})
    return pd.DataFrame(rows)


def _plain(name):
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().replace(".", "").strip()


def starting_goalies(day, data):
    """{team abbreviation: (goalie id, status)} for tonight's starters.

    Projected starters come from DailyFaceoff. Any team it doesn't cover gets
    the goalie who started most of their last 10 games.
    """
    goalies, teams = data["goalies"], data["teams"]
    recent = goalies[goalies["season"] >= goalies["season"].max() - 10001]
    by_name = {(_plain(n), t): pid for n, t, pid in recent[["name", "team", "playerId"]].itertuples(index=False)}
    by_name.update({_plain(n): pid for n, pid in recent[["name", "playerId"]].itertuples(index=False)})
    abbrev = dict(teams.drop_duplicates("teamFullName", keep="last")[["teamFullName", "team"]].values)

    out = {}
    try:
        html = requests.get(f"https://www.dailyfaceoff.com/starting-goalies/{day}", headers=BROWSER, timeout=30).text
        games = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S).group(1))
        for g in games["props"]["pageProps"]["data"]:
            for side in ("home", "away"):
                team = abbrev.get(g[f"{side}TeamName"])
                name = _plain(g.get(f"{side}GoalieName") or "")
                pid = by_name.get((name, team)) or by_name.get(name)
                if team and pid:
                    out[team] = (pid, g.get(f"{side}NewsStrengthName") or "Unconfirmed")
    except (requests.RequestException, AttributeError, KeyError, ValueError) as e:
        print(f"  Couldn't get projected starting goalies ({e}), using recent starters")

    last10 = goalies[goalies["started"] == 1].sort_values("gameDate").groupby("team").tail(10)
    for team, pid in last10.groupby("team")["playerId"].agg(lambda s: s.value_counts().idxmax()).items():
        out.setdefault(team, (pid, "Recent starter"))
    return out


def predict(targets, day, by):
    bundle = joblib.load(MODEL_FILE)
    season = tims_data.current_season(day)
    #Two seasons back covers every feature (82 game windows + last season)
    data = tims_features.load_all(season - 20002)

    starters = starting_goalies(day, data)
    targets = targets.assign(gameDate=pd.Timestamp(day), season=season)
    targets["opp_goalie"] = targets["opp"].map(lambda t: starters.get(t, (None, ""))[0])
    targets["goalie_status"] = targets["opp"].map(lambda t: starters.get(t, (None, ""))[1])
    names = {pid: n for pid, n in data["goalies"][["playerId", "name"]].itertuples(index=False)}
    targets["goalie_name"] = targets["opp_goalie"].map(names)

    df = tims_features.build_features(targets, data)
    #Skip players with no NHL history (call-ups making their debut)
    df = df[df["career_gp"].notna()].copy()
    feats = bundle["features"]
    df["gbm"] = bundle["gbm"].predict_proba(df[feats])[:, 1]
    df["lambdarank"] = bundle["lambdarank"].predict(df[feats])
    df["bayes"] = bundle["bayes"].predict(df)
    df = tims_rank.add_blend(df, by=by)
    return df, bundle["deploy"]


def show(df, deploy, title):
    print(f"\n{title}   (ranked by {METHODS[deploy]})\n")
    print(f"{'#':>2} {'Player':23}{'Team':>5}{'Opp':>6}{'P(goal)':>9}{'G':>4}{'GP':>4}{'xG/GP':>7}"
          f"{'Line':>6}{'PP':>4}   Opposing goalie")
    for i, (_, p) in enumerate(df.sort_values(deploy, ascending=False).iterrows(), start=1):
        line = f"{'D' if p['positionCode'] == 'D' else 'F'}{p['line']:.0f}" if pd.notna(p["line"]) else "-"
        pp = f"PP{p['pp_unit']:.0f}" if p["pp_unit"] else "-"
        goalie = f"{p['goalie_name']} ({p['goalie_status']})" if pd.notna(p.get("goalie_name")) else "unknown"
        print(f"{i:>2} {p['name'][:22]:23}{p['team']:>5}{('@' if p['homeRoad'] == 'R' else '') + p['opp']:>6}"
              f"{p['gbm']:>9.1%}{p['season_goals']:>4.0f}{p['season_gp']:>4.0f}{p['ixg_l20']:>7.2f}"
              f"{line:>6}{pp:>4}   {goalie}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--all", action="store_true", help="rank every skater playing, not just the Tim's players")
    parser.add_argument("--top", type=int, default=30, help="how many players to show with --all")
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        sys.exit("No model found - run 'python train_model.py' first.")
    sys.stdout.reconfigure(encoding="utf-8")

    print("Getting today's games and rosters...")
    playing = todays_skaters(args.date)
    if playing.empty:
        sys.exit(f"No NHL games on {args.date}.")

    if args.all:
        targets, by = playing.assign(name=playing["rosterName"], list=0), ("gameDate",)
    else:
        print("Getting Tim's lists...")
        tims = get_tims_players()
        if tims.empty:
            sys.exit("Couldn't find any Tim's lists - try --all to rank everyone playing.")
        missing = tims[~tims["playerId"].isin(playing["playerId"])]
        for name in missing["name"]:
            print(f"  {name} isn't on a roster playing today, skipping")
        targets, by = tims.merge(playing, on="playerId"), ("gameDate", "list")

    print("Loading stats and predicting...")
    df, deploy = predict(targets, args.date, by)

    if args.all:
        show(df.nlargest(args.top, deploy), deploy, f"TOP {args.top} - {args.date}")
    else:
        for l in sorted(df["list"].unique()):
            show(df[df["list"] == l], deploy, f"LIST #{l}")


if __name__ == "__main__":
    main()
