"""Turn game logs into model features.

Every feature for a game is built only from games played *before* that day, so
the model never sees information it wouldn't have had when making a pick.

The same code builds features for historical games (training) and for
today's games (predicting), by:
  1. Building a running "state" for every player and team after each game
     (rolling goals, shots, TOI, PP%, etc).
  2. For each target game, looking up the latest state from before that day.
"""
import numpy as np
import pandas as pd

import tims_data
import tims_shots
import tims_xg

PLAYER_WINDOWS = [5, 10, 20, 40, 82]
TEAM_WINDOWS   = [10, 40]
#Half-lives (in games) for the Bayesian model's decayed totals: scoring chance
#rate adapts quickly, finishing skill slowly
BAYES_RATE_HL   = [20, 50]
BAYES_FINISH_HL = [100, 300]

FEATURES = [
    #Player form, recent to long term
    *[f"goals_l{n}" for n in PLAYER_WINDOWS],
    *[f"shots_l{n}" for n in PLAYER_WINDOWS],
    "toi_l5", "toi_l20", "pp_toi_l5", "pp_toi_l20", "ev_toi_l20",
    "ppgoals_l82", "points_l20",
    "goals_ewm", "shots_ewm", "toi_ewm",
    "sh_pct_l82", "exp_goals",
    #Season so far and last season
    "season_gp", "season_gpg", "season_spg", "prev_gpg", "prev_spg", "prev_gp",
    "career_gp", "days_rest", "is_defense",
    #Game context
    "home", "playoffs",
    #Player's team
    "team_gf_l10", "team_gf_l40", "team_sf_l40", "team_pp_pct_l40", "team_pp_opps_l40",
    #Opponent
    "opp_ga_l10", "opp_ga_l40", "opp_sa_l40", "opp_pk_pct_l40", "opp_times_sh_l40",
    "opp_sv_pct_l40", "opp_win_pct_l40",
    #Shot quality from MoneyPuck shot data (our own expected goals model)
    "ixg_l10", "ixg_l20", "ixg_l40", "ixg_l82", "ixg_ewm", "attempts_l10", "hd_l20", "pp_ixg_l20",
    "finishing_l82",
    #Deployment: line (forwards 1-4, defence pairs 1-3) and power play unit (0 = none)
    "line", "pp_unit",
    #Team and opponent shot quality, opposing starting goalie
    "team_xgf_l20", "opp_xga_l20", "goalie_gsax_l40", "goalie_sv_l40",
]


def load_all(first_season, refresh_current=True):
    """Every data source the features use, merged per game.

    Returns {"skaters", "teams", "goalies"}: the game logs from tims_data with
    play-by-play expected goals added for players, teams and goalies.
    """
    skaters, teams = tims_data.load_history(first_season, refresh_current=refresh_current)
    goalies = tims_data.load_goalies(first_season, refresh_current=refresh_current)
    shots = tims_xg.add_xg(tims_shots.load_shots(first_season))

    skaters = skaters.merge(tims_xg.player_games(shots), on=["playerId", "gameId"], how="left")
    skaters[["ixg", "attempts", "hd", "pp_ixg"]] = skaters[["ixg", "attempts", "hd", "pp_ixg"]].fillna(0)
    teams = teams.merge(tims_xg.team_games(shots), on=["gameId", "homeRoad"], how="left")
    goalies = goalies.merge(tims_xg.goalie_games(shots), on=["playerId", "gameId"], how="left")
    goalies[["xg_faced", "goals_allowed"]] = goalies[["xg_faced", "goals_allowed"]].fillna(0)
    return {"skaters": skaters, "teams": teams, "goalies": goalies}


def deployment(skaters):
    """Line and power play unit each player had in each game, from ice time.

    Forwards are ranked by even strength minutes within their team (top 3 =
    line 1...), defencemen likewise in pairs, and the top 5 / next 5 players
    by power play minutes make up PP units 1 and 2.
    """
    df = skaters[["playerId", "gameId", "team", "positionCode", "ev_toi", "pp_toi"]].copy()
    df["is_d"] = df["positionCode"] == "D"
    ev_rank = df.groupby(["gameId", "team", "is_d"])["ev_toi"].rank(ascending=False, method="first")
    df["line"] = np.where(df["is_d"], np.ceil(ev_rank / 2).clip(upper=3), np.ceil(ev_rank / 3).clip(upper=4))
    pp_rank = df.groupby(["gameId", "team"])["pp_toi"].rank(ascending=False, method="first")
    df["pp_unit"] = np.where(df["pp_toi"].fillna(0) <= 0, 0, np.where(pp_rank <= 5, 1, np.where(pp_rank <= 10, 2, 0)))
    return df[["playerId", "gameId", "line", "pp_unit"]]


def _decayed_sum(df, col, half_life):
    """Sum of a column over each player's history, older games counting for less."""
    d = 0.5 ** (1 / half_life)
    idx = df.groupby("key").cumcount()
    mean = df.groupby("key")[col].transform(lambda x: x.ewm(alpha=1 - d, adjust=True).mean())
    return mean * (1 - d ** (idx + 1)) / (1 - d)


def _rolling_sum(df, col, n):
    """Sum of the last n rows of a column within each player/team's history."""
    cs = df.groupby("key")[col].cumsum()
    return cs - cs.groupby(df["key"]).shift(n).fillna(0)


def player_state(skaters):
    """Each player's rolling stats as of the end of each of their games."""
    df = skaters.sort_values(["playerId", "gameDate"]).reset_index(drop=True)
    df["key"] = df["playerId"]
    df["points"] = df["goals"] + df["assists"]
    for col in ["toi", "pp_toi", "ev_toi"]:
        df[col] = df[col].fillna(0) / 60     #seconds -> minutes

    state = df[["playerId", "gameDate"]].copy()
    state["state_season"] = df["season"]
    games = df.groupby("key").cumcount() + 1
    state["career_gp"] = games

    for n in PLAYER_WINDOWS:
        played = np.minimum(games, n)
        state[f"goals_l{n}"] = _rolling_sum(df, "goals", n) / played
        state[f"shots_l{n}"] = _rolling_sum(df, "shots", n) / played
    for col, n in [("toi", 5), ("toi", 20), ("pp_toi", 5), ("pp_toi", 20), ("ev_toi", 20),
                   ("ppGoals", 82), ("points", 20)]:
        state[f"{col.lower()}_l{n}"] = _rolling_sum(df, col, n) / np.minimum(games, n)

    for col in ["goals", "shots", "toi"]:
        state[f"{col}_ewm"] = df.groupby("key")[col].transform(lambda x: x.ewm(halflife=15).mean())

    #Shooting % over the last 82 games, pulled towards the league average
    #(~10%) so a 3-for-5 hot streak doesn't look like a 60% shooter
    g82 = _rolling_sum(df, "goals", 82)
    s82 = _rolling_sum(df, "shots", 82)
    state["sh_pct_l82"] = (g82 + 5) / (s82 + 50)
    state["exp_goals"]  = state["shots_l20"] * state["sh_pct_l82"]

    #Shot quality
    for n in [10, 20, 40, 82]:
        state[f"ixg_l{n}"] = _rolling_sum(df, "ixg", n) / np.minimum(games, n)
    state["ixg_ewm"]      = df.groupby("key")["ixg"].transform(lambda x: x.ewm(halflife=15).mean())
    state["attempts_l10"] = _rolling_sum(df, "attempts", 10) / np.minimum(games, 10)
    state["hd_l20"]       = _rolling_sum(df, "hd", 20) / np.minimum(games, 20)
    state["pp_ixg_l20"]   = _rolling_sum(df, "pp_ixg", 20) / np.minimum(games, 20)
    #Goals per expected goal, pulled towards 1 (average finishing)
    state["finishing_l82"] = (g82 + 10) / (_rolling_sum(df, "ixg", 82) + 10)

    #Line and PP unit in their most recent game
    dep = df[["playerId", "gameId"]].merge(deployment(df), on=["playerId", "gameId"], how="left")
    state["line"], state["pp_unit"] = dep["line"].values, dep["pp_unit"].values

    #Decayed totals for the Bayesian talent model (tims_bayes.py)
    df["one"] = 1.0
    for hl in BAYES_RATE_HL:
        state[f"b_n_{hl}"] = _decayed_sum(df, "one", hl)
        state[f"b_x_{hl}"] = _decayed_sum(df, "ixg", hl)
    for hl in BAYES_FINISH_HL:
        state[f"b_g_{hl}"]  = _decayed_sum(df, "goals", hl)
        state[f"b_xf_{hl}"] = _decayed_sum(df, "ixg", hl)

    by_season = df.groupby(["playerId", "season"])
    state["season_gp"]    = by_season.cumcount() + 1
    state["season_goals"] = by_season["goals"].cumsum()
    state["season_shots"] = by_season["shots"].cumsum()
    return state


def previous_seasons(skaters):
    """Per player per season totals, keyed to the *following* season."""
    prev = skaters.groupby(["playerId", "season"]).agg(
        prev_gp=("goals", "size"), prev_goals=("goals", "sum"), prev_shots=("shots", "sum")).reset_index()
    prev["prev_gpg"] = prev["prev_goals"] / prev["prev_gp"]
    prev["prev_spg"] = prev["prev_shots"] / prev["prev_gp"]
    prev["season"] = prev["season"] + 10001       #20242025 -> 20252026
    return prev[["playerId", "season", "prev_gp", "prev_gpg", "prev_spg"]]


def team_state(teams):
    """Each team's rolling stats as of the end of each of their games."""
    df = teams.sort_values(["team", "gameDate"]).reset_index(drop=True)
    df["key"] = df["team"]
    df[["ppOpps", "ppGoals", "timesSH", "ppGoalsAgainst"]] = \
        df[["ppOpps", "ppGoals", "timesSH", "ppGoalsAgainst"]].fillna(0)

    state = df[["team", "gameDate"]].copy()
    games = df.groupby("key").cumcount() + 1
    for n in TEAM_WINDOWS:
        played = np.minimum(games, n)
        s = {c: _rolling_sum(df, c, n) for c in
             ["goalsFor", "goalsAgainst", "shotsFor", "shotsAgainst", "ppOpps", "ppGoals",
              "timesSH", "ppGoalsAgainst", "wins"]}
        state[f"gf_l{n}"]       = s["goalsFor"] / played
        state[f"ga_l{n}"]       = s["goalsAgainst"] / played
        state[f"sf_l{n}"]       = s["shotsFor"] / played
        state[f"sa_l{n}"]       = s["shotsAgainst"] / played
        state[f"pp_pct_l{n}"]   = s["ppGoals"] / s["ppOpps"].replace(0, np.nan)
        state[f"pp_opps_l{n}"]  = s["ppOpps"] / played
        state[f"pk_pct_l{n}"]   = 1 - s["ppGoalsAgainst"] / s["timesSH"].replace(0, np.nan)
        state[f"times_sh_l{n}"] = s["timesSH"] / played
        state[f"sv_pct_l{n}"]   = 1 - s["goalsAgainst"] / s["shotsAgainst"].replace(0, np.nan)
        state[f"win_pct_l{n}"]  = s["wins"] / played
    for col in ["xgf", "xga"]:
        df[col] = df[col].fillna(0)
        state[f"{col}_l20"] = _rolling_sum(df, col, 20) / np.minimum(games, 20)
    return state


def goalie_state(goalies):
    """Each goalie's form as of the end of each game: goals saved above expected per
    expected goal faced, and save %, over their last 40 games (pulled to average)."""
    df = goalies.sort_values(["playerId", "gameDate"]).reset_index(drop=True)
    df["key"] = df["playerId"]
    state = df[["playerId", "gameDate"]].copy()
    xg = _rolling_sum(df, "xg_faced", 40)
    ga = _rolling_sum(df, "goals_allowed", 40)
    state["gsax_l40"] = (xg - ga) / (xg + 20)
    shots, goals = _rolling_sum(df, "shotsAgainst", 40), _rolling_sum(df, "goalsAgainst", 40)
    state["sv_l40"] = 1 - (goals + 0.1 * 300) / (shots + 300)
    return state


def starters(goalies):
    """The goalie who started each game for each team."""
    s = goalies[goalies["started"] == 1].sort_values("toi", ascending=False)
    return s.drop_duplicates(["gameId", "team"])[["gameId", "team", "playerId"]]


def _latest_before(targets, state, by, prefix=""):
    """Attach the most recent state strictly before each target's game date."""
    state = state.sort_values("gameDate").rename(
        columns={c: prefix + c for c in state.columns if c not in (by, "gameDate")})
    state = state.rename(columns={"gameDate": prefix + "stateDate"})
    targets = targets.sort_values("gameDate")
    return pd.merge_asof(targets, state, left_on="gameDate", right_on=prefix + "stateDate",
                         by=by, allow_exact_matches=False)


def build_features(targets, data):
    """Add the FEATURES columns to a table of upcoming or past games.

    targets needs: playerId, gameDate, season, team, opp, homeRoad, positionCode,
    gameType and opp_goalie (the opposing starter's id, if known). Optional
    line_override / pp_override columns (tonight's projected lines) replace the
    line and PP unit from the player's last game.
    """
    skaters, teams = data["skaters"], data["teams"]
    targets = targets.copy()
    targets["gameDate"] = pd.to_datetime(targets["gameDate"]).astype("datetime64[ns]")

    out = _latest_before(targets, player_state(skaters), "playerId")
    out = out.merge(previous_seasons(skaters), on=["playerId", "season"], how="left")

    #Season-to-date stats only count if the last game was in this season
    same_season = out["state_season"] == out["season"]
    for c in ["season_gp", "season_goals", "season_shots"]:
        out[c] = out[c].where(same_season, 0)
    #Small-sample season rates get pulled towards last season's rate
    prior_gpg = out["prev_gpg"].fillna(out["goals_l82"]).fillna(0.1)
    prior_spg = out["prev_spg"].fillna(out["shots_l82"]).fillna(1.5)
    out["season_gpg"] = (out["season_goals"] + 10 * prior_gpg) / (out["season_gp"] + 10)
    out["season_spg"] = (out["season_shots"] + 10 * prior_spg) / (out["season_gp"] + 10)

    out["days_rest"]  = (out["gameDate"] - out["stateDate"]).dt.days.clip(upper=30)
    out["is_defense"] = (out["positionCode"] == "D").astype(int)
    out["home"]       = (out["homeRoad"] == "H").astype(int)
    out["playoffs"]   = (out["gameType"] == 3).astype(int)

    for col, override in [("line", "line_override"), ("pp_unit", "pp_override")]:
        if override in out:
            out[col] = out[override].fillna(out[col])

    tstate = team_state(teams)
    out = _latest_before(out, tstate, "team", prefix="team_")
    out = _latest_before(out, tstate.rename(columns={"team": "opp"}), "opp", prefix="opp_")

    #Opposing starting goalie's form (left blank when the starter isn't known)
    gstate = goalie_state(data["goalies"]).rename(columns={"playerId": "opp_goalie"})
    if "opp_goalie" not in out:
        out["opp_goalie"] = np.nan
    known = out["opp_goalie"].notna()
    has = _latest_before(out[known].astype({"opp_goalie": "int64"}), gstate, "opp_goalie", prefix="goalie_")
    out = pd.concat([has, out[~known]], ignore_index=True)

    return out.reset_index(drop=True)


def training_table(data, first_season=None):
    """Every historical skater-game with its features and whether they scored."""
    targets = data["skaters"][["playerId", "name", "gameDate", "season", "team", "opp", "homeRoad",
                               "positionCode", "gameType", "gameId", "goals"]]
    if first_season:
        targets = targets[targets["season"] >= first_season]
    #The opposing team's actual starting goalie
    opp_start = starters(data["goalies"]).rename(columns={"team": "opp", "playerId": "opp_goalie"})
    targets = targets.merge(opp_start, on=["gameId", "opp"], how="left")
    df = build_features(targets, data)
    df["scored"] = (df["goals"] > 0).astype(int)
    return df
