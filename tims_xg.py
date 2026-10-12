"""Expected goals (xG): the chance each shot becomes a goal, from where and how it was taken.

The xG model only learns from shots before the backtest seasons (see
XG_LAST_SEASON), so the Tim's backtest never sees shot outcomes it's tested on.
From it come per-game totals for players (individual xG), teams (xG for and
against) and goalies (goals saved above expected).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

XG_FILE = Path(__file__).parent / "models" / "xg_model.joblib"
XG_LAST_SEASON = 20222023

UNBLOCKED = ["goal", "shot-on-goal", "missed-shot"]
SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around"]
XG_FEATURES = ["dist", "angle", "shot_type", "rebound", "rush", "dt_prev", "skaters_for", "skaters_against",
               "empty_net", "overtime", "prev_shot"]


def shot_features(shots):
    df = shots[["dist", "angle", "rebound", "rush", "skaters_for", "skaters_against", "empty_net",
                "prev_shot"]].copy()
    df["shot_type"] = pd.Categorical(shots["shotType"], categories=SHOT_TYPES).codes
    df["dt_prev"] = shots["dt_prev"].clip(upper=60)
    df["overtime"] = (shots["period"] > 3).astype(int)
    return df


def fit_xg(shots):
    train = shots[shots["type"].isin(UNBLOCKED) & (shots["season"] <= XG_LAST_SEASON)]
    model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=200,
                                           categorical_features=[XG_FEATURES.index("shot_type")],
                                           early_stopping=True, random_state=0)
    model.fit(shot_features(train)[XG_FEATURES], train["goal"])
    XG_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, XG_FILE)
    return model


def add_xg(shots, model=None):
    if model is None:
        model = joblib.load(XG_FILE) if XG_FILE.exists() else fit_xg(shots)
    shots = shots.copy()
    shots["xg"] = 0.0
    unblocked = shots["type"].isin(UNBLOCKED)
    shots.loc[unblocked, "xg"] = model.predict_proba(shot_features(shots[unblocked])[XG_FEATURES])[:, 1]

    #How shots are recorded changes over the years (rebounds, angles, shot
    #types), which drifts the model's overall scale. Rescale each game by the
    #league's actual goals / xG over the previous 1,000 games - only games
    #already played, so nothing leaks from the future.
    games = shots.groupby("gameId")[["goal", "xg"]].sum().sort_index()
    roll = games.rolling(1000, min_periods=200).sum().shift(1)
    ratio = (roll["goal"] / roll["xg"]).fillna(1.0)
    shots["xg"] *= shots["gameId"].map(ratio).values
    return shots


def player_games(shots):
    """Per player per game: individual xG, unblocked shot attempts, high danger chances, PP xG."""
    s = shots.dropna(subset=["shooter"]).copy()
    s["shooter"] = s["shooter"].astype(int)
    s["pp_xg"] = s["xg"].where(s["skaters_for"] > s["skaters_against"], 0)
    s["hd"] = (s["xg"] >= 0.15).astype(int)
    g = s.groupby(["shooter", "gameId"]).agg(ixg=("xg", "sum"), attempts=("type", "size"), hd=("hd", "sum"),
                                              pp_ixg=("pp_xg", "sum")).reset_index()
    return g.rename(columns={"shooter": "playerId"})


def team_games(shots):
    """Per team per game (home or road): xG for and against, empty net shots left out."""
    s = shots[~shots["empty_net"].astype(bool)]
    xg = s.pivot_table(index="gameId", columns="is_home", values="xg", aggfunc="sum", fill_value=0)
    home = pd.DataFrame({"gameId": xg.index, "homeRoad": "H", "xgf": xg.get(1, 0).values, "xga": xg.get(0, 0).values})
    road = pd.DataFrame({"gameId": xg.index, "homeRoad": "R", "xgf": xg.get(0, 0).values, "xga": xg.get(1, 0).values})
    return pd.concat([home, road], ignore_index=True)


def goalie_games(shots):
    """Per goalie per game: xG faced and goals allowed on unblocked shots."""
    s = shots[shots["type"].isin(UNBLOCKED) & shots["goalie"].notna()].copy()
    s["goalie"] = s["goalie"].astype(int)
    g = s.groupby(["goalie", "gameId"]).agg(xg_faced=("xg", "sum"), goals_allowed=("goal", "sum")).reset_index()
    return g.rename(columns={"goalie": "playerId"})
