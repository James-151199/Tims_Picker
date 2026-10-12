"""Train the Tim's Picks models and backtest them on the real Tim's lists.

    python train_model.py                       #download/refresh data, backtest, train, save
    python train_model.py --backtest-seasons 2  #backtest fewer seasons (faster)

Scoring metric: the list hit rate - for each real Tim's list (from
hockeychallengehelper.com), did the player a method ranks first score?

Methods compared, each trained only on seasons before the one it's tested on:
  gbm         gradient boosted trees predicting P(goal) for every player-game
  lambdarank  LightGBM LambdaRank, trained to put scorers at the top of each list
  bayes       Bayesian player-talent model (tims_bayes.py)
  blend       50/50 blend of the GBM's and goals per game's rankings within a list
  season_gpg  goals per game this season (the simple baseline)

The method with the best hit rate across the backtest seasons is what
Tims_Picker.py ranks by. Figures and metrics are saved to reports/.
"""
import argparse
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import tims_data
import tims_features
import tims_rank
import tims_report
from tims_bayes import BayesTalent
from tims_features import FEATURES

MODEL_FILE = Path(__file__).parent / "models" / "tims_model.joblib"

METHODS = {"gbm": "Gradient boosting", "lambdarank": "LambdaRank", "bayes": "Bayesian talent",
           "blend": "GBM + goals/game blend", "season_gpg": "Goals per game"}
MODEL_METHODS = ["gbm", "lambdarank", "bayes", "blend"]


def make_model(max_iter=1000, early_stopping=True):
    """Gradient boosted trees. With early_stopping it stops adding trees once a
    held out 10% stops improving; the final model reuses that tree count."""
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=500,
        l2_regularization=1.0,
        early_stopping=early_stopping,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=0,
    )


def backtest(df, seasons, real_lists):
    """Train every method on the seasons before each backtest season and score
    them on that season's real Tim's lists."""
    all_rows, last = [], {}
    for season in seasons:
        label = tims_report.season_label(season)
        train = df[df["season"] < season]
        test = df[df["season"] == season].copy()

        print(f"  {label}: gradient boosting...", flush=True)
        gbm = make_model().fit(train[FEATURES], train["scored"])
        test["gbm"] = gbm.predict_proba(test[FEATURES])[:, 1]

        print(f"  {label}: LambdaRank...", flush=True)
        ranker = tims_rank.fit_lambdarank(train, real_lists)
        test["lambdarank"] = ranker.predict(test[FEATURES])

        print(f"  {label}: Bayesian talent model...", flush=True)
        bayes = BayesTalent().fit(df[df["season"] == season - 10001])
        test["bayes"] = bayes.predict(test)

        rows = real_lists[["gameDate", "list", "playerId"]].merge(test, on=["gameDate", "playerId"])
        rows = tims_rank.add_blend(rows)
        all_rows.append(rows)
        last = {"gbm": gbm, "bayes": bayes, "test": test, "rows": rows, "season": season}

    rows = pd.concat(all_rows, ignore_index=True)
    return tims_rank.evaluate_lists(rows, list(METHODS)), last


def print_results(results):
    s = tims_rank.summary(results, list(METHODS))
    print(f"\n{'':26}{'hit rate':>10}{'top 3':>8}{'list AUC':>10}   vs goals/game (95% interval)")
    for m in METHODS:
        line = f"{METHODS[m]:26}{s.loc[m, 'hit']:>10.1%}{s.loc[m, 'top3']:>8.1%}{s.loc[m, 'auc']:>10.3f}"
        if m != "season_gpg":
            d, lo, hi = tims_rank.bootstrap_vs(results, m, "season_gpg")
            line += f"   {d:+.1%} ({lo:+.1%} to {hi:+.1%})"
        print(line)
    print(f"{'Random':26}{results['random'].mean():>10.1%}")
    print(f"({len(results):,} real Tim's lists. Hit rate = list's top-ranked player scored; "
          "top 3 = any of the top 3 scored)")
    print("\nBy season (hit rate):")
    by = results.groupby("season")[[f"{m}_hit" for m in METHODS]].mean()
    by.columns = list(METHODS)
    print(by.rename(index=tims_report.season_label).to_string(float_format="{:.1%}".format))
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=2010, help="first season start year to use")
    parser.add_argument("--backtest-seasons", type=int, default=3,
                        help="how many recent completed seasons to backtest on the real Tim's lists")
    args = parser.parse_args()

    first = int(f"{args.first}{args.first + 1}")
    print("Loading game logs, play-by-play and goalie data...")
    data = tims_features.load_all(first)

    print("Building features...")
    #The first season only provides history for the following ones
    df = tims_features.training_table(data, first_season=first + 10001)
    now = tims_data.current_season()
    completed = sorted(s for s in df["season"].unique() if s < now)
    seasons = completed[-args.backtest_seasons:]

    print("Getting real Tim's lists...")
    real_lists = tims_data.tims_history(df.loc[df["season"] >= seasons[0], "gameDate"].unique())

    print(f"\nBacktesting {len(seasons)} season(s) on the real Tim's lists...")
    results, last = backtest(df, seasons, real_lists)
    s = print_results(results)
    deploy = s.loc[MODEL_METHODS, "hit"].idxmax()
    print(f"\nBest list hit rate: {METHODS[deploy]} - Tims_Picker.py will rank by it")
    print(f"Bayesian model ({tims_report.season_label(last['season'])}): {last['bayes'].describe()}")

    tims_report.save_figures(df, results, last, METHODS, deploy)

    print(f"\nTraining final models on all {len(df):,} games...")
    gbm = make_model(max_iter=last["gbm"].n_iter_, early_stopping=False).fit(df[FEATURES], df["scored"])
    ranker = tims_rank.fit_lambdarank(df, real_lists)
    bayes = BayesTalent().fit(df[df["season"] == completed[-1]])

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"gbm": gbm, "lambdarank": ranker, "bayes": bayes, "deploy": deploy, "features": FEATURES,
                 "first_season": first, "trained_on": date.today().isoformat(), "games": len(df),
                 "backtest": s.to_dict()}, MODEL_FILE)
    print(f"Saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
