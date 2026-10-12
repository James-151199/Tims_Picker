"""Ranking within Tim's lists: the scoring metric, LambdaRank and the blend.

The metric that matters for Tim's is the list hit rate: in each list, did the
player a method ranks first score? Everything here is measured per list.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from tims_features import FEATURES

LIST_SIZE = 33      #Real Tim's lists average about 33 players
BLEND_WEIGHT = 0.5  #Share of the GBM in the GBM / goals-per-game blend


def synthetic_lists(df):
    """Tim's-style lists for days without real ones: each day's players ranked
    by goals per game, split into three lists of 33 (stars first)."""
    df = df.sort_values(["gameDate", "season_gpg"], ascending=[True, False])
    pos = df.groupby("gameDate").cumcount()
    size = np.minimum(LIST_SIZE, df.groupby("gameDate")["playerId"].transform("size") // 3)
    df = df[pos < 3 * size].copy()
    df["list"] = (pos[pos < 3 * size] // size[pos < 3 * size]) + 1
    return df


def list_rows(df, real_lists):
    """Feature rows grouped into lists: real Tim's lists where we have them,
    synthetic ones for every other day."""
    real = real_lists[["gameDate", "list", "playerId"]].merge(df, on=["gameDate", "playerId"])
    other = synthetic_lists(df[~df["gameDate"].isin(real_lists["gameDate"].unique())])
    return pd.concat([real, other], ignore_index=True).sort_values(["gameDate", "list"]).reset_index(drop=True)


def fit_lambdarank(train, real_lists):
    """LightGBM LambdaRank: learns to put scorers at the top of each list.

    The last training season is held out to choose the number of trees.
    """
    rows = list_rows(train, real_lists)
    last = rows["season"].max()
    fit, val = rows[rows["season"] < last], rows[rows["season"] == last]
    group = lambda d: d.groupby(["gameDate", "list"], sort=False).size().values
    params = dict(objective="lambdarank", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  lambdarank_truncation_level=10, n_estimators=1500, verbose=-1, random_state=0)
    model = lgb.LGBMRanker(**params)
    model.fit(fit[FEATURES], fit["scored"], group=group(fit), eval_set=[(val[FEATURES], val["scored"])],
              eval_group=[group(val)], eval_at=[1, 3], callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    best = model.best_iteration_ or params["n_estimators"]
    #Refit on every training season with the chosen number of trees
    final = lgb.LGBMRanker(**{**params, "n_estimators": best})
    final.fit(rows[FEATURES], rows["scored"], group=group(rows))
    return final


def add_blend(rows, by=("gameDate", "list")):
    """Blend = average of each player's percentile rank within the list by the GBM
    and by goals per game."""
    g = rows.groupby(list(by))
    rows["blend"] = BLEND_WEIGHT * g["gbm"].rank(pct=True) + (1 - BLEND_WEIGHT) * g["season_gpg"].rank(pct=True)
    return rows


def evaluate_lists(rows, methods, max_k=5):
    """Per list, for each method: did its 1st..max_k-th choice score, and its
    within-list AUC (chance a scorer is ranked above a non-scorer).

    rows: one row per player per list with `scored` and a score column per method.
    Returns one row per list.
    """
    keys = ["gameDate", "list"]
    g = rows.groupby(keys)
    out = g["scored"].agg(["mean", "sum", "size"]).rename(columns={"mean": "random", "sum": "scorers", "size": "n"})
    for m in methods:
        rank = g[m].rank(ascending=False, method="first")
        for k in range(1, max_k + 1):
            out[f"{m}_k{k}"] = rows["scored"].where(rank == k).groupby([rows[c] for c in keys]).max()
        out[f"{m}_hit"] = out[f"{m}_k1"]
        #Within-list AUC from the ranks of the scorers
        asc = g[m].rank(method="average")
        pos_rank = asc.where(rows["scored"] == 1).groupby([rows[c] for c in keys]).sum()
        npos, nneg = out["scorers"], out["n"] - out["scorers"]
        out[f"{m}_auc"] = (pos_rank - npos * (npos + 1) / 2) / (npos * nneg)
    out = out.reset_index()
    out["season"] = rows.groupby(keys)["season"].first().values
    return out


def summary(results, methods):
    """Hit rate, top-3 hit rate and within-list AUC per method."""
    rows = []
    for m in methods:
        top3 = results[[f"{m}_k{k}" for k in (1, 2, 3)]].max(axis=1)
        rows.append({"method": m, "hit": results[f"{m}_hit"].mean(), "top3": top3.mean(),
                     "auc": results[f"{m}_auc"].mean()})
    return pd.DataFrame(rows).set_index("method")


def bootstrap_vs(results, method, baseline, n=2000, seed=0):
    """Difference in hit rate (method - baseline) with a 95% interval, resampling lists."""
    d = (results[f"{method}_hit"] - results[f"{baseline}_hit"]).values
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
