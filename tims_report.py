"""Figures for the Tim's Picks backtest, saved as PNGs in reports/.

Called by train_model.py after the backtest on the real Tim's lists, so the
figures are regenerated every time the models are retrained. Everything is
measured the way Tim's is played: within each list, did the top-ranked
player score?
"""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")      #Draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

import tims_rank
from tims_features import FEATURES

REPORT_DIR = Path(__file__).parent / "reports"

INK    = "#0f1b2d"
MUTED  = "#586577"
RANDOM = "#9aa4b1"
#Fixed colour per method, so it means the same thing in every figure
COLORS = {"gbm": "#2a78d6", "season_gpg": "#eb6834", "lambdarank": "#1baf7a", "bayes": "#eda100",
          "blend": "#e87ba4", "random": RANDOM}

LABELS = {
    "is_defense": "Defenceman or forward", "exp_goals": "Shots x shooting %", "goals_l82": "Goals / game, last 82",
    "toi_l5": "Ice time, last 5", "pp_toi_l5": "PP ice time, last 5", "season_spg": "Shots / game, season",
    "prev_spg": "Shots / game, last season", "career_gp": "Career games", "points_l20": "Points / game, last 20",
    "season_gpg": "Goals / game, season", "prev_gpg": "Goals / game, last season",
    "pp_toi_l20": "PP ice time, last 20", "opp_sa_l40": "Opponent shots against, last 40",
    "sh_pct_l82": "Shooting %, last 82", "shots_l82": "Shots / game, last 82", "shots_l5": "Shots / game, last 5",
    "opp_ga_l40": "Opponent goals against, last 40", "days_rest": "Days since last game",
    "ixg_l10": "xG / game, last 10", "ixg_l20": "xG / game, last 20", "ixg_l40": "xG / game, last 40",
    "ixg_l82": "xG / game, last 82", "ixg_ewm": "xG / game, weighted recent", "attempts_l10": "Shot attempts, last 10",
    "hd_l20": "High-danger chances, last 20", "pp_ixg_l20": "PP xG / game, last 20",
    "finishing_l82": "Finishing (goals / xG), last 82", "line": "Line / pair", "pp_unit": "Power play unit",
    "team_xgf_l20": "Team xG for, last 20", "opp_xga_l20": "Opponent xG against, last 20",
    "goalie_gsax_l40": "Opposing goalie, saves above expected", "goalie_sv_l40": "Opposing goalie save %",
    "goals_ewm": "Goals, weighted recent", "shots_ewm": "Shots, weighted recent", "toi_ewm": "Ice time, weighted recent",
    "opp_win_pct_l40": "Opponent win %, last 40", "toi_l20": "Ice time, last 20", "home": "Home game",
    "ev_toi_l20": "Even strength ice time, last 20", "goals_l5": "Goals / game, last 5", "goals_l10": "Goals / game, last 10",
    "goals_l20": "Goals / game, last 20", "goals_l40": "Goals / game, last 40", "shots_l10": "Shots / game, last 10",
    "shots_l20": "Shots / game, last 20", "shots_l40": "Shots / game, last 40", "ppgoals_l82": "PP goals / game, last 82",
    "season_gp": "Games played, season", "prev_gp": "Games played, last season", "playoffs": "Playoff game",
    "team_gf_l10": "Team goals for, last 10", "team_gf_l40": "Team goals for, last 40",
    "team_sf_l40": "Team shots for, last 40", "team_pp_pct_l40": "Team PP %, last 40",
    "team_pp_opps_l40": "Team PP chances, last 40", "opp_ga_l10": "Opponent goals against, last 10",
    "opp_pk_pct_l40": "Opponent PK %, last 40", "opp_times_sh_l40": "Opponent penalties, last 40",
    "opp_sv_pct_l40": "Opponent save %, last 40",
}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9d0d8", "axes.labelcolor": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e9edf1", "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "lines.linewidth": 2,
})


def pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def season_label(s):
    return f"{str(s)[2:4]}-{str(s)[6:]}"


def data_by_season(df, path):
    by = df.groupby("season").agg(rows=("scored", "size"), rate=("scored", "mean"))
    labels = [season_label(s) for s in by.index]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.bar(labels, by["rows"], color=COLORS["gbm"], width=0.6)
    a.set_title("Player-games per season")
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    a.tick_params(axis="x", rotation=60)
    a.grid(axis="x", visible=False)
    done = by.iloc[:-1]     #The current season is too short for a meaningful rate
    b.plot([season_label(s) for s in done.index], done["rate"], color=COLORS["gbm"], marker="o", markersize=4)
    b.set_title("Share of player-games with a goal")
    b.tick_params(axis="x", rotation=60)
    pct(b, decimals=1)
    fig.savefig(path)
    plt.close(fig)


def ranking_curve(gbm, rows, season, path):
    """List hit rate and within-list AUC as trees are added to the GBM."""
    steps, hits, aucs = [], [], []
    r = rows[["gameDate", "list", "season", "scored"]].copy()
    for i, proba in enumerate(gbm.staged_predict_proba(rows[FEATURES]), start=1):
        if i % 5 and i != gbm.n_iter_:
            continue
        r["gbm"] = proba[:, 1]
        res = tims_rank.evaluate_lists(r, ["gbm"], max_k=1)
        steps.append(i); hits.append(res["gbm_hit"].mean()); aucs.append(res["gbm_auc"].mean())
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.plot(steps, hits, color=COLORS["gbm"])
    a.set_title(f"List hit rate as trees are added ({season_label(season)})")
    a.set_xlabel("trees")
    pct(a, decimals=1)
    b.plot(steps, aucs, color=COLORS["gbm"])
    b.set_title("Within-list AUC as trees are added")
    b.set_xlabel("trees")
    for ax, ys in [(a, hits), (b, aucs)]:
        ax.annotate(f"{ys[-1]:.3f}" if ax is b else f"{ys[-1]:.1%}", (steps[-1], ys[-1]),
                    textcoords="offset points", xytext=(-4, 8), ha="right", fontsize=9, color=INK, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    boots = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(axis=1)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def methods_hit(results, methods, deploy, path):
    keys = list(methods) + ["random"]
    names = {**methods, "random": "Random"}
    cols = {m: (f"{m}_hit" if m != "random" else "random") for m in keys}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    order = sorted(keys, key=lambda m: results[cols[m]].mean())
    for i, m in enumerate(order):
        v = results[cols[m]].mean()
        lo, hi = _boot_ci(results[cols[m]])
        a.barh(i, v, color=COLORS[m], height=0.6)
        a.errorbar(v, i, xerr=[[v - lo], [hi - v]], color=INK, capsize=3, linewidth=1)
        a.text(hi + 0.005, i, f"{v:.1%}", va="center", fontsize=9, color=INK, fontweight="bold" if m == deploy else None)
    a.set_yticks(range(len(order)), [names[m] + (" (used)" if m == deploy else "") for m in order])
    a.grid(axis="y", visible=False)
    pct(a, "x")
    a.set_xlim(0, max(results[cols[m]].mean() for m in keys) * 1.3)
    a.set_title(f"List hit rate, {len(results):,} real Tim's lists (95% interval)")

    by = results.groupby("season")
    seasons = [season_label(s) for s in by.groups]
    for m in keys:
        b.plot(seasons, by[cols[m]].mean().values, color=COLORS[m], marker="o", markersize=4, label=names[m],
               linewidth=2.5 if m == deploy else 1.5)
    pct(b)
    b.set_title("By season")
    b.legend(fontsize=8, ncols=2, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def vs_baseline(results, methods, path):
    others = [m for m in methods if m != "season_gpg"]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(others) + 1.4))
    for i, m in enumerate(others):
        d, lo, hi = tims_rank.bootstrap_vs(results, m, "season_gpg")
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=COLORS[m], ecolor=COLORS[m],
                    markersize=8, capsize=4, linewidth=2, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(hi + 0.002, i, f"{d:+.1%}  ({lo:+.1%} to {hi:+.1%})", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(others)), [methods[m] for m in others])
    ax.set_ylim(-0.6, len(others) - 0.4)
    ax.grid(axis="y", visible=False)
    pct(ax, "x", decimals=0)
    lim = max(abs(v) for m in others for v in tims_rank.bootstrap_vs(results, m, "season_gpg")[1:]) * 1.8
    ax.set_xlim(-lim, lim)
    ax.set_title("Hit rate difference vs picking by goals per game")
    ax.set_xlabel("percentage points (right of 0 = better than goals per game)")
    fig.savefig(path)
    plt.close(fig)


def choice_rank(results, methods, path):
    """How often each method's 1st, 2nd ... 5th choice in a list scored."""
    ks = range(1, 6)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for m in methods:
        ax.plot(list(ks), [results[f"{m}_k{k}"].mean() for k in ks], color=COLORS[m], marker="o", markersize=5,
                label=methods[m])
    ax.axhline(results["random"].mean(), color=RANDOM, linewidth=1.5, label="Random pick")
    ax.set_xticks(list(ks), ["1st choice", "2nd", "3rd", "4th", "5th"])
    pct(ax)
    ax.set_title("Scoring rate of each method's 1st to 5th choice in a list")
    ax.legend(fontsize=8, ncols=2)
    fig.savefig(path)
    plt.close(fig)


def importance(gbm, rows, path, repeats=3):
    """Drop in within-list AUC when each input is shuffled (GBM)."""
    r = rows[["gameDate", "list", "season", "scored"]].copy()
    r["gbm"] = gbm.predict_proba(rows[FEATURES])[:, 1]
    base = tims_rank.evaluate_lists(r, ["gbm"], max_k=1)["gbm_auc"].mean()
    rng = np.random.default_rng(0)
    drops = {}
    for f in FEATURES:
        vals = []
        for _ in range(repeats):
            X = rows[FEATURES].copy()
            X[f] = rng.permutation(X[f].values)
            r["gbm"] = gbm.predict_proba(X)[:, 1]
            vals.append(base - tims_rank.evaluate_lists(r, ["gbm"], max_k=1)["gbm_auc"].mean())
        drops[f] = np.mean(vals)
    top = pd.Series(drops).sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh([LABELS.get(k, k) for k in top.index], top.values * 1000, color=COLORS["gbm"], height=0.6)
    ax.bar_label(bars, labels=[f"{v * 1000:.1f}" for v in top.values], fontsize=8, color=MUTED, padding=3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("drop in within-list AUC x1000 when shuffled")
    ax.set_title("Most important inputs for ranking within a list (gradient boosting)")
    fig.savefig(path)
    plt.close(fig)
    return top[::-1]


def partial_dependence_grid(gbm, test, path):
    feats = ["ixg_l40", "finishing_l82", "toi_l5", "pp_ixg_l20", "opp_xga_l20", "goalie_gsax_l40"]
    sample = test[FEATURES].sample(min(len(test), 5000), random_state=0)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, f in zip(axes.flat, feats):
        r = partial_dependence(gbm, sample, [f], kind="average", grid_resolution=25, method="brute",
                               percentiles=(0.02, 0.98), response_method="predict_proba")
        ax.plot(r["grid_values"][0], r["average"][0], color=COLORS["gbm"])
        ax.set_title(LABELS[f], fontsize=10)
        pct(ax)
    fig.suptitle("How P(goal) changes with one input, holding the rest as they were (gradient boosting)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def example_player(test, path):
    name = test.groupby("name")["goals"].sum().idxmax()
    p = test[test["name"] == name].sort_values("gameDate")
    scored = p["goals"] > 0
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(p["gameDate"], p["gbm"], color=COLORS["gbm"], alpha=0.1, linewidth=0)
    ax.plot(p["gameDate"], p["gbm"], color=COLORS["gbm"], label="P(goal) before the game")
    ax.scatter(p["gameDate"][scored], p["gbm"][scored], color=INK, s=22, zorder=3,
               edgecolors="white", linewidths=1, label="Scored")
    ax.scatter(p["gameDate"][~scored], p["gbm"][~scored], facecolors="white", edgecolors=RANDOM,
               s=16, zorder=3, label="Didn't score")
    pct(ax)
    ax.set_ylim(0, None)
    ax.set_title(f"{name}: scored in {scored.sum()} of {len(p)} games, model averaged {p['gbm'].mean():.0%}")
    ax.legend(loc="upper left", ncols=3)
    fig.savefig(path)
    plt.close(fig)


def write_summary(results, methods, deploy, last, top_inputs, path):
    s = tims_rank.summary(results, list(methods))
    lines = [f"Tim's Picks backtest on real Tim's lists - generated {date.today()}",
             f"{len(results):,} lists over {results['season'].nunique()} seasons "
             f"({', '.join(season_label(x) for x in sorted(results['season'].unique()))}). "
             "Each season's models were trained only on earlier seasons.",
             "Hit rate = the list's top-ranked player scored. Top 3 = any of the top 3 scored.",
             "List AUC = chance a scorer is ranked above a non-scorer in the same list.", "",
             f"{'':26}{'hit rate':>10}{'top 3':>8}{'list AUC':>10}   vs goals/game (95% interval)"]
    for m in methods:
        line = f"{methods[m]:26}{s.loc[m, 'hit']:>10.1%}{s.loc[m, 'top3']:>8.1%}{s.loc[m, 'auc']:>10.3f}"
        if m != "season_gpg":
            d, lo, hi = tims_rank.bootstrap_vs(results, m, "season_gpg")
            line += f"   {d:+.1%} ({lo:+.1%} to {hi:+.1%})"
        lines.append(line)
    lines.append(f"{'Random':26}{results['random'].mean():>10.1%}")
    lines += ["", "Hit rate by season:"]
    by = results.groupby("season")[[f"{m}_hit" for m in methods]].mean()
    by.columns = [methods[m] for m in methods]
    lines.append(by.rename(index=season_label).to_string(float_format="{:.1%}".format))
    lines += ["", "Hit rate by list:"]
    by = results.groupby("list")[[f"{m}_hit" for m in methods]].mean()
    by.columns = [methods[m] for m in methods]
    lines.append(by.to_string(float_format="{:.1%}".format))
    lines += ["", f"Used by Tims_Picker.py: {methods[deploy]} (best hit rate)",
              f"Bayesian model ({season_label(last['season'])}): {last['bayes'].describe()}",
              "", "Top inputs for ranking within a list (gradient boosting, drop in list AUC x1000):"]
    lines += [f"  {LABELS.get(k, k):42}{v * 1000:6.1f}" for k, v in top_inputs.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figures(df, results, last, methods, deploy):
    REPORT_DIR.mkdir(exist_ok=True)
    print(f"\nSaving figures to {REPORT_DIR}...")
    for old in REPORT_DIR.glob("*.png"):
        old.unlink()
    gbm, rows, test = last["gbm"], last["rows"], last["test"]
    data_by_season(df, REPORT_DIR / "1_data_by_season.png")
    ranking_curve(gbm, rows, last["season"], REPORT_DIR / "2_ranking_learning_curve.png")
    methods_hit(results, methods, deploy, REPORT_DIR / "3_list_hit_rate_by_method.png")
    vs_baseline(results, methods, REPORT_DIR / "4_vs_goals_per_game.png")
    choice_rank(results, methods, REPORT_DIR / "5_hit_rate_by_choice.png")
    top_inputs = importance(gbm, rows, REPORT_DIR / "6_ranking_importance.png")
    partial_dependence_grid(gbm, test, REPORT_DIR / "7_partial_dependence.png")
    example_player(test, REPORT_DIR / "8_example_player.png")
    write_summary(results, methods, deploy, last, top_inputs, REPORT_DIR / "metrics.txt")
