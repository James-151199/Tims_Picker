TIMS PICKER

Ranks the players in each Tim's Hockey Challenge list by how likely they are
to score today, using models trained on every NHL game since 2010 and judged
on how often their top pick in a real Tim's list scored.

SETUP (once)

1. Install the packages:
       pip install -r requirements.txt

2. Train the models:
       python train_model.py

   The first run downloads ~16 seasons of game logs and goalie logs (NHL stats
   API) and MoneyPuck's shot data into data/, plus the real Tim's lists for
   the last 3 seasons (hockeychallengehelper.com). After that only the
   current season is refreshed.

   It backtests four methods on the real Tim's lists, each season using
   models trained only on earlier seasons, and compares them with picking by
   goals per game:
       gbm          gradient boosted trees predicting P(goal)
       lambdarank   LightGBM LambdaRank, trained to rank scorers first in a list
       bayes        Bayesian player-talent model (chance rate x finishing)
       blend        50/50 blend of the GBM's and goals per game's rankings
   The scoring metric is the list hit rate: how often the player a method
   ranks first in a list scored. The best method is the one the picker uses.
   Charts are saved to reports/ as PNGs, with the numbers in reports/metrics.txt.

DAILY

       python Tims_Picker.py              today's Tim's lists, best pick first
       python Tims_Picker.py --all        top 30 of everyone playing today
       python Tims_Picker.py --date 2026-10-14

   The Tim's lists, with tonight's projected lines and power play units, come
   from hockeychallengehelper.com; projected starting goalies from
   dailyfaceoff.com. P(goal) is the gradient boosting model's chance the
   player scores at least one goal.

Retrain every few weeks (python train_model.py) so the models learn from the
newest games. The daily picker always uses the latest stats even without
retraining.

HOW IT WORKS

tims_data.py      NHL stats API game logs (skaters, teams, goalies), schedule,
                  rosters, and the real Tim's lists history
tims_shots.py     MoneyPuck shot data: every unblocked shot since 2010
tims_xg.py        our expected goals (xG) model for each shot, and per-game xG
                  for players, teams and goalies
tims_features.py  what was known before each game: goals, shots, xG, ice time,
                  line and PP unit, finishing, season rates, team and opponent
                  form, opposing starting goalie's saves above expected
tims_bayes.py     Bayesian player-talent model
tims_rank.py      list hit rate and other within-list metrics, LambdaRank, blend
train_model.py    backtest, choose the best method, train the final models
tims_report.py    figures for reports/
Tims_Picker.py    builds the same features for today and ranks each list
