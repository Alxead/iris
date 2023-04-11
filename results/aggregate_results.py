import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from matplotlib import rc
import numpy as np
import pandas as pd
import seaborn as sns

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

rcParams['legend.loc'] = 'best'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rc('text', usetex=False)
RAND_STATE = np.random.RandomState(42)
sns.set_style("white")

Path('figures').mkdir(exist_ok=True, parents=False)


def score_normalization(res_dict, min_scores, max_scores):
    games = res_dict.keys()
    norm_scores = {}
    for game, scores in res_dict.items():
        norm_scores[game] = (scores - min_scores[game]) / (max_scores[game] - min_scores[game])
    return norm_scores


def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


StratifiedBootstrap = rly.StratifiedBootstrap

IQM = lambda x: metrics.aggregate_iqm(x)                  # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)   # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)


ATARI_100K_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
    'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Seaquest', 'UpNDown'
]


def load_json_scores(algorithm_name, normalize=True, atari_games=ATARI_100K_GAMES):
    path = Path('data') / f'{algorithm_name}.json'
    with path.open('r') as f:
        raw_scores = json.load(f)
    raw_scores = {game: np.array(val) for game, val in raw_scores.items() if game in atari_games}

    if normalize:
        hn_scores = score_normalization(raw_scores, RANDOM_SCORES, HUMAN_SCORES)
        hn_score_matrix = convert_to_matrix(hn_scores)
    else:
        hn_scores, hn_score_matrix = None, None
    return hn_scores, hn_score_matrix, raw_scores


def load_csv_scores(file_name, normalize=True, atari_games=ATARI_100K_GAMES):
    df = pd.read_csv(Path('data') / file_name)
    raw_scores = {}
    for task in atari_games:
        task_df = df[df['game'] == task]
        raw_scores[task] = task_df['return'].values

    raw_scores = {game.replace('NoFrameskip-v4', ''): val for game, val in raw_scores.items()}

    if normalize:
        hn_scores = score_normalization(raw_scores, RANDOM_SCORES, HUMAN_SCORES)
        hn_score_matrix = convert_to_matrix(hn_scores)
    else:
        hn_scores, hn_score_matrix = None, None
    return hn_scores, hn_score_matrix, raw_scores


_, _, raw_scores_random = load_json_scores('RANDOM', normalize=False)
_, _, raw_scores_human = load_json_scores('HUMAN', normalize=False)
RANDOM_SCORES = {k: v[0] for k, v in raw_scores_random.items()}
HUMAN_SCORES = {k: v[0] for k, v in raw_scores_human.items()}

############################
# Scores on 8 selected games
############################
ATARI_100K_GAMES_SUBSET = [
    'Alien', 'Asterix', 'Breakout', 'MsPacman', 'Pong', 'Qbert', 'Seaquest', 'UpNDown'
]

ATARINAME_100K_GAMES_SUBSET = [
    'AlienNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'BreakoutNoFrameskip-v4',
    'MsPacmanNoFrameskip-v4', 'PongNoFrameskip-v4', 'QbertNoFrameskip-v4',
    'SeaquestNoFrameskip-v4', 'UpNDownNoFrameskip-v4'
]

# score_dict_muzero, score_muzero, raw_scores_muzero = load_json_scores('MuZero', atari_games=ATARI_100K_GAMES_SUBSET)
# score_dict_efficientzero, score_efficientzero, raw_scores_efficientzero = load_json_scores('EfficientZero', atari_games=ATARI_100K_GAMES_SUBSET)
# score_dict_simple, score_simple, _ = load_json_scores('SimPLe', atari_games=ATARI_100K_GAMES_SUBSET)
# score_dict_curl, score_curl = read_curl_scores(atari_games=ATARI_100K_GAMES_SUBSET)
# score_dict_drq_eps, score_drq_eps, _ = load_json_scores('DrQ(eps)', atari_games=ATARI_100K_GAMES_SUBSET)
# score_dict_spr, score_spr, _ = load_json_scores('SPR', atari_games=ATARI_100K_GAMES_SUBSET)

score_dict_iris, score_iris, _ = load_json_scores('IRIS', atari_games=ATARI_100K_GAMES_SUBSET)
score_dict_myrun, score_myrun, _ = load_csv_scores(file_name='BlockCausal-64tokens-smallmodel-LongSchedule.csv', atari_games=ATARINAME_100K_GAMES_SUBSET)


score_data_dict_games = {
    'IRIS (paper)': score_dict_iris,
    'my run': score_dict_myrun,
}

all_score_dict = {
    'IRIS (paper)': score_iris,
    'my run': score_myrun
}

aggregate_func = lambda x: np.array([MEAN(x), MEDIAN(x), IQM(x), OG(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    all_score_dict, aggregate_func, reps=50000)

for algo in aggregate_scores.keys():
    n_runs, n_games = all_score_dict[algo].shape
    assert n_games == len(ATARI_100K_GAMES_SUBSET)
    print(f"{algo.ljust(14)}: {n_runs:3d} runs")

for k in aggregate_scores.keys():
    n_runs, n_games = all_score_dict[k].shape
    assert n_games == 8
    score_dict = score_data_dict_games[k]
    mean, median, iqm, og = aggregate_scores[k]
    mean_min, mean_max = aggregate_interval_estimates[k][0, 0], aggregate_interval_estimates[k][1, 0]
    median_min, median_max = aggregate_interval_estimates[k][0, 1], aggregate_interval_estimates[k][1, 1]
    iqm_min, iqm_max = aggregate_interval_estimates[k][0, 2], aggregate_interval_estimates[k][1, 2]
    og_min, og_max = aggregate_interval_estimates[k][0, 3], aggregate_interval_estimates[k][1, 3]
    sh = np.sum(np.mean(all_score_dict[k], axis=0) >= 1)
    print(f"\n####################\n{k}\n####################\n")
    print(f"{n_runs} runs")
    print(f"#superhuman: {sh}\nMean: {mean:.3f} ({mean_min:.3f}, {mean_max:.3f})\nMedian: {median:.3f} ({median_min:.3f}, {median_max:.3f})\nIQM: {iqm:.3f} ({iqm_min:.3f}, {iqm_max:.3f})\nOptimality gap: {og:.3f} ({og_min:.3f}, {og_max:.3f})\n")

    for game in score_dict.keys():
        h, r = HUMAN_SCORES[game], RANDOM_SCORES[game]
        raw_score = score_dict[game] * (h - r) + r
        print(f"{game}: {np.mean(raw_score): .1f}")