# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:40:08 2024

@author: ww1a23
"""
from datetime import datetime
import os
from scenv0703 import SupplyChainEnvironment,Action, State
import matplotlib.pyplot as plt
import numpy as np
import pickle
now = datetime.now()
env = SupplyChainEnvironment()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
local_dir = f"{env.product_types_num}P{env.distr_warehouses_num}W_{now_str}"
plots_dir = 'plots'



def calculate_cum_profit(returns_trace, print_reward=True):
    """
    Calculate the cumulative profit for each episode.
    """
    rewards_trace = []
    for return_trace in returns_trace:
        rewards_trace.append(abs(
            np.sum(return_trace.T[2])))

    if print_reward:
        print(f"reward: mean "
              f"{np.mean(rewards_trace)}, "
              f"std "
              f"{np.std(rewards_trace)}, "
              f"max "
              f"{np.max(rewards_trace)}, "
              f"min "
              f"{np.min(rewards_trace)}")

    return rewards_trace


def visualize_cum_profit(rewards_trace, algorithm,
                         local_dir=local_dir, plots_dir=plots_dir):
    """
    Visualize the cumulative profit boxplot along the episodes.
    """
    xticks = []
    if not isinstance(algorithm, list):
        xticks.append(algorithm)
    else:
        xticks = algorithm

    plt.figure(figsize=(15, 5))
    plt.boxplot(rewards_trace)

    plt.ylabel('Cumulative Costs')
    plt.xticks(np.arange(1,
                         len(xticks)+1),
               xticks)
    plt.tick_params(axis='x', which='both',
                    top=False, bottom=True,
                    labelbottom=True)
    plt.ticklabel_format(axis='y', style='plain',
                         useOffset=False)
    plt.tight_layout()

    # creating necessary subdir and saving plot
    if not os.path.exists(f"{local_dir}/{plots_dir}/{algorithm}"):
        os.makedirs(f"{local_dir}/{plots_dir}/{algorithm}")
    plt.savefig(f"{local_dir}/{plots_dir}/{algorithm}"
                f"/cum_profit_{algorithm}.pdf",
                format='pdf', bbox_inches='tight')

    # saving the cumulative profit as text
    if not isinstance(algorithm, list):
        f = open(f"{local_dir}/{plots_dir}/{algorithm}"
                 f"/cum_profit_{algorithm}.txt",
                 'w', encoding='utf-8')
        f.write(f"reward: mean "
                f"{np.mean(rewards_trace)}, "
                f"std "
                f"{np.std(rewards_trace)}, "
                f"max "
                f"{np.max(rewards_trace)}, "
                f"min "
                f"{np.min(rewards_trace)}")
        f.close()


def save_checkpoint(checkpoint, algorithm,
                    local_dir=local_dir, plots_dir=plots_dir):
    """
    Save Ax BS/(s, Q)-policy parameters or RLib Agent checkpoint.
    """
    f = open(f"{local_dir}/{plots_dir}/{algorithm}"
             f"/best_checkpoint_{algorithm}.txt",
             'w', encoding='utf-8')
    f.write(checkpoint)
    f.close()


def save_object(obj, obj_name, algorithm,
                local_dir=local_dir, plots_dir=plots_dir):
    try:
        with open(f"{local_dir}/{plots_dir}/{algorithm}/"
                  f"/{obj_name}_{algorithm}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"{e.__class__} occurred!")


def convert_seconds_to_min_sec(seconds_float):
    # Get the minutes part and the remaining seconds
    minutes, seconds = divmod(seconds_float, 60)

    # Combine minutes and seconds in requested format
    min_sec = minutes + seconds / 100

    return min_sec