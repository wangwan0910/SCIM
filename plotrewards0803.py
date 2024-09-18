# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 18:53:24 2024

@author: ww1a23
"""

import os
from datetime import datetime
from scenv0703 import SupplyChainEnvironment,Action, State
from env0703 import SupplyChain

import matplotlib.pyplot as plt


# supply chain env
env = SupplyChainEnvironment()
now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
ray_dir = 'ray_results'
local_dir = f"{env.product_types_num}P{env.distr_warehouses_num}W_{now_str}"
plots_dir = 'plots'


def apply_style_plt(ax,xticks_total,episodes_total,legend):
    ax.set_xticks([0,
                   xticks_total//2,
                   xticks_total])
    ax.set_xticklabels([str(0),
                        str(episodes_total//2),
                        str(episodes_total)])
    
    ax.set_label('Episodes')
    ax.legend(legend, bbox_to_anchor = (1.04, .5), borderaxespad = 0,
              frameon  = False, loc = 'center_left', fancybox = True,
              shadow = True)
    ax.ticklabel_format(axis = 'y',style = 'plain',
                        useOffset = False)
    
    
def visualize_rewards(results,best_result, algorithm, legend = [],
                      local_dir = local_dir, plots_dir =plots_dir):
    if not os.path.exists(f'{local_dir}/{plots_dir}/{algorithm}'):
        os.makedirs(f"{local_dir}/{plots_dir}/{algorithm}")
    xticks_total = len(list(results.values())[0])-1 
    episodes_total = calculate_training_episodes(best_result)
    
    fig, ax = plt.subplots(figsize = (15,5))
    for result in results.values():
        ax = result.episode_reward_min.plot(ax= ax)

    apply_style_plt(ax, xticks_total, episodes_total, legend)        
        
        
        
        
        
        