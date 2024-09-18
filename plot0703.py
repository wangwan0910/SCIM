# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:11:06 2024

@author: ww1a23
"""



from datetime import datetime
from scenv0703 import SupplyChainEnvironment,Action, State
from env0703 import SupplyChain
import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt
from datetime import datetime
import os


now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

SEED =2023
env = SupplyChainEnvironment()
# env = SupplyChain()
local_dir = f"{env.product_types_num}P{env.distr_warehouses_num}W_{now_str}"
plots_dir = 'plots'


def visualize_demand(env, num_episodes =1,
                     local_dir = local_dir,
                     plots_dir=plots_dir,
                     seed = SEED):
    if env.distr_warehouses_num <= 3 and env.product_types_num <= 2:
        env.reset(seed)
        episode_duration = env.T- (2*env.lead_times_len)
        demands_episodes = []
        for _ in range(num_episodes):
            demands_episode = env.generate_episode_demands() #
            # print('demands_episode',demands_episode) #[[[ 7.]
             #  [ 2.]]

             # [[ 6.]
             #  [ 1.]]

             # [[ 5.]
             #  [10.]]

             # [[ 1.]
             #  [ 1.]]

             # [[ 6.]
             #  [ 1.]]

             # [[ 5.]
             #  [ 5.]]

             # [[ 6.]
             #  [ 6.]]]
            demands_episodes.append(demands_episode)
            # print('demands_episodes',demands_episodes)
        demands_episodes = np.array(demands_episodes)
        demands_mean = np.array([np.mean(d,axis = 0)
                                 for d in zip(*demands_episodes)])
        demands_std = np.array([np.std(d,axis = 0)
                                 for d in zip(*demands_episodes)])
        plt.figure(figsize=(15,5))
        plt.xlabel('time steps')
        plt.ylabel('demand value')
        
        plt.xticks(np.arange(1, episode_duration+1))
        plt.tick_params(axis = 'x', which = 'both', top= True, bottom= True,
                        labelbottom = True)
        plt.ticklabel_format(axis='y',style = 'plain', useOffset=False)
        plt.tight_layout()
        
        color = [['b','b'],['g','g'],['r','r']]
        line_style = [['b-','b--'],['g-','g--'],['r-'],['r--']]
        
        timesteps = np.arange(1, episode_duration+1)
        for j in range(env.distr_warehouses_num):
            for i in range(env.product_types_num):
                if env.product_types_num == 1:
                    plt.plot(timesteps,demands_mean[:,j,i],
                             line_style[j][i],
                             label = f'WH{j+1}')
                else:
                    plt.plot(timesteps, demands_mean[:,j,i],
                             line_style[j][i],
                             label=f'WH{j+1}, P{i+1}')
                plt.fill_between(timesteps,
                                 demands_mean[:,j,i]-demands_std[:,j,i],
                                 demands_mean[:,j,i] + demands_std[:,j,i],
                                 color = color[j][i],alpha = .2)
                
        plt.legend()
        
        
        
        
        plt.savefig(f'{local_dir}/{plots_dir}'
                    f'/demand.pdf',
                    format = 'pdf', 
                    bbox_inches = 'tight')
                    

def save_env_settings(env,local_dir = local_dir, plots_dir = plots_dir):
    f =open(f'{local_dir}/{plots_dir}'
            f'/env_settings.txt',
            'w',encoding='utf-8')
    f.write(f'--supplychainenvironment----'
            f'\nproduct_types_num is'
            f'{env.product_types_num}'
            f'\ndistr_warehouses_num is'
            f'\nT is'
            f'{env.T}'
            F'\ndemand_type is'
            f'{env.demand_type}'
            f'\nd_max is'
            f'{env.d_max}'
            f'\d_var is'
            f'{env.d_var}')
    f.close()
    

if __name__ =='__main__':
    np.random.seed(SEED)
    env.reset()
    NUM_EPISODE = 250
    env = SupplyChainEnvironment()
    NUM_EPISODES = 250
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    local_dir = f"{env.product_types_num}P{env.distr_warehouses_num}W_{now_str}"
    plots_dir = 'plots'
    if not os.path.exists(f'{local_dir}'):
        os.makedirs(f'{local_dir}')
    if not os.path.exists(f"{local_dir+'/'+plots_dir}"):
    #     os.makedirs(f'{local_dir}')
    # if not os.path.exists(f"{local_dir +'/' +plots_dir}"):
        os.makedirs(f"{local_dir+'/'+plots_dir}")
    
    
    visualize_demand(env, NUM_EPISODE)
    save_env_settings(env)
    
    
    
    
    
    
    
    