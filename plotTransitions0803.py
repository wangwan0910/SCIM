# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:11:49 2024

@author: ww1a23
"""

import matplotlib.pyplot as plt
from scenv0703 import SupplyChainEnvironment,Action, State
import numpy as np
from datetime import datetime
import os




now = datetime.now()
env = SupplyChainEnvironment()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
local_dir = f"{env.product_types_num}P{env.distr_warehouses_num}W_{now_str}"
plots_dir = 'plots'

def prepare_metric_plot(ylabel,n,
                        episode_duration = env.T-(2*env.lead_times_len),
                        plots_n = 4+(2*env.distr_warehouses_num)+(
                            env.lead_times_len*env.distr_warehouses_num)):
    plt.subplot(plots_n,1,n)
    plt.ylabel(ylabel,fontsize='medium',ha='center')
    plt.xticks(np.arange(min(range(episode_duration)),
                         max(range(episode_duration))+1))
    plt.tick_params(axis='x', which = 'both', top=True, 
                    bottom=True, labelbottom = False)
    
    
def visualize_transitions(returns_trace,algorithm,
                         local_dir = local_dir, plots_dir = plots_dir):
    if env.distr_warehouses_num <= 3 and env.product_types_num <=1 :
        transitions = np.array(
            [(return_trace)
             for return_trace in zip(*returns_trace)])
        states_trace, actions_trace,rewards_trace = (
            transitions.T[0],
            transitions.T[1],
            abs(transitions.T[2]))
        episode_duration = env.T - (2*env.lead_times_len)
        
        if env.distr_warehouses_num == 3:
            fig = plt.figure(figsize=(10,40))
        elif env.distr_warehouses_num ==2:
            fig = plt.figure(figsize=(10,30))
        else:
            fig = plt.figure(figsize=(10,20))
        states = np.array(
            [(state_trace)
             for state_trace in zip(*states_trace)])
        
        prepare_metric_plot('Stock, \nFactory',
                            1)
        tmp_mean = []
        for t in range(len(states)):
            tmp_mean.append(
                np.mean(
                    [np.sum(state.factoy_stocks)
                     for state in states[t]], axis=0))
            
        tmp_std = []    
        for t in range(len(states)):
            tmp_mean.append(
                np.std(
                    [np.sum(state.factoy_stocks)
                     for state in states[t]], axis=0))
            
        plt.plot(range(episode_duration), tmp_mean,
                 color='purple', alpha=.5)
        plt.fill_between(range(episode_duration), 
                         list(np.array(tmp_mean), 
                              np.array(tmp_std)),
                         list(np.array(tmp_mean) + 
                              np.array(tmp_std)),
                         color = 'purple', alpha=.2)
        
        
        actions = np.array(
            [(action_trace)
             for action_trace in zip(*actions_trace)])
        
        
        for  j in range(env.distr_warehouses_num):
            prepare_metric_plot(f'Ship, \nWH{j+1}',
                                3+env.distr_warehouses_num+j)
            tmp_mean = []
            for t in range(len(actions)):
                tmp_mean.append(
                    np.mean(
                        [np.sum(action.shipped_stocks[j])
                         for action in actions[t]], axis=0))
                
                
            tmp_std = []
            for t in range(len(actions)):
                tmp_std.append(
                    np.std(
                        [np.sum(action.shipped_stocks[j])
                         for action in actions[t]], axis=0))
            plt.plot(range(episode_duration), tmp_mean,
                     color = 'blue',alpha = .5)
            plt.fill_between(range(episode_duration), 
                             list(np.array(tmp_mean)-
                                  np.array(tmp_std)),
                             list(np.array(tmp_mean)+
                                  np.array(tmp_std)),
                             color = 'blue',alpha = .2)
            
            cont = 0
            for l in range(env.lead_times_len):
                for j in range(env.distr_warehouses_num):
                    prepare_metric_plot(f'In Transit, \n{l+1} WH{j+1}',
                                        3+(2*env.distr_warehouses_num)+j+cont)
                    tmp_mean = []
                    for t in range(len(states)):
                        tmp_mean.append(
                            np.mean(
                                [np.sum(state.lead_times_len[l][j])
                                 for state in states[t]], axis = 0))
                    tmp_std = []
                    for t in range(len(states)):
                        tmp_std.append(
                            np.std(
                                [np.sum(state.lead_time[l][j])
                                 for state in states[t]], axis = 0))
                    plt.plot(range(episode_duration), 
                             tmp_mean,
                             color = 'green',
                             alpha = .5)
                    plt.fill_between(range(episode_duration), 
                                     list(np.array(tmp_mean) -
                                          np.array(tmp_std)),
                                     list(np.array(tmp_mean)+
                                          np.arange(tmp_std)),
                                     color = 'green', alpha = .2)
                cont += env.distr_warehouses_num
                
            prepare_metric_plot('\nCosts', 
                                3+(2*env.distr_warehouses_num) + (
                                    env.lead_times_len * env.distr_warehouses_num))
            
            reward_mean = np.array(
                np.mean(rewards_trace,axis=0),
                dtype = np.int32)
            reward_std = np.array(
                np.std(rewards_trace.astype(np.int32), axis = 0),
                dtype = np.int32)
            plt.plot(range(episode_duration),
                     reward_mean,
                     linewidth = 2,
                     color = 'red',
                     alpha = .5)
            plt.fill_between(range(episode_duration),
                             reward_mean - reward_std,
                             reward_mean-reward_std,
                             color = 'red',
                             alpha = .2)
            
            prepare_metric_plot('Cum\nCosts',
                                4+(2*env.distr_warehouses_num)+(
                                    env.lead_times_len * env.distr_warehouses_num))
            
                
                
                
                
            cum_reward = np.array(
                [np.cumsum(rewards_trace)
                                 for reward_trace in rewards_trace])
            cum_reward_mean = np.array(
                np.mean(cum_reward,axis=0),
                dtype = np.int32)
            
            cum_reward_std = np.array(
                np.std(cum_reward.astype(np.int32), axis = 0),
                dtype = np.int32)
            plt.plot(range(episode_duration), 
                     cum_reward_mean,
                     linewidth =2,
                     color = 'red',
                     alpha = .5)
            plt.fill_between(range(episode_duration),
                             cum_reward_mean - cum_reward_std,
                             cum_reward_mean + cum_reward_std,
                             color = 'red',
                             alpha = .2)
            
            fig.align_labels()
            plt.ticklabel_format(axis = 'y',
                                 style='plain',
                                 useOffset=False)
            plt.tight_layout()
            
            
            if not os.path.exists(f'{local_dir}/{plots_dir}/{algorithm}'):
                os.makedirs(f'{local_dir}/{plots_dir}/{algorithm}')
                
            plt.savefig(f'{local_dir}/{plots_dir}/{algorithm}', 
                        f'/transitions_{algorithm}.pdf',
                        format = 'pdf',
                        bbox_inches = 'tight')
            
            
        
                    
                    
                    
                    
                    
                    
                    
                        
        
        
    