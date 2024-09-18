# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:06:28 2024

@author: ww1a23
"""

from scenv0703 import SupplyChainEnvironment,Action, State

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from itertools import chain
from datetime import datetime
import os
from ray.rllib.algorithms.dqn import DQNConfig  
from ray.tune.logger import pretty_print 
from ray.rllib.algorithms.ppo import PPOConfig  


class SupplyChain(gymnasium.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.supply_chain = SupplyChainEnvironment()
        self.reset()
        factory_size = self.supply_chain.product_types_num
        lead_times_size =(self.supply_chain.lead_times_len*
                          self.supply_chain.distr_warehouses_num*
                          self.supply_chain.product_types_num)

        distr_warehouses_size =(self.supply_chain.distr_warehouses_num*
                                self.supply_chain.product_types_num)
        
        low_act = np.zeros(
            ((self.supply_chain.distr_warehouses_num+1)*
             self.supply_chain.product_types_num),
            dtype = np.int32)
        
        high_act = np.zeros(
            ((self.supply_chain.distr_warehouses_num+1)*
             self.supply_chain.product_types_num),
            dtype = np.int32)
        
        high_act[
            :factory_size
            ] = self.supply_chain.prod_level_max
        high_act[factory_size:
                 ] = self.supply_chain.storage_capacities.flatten()[
                     self.supply_chain.product_types_num:
                         ]
        self.action_space = Box(low=low_act,
                                high = high_act,
                                dtype = np.int32)
        low_obs = np.zeros(
            (len(self.supply_chain.initial_state().to_array()),),
            dtype=np.int32) 
        
        low_obs[factory_size+lead_times_size:
                factory_size+lead_times_size+ distr_warehouses_size
                ] = np.array(
                    [-(self.supply_chain.d_max+self.supply_chain.d_var)*
                     self.supply_chain.T]*
                    self.supply_chain.distr_warehouses_num).flatten()
        high_obs = np.zeros(
            (len(self.supply_chain.initial_state().to_array()),),
            dtype = np.int32)
        high_obs[:factory_size
                 ] = self.supply_chain.storage_capacities[:1].flatten()
        high_obs[factory_size:
                 factory_size+lead_times_size
                 ] = np.repeat(self.supply_chain.storage_capacities[:1], 
                               self.supply_chain.distr_warehouses_num*
                               self.supply_chain.lead_times_len)
        high_obs[factory_size+lead_times_size:
                 factory_size+lead_times_size+distr_warehouses_size
                 ] = self.supply_chain.storage_capacities[1:].flatten()
            
        high_obs[
            factory_size+lead_times_size+distr_warehouses_size:
                len(high_obs)-1
                ] = np.array(
                    [self.supply_chain.d_max+self.supply_chain.d_var]*
                    len(list(chain(*self.supply_chain.demand_history)))).flatten()
                    
        high_obs[len(high_obs)-1] = self.supply_chain.T
        self.observation_space = Box(low=low_obs,
                                     high=high_obs,
                                     dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        self.supply_chain.reset(seed)
        self.state = self.supply_chain.initial_state()
        return self.state.to_array(), {}  # 添加空字典作为第二个返回值

    def step(self, action):
        action_obj = Action(
            self.supply_chain.product_types_num,
            self.supply_chain.distr_warehouses_num)
        action_obj.production_level = action[:self.supply_chain.product_types_num].astype(np.int32)
        action_obj.shipped_stocks = action[self.supply_chain.product_types_num:].reshape(
            (self.supply_chain.distr_warehouses_num, self.supply_chain.product_types_num)
        ).astype(np.int32)

        self.state, reward, done = self.supply_chain.step(self.state, action_obj)
        info = {}  # 任何额外的信息
        truncated = False  # 或者其他逻辑来确定是否截断

        return self.state.to_array(), reward, done, truncated, info
    
    
if __name__ == "__main__":
    # config =
    env = SupplyChain()
    # env = SupplyChainEnvironment()
    # SEED =2023
    # env.reset()
    # # print('state',state)
    # NUM_EPISODES = 250
    # now = datetime.now()
    
    # now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # local_dir = f"{env.product_types_num}P{env.distr_warehouses_num}W_{now_str}"
    # plots_dir = 'plots'
    # if not os.path.exists(f'{local_dir}'):
    #     os.makedirs(f'{local_dir}')
    # if not os.path.exists(f"{local_dir+'/'+plots_dir}"):
    # #     os.makedirs(f'{local_dir}')
    # # if not os.path.exists(f"{local_dir +'/' +plots_dir}"):
    #     os.makedirs(f"{local_dir+'/'+plots_dir}")
    episodes = 10
    for ep in range(episodes):
        actions = env.action_space.sample()
        print('999',actions)
        state, reward, done, info =  env.step(actions)
        print(state)
        
    config = PPOConfig().environment(SupplyChain) 

              # .rollouts(num_rollout_workers=2, 

              #           create_env_on_local_worker=True)) 

     

    # pretty_print(config.to_dict()) 

     

    algo = config.build() 
    
    for i in range(10): 

        result = algo.train() 

     

    pretty_print(result) 
        
        
    algo = PPOConfig().environment(env=SupplyChain).multi_agent(  

            policies={  

  

                "policy_1": (  

  

                    None, env.observation_space, env.action_space, {"gamma": 0.80}  

  

                ),  

  

                "policy_2": (  

  

                    None, env.observation_space, env.action_space, {"gamma": 0.95}  

  

                ),  

  

            },  

  

            policy_mapping_fn = lambda agent_id: 

        f"policy_{agent_id}",  

        ).build()  

  

      

  

    print(algo.train())  

  

    print('rew------333')  
        
 
        
    
    
