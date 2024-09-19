# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:53:44 2024

@author: ww1a23
"""

# -*- coding: utf-8 -*-  

 
 

"""  

Created on Mon Sep 16 11:51:47 2024  

@author: ww1a23  

 
 

"""  

from gymnasium.spaces import Discrete,MultiDiscrete   

import datetime  

import os   

import gymnasium   

import csv  

import numpy as np   

import time  

from gym import utils   

from gym.utils import seeding   

from ray.rllib.utils.typing import AgentID  

from ray.tune.logger import pretty_print  

# from ray.rllib.algorithms.algorithm import Algorithm  

 
 

# from ray.rllib.env.policy_server_input import PolicyServerInput  

 
 

# from ray import tune  

 
 

import matplotlib.colors as mcolors  

import matplotlib.pyplot as plt  

from scipy.stats import poisson, binom, randint, geom   

import time  

from itertools import chain  

import collections  

from ray.tune.logger import pretty_print  

from ray.rllib.algorithms.ppo import PPOConfig  

from ray.rllib.algorithms.dqn import DQNConfig  

# import gym  

import gymnasium 

from gymnasium.spaces import Discrete,MultiDiscrete 

import os  

from gym.utils import seeding   

import numpy as np   

import logging   

 
 

logger = logging.getLogger(__name__)   

 
 

   

 
 

ep_count = 0  

class POSingAgent1W1F:   

    def __init__(self, *args, **kwargs):  

        # super().__init__(*args, **kwargs)  

        self.factory_capacity = 60  

        self.retailer_capacity = 20   

        self.price_capacity = 60  

        factory_capacity = self.factory_capacity   

        retailer_capacity = self.retailer_capacity   

        price_capacity = self.price_capacity   

        self.v_f = 2 # 1:the factory does not show its stock; 2:factory can choose to show its stock or not.  

        self.action_space = MultiDiscrete([retailer_capacity + 1,price_capacity+1,self.v_f,retailer_capacity + 1,price_capacity+1,self.v_f])  #[order , price, show]  

        self.periods = 30 #simulation days  

        self.observation_space = MultiDiscrete([factory_capacity,factory_capacity,factory_capacity,factory_capacity,factory_capacity,factory_capacity,factory_capacity,self.periods+1,factory_capacity,factory_capacity,factory_capacity,factory_capacity,factory_capacity,factory_capacity,factory_capacity,self.periods+1]) # [inventory,backlog,stockout,Factory_inventory]  

        self.state =  np.array([10,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0])  

        self.r = {1:0, 2:0}   

        self.info =  self.state  

        self.k = 5   

        self.c = 2   

        self.h = 2   

        self.b = 5  

        self.s = [200,100]  

        self.re_num = 0  

        self.days =  30  

        self.day =  0  

        self.stockoutCount  = 0  

        self.stockoutCountS  = 12  

        self._seed()   

        self.dist = 1  #This parameter can be selected for 1-5 different demand distributions.  

        self.ALPHA = 0.5  

        self.dist_param = {'mu':10}  

        self.d_max = 30  

        self.d_var = 5  

        self.task = 1 #1: rewards set remains the same, 2: factory rewards need to be subtracted from retailer rewards, 3: retailer incentives need to be subtracted from factory rewards  

        self.demand_len=3  

 
 

   

 
 

        # self.state_history = []  

 
 

        # self.action_history = []  

 
 

        # self.reward_history = []  

 
 

        # self.demand_history = []  

 
 

          

 
 

        self.reset()   

        self.agent1_action_1_count = 0  

        self.agent1_action_0_count = 0  

        self.agent2_action_1_count = 0  

        self.agent2_action_0_count = 0  

        self.total_steps = 0   

 
 

   

 
 

   

 
 

   

 
 

   

 
 

    def demand(self):  

        distributions = {  

            1: {'dist': poisson, 'params': {'mu': self.dist_param['mu']}},  

            2: {'dist': binom, 'params': {'n': 10, 'p': 0.5}},  

            3: {'dist': randint, 'params': {'low': 0, 'high': 50}},  

            4: {'dist': geom, 'params': {'p': 0.5}},  

            5: {'dist': 'seasonal', 'params': None}  

        }  

 
 

        if self.dist < 5:  

            dist_info = distributions[self.dist]  

            dist_instance = dist_info['dist']  

            dist_params = dist_info['params']  

            if dist_instance is not None:  

                return dist_instance.rvs(**dist_params)  

        else:  

            current_date = datetime.datetime.now()  

            # Calculate the number of days in a year for that day (1 to 365) 

            t = current_date.timetuple().tm_yday  

            demand = np.round(  

                self.d_max / 3 +  

                self.d_max / 5 * np.cos(4 * np.pi * (2  + t) / self.periods) +  

                np.random.randint(0, self.d_var+ 1)  

            )  

            demand = min(demand, 30)  

            return demand  

 
 

   

 
 

   

 
 

    def transition(self, x, a_factory, a_retailer,V_F,d):  

        print('11',x)  

        print(a_factory)  

        print(a_retailer)  

        I = [x[0],x[8]]  

        BL = [x[1],x[9]]  

        SO =  [x[2],x[10]]  

        total_inventory_factory = a_factory + I[1]  

        factory_stockout = False  

        if  0 <= total_inventory_factory <= self.factory_capacity:  #### blance  

            I[1] = total_inventory_factory  

            if  I[1] < a_retailer: #### stockout  

                SO[1] = a_retailer-I[1]  

                if SO[1] > 0:  

                    factory_stockout = True  

                    # for agent_id in agent_ids:  

                    self.stockoutCount += 1  

                    a_retailer = I[1] #*********     order changed = real order/demand  

                I[1] = 0  

                BL[1] = 0  

            else:  

                I[1] -= a_retailer  

                SO[1] = 0  

                BL[1] = 0  

        elif total_inventory_factory > self.factory_capacity:  

                BL[1] = total_inventory_factory-self.factory_capacity  

                I[1] = total_inventory_factory  

                if I[1] < a_retailer: #### stockout  

                    SO[1] = a_retailer-I[1]  

                    if SO[1] > 0:  

                        factory_stockout = True  

                        # for agent_id in agent_ids:  

                        self.stockoutCount += 1  

                    I[1] = 0         

                else:  

                    I[1] -= a_retailer  

                    SO[1] = 0  

                    BL[1] = np.where(I[1] < self.factory_capacity, 0, BL[1])   

 
 

        I[1] = np.clip(I[1],0, self.factory_capacity-1)  

        BL[1] = np.clip(BL[1],0, self.factory_capacity-1)  

        SO[1] = np.clip(SO[1],0, self.factory_capacity-1)  

        if not factory_stockout:  

            total_inventory_retailer = a_retailer + I[0]  

            if  0 < total_inventory_retailer <= self.retailer_capacity:  

                I[0] = total_inventory_retailer  

                if  I[0] < d:  

                    SO[0] = d-I[0]  

                    if SO[0] > 0:  

                        # for agent_id in agent_ids:  

                        self.stockoutCount += 1  

                        d = I[0] #*********     order changed = real order/demand  

                    I[0] = 0  

                    BL[0] = 0  

                else:  

                    I[0] -= d  

                    SO[0] = 0  

                    BL[0] = 0  

            elif total_inventory_retailer > self.retailer_capacity:  

                # self.retailer_capacity < a_retailer+I[0]:  

                    BL[0] = total_inventory_retailer-self.retailer_capacity  

                    I[0] = total_inventory_retailer  

                    if d > I[0]:  

                        SO[0] = d-I[0]  

                        if SO[0] > 0:  

                            # for agent_id in agent_ids:  

                            self.stockoutCount += 1  

                        I[0] = 0  

                        # BL[1] = 0  

                    else:  

                        I[0] -= d  

                        SO[0] = 0  

                        BL[0] = np.where(I[0] < self.retailer_capacity, 0, BL[0])  

 
 

            I[0] = np.clip(I[0],0, self.retailer_capacity-1)  

            BL[0] = np.clip(BL[0],0, self.retailer_capacity-1)  

            SO[0] = np.clip(SO[0],0, self.retailer_capacity-1)  

 
 

        else:  

            a_retailer = 0 #*********     order changed = real order/demand  

            total_inventory_retailer = I[0]  

                # BL[0] = 0  

 
 

                # SO[0] = 0  

 
 

            if  I[0] < d:  

                SO[0] = d-I[0]  

                if SO[0] > 0:  

                    # for agent_id in agent_ids:  

                    # self.stockoutCount[0] += 1  

                    d = I[0] #*********     order changed = real order/demand  

                I[0] = 0  

                BL[0] = 0  

            else:  

                I[0] -= d  

                SO[0] = 0  

                BL[0] = 0  

    

 
 

        self.o_history.append(np.array([[a_retailer]], dtype=np.int64))  

        o_history = np.hstack(list(chain(*chain(*self.o_history))))  

        self.d_history.append(np.array([[d]], dtype=np.int64))  

        d_history = np.hstack(list(chain(*chain(*self.d_history))))  

        self.period +=1  

 
 

         

 
 

        x =  np.array([I[0],BL[0],SO[0],d_history[0],d_history[1],d_history[2],I[1]*V_F,self.period,I[1],BL[1],SO[1],o_history[0],o_history[1],o_history[2],0,self.period])  

        return x  

 
 

    def reward(self, x, a_retailer,p_retailer,a_factory, p_factory,d, y):   

        k = self.k   

        c = self.c   

        h = self.h  

        factory_reward = - c * max(min(y[8] + a_factory, self.factory_capacity) - y[8], 0)-h * y[8] -self.b*y[9]-self.s[1]*y[10]+ p_factory * max(min(x[8] + x[13], self.factory_capacity) - x[8], 0)   

        if self.task == 2:  

            factory_reward -= self.s[0]*y[1][2]  

        self.r[2] = factory_reward  

        # print('factory_reward',factory_reward)   

 
 

        # print('p_factory',p_factory)   

 
 

        self.day+= 1  

        retailer_reward =- p_factory * max(min(y[0] + y[13], self.retailer_capacity) - y[0], 0)-h * y[0] -self.b*y[1]-self.s[0]*y[2]  + p_retailer * max(min(x[0] +x[5], self.retailer_capacity) - x[0], 0)  

        # retailer_reward = -k * (a_retailer > 0) - p_factory * max(min(x[1][0] + a_retailer, self.retailer_capacity) - x[1][0], 0)-h * x[1][0] -self.b*x[1][1]-self.s[0]*x[1][2]  + p_retailer * max(min(x[1][0] + d, self.retailer_capacity) - y[1][0], 0)   

        if self.task == 3:  

            retailer_reward -= self.s[1]*y[2][2]  

        self.r[1] = retailer_reward 

        r = self.r[1] + self.r[2]  

        return r  

 
 

    def _seed(self, seed=None):   

        self.np_random, seed = seeding.np_random(seed)   

        return [seed]   

 
 

    def step(self, action):   

        global ep_count  

        ep_count += 1  

        obs_dict = self.state.copy()   

        demand = self.demand()   

        # actions[1][2] = 0  

 
 

        # if self.v_f == 2:  

 
 

        #     actions[2][2] = np.where(actions[2][2] >= 1, 1, 0)  

 
 

        # elif self.v_f == 3:  

 
 

        #     actions[2][2] = 1  

 
 

        # else:  

 
 

        #     actions[2][2] = 0  

 
 

              

 
 

              

 
 

        # Track the third value actions  

 
 

        # if actions[1][2] == 0:  

 
 

        #     self.agent1_action_1_count = 0  

 
 

        #     self.agent1_action_0_count += 1  

 
 

   

 
 

        # if actions[2][2] == 1:  

 
 

        #     self.agent2_action_1_count += 1  

 
 

        # else:  

 
 

        #     self.agent2_action_0_count += 1  

        # Increment the total step counter  

 
 

        self.total_steps += 1  

        # print('000',type(action))  

        # print('000',action[3])  

        observations_dict = self.transition(obs_dict, action[3], action[0],action[5], demand)   

        self.state = observations_dict   

        rewards = self.reward(obs_dict, action[0],action[1],  action[3], action[4], demand,observations_dict)   

        done = self.is_done()  

        # done["__all__"] = all(done.values())   

 
 

        # self.state_history.append(self.state.copy())  

 
 

        # self.reward_history.append(rewards.copy())  

 
 

        # self.demand_history.append(demand) 

 
 

        # self.action_history.append(actions)  

        return observations_dict, rewards, done,done, {}   

 
 

      

 
 

   

 
 

    

 
 

      

 
 

    def is_done(self):   

        # if  self.day[1] >= self.days[1]:  

        if self.stockoutCount >= self.stockoutCountS or self.day >= self.days:  

        # if self.day[agent_id] >= self.days[agent_id]:  

            done = True  

        else:  

            done = False 

        return done   

 
 
 

    # def reset(self,seed=2024): 

    def reset(self, seed=None, options=None): 

        self.re_num += 1  

        self.stockoutCount = 0  

        self.d_history = collections.deque(maxlen=self.demand_len)  

        for d in range(self.demand_len):  

 
 

            self.d_history.append(np.zeros((1,1), dtype = np.int32))  

        self.o_history = collections.deque(maxlen=self.demand_len)  

        for d in range(self.demand_len):  

            self.o_history.append(np.zeros((1,1), dtype = np.int32))  

        self.t = 0  

        self.day = 0  

        self.period = 0  

        self.r = {1: 0, 2: 0}  

        self.agent1_action_1_count = 0  

        self.agent1_action_0_count = 0  

        self.agent2_action_1_count = 0  

        self.agent2_action_0_count = 0  

        self.total_steps = 0   

 
 

        return self.state ,{} 

 
 

   

 
 

     

 
 

          

 
 

          

 
 

          

 
 

   

 
 

if __name__ == '__main__':  

 
 

     

 
 

    env = POSingAgent1W1F()  

 
 

   

 
 

   

 
 

    episides = 10  

 
 

    for i in range(episides):   

 
 

        obs,_ = env.reset()  

 
 

        while True:  

 
 

            print('+++++++++++++++++egthrget5++++++++',i,'++++++++++++++++++++++++++')  

 
 

   

 
 

            obs, rew, done, _,info = env.step(   

 
 

   

 
 

                env.action_space.sample()   

 
 

   

 
 

            )   

 
 

            if done :  

 
 

                break  

 
 

    assert done  

 
 

   

 
 

    print('obs',obs)   

 
 

   

 
 

    print('rew',rew)   

 
 

   

 
 

    print('done',done)   

 
 

   

 
 

    print('info',info)   

 
 

   

 
 

     

 