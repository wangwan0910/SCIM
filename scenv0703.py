# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:34:12 2024

@author: ww1a23
"""
import numpy as np
from itertools import chain
import collections
from gymnasium.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv


SEED = 2023


class State:
    def __init__(self,product_types_num,distr_warehouses_num,T,lead_times,demand_history,t=0):
        self.product_types_num = product_types_num
        self.factory_stocks = np.zeros(
            (self.product_types_num,),
            dtype = np.int32)
        self.distr_warehouses_num = distr_warehouses_num
        self.distr_warehouses_stocks =np.zeros(
           ( self.distr_warehouses_num, self.product_types_num),
        dtype = np.int32)
        # print(' self.distr_warehouses_stocks', self.distr_warehouses_stocks) #[[0] [0]]
        self.T = T
        self.lead_times = lead_times
        
        self.demand_history = demand_history
        self.t = t
    def to_array(self):
        if len(self.lead_times) >0:
            # print( '333333333333333',np.concatenate((self.factory_stocks, 
            #                        np.hstack(list(chain(*chain(*self.lead_times)))),
            #                        self.distr_warehouses_stocks.flatten(),
            #                        np.hstack(list(chain(*chain(*self.demand_history)))),
            #                        [self.t])))
            return np.concatenate((self.factory_stocks, 
                                   np.hstack(list(chain(*chain(*self.lead_times)))),
                                   self.distr_warehouses_stocks.flatten(),
                                   np.hstack(list(chain(*chain(*self.demand_history)))),
                                   [self.t]))
        else:
            # print('555555',self.factory_stocks)
            # print('666666',self.distr_warehouses_stocks.flatten())
            # print('777777',self.demand_history)
            # print('88888',self.t)
            # print('444444444444',np.concatenate((self.factory_stocks, 
            #                       self.distr_warehouses_stocks.flatten(),
            #                       np.hstack(list(chain(*chain(*self.demand_history)))),
            #                       [self.t])))
            return np.concatenate((self.factory_stocks, 
                                  self.distr_warehouses_stocks.flatten(),
                                  np.hstack(list(chain(*chain(*self.demand_history)))),
                                  [self.t]))
    def stock_levels(self):
        # print('555555555',np.concatenate((
        #     self.factory_stocks, 
        #     self.distr_warehouses_stocks.flatten())))
        return np.concatenate((
            self.factory_stocks, 
            self.distr_warehouses_stocks.flatten()))
    
class Action:
    def __init__(self, product_types_num,distr_warehouses_num):
        self.production_level = np.zeros(
            (product_types_num,),
            dtype = np.int32)
        self.shipped_stocks = np.zeros(
            (distr_warehouses_num, product_types_num),
            dtype = np.int32)

class SupplyChainEnvironment(MultiAgentEnv):
    def __init__(self, seed=2023):
        super().__init__()
        self.product_types_num = 1
        self.distr_warehouses_num = 2
        self.lead_times_len = 0
        self.T = 7 + (2 * self.lead_times_len)
        self.demand_type = 'probabilistic'
        self.d_max = np.array([5], np.int32)
        self.d_var = np.array([5], np.int32)
        self.sale_prices = np.array([0], np.int32)
        self.production_costs = np.array([1], np.int32)
        self.storage_capacities = np.array([[20], [10], [10]], np.int32)
        self.storage_costs = np.array([[.1], [1], [1]], np.float32)
        self.prod_level_max = np.array([15], np.int32)
        self.transportation_capacities = np.array([[3], [3]], np.int32)
        self.transportation_costs_fixed = np.array([[.7], [.7]], np.float32)
        self.transportation_costs_unit = np.array([[.03], [.03]], np.float32)
        self.penalty_costs = 10 * self.production_costs
        self.excess_demand = 'backorder'
        self.demand_history_len = 1
        self.reset(seed=seed)
        factory_size = self.product_types_num
        lead_times_size = (self.lead_times_len * self.distr_warehouses_num * self.product_types_num)
        distr_warehouses_size = (self.distr_warehouses_num * self.product_types_num)
        low_act = np.zeros(((self.distr_warehouses_num + 1) * self.product_types_num), dtype=np.int32)
        high_act = np.zeros(((self.distr_warehouses_num + 1) * self.product_types_num), dtype=np.int32)
        high_act[:factory_size] = self.prod_level_max
        high_act[factory_size:] = self.storage_capacities.flatten()[self.product_types_num:]
        self.action_space = Box(low=low_act, high=high_act, dtype=np.int32)
        low_obs = np.zeros((len(self.initial_state().to_array()),), dtype=np.int32)
        low_obs[factory_size + lead_times_size: factory_size + lead_times_size + distr_warehouses_size] = np.array(
            [-(self.d_max + self.d_var) * self.T] * self.distr_warehouses_num).flatten()
        high_obs = np.zeros((len(self.initial_state().to_array()),), dtype=np.int32)
        high_obs[:factory_size] = self.storage_capacities[:1].flatten()
        high_obs[factory_size: factory_size + lead_times_size] = np.repeat(self.storage_capacities[:1],
                                                                           self.distr_warehouses_num * self.lead_times_len)
        high_obs[
        factory_size + lead_times_size: factory_size + lead_times_size + distr_warehouses_size] = self.storage_capacities[
                                                                                                  1:].flatten()
        high_obs[factory_size + lead_times_size + distr_warehouses_size: len(high_obs) - 1] = np.array(
            [self.d_max + self.d_var] * len(list(chain(*self.demand_history)))).flatten()
        high_obs[len(high_obs) - 1] = self.T
        self.observation_space = Box(low=low_obs, high=high_obs, dtype=np.int32)

        print(f"\n-- supplychainenvironment-- __init__"
              f"\n product_types_num_is"
              f"{self.product_types_num}"
              f"\n distr_warehouses_num is"
              f"{self.distr_warehouses_num}"
              f"\nT is"
              f"{self.T}"
              f"\ndemand_type is"
              f"{self.demand_type}"
              f"\nd_max is "
              f"{self.d_max}"
              f"\nd_var is "
              f"{self.d_var}"
              f"\nsale_prices is "
              f"{self.sale_prices}"
              f"\nproduction_costs is "
              f"{self.production_costs}"
              f"\nstorage_capacities is "
              f"{self.storage_capacities}"
              f"\nstorage_costs is "
              f"{self.storage_costs}"
              f"\nprod_level_max is "
              f"{self.prod_level_max}"
              f"\ntransportation_capacities is "
              f"{self.transportation_capacities}"
              f"\ntransportation_costs_fixed is "
              f"{self.transportation_costs_fixed}"
              f"\ntransportation_costs_unit is "
              f"{self.transportation_costs_unit}"
              f"\npenalty_costs is "
              f"{self.penalty_costs}"
              f"\nexcess_demand is "
              f"{self.excess_demand}"
              f"\nlead_times_len is "
              f"{self.lead_times_len}"
              f"\ndemand_history_len is "
              f"{self.demand_history_len}")

        self.reset(seed =seed)

    def reset(self, seed=None, options=None):
        if seed:
            self.demand_random_generator = np.random.default_rng(seed=seed)
        self.lead_times = collections.deque(maxlen=self.lead_times_len)
        self.demand_history = collections.deque(maxlen=self.demand_history_len)
        if self.lead_times_len > 0:
            for l in range(self.lead_times_len):
                self.lead_times.appendleft(
                    np.zeros((self.distr_warehouses_num, self.product_types_num), dtype=np.int32))

        for d in range(self.demand_history_len):
            self.demand_history.append(np.zeros((self.distr_warehouses_num, self.product_types_num), dtype=np.int32))
        self.t = 0
        return {
            "warehouse": self.initial_state().to_array(),
            "retailer": self.initial_state().to_array()
        }, {}

    def generate_demand(self,t):
        if self.demand_type  == 'stationary':
            demands = np.fromfunction(
                lambda j ,i : self.stationary_demand(j+1,i+1,t),
                (self.distr_warehouses_num,self.product_types_num),
                dtype = np.int32)
            
        elif self.demand_type == 'probabilistic':
            # print(self.stationary_demand(j+1,i+1,t))
            # print(self.distr_warehouses_num) #2
            # print(self.product_types_num) #1
            # print(self.d_var[0]) #5
           
            demands = np.fromfunction(
                lambda j, i :self.stationary_demand(j+1,i+1,t),
                (self.distr_warehouses_num,self.product_types_num),
                dtype = np.int32)+ \
                    self.demand_random_generator.choice(
                        [0,self.d_var[0]],
                        p = [.5,.5],
                        size = (self.distr_warehouses_num,
                                self.product_types_num))
                    # print('p:',p)
        elif self.demand_type == 'negbinomial':
            demands = np.fromfunction(
                lambda j ,i :self.stationary_demand(j+1,i+1,t),
                (self.distr_warehouses_num, self.product_types_num),
                dtype = np.int32) + \
                self.demand_random_generator.negative_binomial(
                    self.d_var[0], 
                    .7,
                size =(self.distr_warehouses_num,
                        self.product_types_num))
        elif self.demand_type == 'stochastic':
            demands = np.fromfunction(lambda j,i :self.stationary_demand(j+1, i+1,t),
                                      (self.distr_warehouses_num,self.product_types_num),
                                        dtype = np.int32) + \
                                          self.demand_random_generator.integers(
                                              0, self.d_var[0]+1, 
                                              size =(self.distr_warehouses_num, self.product_types_num))
        # demands = [[2],[2]]
        print('demands',demands)
        return demands
    
    
    
    
    
    
    
    
    
    def generate_episode_demands(self):
        demands_episode = []
        for t in range(self.T):
            if (t<self.lead_times_len):
                demands = np.zeros(
                    (self.distr_warehouses_num,self.product_types_num),
                    dtype = np.int32)
                print('222',demands)
            else:
                demands = self.generate_demand(t)
                # print('333',demands) # [[7.] [7.]]
            demands_episode.append(demands)
        demands_episode = np.array(demands_episode)
        print('11111self.lead_times_len',self.lead_times_len) 
        return demands_episode
    
    def stationary_demand(self, j ,i, t):
        demand = np.round(
            self.d_max[i-1]/2 +
            self.d_max[i-1]/2*np.sin(-5*np.pi*(t)/self.T))
        # print('demand1111111111',demand)  #[[2.] [2.]]
        return demand





    def initial_state(self):
        return State(self.product_types_num, self.distr_warehouses_num, self.T, list(self.lead_times), list(self.demand_history))

    def step(self, action_dict):
        state = self.initial_state()
        action_warehouse = action_dict["warehouse"]
        action_retailer = action_dict["retailer"]
        action_obj_warehouse = Action(self.product_types_num, self.distr_warehouses_num)
        action_obj_retailer = Action(self.product_types_num, self.distr_warehouses_num)
        action_obj_warehouse.production_level = np.minimum(
            action_warehouse[:self.product_types_num],
            self.prod_level_max
        )
        action_obj_warehouse.shipped_stocks = action_warehouse[
                                              self.product_types_num:
                                              ].reshape(self.distr_warehouses_num, self.product_types_num)

        action_obj_retailer.production_level = np.minimum(
            action_retailer[:self.product_types_num],
            self.prod_level_max
        )
        action_obj_retailer.shipped_stocks = action_retailer[
                                             self.product_types_num:
                                             ].reshape(self.distr_warehouses_num, self.product_types_num)

        next_state_warehouse, reward_warehouse, done_warehouse = self.supply_chain_step(state, action_obj_warehouse)
        next_state_retailer, reward_retailer, done_retailer = self.supply_chain_step(state, action_obj_retailer)

        done = done_warehouse or done_retailer
        truncated = {"warehouse": done, "retailer": done, "__all__": done}
        terminated = {"warehouse": done, "retailer": done, "__all__": done}
        info = {}  # 任何额外的信息

        return {
            "warehouse": next_state_warehouse.to_array(),
            "retailer": next_state_retailer.to_array()
        }, {
            "warehouse": reward_warehouse,
            "retailer": reward_retailer
        }, terminated, truncated, {
            "warehouse": info,
            "retailer": info
        }
    def supply_chain_step(self, state, action):
        if self.t < self.lead_times_len:
            demands = np.zeros(
                (self.distr_warehouses_num, self.product_types_num),
                dtype=np.int32
            )
        else:
            demands = self.generate_demand(self.t)

        next_state = State(self.product_types_num, self.distr_warehouses_num,
                           self.T, list(self.lead_times), list(self.demand_history),
                           self.t + 1)
        if self.lead_times_len > 0:
            distr_warehouses_stocks = np.minimum(
                np.add(state.distr_warehouses_stocks,
                       self.lead_times[self.lead_times_len - 1]),
                self.storage_capacities[1:]
            )
            next_state.distr_warehouses_stocks = np.subtract(
                distr_warehouses_stocks,
                demands
            )
        factory_stocks = np.minimum(
            np.add(state.factory_stocks,
                   np.minimum(
                       action.production_level,
                       self.prod_level_max
                   )),
            self.storage_capacities[0]
        )
        shipped_stocks = np.minimum(
            np.maximum(factory_stocks,
                       np.zeros((self.product_types_num,), dtype=np.int32)),
            np.sum(action.shipped_stocks, axis=0)
        )
        # Create a writable copy of action.shipped_stocks
        action_shipped_stocks_copy = np.copy(action.shipped_stocks)
        for i in range(self.product_types_num):
            if np.sum(action_shipped_stocks_copy, axis=0)[i] > shipped_stocks[i]:
                j_indexes = np.arange(self.distr_warehouses_num, dtype=int)
                while np.sum(action_shipped_stocks_copy, axis=0)[i] > shipped_stocks[i]:
                    j_index = np.random.choice(j_indexes)
                    if action_shipped_stocks_copy[j_index][i] > 0:
                        action_shipped_stocks_copy[j_index][i] -= 1
                    else:
                        j_indexes = j_indexes[j_indexes != j_index]
        next_state.factory_stocks = np.subtract(
            factory_stocks,
            np.sum(action_shipped_stocks_copy, axis=0)
        )
        if self.lead_times_len > 0:
            self.lead_times.append(action_shipped_stocks_copy)
        else:
            distr_warehouses_stocks = np.minimum(
                np.add(state.distr_warehouses_stocks,
                       action_shipped_stocks_copy),
                self.storage_capacities[1:]
            )
            next_state.distr_warehouses_stocks = np.subtract(
                distr_warehouses_stocks, demands
            )
        self.demand_history.append(demands)
        self.t += 1
        unsatisfied_demands = np.zeros(
            (self.distr_warehouses_num, self.product_types_num),
            dtype=np.int32
        )
        for i in range(self.product_types_num):
            for j in range(self.distr_warehouses_num):
                if distr_warehouses_stocks[j][i] >= 0 and next_state.distr_warehouses_stocks[j][i] < 0:
                    unsatisfied_demands[j][i] = np.abs(next_state.distr_warehouses_stocks[j][i])
                elif distr_warehouses_stocks[j][i] < 0 and next_state.distr_warehouses_stocks[j][i] < 0:
                    unsatisfied_demands[j][i] = demands[j][i]
        total_revenues = np.dot(
            self.sale_prices,
            np.sum(np.subtract(demands, unsatisfied_demands), axis=0)
        )
        total_production_costs = np.dot(self.production_costs, action.production_level)
        total_transportation_costs_fixed = np.dot(
            np.ceil(np.divide(action_shipped_stocks_copy,
                              self.transportation_capacities)).flatten(),
            self.transportation_costs_fixed.flatten()
        )
        total_transportation_costs_unit = np.dot(
            self.transportation_costs_unit.flatten(),
            action_shipped_stocks_copy.flatten()
        )
        total_transportation_costs = np.add(total_transportation_costs_fixed,
                                            total_transportation_costs_unit)
        total_storage_costs = np.dot(
            self.storage_costs.flatten(),
            np.maximum(next_state.stock_levels(),
                       np.zeros(((self.distr_warehouses_num + 1) * self.product_types_num),
                                dtype=np.int32))
        )
        total_penalty_costs = -np.dot(
            self.penalty_costs,
            np.add(
                np.sum(
                    np.minimum(next_state.distr_warehouses_stocks,
                               np.zeros((self.distr_warehouses_num, self.product_types_num),
                                        dtype=np.int32)),
                    axis=0),
                np.minimum(next_state.factory_stocks,
                           np.zeros((self.product_types_num,),
                                    dtype=np.int32))
            )
        )
        reward = total_revenues - total_production_costs - \
                 total_storage_costs - total_transportation_costs - \
                 total_penalty_costs
        if self.excess_demand == 'lost-sales':
            next_state.distr_warehouses_stocks = np.maximum(
                next_state.distr_warehouses_stocks,
                np.zeros((self.distr_warehouses_num, self.product_types_num),
                         dtype=np.int32)
            )
        return next_state, reward, self.t == self.T - 1


                
        
        
        
        
        
        
        
        