from ray.tune.logger import pretty_print
import os
from datetime import datetime
import matplotlib.pyplot as plt
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, ExperimentPlateauStopper
import logging
import numpy as np
from scenv0703 import SupplyChainEnvironment, Action, State
from plotrewards0803 import visualize_rewards
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
# Register the environment if not already registered
import torch
from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
env_instance = SupplyChainEnvironment()
NUM_EPISODES = 250
num_episodes_ray = 7500
grace_period_ray = num_episodes_ray / 10
std_episodes_ray = 5
top_episodes_ray = NUM_EPISODES
SEED = 2023
plots_dir = 'plots'
now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
ray_dir = 'ray_results'
local_dir = f"{env_instance.product_types_num}P{env_instance.distr_warehouses_num}W_{now_str}"

def trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

def train_multi_agent(config, verbose, num_episodes_ray, grace_period_ray, std_episodes_ray, top_episodes_ray,
                      local_dir, ray_dir):
    ray.shutdown()
    ray.init(log_to_driver=False)
    print('num_episodes_ray', num_episodes_ray)
    analysis = tune.run(
        "PPO",
        config=config,
        metric='episode_reward_mean',
        mode='max',
        scheduler=ASHAScheduler(
            time_attr='episodes_total',
            max_t=num_episodes_ray,
            grace_period=grace_period_ray,
            reduction_factor=5),
        stop=CombinedStopper(
            ExperimentPlateauStopper(
                metric='episode_reward_mean',
                patience=5),
            MaximumIterationStopper(max_iter=num_episodes_ray)
        ),
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        checkpoint_score_attr='episode_reward_mean',
        max_failures=5,
        verbose=verbose,
        trial_dirname_creator=trial_dirname_creator,
        local_dir=os.getcwd() + '/' + local_dir + '/' + ray_dir
    )
    results_df = analysis.dataframe()
    best_result_df = results_df.sort_values('episode_reward_mean', ascending=False).iloc[0]
    best_config = analysis.best_config
    best_checkpoint = analysis.best_checkpoint
    print(f'\n checkpoint saved at {best_checkpoint}')
    ray.shutdown()
    return results_df, best_result_df, best_config, best_checkpoint

def result_df_as_image(result_df, algorithm, local_dir=local_dir, plots_dir=plots_dir):
    if not os.path.exists(f"{local_dir}/{plots_dir}/{algorithm}"):
        os.makedirs(f"{local_dir}/{plots_dir}/{algorithm}")
    f = open(f"{local_dir}/{plots_dir}/{algorithm}/best_result_{algorithm}.tex", 'w', encoding='utf-8')
    f.write(result_df.to_frame().T.style.to_latex())
    f.close()

def visualize_rewards(results_df, algorithm, local_dir=local_dir, plots_dir=plots_dir):
    rewards = results_df['episode_reward_mean'].values
    plt.figure(figsize=(15, 5))
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title(f'Mean Reward for {algorithm}')
    plot_path = f'{local_dir}/{plots_dir}/{algorithm}'
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f'{plot_path}/rewards.pdf', format='pdf', bbox_inches='tight')

def visualize_demand(env, num_episodes=1, local_dir=local_dir, plots_dir=plots_dir, seed=SEED):
    if env.distr_warehouses_num <= 3 and env.product_types_num <= 2:
        env.reset(seed)
        episode_duration = env.T - (2 * env.lead_times_len)
        demands_episodes = []
        for _ in range(num_episodes):
            demands_episode = env.generate_episode_demands()
            demands_episodes.append(demands_episode)
        demands_episodes = np.array(demands_episodes)
        demands_mean = np.array([np.mean(d, axis=0) for d in zip(*demands_episodes)])
        demands_std = np.array([np.std(d, axis=0) for d in zip(*demands_episodes)])
        plt.figure(figsize=(15, 5))
        plt.xlabel('time steps')
        plt.ylabel('demand value')
        plt.xticks(np.arange(1, episode_duration + 1))
        plt.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=True)
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.tight_layout()
        color = [['b', 'b'], ['g', 'g'], ['r', 'r']]
        line_style = [['b-', 'b--'], ['g-', 'g--'], ['r-'], ['r--']]
        timesteps = np.arange(1, episode_duration + 1)
        for j in range(env.distr_warehouses_num):
            for i in range(env.product_types_num):
                if env.product_types_num == 1:
                    plt.plot(timesteps, demands_mean[:, j, i], line_style[j][i], label=f'WH{j + 1}')
                else:
                    plt.plot(timesteps, demands_mean[:, j, i], line_style[j][i], label=f'WH{j + 1}, P{i + 1}')
                plt.fill_between(timesteps, demands_mean[:, j, i] - demands_std[:, j, i],
                                 demands_mean[:, j, i] + demands_std[:, j][i], color=color[j][i], alpha=.2)
        plt.legend()
        plot_path = f'{local_dir}/{plots_dir}'
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f'{local_dir}/{plots_dir}/demand.pdf', format='pdf', bbox_inches='tight')

def save_env_settings(env, local_dir=local_dir, plots_dir=plots_dir):
    f = open(f'{local_dir}/{plots_dir}/env_settings.txt', 'w', encoding='utf-8')
    f.write(f'--supplychainenvironment----'
            f'\nproduct_types_num is {env.product_types_num}'
            f'\ndistr_warehouses_num is {env.distr_warehouses_num}'
            f'\nT is {env.T}'
            f'\ndemand_type is {env.demand_type}'
            f'\nd_max is {env.d_max}'
            f'\nd_var is {env.d_var}')
    f.close()

if __name__ == '__main__':
    num_episodes_ray = 7500
    grace_period_ray = num_episodes_ray / 10
    std_episodes_ray = 5.0
    top_episodes_ray = NUM_EPISODES
    env_instance = SupplyChainEnvironment()
    def env_creator(env_config):
        return SupplyChainEnvironment()
    register_env("supply_chain_env", env_creator)

    config = {
        "env": "supply_chain_env",
        "env_config": {},
        "multiagent": {
            "policies": {
                "policy1": (None, env_instance.observation_space, env_instance.action_space, {
                    "gamma": 0.99,
                    "model": {
                        "fcnet_hiddens": [256, 256],
                        "fcnet_activation": "relu",
                    },
                    "lr": 0.001,
                }),
                "policy2": (None, env_instance.observation_space, env_instance.action_space, {
                    "gamma": 0.95,
                    "model": {
                        "fcnet_hiddens": [128, 128],
                        "fcnet_activation": "tanh",
                    },
                    "lr": 0.0001,
                }),
            },
            "policy_mapping_fn": lambda agent_id, episode, **kwargs: "policy1" if agent_id == "warehouse" else "policy2",
        },
        "framework": "torch",
        "seed": 2023,
        "log_level": "DEBUG",
        "horizon": env_instance.T - 1,
        "batch_mode": "complete_episodes",
        "lr": 0.0001,
        "gamma": 0.99,
        "rollout_fragment_length": 20,
        "train_batch_size": 4000,
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 512,
    }

    logger = logging.getLogger('LOGGING_SCIMAI_GYM_VS')
    logger.setLevel((logging.INFO))
    VERBOSE = 3 if logger.level == 10 else 0

    # Training
    results_df, best_result, best_config, checkpoint = train_multi_agent(
        config, VERBOSE, num_episodes_ray, grace_period_ray, std_episodes_ray, top_episodes_ray, local_dir, ray_dir
    )

    visualize_demand(env_instance, NUM_EPISODES)
    save_env_settings(env_instance)
    result_df_as_image(best_result, 'PPO')
    visualize_rewards(results_df, 'PPO')
    def env_creator(env_config):
        return SupplyChainEnvironment()
    register_env("supply_chain_env", env_creator)
    ray.init(log_to_driver=False)
    ppo_trainer = PPO(config=config)
    ppo_trainer.restore(checkpoint)
    env = env_creator({})
    state = env.reset()
    done = False
    cumulative_reward = 0
    def policy_mapping_fn(agent_id):
        if agent_id == "warehouse":
            return "policy1"
        elif agent_id == "retailer":
            return "policy2"
        else:
            return "default_policy"
    while not done:
        if isinstance(state, tuple) and isinstance(state[0], dict):
            observation_dict, _ = state
        elif isinstance(state, dict):
            observation_dict = state
        else:
            raise ValueError(
                f"Expected state to be a tuple with a dictionary or a dictionary, but got {type(state)}: {state}")
        actions = {}
        for agent_id, observation in observation_dict.items():
            observation_array = np.array(observation, dtype=np.float32)
            input_dict = torch.tensor(observation_array)
            policy_id = policy_mapping_fn(agent_id)
            action, _, _ = ppo_trainer.get_policy(policy_id).compute_single_action(input_dict, explore=False)
            actions[agent_id] = action
        step_output = env.step(actions)
        print(f"step_output: {step_output}")
        if not (isinstance(step_output, tuple) and len(step_output) == 5):
            raise ValueError(
                f"Expected step_output to be a tuple of length 5, but got {type(step_output)}: {step_output}")
        state, rewards, dones, infos, _ = step_output

        # 检查 state 是否是期望的类型
        if isinstance(state, dict):
            state = (state, None)  # 将字典包装成元组

        cumulative_reward += sum(rewards.values())

        if dones['__all__']:
            done = True

    print("Cumulative reward:", cumulative_reward)
    ray.shutdown()