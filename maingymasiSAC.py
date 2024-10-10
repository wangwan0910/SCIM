
import time
import ray  
import argparse
from ray import air  
from ray import tune  
from ray.tune.logger import pretty_print  
from ray.rllib.algorithms.sac import SACConfig  
import gymnasium  
from gymnasium.spaces import Discrete,MultiDiscrete   
import os  
from gymnasium.utils import seeding   
import numpy as np   
import logging   
from datetime import datetime  
import logging  
import numpy as np  
from sgymnasi import (POSingAgent1W1F,POSAgent1W1F_V1T1,POSAgent1W1F_V1T2,POSAgent1W1F_V1T3,
                      POSAgent1W1F_V2T1,POSAgent1W1F_V2T2,POSAgent1W1F_V2T3,
                      POSAgent1W1F_V3T1,POSAgent1W1F_V3T2,POSAgent1W1F_V3T3)
from ray.tune.schedulers import ASHAScheduler  
from ray.tune.stopper import (CombinedStopper,  
                              MaximumIterationStopper,  
                              ExperimentPlateauStopper)  
import ray.rllib.algorithms.sac as sac  
from ray.rllib.utils import try_import_torch   

logger = logging.getLogger(__name__)   

# env = GymEnvironment()  
from ray.tune.logger import TBXLoggerCallback,TBXLogger #2.2.0
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
# from scenv0703 import SupplyChainEnvironment, Action, State 
# from plotrewards0803 import visualize_rewards 
from ray.rllib.env.multi_agent_env import MultiAgentEnv 
from ray.tune.registry import register_env  
from ray.rllib.algorithms.sac import SAC 
# Register the environment if not already registered 
import torch 
from ray.rllib.policy.sample_batch import SampleBatch 
import numpy as np 
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune import CLIReporter
 
 

def trial_dirname_creator(trial): 
    return f"trial_{trial.trial_id}" 

 


def parse_args():
    parser = argparse.ArgumentParser(description="POSAgent1W1F Environment Configuration")
    parser.add_argument('--env1', action='store_true', help='Use POSAgent1W1F_V1T1 environment')
    parser.add_argument('--env2', action='store_true', help='Use POSAgent1W1F_V1T2 environment')
    parser.add_argument('--env3', action='store_true', help='Use POSAgent1W1F_V1T3 environment')
    parser.add_argument('--env4', action='store_true', help='Use POSAgent1W1F_V2T1 environment')
    parser.add_argument('--env5', action='store_true', help='Use POSAgent1W1F_V2T2 environment')
    parser.add_argument('--env6', action='store_true', help='Use POSAgent1W1F_V2T3 environment')
    parser.add_argument('--env7', action='store_true', help='Use POSAgent1W1F_V3T1 environment')
    parser.add_argument('--env8', action='store_true', help='Use POSAgent1W1F_V3T2 environment')
    parser.add_argument('--env9', action='store_true', help='Use POSAgent1W1F_V3T3 environment')
    return parser.parse_args() 
 

if __name__ == '__main__':
    args = parse_args()
    
    if args.env1:
        envs = POSAgent1W1F_V1T1
    elif args.env2:
        envs = POSAgent1W1F_V1T2
    elif args.env3:
        envs = POSAgent1W1F_V1T3
    elif args.env4:
        envs = POSAgent1W1F_V2T1
    elif args.env5:
        envs = POSAgent1W1F_V2T2
    elif args.env6:
        envs = POSAgent1W1F_V2T3
    elif args.env7:
        envs = POSAgent1W1F_V3T1
    elif args.env8:
        envs = POSAgent1W1F_V3T2
    elif args.env9:
        envs = POSAgent1W1F_V3T3
    else:
        raise ValueError("Please specify one of --env1, ..., --env9")



    

    # # training a PPO agent
    # (results_PPO, best_result_PPO,
    # best_config_PPO, checkpoint_PPO) = train(algorithms['PPO'],
    #                                         config_PPO,
    #                                         VERBOSE)
    
    NUM_EPISODES = 250
    # number of episodes for RLib agents
    num_episodes_ray = 75000
    # stop trials at least from this number of episodes
    grace_period_ray = num_episodes_ray / 10
    # number of episodes to consider
    std_episodes_ray = 5.0
    # number of epochs to wait for a change in the episodes
    top_episodes_ray = NUM_EPISODES
    # name of the experiment (e.g., '2P2W' stands for two product types and two
    # distribution warehouses)
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    
 
     
     
    
    env = envs()
    save_dir = f"D{env.dist}B{env.b}_S{env.s}_V{env.v_f}T{env.task}_{now_str}"
    # dir to save plots
    plots_dir = 'plots'
    # creating necessary dirs
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    if not os.path.exists(f"{save_dir+'/'+plots_dir}"):
        os.makedirs(f"{save_dir+'/'+plots_dir}")
    # dir for saving Ray results
    ray_dir = 'ray_results'
    # creating necessary dir
    if not os.path.exists(f"{save_dir+'/'+ray_dir}"):
        os.makedirs(f"{save_dir+'/'+ray_dir}")
    
    
    # Set the framework to 'tf2' and enable eager tracing if necessary
    config = SACConfig().framework('tf2', eager_tracing=True)

    config = SACConfig().environment(env=envs)
    # Single-agent policies: define a single policy
    config = config.training(
        gamma=0.95,  # Example hyperparameter, adjust as needed
        tau=0.005,
        lr=3e-4, 
        #buffer_size=1000000,
        #exploration_config={
         #   "type": "StochasticSampling",
          #  "temperature": 1.0,
        #},
    )
   

    config = (
    SACConfig()
    .environment(env=envs)
    .training(
        train_batch_size=128,
        gamma=0.95,
    )
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    # Directly specify the policy if the API all
)
    

    

    # save_dir = './envs'
    stop_cf = {"timesteps_total":  200000}#10000000


    # 配置调度器和停止条件
    scheduler = ASHAScheduler(
        time_attr='episode_reward_mean',
        max_t=num_episodes_ray,
        grace_period=grace_period_ray,
        reduction_factor=5
    )

    stopper = CombinedStopper(
        ExperimentPlateauStopper(
            metric='episode_reward_mean',
            std=std_episodes_ray,
            top=top_episodes_ray,
            mode='max',
            patience=5
        ),
        MaximumIterationStopper(max_iter=num_episodes_ray)
    )

    # reporter = JupyterNotebookReporter(overwrite=True)
    reporter = CLIReporter(
    metric_columns=["episode_reward_mean", "episodes_total", "training_iteration"])
    
    run_config = RunConfig(
        # local_dir=save_dir,
        stop=stop_cf,
        name="single_agent_train_run",
  
        checkpoint_config=CheckpointConfig(
            checkpoint_score_attribute = "episode_reward_mean",
            checkpoint_score_order = "max",
            checkpoint_frequency=2,
            checkpoint_at_end=True,
            num_to_keep = 10
        ),
        progress_reporter=reporter,
        local_dir=os.path.join(os.getcwd(), save_dir, ray_dir),
        callbacks=[
            TBXLoggerCallback(),
        ],
        verbose=3, #3
        
        log_to_file = True,
        )
    
    
    # tuner = tune.Tuner(
    #         "PPO", run_config=run_config, param_space=config,
    #     )
    
    # results = tuner.fit()

    start = time.time()
    #iterator = trange(3)
    #for epoch in iterator:
    # 定义Tuner
    tuner = tune.Tuner(
        "SAC",
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(
            metric='episode_reward_mean',
            mode='max',
            scheduler=scheduler,
            num_samples=5,  # 调优的样本数量，可以根据需要调整
        ),
        # progress_reporter=reporter,
        run_config=run_config,
        # max_failures=5,
        # verbose=3
    )

    # 运行调优
    results = tuner.fit()

    total_time_taken = time.time() - start
    print(f"Total number of models: {len(results)}")
    print(f"Total time taken: {total_time_taken/60:.2f} minutes")
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    best_result_df = best_result.metrics_dataframe
    best_config = best_result.config
    best_checkpoint = best_result.checkpoint
   

    print(f"\nbest_result saved at {best_result}")
    print(f"\nbest_result_df saved at {best_result_df}")
    print(f"\nbest_config saved at {best_config}")
    print(f"\nbest_checkpoint saved at {best_checkpoint}")

    # 停止 Ray
    ray.shutdown()
