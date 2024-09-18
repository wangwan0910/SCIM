import os

path = "/\\1P2W_2024-05-26_20-52-22\\ray_results\\PPO_2024-05-26_20-52-37\\PPO_SupplyChain_d47ea_00000_0_lr=0.0001,fcnet_activation=relu,fcnet_hiddens=64_64,num_sgd_iter=15,rollout_fragment_length=auto,sgd_2024-05-26_20-52-37"
# sub_paths = path.split("/")
# current_path = ""
# for sub_path in sub_paths:
#     current_path = os.path.join(current_path, sub_path)
#     if not os.path.exists(current_path):
#         os.mkdir(current_path)
os.makedirs(path)