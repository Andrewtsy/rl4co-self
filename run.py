import torch
import importlib
import time
import csv

from rl4co.models import AttentionModelPolicy
from rl4co.envs import CVRPEnv
from rl4co.models.zoo.decomposition import DECOMP, BaryCenterDecomp

torch.manual_seed(42)

env = CVRPEnv(generator_params={'num_loc': 200})
policy = AttentionModelPolicy(env_name=env.name, 
                              embed_dim=128,
                              num_encoder_layers=3,
                              num_heads=8,
                            )
decomp_pol = BaryCenterDecomp(4, policy=policy, naive_policy=policy)
model = DECOMP(env,
               policy=decomp_pol,
               baseline='rollout',
               train_data_size=1_000,
               val_data_size=100)

td = env.reset(batch_size=[4])

st = time.time()
out, inters = model.policy(td, env, 100, 0.1, 0.98)
t = time.time() - st

print(f"{out['reward'].mean()} reward in {t} seconds.")

# with open("history.csv", "w") as fhandle:
#     writer = csv.writer(fhandle)
#     writer.writerows([float(el)] for el in inters)
