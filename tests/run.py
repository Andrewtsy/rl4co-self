import torch
import importlib
import time
import csv

from ..rl4co.models import AttentionModelPolicy
from ..rl4co.envs import CVRPEnv
from ..rl4co.models.zoo.decomposition import DECOMP, BaryCenterDecomp

torch.manual_seed(42)

env = CVRPEnv(generator_params={'num_loc': 500})
policy = AttentionModelPolicy(env_name=env.name)
decomp_pol = BaryCenterDecomp(4, policy=model_a2c50.policy, naive_policy=model_a2c50.policy)

model = DECOMP(env,
               policy=decomp_pol,
               baseline='rollout',
               train_data_size=1_000,
               val_data_size=100)

td = env.reset(batch_size=[10])

st = time.time()
out, inters = model.policy(td, env, 10, 0.1, 0.98)
t = time.time() - st

print(f"{out['reward'].mean()} reward in {t} seconds.\n {b} solutions improved.")

with open("history.csv", "w") as fhandle:
    writer = csv.writer(fhandle)
    writer.writerows([float(el)] for el in inters)
