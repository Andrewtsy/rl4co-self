import torch
import importlib
import time
import csv

from rl4co.models import AttentionModelPolicy
from rl4co.envs import CVRPEnv
from rl4co.models.zoo.decomposition import DECOMP, BaryCenterDecomp

torch.manual_seed(42)

env = CVRPEnv(generator_params={'num_loc': 500})
td = env.reset(batch_size=[10])

PATH = "checkpoints/am-a2c-cvrp50.ckpt"
model_root = importlib.import_module("rl4co.models.rl")
model_cls = getattr(model_root, "A2C")
model_a2c50 = model_cls.load_from_checkpoint(PATH, load_baseline=False)
model_a2c50 = model_a2c50.to(torch.device("cpu"))
policy = model_a2c50.policy

# policy = AttentionModelPolicy(env_name=env.name)
decomp_pol = BaryCenterDecomp(10, policy=policy, naive_policy=policy)

model = DECOMP(env,
               policy=decomp_pol,
               baseline='rollout',
               train_data_size=1_000,
               val_data_size=100)

st = time.time()
out, inters = model.policy(td, env, 50, 0.1, 0.98)
t = time.time() - st

print(f"{out['reward'].mean()} reward in {t} seconds.")
