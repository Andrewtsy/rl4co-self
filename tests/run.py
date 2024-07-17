import torch
import importlib
import time

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.decomposition import DECOMP, BaryCenterDecomp

torch.manual_seed(42)

PATH = "checkpoints/am-a2c-cvrp50.ckpt"
model_root = importlib.import_module("rl4co.models.rl")
model_cls = getattr(model_root, "A2C")
model_a2c50 = model_cls.load_from_checkpoint(PATH, load_baseline=False)
model_a2c50 = model_a2c50.to(torch.device("cpu"))

decomp_pol = BaryCenterDecomp(4, policy=model_a2c50.policy, naive_policy=model_a2c50.policy)

env = CVRPEnv(generator_params={'num_loc': 500})
model = DECOMP(env,
               policy=decomp_pol,
               baseline='rollout',
               train_data_size=1_000,
               val_data_size=100,
               policy_kwargs={"num_subproblems": 10})

td = env.reset(batch_size=[4])

st = time.time()
out, b = model.policy(td, env, 5, 1, 1)
t = time.time() - st

print(f"{out['reward']} reward in {t} seconds.\n {b} solutions improved.")