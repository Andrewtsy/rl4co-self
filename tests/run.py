import torch
import importlib
import time
import csv

from rl4co.models import AttentionModelPolicy
from rl4co.envs import CVRPEnv
from rl4co.models.zoo.decomposition import DECOMP, BaryCenterDecomp, Naive

torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = CVRPEnv(generator_params={'num_loc': 500})
td = env.reset(batch_size=[10]).to(device)

# PATH = "checkpoints/am-a2c-cvrp50.ckpt"
# model_root = importlib.import_module("rl4co.models.rl")
# model_cls = getattr(model_root, "A2C")
# model_a2c50 = model_cls.load_from_checkpoint(PATH, load_baseline=False)
# model_a2c50 = model_a2c50
# policy = model_a2c50.policy.to(device)

PATH = "checkpoints/am_cvrp50_2024_7_25.ckpt"
model_root = importlib.import_module("rl4co.models")
model_cls = getattr(model_root, "AttentionModel")
model_am = model_cls.load_from_checkpoint(PATH, load_baseline=False)
policy = model_am.policy.to(device)
naive_pol = Naive()

decomp_pol = BaryCenterDecomp(10, policy=policy, naive_policy=naive_pol).to(device)

model = DECOMP(env,
               policy=decomp_pol,
               baseline='rollout',
               train_data_size=1_000,
               val_data_size=100)

st = time.time()
out, inters = model.policy(td, env, 100, 0.1, 0.98, history=True)
t = time.time() - st

print(f"{out['reward'].mean()} reward in {t} seconds.")
