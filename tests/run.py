import torch
import importlib
import time
import csv

from rl4co.envs import CVRPEnv
from rl4co.models import REINFORCE
from rl4co.models import AttentionModelPolicy, PolyNet
from rl4co.models.zoo.decomposition import DECOMP, BaryCenterDecomp, Naive
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.loggers import WandbLogger

torch.manual_seed(42)

accel = str("cuda" if torch.cuda.is_available()
    else (
        "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
        else "cpu"
    )
)

device = torch.device(accel)
print(device)

env = CVRPEnv(generator_params={'num_loc': 150}).to(device)

# td = env.reset(batch_size=[10]).to(device)

policy_params = {"its": 4, "temp": 0.1, "r": 0.9}
policy = AttentionModelPolicy(env_name=env.name, 
                              embed_dim=128,
                              num_encoder_layers=6,
                              num_heads=8,
                              **policy_params
                            ).to(device)
naive_pol = Naive().to(device)
decomp_pol = BaryCenterDecomp(3, policy=policy, naive_policy=naive_pol).to(device)

model = DECOMP(env,
               policy=decomp_pol,
               baseline="rollout",
               batch_size=64,
               train_data_size=128,
               val_data_size=64,
               optimizer_kwargs={"lr": 1e-4},
               **policy_params
               ).to(device)

# model = PolyNet(env,
#                 k=10,
#                 val_num_solutions=16,
#                 batch_size=64,
#                 train_data_size=1_000,
#                 val_data_size=500,
#                 optimizer_kwargs={"lr": 1e-4}).to(device)

# logger = WandbLogger(project="rl4co", name="polynet-train-0")

trainer = RL4COTrainer(
    max_epochs=4,
#     logger=logger,
#     log_every_n_steps=20,
    accelerator="cpu"
)

trainer.fit(model)
