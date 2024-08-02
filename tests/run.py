import torch
import importlib
import time
import csv

from rl4co.models import AttentionModelPolicy
from rl4co.envs import CVRPEnv
from rl4co.models.zoo.decomposition import DECOMP, BaryCenterDecomp, Naive
from rl4co.models import REINFORCE
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.loggers import WandbLogger

torch.manual_seed(42)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else (
        "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
        else "cpu"
    )
)
print(device)

env = CVRPEnv(generator_params={'num_loc': 100})

# td = env.reset(batch_size=[10]).to(device)

# PATH = "checkpoints/am-a2c-cvrp50.ckpt"
# model_root = importlib.import_module("rl4co.models.rl")
# model_cls = getattr(model_root, "A2C")
# model_a2c50 = model_cls.load_from_checkpoint(PATH, load_baseline=False)
# model_a2c50 = model_a2c50
# policy = model_a2c50.policy.to(device)

# PATH = "checkpoints/am_cvrp50_2024_7_25.ckpt"
# model_root = importlib.import_module("rl4co.models")
# model_cls = getattr(model_root, "AttentionModel")
# model_am = model_cls.load_from_checkpoint(PATH, load_baseline=False)
# policy = model_am.policy.to(device)
# naive_pol = Naive()

policy = AttentionModelPolicy(env_name=env.name, 
                              embed_dim=128,
                              num_encoder_layers=3,
                              num_heads=8,
                            )
naive_pol = Naive()
decomp_pol = BaryCenterDecomp(2, policy=policy, naive_policy=naive_pol).to(device)

# model = REINFORCE(env, 
#                     policy,
#                     baseline="rollout",
#                     batch_size=128,
#                     train_data_size=512,
#                     val_data_size=128,
#                     optimizer_kwargs={"lr": 1e-4},
#                     ) 

model = DECOMP(env,
               policy=decomp_pol,
               baseline='no',
               batch_size=128,
               train_data_size=1024,
               val_data_size=128,
               optimizer_kwargs={"lr": 1e-4}
               )

logger = WandbLogger(project="rl4co", name="am-decomp-train-0")

trainer = RL4COTrainer(
    max_epochs=10,
    logger=logger
)

trainer.fit(model)
