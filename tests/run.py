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
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

torch.manual_seed(42)

accel = str("cuda" if torch.cuda.is_available()
    else (
        "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
        else "cpu"
    )
)

device = torch.device(accel)
print(device)

env = CVRPEnv(generator_params={'num_loc': 500, 'loc_distribution': 'cluster',
                                'n_cluster': 10}).to(device)

# td = env.reset(batch_size=[10]).to(device)

policy_params = {"its": 4, "temp": 0.1, "r": 0.9}
policy = AttentionModelPolicy(env_name=env.name, 
                              embed_dim=128,
                              num_encoder_layers=3,
                              num_heads=8,
                              **policy_params
                            ).to(device)
naive_pol = Naive().to(device)
decomp_pol = BaryCenterDecomp(4, policy=policy, naive_policy=naive_pol).to(device)

model = DECOMP(env,
               policy=decomp_pol,
               baseline="rollout",
               baseline_kwargs={"n_epochs": 2},
               batch_size=16,
               train_data_size=10_000,
               val_data_size=1_000,
               optimizer_kwargs={"lr": 1e-4},
               **policy_params
               ).to(device)

# Checkpointing callback: save models when validation reward improves
checkpoint_callback = ModelCheckpoint(  dirpath="checkpoints", # save to checkpoints/
                                        filename="run5_epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
                                        save_top_k=1, # save only the best model
                                        save_last=True, # save the last model
                                        monitor="val/reward", # monitor validation reward
                                        mode="max") # maximize validation reward

# Print model summary
rich_model_summary = RichModelSummary(max_depth=3)

# Callbacks list
callbacks = [checkpoint_callback, rich_model_summary]

# model = REINFORCE(env,
#                policy=policy,
#                baseline="rollout",
#                baseline_kwargs={"n_epochs": 1},
#                batch_size=32,
#                train_data_size=64,
#                val_data_size=32,
#                optimizer_kwargs={"lr": 1e-4},
#                ).to(device)

# model = PolyNet(env,
#                 k=8,
#                 val_num_solutions=16,
#                 batch_size=64,
#                 train_data_size=50_000,
#                 val_data_size=5_000,
#                 optimizer_kwargs={"lr": 1e-4}).to(device)

logger = WandbLogger(project="rl4co", name="am-decomp-5")

trainer = RL4COTrainer(
    max_epochs=5,
    logger=logger,
    log_every_n_steps=12,
    accelerator=accel,
    precision="16-mixed",
    callbacks=callbacks
)

trainer.fit(model)

