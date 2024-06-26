import torch

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.decomposition import DECOMP, KmeansPol, BaryCenterDecomp, Delegate
from rl4co.models.zoo.am import AttentionModelPolicy

torch.manual_seed(42)

env = CVRPEnv(generator_params={'num_loc': 10})
model = DECOMP(env,
                baseline='rollout',
                train_data_size=1_000,
                val_data_size=100,
                decomp_policy_kwargs={"num_subproblems": 2})

td = env.reset(batch_size=[2])
decomp_policy = model.policy
out = decomp_policy(td, env, 2)

print(out["reward"])