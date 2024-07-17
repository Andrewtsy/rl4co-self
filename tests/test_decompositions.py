import pytest
import importlib

import torch
from tensordict.tensordict import TensorDict
from rl4co.models.zoo.decomposition import DECOMP, BaryCenterDecomp, DecompPolicy
from rl4co.envs import CVRPEnv

torch.manual_seed(42)

env = CVRPEnv(generator_params={'num_loc': 500})

PATH = "checkpoints/am-a2c-cvrp50.ckpt"
model_root = importlib.import_module("rl4co.models.rl")
model_cls = getattr(model_root, "A2C")
model_a2c50 = model_cls.load_from_checkpoint(PATH, load_baseline=False)
model_a2c50 = model_a2c50.to(torch.device("cpu"))

td = env.reset(batch_size=[4])

decomp_pol = BaryCenterDecomp(10, policy=model_a2c50.policy, naive_policy=model_a2c50.policy)
sol = decomp_pol(td, env, 8, 0.2, 0.75)

class TestBaryCenter:
    def test_full(self):
        sol = decomp_pol(td, env, 8, 0.4, 0.75)
        pass

    def test_assign_routes(self):
        groups_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"], sol['actions'])
        ggroups_idx, gbarycenters = BaryCenterDecomp._assign_routes_old(td["locs"], sol['actions'])
        assert torch.equal(ggroups_idx[:, 1:] + 1, groups_idx)
        assert torch.allclose(gbarycenters, barycenters)

    def test_recompose(self):
        global td, sol
        sol = decomp_pol.recompose(td, sol)
        pass

class TestDelegate:
    def test_call(self):
        idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"], sol['actions'])
        idx = decomp_pol.partition(barycenters, idx)

        td_new, res = decomp_pol.delegate(td, idx, 4, sol, env)
        td_new = env.reset(td_new)
        reward = env._get_reward(td_new, res)
        s = TensorDict({"actions": res, "reward": reward, "log_likelihood": torch.zeros((16))}, batch_size=16)
        rec = decomp_pol.recompose(td_new, s)
        assert torch.allclose(rec["reward"], sol["reward"])
        pass

    def test_pad(self):
        bs = 4
        t = torch.tensor([[4, 4, 0, 2], [0, 0, 4, 1], [4, 0, 4, 0], [1, 2, 3, 0]])
        mask = torch.tensor([[True, True, False, True], [False, False, True, False], 
                             [True, False, True, False], [True, True, True, False]])
        t = DecompPolicy._shift_pad(bs, t, mask)
        pass

if __name__ == "__main__":
    pass