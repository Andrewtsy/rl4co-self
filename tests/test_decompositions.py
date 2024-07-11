import pytest
import importlib

import torch
from rl4co.models.zoo.decomposition import DECOMP, KmeansPol, BaryCenterDecomp, Delegate
from rl4co.envs import CVRPEnv
from rl4co.models.zoo import AttentionModel

torch.manual_seed(42)

env = CVRPEnv(generator_params={'num_loc': 20})

PATH = "checkpoints/am-a2c-cvrp50.ckpt"
model_root = importlib.import_module("rl4co.models.rl")
model_cls = getattr(model_root, "A2C")
model_a2c50 = model_cls.load_from_checkpoint(PATH, load_baseline=False)
model_a2c50 = model_a2c50.to(torch.device("cpu"))

td = env.reset(batch_size=[4])

decomp_pol = BaryCenterDecomp(2, policy=model_a2c50.policy, naive_policy=model_a2c50.policy)
sol = decomp_pol.naive_policy(td, env, return_actions=True)


td = decomp_pol.decompose(td, env, sol)
td = env.reset(td)
td["action_mask"] = torch.concat((torch.ones((td.batch_size[0], 1), dtype=bool), td["problem_mask"]), dim=1)
env.check_solution = False
sol = decomp_pol.policy(td, env, problem_mask=True, return_actions=True)

class TestBaryCenter:
    def test_assign_routes(self):
        groups_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], sol['actions'])
        pass

    def test_recompose(self):
        global td, sol
        sol = decomp_pol.recompose(td, sol)
        pass

class TestDelegate:
    def test_call(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], sol['actions'])
        loc_idx = BaryCenterDecomp.kmeans(barycenters, loc_idx)
        td_new = Delegate.delegate(td, loc_idx, 2)
        pass

    def test_del_sp(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], sol['actions'])
        labels = BaryCenterDecomp.kmeans(td, barycenters)
        loc_idx = Delegate.del_sp(loc_idx, labels)
        pass

    def test_find_sps(self):
        loc_idx = torch.tensor([[1, 1, 1, 2], [1, 1, 2, 2]], dtype=torch.int16)
        assert Delegate._find_sps(loc_idx) == 3

class TestKmeans:
    def test_call(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], sol['actions'])
        labels = kmeans(td, barycenters)
        pass

if __name__ == "__main__":
    pass