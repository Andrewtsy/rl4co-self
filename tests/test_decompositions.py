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

decomp_pol = BaryCenterDecomp(4, policy=model_a2c50.policy, naive_policy=model_a2c50.policy)
sol = decomp_pol.naive_policy(td, env, return_actions=True)
kmeans = KmeansPol(4, env)

class TestBaryCenter:
    def test_assign_routes(self):
        groups_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], sol['actions'])
        
        # groups_idx_gold = torch.Tensor([[1, 3, 2, 2, 1, 0, 1, 0, 2, 3, 1, 0, 3, 2, 0, 3, 1, 1, 0, 0],
        #                              [0, 2, 1, 0, 0, 2, 3, 2, 1, 2, 1, 3, 0, 2, 1, 3, 0, 3, 3, 1]])
        # assert torch.equal(groups_idx[:2], groups_idx_gold)

        # barycenters_gold = torch.Tensor([[[0.6191, 0.6274], [0.5764, 0.3365], [0.1292, 0.2231], [0.1999, 0.7730]],
        #                                  [[0.5701, 0.7364], [0.1841, 0.8193], [0.4152, 0.1926], [0.2002, 0.4218]]])
        # assert torch.equal(barycenters_gold[:2], barycenters[:2])

class TestDelegate:
    def test_call(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], sol['actions'])
        delegate = Delegate()
        td_new = delegate(td, loc_idx, 9)
        pass

    def test_del_sp(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], sol['actions'])
        labels = kmeans(td, barycenters)
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