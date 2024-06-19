import pytest

import torch
from rl4co.models.zoo.decomposition import DECOMP, KmeansPol, BaryCenterDecomp, Delegate
from rl4co.envs import CVRPEnv
from rl4co.models.zoo import AttentionModel

torch.manual_seed(42)

env = CVRPEnv(generator_params={'num_loc': 10})

model = AttentionModel(env,
                baseline='rollout',
                train_data_size=1_000,
                val_data_size=100)

td = env.reset(batch_size=[2])
policy = model.policy
out = policy(td.clone(), env, phase="test", decode_type="greedy", return_actions=True)

decomp = BaryCenterDecomp(4)
kmeans = KmeansPol(4, env)

class TestBaryCenter:
    def test_assign_routes(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], out['actions'])
        ground_truth = torch.Tensor([3, 1, 2, 8, 6, 4, 7, 0, 5, 0])
        assert torch.equal(ground_truth, loc_idx[0])

class TestDelegate:
    def test_call(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], out['actions'])
        delegate = Delegate()
        td_new = delegate(td, loc_idx, 9)
        pass

    def test_del_sp(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], out['actions'])
        labels = kmeans(td, barycenters)
        loc_idx = Delegate.del_sp(loc_idx, labels)
        pass

    def test_find_sps(self):
        loc_idx = torch.tensor([[1, 1, 1, 2], [1, 1, 2, 2]], dtype=torch.int16)
        assert Delegate._find_sps(loc_idx) == 3

class TestKmeans:
    def test_call(self):
        loc_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], out['actions'])
        labels = kmeans(td, barycenters)
        pass

if __name__ == "__main__":
    pass