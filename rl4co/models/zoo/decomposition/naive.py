from typing import Any, Union, Optional
from sklearn.cluster import KMeans
import torch
from tensordict.tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.models.common import ConstructivePolicy
from rl4co.models.zoo.decomposition.decomp import DecompPolicy

class BaryCenterDecomp(DecompPolicy):
    """Baseline Barycenter decomposition policy using Kmeans clustering.
    First, calculate the barycenters of the current solution's routes. 
    Then, use Kmeans clustering to cluster the nearest routes into k CVRP subproblems.
    
    Args:
        num_subproblems: Number of clusters/subproblems
        env_name: Problem environment,
        policy: Constructive policy used iteratively on subproblems
        policy_kwargs: Keyword arguments
        naive_policy: Constructive policy to generate initial solution. Should be highly efficient.
        naive_policy_kwargs: Keyword arguments
        """
    
    def __init__(
        self,
        num_subproblems: int = None,
        env_name: str = "cvrp",
        policy: ConstructivePolicy = None,
        policy_kwargs: dict = {},
        naive_policy: ConstructivePolicy = None,
        naive_policy_kwargs: dict = {}
    ):
        super().__init__(num_subproblems, env_name, policy, policy_kwargs, naive_policy, naive_policy_kwargs)

        self.k = num_subproblems
        self.kmeans = KMeans(n_clusters=self.k)

    def partition(
        self,
        barycenters: torch.Tensor = None,
        locs_idx: torch.Tensor = None
    ) -> RL4COEnvBase:
        """Kmeans split the environment
        Args:
            barycenters: Tensor of barycenters
            locs_idx: Tensor of locations by their tours
        Returns:
            locs_idx: Tensor of locations by their subproblems
        """
        device = locs_idx.device
        # USE BATCHING TO PARALLELIZE
        subproblems = torch.zeros_like(locs_idx, dtype=torch.int64, device=device)

        # Impute padded nan values
        mask = torch.isnan(barycenters)
        means = torch.nanmean(barycenters, dim=1)
        barycenters = torch.where(mask, means.unsqueeze(1).expand_as(barycenters), barycenters).to("cpu")

        for i in range(locs_idx.shape[0]):
            # For each batch, classify each route into subproblems
            labels = torch.tensor(self.kmeans.fit_predict(BaryCenterDecomp.no_nan(barycenters[i])), dtype=torch.int64, device=device)
            for j in range(len(labels)):
                # For each subproblem, note which nodes belong to them
                subproblems[i, locs_idx[i] == j] = labels[j]

        return subproblems
    
    @staticmethod
    def no_nan(row):
        msk = torch.isnan(row)
        return row[:torch.nonzero(msk)[0][0]] if torch.nonzero(msk).numel() > 0 else row

    def _kmeans(
        self
    ):
        """Custom Kmeans implementation that keeps size of clusters roughly the same
        """
        raise NotImplementedError
    
class Naive(ConstructivePolicy):
    """Extremely Naive constructive policy that makes a single tour for every single node in
    our environment.
    """
    def __init__(
        self,
        env_name: str = "cvrp",
    ):
        super(Naive, self).__init__(None, None, env_name)

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        calc_reward: bool = True,
        return_actions: bool = False,
        actions=None,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """
        device = td.device
        actions = torch.zeros((td.batch_size[0], td["demand"].shape[1] * 2), dtype=torch.int64, device=device)
        rng = torch.arange(1, td["locs"].shape[1], device=device)
        actions[:, ::2] = rng
        del rng

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"]
        }

        if return_actions:
            outdict["actions"] = actions

        return outdict