from typing import Any, Union
from sklearn.cluster import KMeans
import torch

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
        # USE BATCHING TO PARALLELIZE
        subproblems = torch.zeros_like(locs_idx)    

        # Impute padded nan values
        mask = torch.isnan(barycenters)
        means = torch.nanmean(barycenters, dim=1)
        barycenters = torch.where(mask, means.unsqueeze(1).expand_as(barycenters), barycenters)

        for i in range(locs_idx.shape[0]):
            # For each batch, classify each route into subproblems
            labels = torch.tensor(self.kmeans.fit_predict(BaryCenterDecomp.no_nan(barycenters[i])))
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