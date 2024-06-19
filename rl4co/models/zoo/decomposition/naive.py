from typing import Any, Union
from sklearn.cluster import KMeans
import torch
from torch import nn
from tensordict.tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.am import AttentionModel
from rl4co.models.common import ImprovementPolicy, ConstructivePolicy
from rl4co.models.zoo.decomposition.delegate import Delegate

def shout():
    print("Hello World")

class BaryCenterDecomp(ImprovementPolicy):
    """Baseline Barycenter decomposition policy using Kmeans clustering.
    First, calculate the barycenters of the current solution's routes. 
    Then, use Kmeans clustering to cluster the nearest routes into k CVRP subproblems.
    
    Args:
        k: Number of clusters/subproblems
        env: Problem environment"""
    
    def __init__(
        self,
        k: int = None,
        env_name: str = "cvrp",
        policy: ConstructivePolicy = None,

    ):
        super().__init__()
        
        self.k = k
        self.env_name = env_name
        self.kmeans = KmeansPol(k)
        self.policy = policy

    def forward(
        self,
        td: TensorDict = None,
        env: Union[str, RL4COEnvBase] = None,
        sol: dict = None,
        its: int = None,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
    ) -> TensorDict:
        """Forward pass of the decomposition policy.
        Full run of decomposition.

        Args:
            td: Tensordict state
            env: Environment our states are in
            sol: Dictionary of current best solution
            its: Number of iterations of the improvement policy
        Returns:
            out: Dictionary containing actions, reward, log likelihood, 
        """
        for it in range(its):
            td = self._pass(td, env, sol)
            td = env.reset(td)
            sol = self.policy(td, env)
        return sol


    def _pass(
        self,
        td: TensorDict = None,
        env: Union[str, RL4COEnvBase] = None,
        sol: dict = None,
    ) -> TensorDict:
        """Forward pass of the decomposition policy.
        Primary override is decomposing a state into a number of subproblems.
        Takes as input a batch of current states and base solution actions run on them.
        Outputs an expanded tensordict state where the batch dimension is divided into
        batches and the subproblems within them.

        Args:
            td: Tensordict state
            env: Environment our states are in
            sol: Dictionary of current best solution
        Returns:
            out: Dictionary containing reward, log likelihood, 
            and importantly, tensordict state decomposed into subproblems.
        """
        # Locations tensor that classifies each location by our current solution routes
        # And barycenters tensor that stores the barycenters of each route
        locs_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"][:, 1:, :], sol['actions'])
        # Kmeans cluster into subproblems
        subproblems = self.kmeans(barycenters)
        # Reformat our td to count each new subproblem as its own batch element
        td = Delegate.delegate(td, subproblems, self.k)

        return td

    def _assign_routes(
        td_locs: torch.Tensor = None,
        actions: torch.Tensor = None
    ):
        """Helper method to divide up the actions tensor of our policy outputs
        by routes, defined as starting at node i and ending at node 0 (the depot).
        To simplify operations, we work on an index tensor of the locations of
        the state.

        Args:
            loc_idx: Zeroed out tensor corresponding to the locations of our state
            actions: Tensor of actions taken by the policy
        Returns:
            loc_idx: Modified tensor containing the route index each of our locations is contained in
        """
        loc_idx = torch.zeros(td_locs.shape[0], td_locs.shape[1], dtype=torch.int16)
        barycenters = torch.zeros_like(td_locs)
        for i in range(len(loc_idx)):
            depot_idx = (actions[i] == 0).nonzero().reshape(-1)
            routes = torch.tensor_split(actions[i], depot_idx)
            for j, route in enumerate(routes):
                coordinate_sum = torch.Tensor([0, 0])
                for node in route:
                    if not node == 0:
                        loc_idx[i, node - 1] = j
                        coordinate_sum += td_locs[i, node - 1]
                barycenters[i, j] = coordinate_sum / len(route)
        return loc_idx, barycenters

class KmeansPol():
    """Baseline K-means clustering for naive decomposition.
    Input a routing environment and outputs an environment split into k subproblems.

    Args:
        k: Number of clusters
        state: Tensordict problem state
    """

    def __init__(
        self,
        k: int = None,
        env: RL4COEnvBase = None,
    ):
        self.k = k
        self.env = env

        # May override with own custom kmeans later
        self.kmeans = KMeans(n_clusters=k)

    def __call__(
        self,
        barycenters: torch.Tensor = None
    ) -> RL4COEnvBase:
        """Kmeans split the environment
        Args:
            td: Tensordict state
        Returns:
            env: Transformed environment
        """
        labels = torch.vstack([torch.tensor(self.kmeans.fit_predict(ex)) for ex in barycenters])

        return labels

    def _kmeans(
        self
    ):
        """Custom Kmeans implementation that keeps size of clusters roughly the same
        """
        raise NotImplementedError

if __name__ == "__main__":
    shout()

    import torch
    from rl4co.envs import CVRPEnv
    from rl4co.models.zoo import AttentionModel

    # RL4CO env based on TorchRL
    env = CVRPEnv(generator_params={'num_loc': 20})

    # Model: default is AM with REINFORCE and greedy rollout baseline
    model = AttentionModel(env,
                    baseline='rollout',
                    train_data_size=1_000, # really small size for demo
                    val_data_size=100)

    td = env.reset(batch_size=[2])
    policy = model.policy
    out = policy(td.clone(), env, phase="test", decode_type="greedy", return_actions=True)

    decomp = BaryCenterDecomp(4)
    td_new = decomp.forward(td)

    input("wait...")
    pass