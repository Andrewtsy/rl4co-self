from typing import Any, Union
from sklearn.cluster import KMeans
import torch
from torch import nn
from tensordict.tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.am import AttentionModel, AttentionModelPolicy
from rl4co.models.common import ImprovementPolicy, ConstructivePolicy
from rl4co.models.zoo.decomposition.delegate import Delegate

class BaryCenterDecomp(ImprovementPolicy):
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
        super().__init__()

        # Set default policies to attention model from 
        # Kool et al. (Attention learn to solve routing problems!)
        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            policy = AttentionModelPolicy(env_name=env_name, **policy_kwargs_with_defaults)
        
        if naive_policy is None:
            naive_policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            naive_policy_kwargs_with_defaults.update(naive_policy_kwargs)
            naive_policy = AttentionModelPolicy(env_name=env_name, **naive_policy_kwargs)

        self.k = num_subproblems
        self.env_name = env_name
        self.kmeans = KmeansPol(self.k)
        self.policy = policy
        self.naive_policy = naive_policy

    def forward(
        self,
        td: TensorDict = None,
        env: Union[str, RL4COEnvBase] = None,
        its: int = None,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
    ) -> dict:
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
        # Cache copy of our state dictionary to assist recomposition.
        td_base = td.copy()
        # Generate initial solution using our naive policy
        # Refrain from checking solution validity until the end of the evaluation
        env.check_solution = False
        sol = self.naive_policy(td, env, return_actions=True)
        intermediates = []
        intermediates.append(sol["actions"].clone())
        # print(sol["reward"])
        # Iterative loop to improve our solution
        for _ in range(its):
            # Decompose our state td and sol tensor into its k subproblems
            # Returned td is reset to prepare for policy
            td = self.decompose(td, env, sol)
            td = env.reset(td)
            td["action_mask"] = torch.concat((torch.ones((td.batch_size[0], 1), dtype=bool), td["problem_mask"]), dim=1)
            # Apply our constructive policy to each subproblem using parallelization
            sol = self.policy(td, env, problem_mask=True, return_actions=True)
            # Recompose our decomposed state and solution to the original problem sizes
            # To save complexity at the cost of some memory, we use our cached state td
            # td_copy = td.clone()
            # sol_copy = sol["actions"].clone()
            # intermediates.append((td_copy, sol_copy))
            sol = self.recompose(td, sol)
            # print(sol["reward"])
            td = td_base.copy()
        env.check_solution = True
        env.check_solution_validity(td, sol["actions"])
        return sol, intermediates


    def decompose(
        self,
        td: TensorDict = None,
        env: Union[str, RL4COEnvBase] = None,
        sol: dict = None,
    ) -> TensorDict:
        """The actual decomposition pass of the decomposition policy.
        Primary override is decomposing a state into a number of subproblems.
        Takes as input a batch of current states and base solution actions run on them.
        Outputs an expanded tensordict state where the batch dimension is divided into
        batches and the subproblems within them.

        Args:
            td: Tensordict state
            env: Environment our states are in
            sol: Dictionary of current best solution
        Returns:
            out: Decomposed tensordict with just [locs, demand, action_mask, depot, capacity]
            ready to be reset by the environment
        """
        # Locations tensor that classifies each location by our current solution routes
        # And barycenters tensor that stores the barycenters of each route
        groups_idx, barycenters = BaryCenterDecomp._assign_routes(td["locs"], sol['actions'])
        # Kmeans cluster into subproblems. locs_idx stores the subproblem of each location
        groups_idx = self.kmeans(barycenters, groups_idx)
        # Reformat our td to count each new subproblem as its own batch element
        td = Delegate.delegate(td, groups_idx, self.k)

        return td
    
    def recompose(self,
                  td: TensorDict = None,
                  sol: dict = None,
    ) -> TensorDict:
        """Recomposition pass of the decomposition policy.
        Takes as input the decomposed tensordict states and solutions, and recomposes
        them into the original problem size with solutions.

        Args:
            td: Tensordict state
            env: Environment our states are in
            sol: Dictionary of current best solution
        Returns:
            out: Dictionary containing reward, log likelihood, 
            and importantly, tensordict state decomposed into subproblems.
        """        
        out = {}
        bs = sol["actions"].shape[0] // self.k # Batch size
        # Tensor of the actions taken in order, sewn together from the subproblem sols
        out["actions"] = torch.zeros(size=(bs, self.k * (sol["actions"].shape[1] + 1)), dtype=torch.int64)
        for i in range(bs):
            row = torch.tensor([])
            # Iteratively add the subproblem solutions together
            for j in range(min(self.k, sol["actions"].shape[0] - (i * self.k))):
                routes = td["locs_idx"][i * self.k + j][Delegate.no_pad(sol["actions"][i * self.k + j])]
                row = torch.cat((row, routes, torch.tensor([0,])))
            out["actions"][i, :len(row)] = row
        sps = max(len(Delegate.no_pad(out["actions"][b])) for b in range(len(out["actions"])))
        out["actions"] = out["actions"][:, :sps]
        # Sum the rewards and dummy zero out the log likelihoods
        out["reward"] = sol["reward"].view((bs, self.k)).sum(dim=1)
        out["log_likelihood"] = torch.zeros((bs,))
        return out


    # def _assign_routes(
    #     td_locs: torch.Tensor = None,
    #     actions: torch.Tensor = None
    # ):
    #     """ NONVECTORIZED
    #     Helper method to divide up the actions tensor of our policy outputs
    #     by routes, defined as starting at node i and ending at node 0 (the depot).
    #     To simplify operations, we work on an index tensor of the locations of
    #     the state.

    #     Args:
    #         td_locs: Tensor of location coordinates
    #         actions: Tensor of actions taken by the policy
    #     Returns:
    #         groups_idx: Tensor containing the route index each of our locations is contained in
    #         barycenters: Tensor containing the coordinates of the barycenters of each of our tours
    #     """
    #     # groups_idx, with the same shape as the td_locs, will tell us the tour number
    #     # node j has in row i
    #     groups_idx = torch.zeros(td_locs.shape[0], td_locs.shape[1], dtype=torch.int16)
    #     # barycenters will tell us the barycenter coordinates of tour i at position i
    #     # First, find the maximum number of tours per any batch. 
    #     # Then, create our tensor shape based on that.
    #     num_routes = max([int((Delegate.no_pad(row) == 0).sum()) + 1 for row in actions])
    #     barycenters = torch.zeros((td_locs.shape[0], num_routes, 2))
    #     # Populate groups_idx tensor
    #     for i in range(len(groups_idx)):
    #         row = Delegate.no_pad(actions[i])
    #         depot_idx = (row == 0).nonzero().reshape(-1)
    #         routes = torch.tensor_split(row, depot_idx)
    #         # After splitting the routes up, we can now assign our route nodes and barycenters
    #         for j, route in enumerate(routes):
    #             route = route[1:] if route[0] == 0 else route
    #             groups_idx[i, route] = j
    #             barycenters[i, j] = td_locs[i, route].mean(dim=0)
    #     return groups_idx, barycenters
    
    def _assign_routes(
            td_locs: torch.Tensor = None,
            actions: torch.Tensor = None
        ):
            """ VECTORIZED
            Helper method to divide up the actions tensor of our policy outputs
            by routes, defined as starting at node i and ending at node 0 (the depot).
            To simplify operations, we work on an index tensor of the locations of
            the state.

            Args:
                td_locs: Tensor of location coordinates
                actions: Tensor of actions taken by the policy
            Returns:
                groups_idx: Tensor containing the route index each of our locations is contained in
                barycenters: Tensor containing the coordinates of the barycenters of each of our tours
            """
            # Construct groups_idx
            # Assign group ids to the unordered actions tensor, based on a cumulative sum of zero borders
            # Note group ids start from 1 and end at num of groups. 0 is reserved for padding.
            grps = torch.cumsum((actions==0), dim=1) + 1
            # Make temporary tensor, where depot actions are assigned the highest index (useful for ordering in next steps)
            tmp = actions.clone()
            mx = actions.max() + 1
            tmp[actions == 0] = mx

            # indices stores the index of the actions sorted by node number
            srted, indices = torch.sort(tmp, dim=1)
            
            # Assign group numbers to each of the ordered nodes
            groups_idx = torch.gather(grps, 1, indices)
            # Mask out the depot nodes at the end of each row
            groups_idx = groups_idx * (srted != mx)


            grps = torch.cat((torch.ones((4, 1), dtype=int), grps), dim=1)
            grp_names = torch.unique(grps)
            msks = grps.unsqueeze(2) == grp_names
            
            tmp = torch.cat((torch.zeros((4, 1), dtype=int), td_locs[:, :, 0]), dim=1)
            tmp = torch.gather(tmp, 1, actions)
            tactions = torch.cat((torch.zeros((4, 1), dtype=int), actions), dim=1)
            tmp = torch.gather(tmp, 1, tactions)
            bx = (tmp.unsqueeze(2) * msks)
            bx = bx.sum(dim=1) / (bx != 0).sum(dim=1)

            return groups_idx, None



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
        subproblems = torch.zeros_like(locs_idx)
        for i in range(locs_idx.shape[0]):
            # For each batch, classify each route into subproblems
            labels = torch.tensor(self.kmeans.fit_predict(Delegate.no_pad(barycenters[i], torch.Tensor([0, 0]))))
            for j in range(len(labels)):
                # For each subproblem, note which nodes belong to them
                subproblems[i, locs_idx[i] == j] = labels[j]

        return subproblems

    def _kmeans(
        self
    ):
        """Custom Kmeans implementation that keeps size of clusters roughly the same
        """
        raise NotImplementedError

if __name__ == "__main__":
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