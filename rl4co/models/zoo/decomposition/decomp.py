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
            print(f"iteration {_}")
            # Decompose our state td and sol tensor into its k subproblems
            # Returned td is reset to prepare for policy
            td = self.decompose(td, env, sol)
            td = env.reset(td)
            td["action_mask"] = torch.concat((torch.ones((td.batch_size[0], 1), dtype=bool), td["problem_mask"]), dim=1)
            # Apply our constructive policy to each subproblem using parallelization
            sol = self.policy(td, env, problem_mask=True, return_actions=True)
            # Recompose our decomposed state and solution to the original problem sizes
            # To save complexity at the cost of some memory, we use our cached state td
            td_copy = td.clone()
            sol_copy = sol["actions"].clone()
            intermediates.append((td_copy, sol_copy))
            sol = self.recompose(td, sol)
            # print(sol["reward"])
            td = td_base.copy()
        # env.check_solution = True
        # env.check_solution_validity(td, sol["actions"])
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
        """ VECTORIZED
        Recomposition pass of the decomposition policy.
        Takes as input the decomposed tensordict states and solutions, and recomposes
        them into the original problem size with solutions.

        Args:
            td: Tensordict state
            env: Environment our states are in
            sol: Dictionary of current best solution
        Returns:
            out: Dictionary containing reward, log likelihood, 
            and importantly, recomposed actions
        """      
        # Store constants
        bs = sol["actions"].shape[0] // self.k
        actions = torch.cat((sol["actions"], torch.zeros((sol["actions"].shape[0], 1), dtype=torch.int64)), dim=1)
        idx = td["locs_idx"]

        # Recomposed tensor. Importantly, we have not removed padded zeros from them middle of rows yet.
        # The rest of the code is devoted to this
        combined = torch.gather(torch.cat((torch.zeros((sol["actions"].shape[0], 1), dtype=torch.int64), idx), dim=1), 1, actions)
        combined = combined.reshape(bs, -1).to(torch.int64)

        # Find the mask of all the nodes in the original subproblems we care about
        pad_indices = torch.argmax(torch.cumsum(actions, dim=1), dim=1)
        rows = torch.arange(actions.shape[1]).unsqueeze(0).expand(actions.shape[0], -1)
        mask = rows <= pad_indices.unsqueeze(1) + 1
        mask = mask.reshape(bs, -1)
        del rows

        # Find the row lengths of each subproblem and the row length needed for the final tensor
        row_lengths = mask.sum(dim=1)
        max_length = torch.max(row_lengths)

        # Make the mask_idx of the places we want to fill in in our resulting tensor
        res = torch.zeros((bs, max_length), dtype=torch.int64)
        idx = torch.arange(max_length).expand(bs, max_length)
        mask_idx = (idx < row_lengths.unsqueeze(1))
        del idx

        # Fill in our resulting tensor!
        res[mask_idx] = combined[mask]

        # Wrap up in output dictionary
        out = {}
        out["actions"] = res
        # Sum the rewards and the log likelihoods
        out["reward"] = sol["reward"].view((bs, self.k)).sum(dim=1)
        out["log_likelihood"] = sol["log_likelihood"].view((bs, self.k)).sum(dim=1)

        return out
   
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
            # Mask out the depot nodes at the end of each row and remove padding
            groups_idx = groups_idx * (srted != mx)
            groups_idx = groups_idx[:, :td_locs.shape[1] - 1]


            # Construct barycenters
            # Find all the group ids and make a mask that divides the nodes by group ids
            grp_names = torch.unique(grps)
            grps = grps * (actions != 0)
            msks = grps.unsqueeze(2) == grp_names
            
            # Find barycenters of x and y coordinates independently
            xy = []
            for i in range(2):
                # Add a depot dummy column to the tensor to correctly offset the indices
                # Then, gather the x/y coordinate to the associated node position
                # tmp = torch.cat((torch.zeros((td_locs.shape[0], 1)), td_locs[:, :, i]), dim=1)
                tmp = torch.gather(td_locs[:, :, i], 1, actions.to(torch.int64))

                # Reformat our node coordinates by group and calculate their group means
                b = (tmp.unsqueeze(2) * msks)
                b = b.sum(dim=1) / (b != 0).sum(dim=1)
                xy.append(b)

            return groups_idx, torch.stack(xy, dim=-1)