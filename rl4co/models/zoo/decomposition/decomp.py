from typing import Any, Union
import torch
from tensordict.tensordict import TensorDict

from rl4co.utils.ops import gather_by_index
from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.common import ImprovementPolicy, ConstructivePolicy

class DecompPolicy(ImprovementPolicy):
    """General policy class    
    Args:
        num_subproblems: Number of clusters/subproblems
        env_name: Problem environment,
        policy: Constructive policy used iteratively on subproblems
        policy_kwargs: Keyword arguments
        naive_policy: Constructive policy to generate initial solution. Should be highly efficient.
        naive_policy_kwargs: Keyword arguments
        its: Number of iterations of the improvement policy
        temp: Probability of accepting worse solution in simulated annealing (starting temperature)
        r: Decay rate
        """
    
    def __init__(
        self,
        num_subproblems: int = None,
        env_name: str = "cvrp",
        policy: ConstructivePolicy = None,
        policy_kwargs: dict = {},
        naive_policy: ConstructivePolicy = None,
        naive_policy_kwargs: dict = {},
        its: int = 1,
        temp: int = 0,
        r: int = 0,
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

        self.its, self.temp, self.r = its, temp, r
        self.k = num_subproblems
        self.env_name = env_name
        self.policy = policy
        self.naive_policy = naive_policy

    def partition(self):
        """To be implemented in subclass"""
        raise NotImplementedError

    def forward(
        self,
        td: TensorDict = None,
        env: Union[str, RL4COEnvBase] = None,
        history: bool = False,
        phase: str = "test",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the decomposition policy.

        Args:
            td: Tensordict state
            env: Environment our states are in
            sol: Dictionary of current best solution
            history: return list of iteration rewards?
        Returns:
            out: Dictionary containing actions, reward
        """
        # Cache copy of our state dictionary to assist recomposition.
        td_base = td.copy()
        # Generate initial solution using our naive policy
        # Refrain from checking solution validity until the end of the evaluation
        env.check_solution = False
        sol = self.naive_policy(td, env, return_actions=True)
        init_reward = sol["reward"].mean()
        # print(init_reward)
        if history:
            intermediates = []
            intermediates.append(sol["reward"])
        # print(sol["reward"])
        # Iterative loop to improve our solution
        # better = 0
        for _ in range(self.its):
            # print(f"iteration {_}")
            # Decompose our state td and sol tensor into its k subproblems
            # Returned td is reset to prepare for policy
            td, actions = self.decompose(td, env, sol)
            td = env.reset(td)
            td["action_mask"] = torch.concat((torch.ones((td.batch_size[0], 1), device=td.device, dtype=bool), td["problem_mask"]), dim=1)
                        
            # Apply our constructive policy to each subproblem using parallelization
            nsol = self.policy(td, env, problem_mask=True, phase=phase, return_actions=True, **decoding_kwargs)

            # Select best solutions
            res = TensorDict({
                "actions": torch.zeros((td.batch_size[0], max(actions.shape[1], nsol["actions"].shape[1])), dtype=torch.int64),
                "reward": torch.zeros((td.batch_size[0]), dtype=torch.float32)},
                batch_size=td.batch_size[0],
                device=td.device
            )
            sol = TensorDict({"actions": actions, "reward": env._get_reward(td, actions)},
                             batch_size=td.batch_size[0],
                             device=td.device)
            
            # td_copy = td.clone()
            # sol_copy = sol.clone()
            # intermediates[f"Iteration {_}"] = [td_copy, sol_copy]

            mask = nsol["reward"] > sol["reward"]
            # better += mask.sum()

            # Simulated Annealing
            tmp = torch.rand(mask.shape, device=td.device)
            mask = mask | (tmp < self.temp)
            self.temp = self.temp * self.r

            res["actions"][mask, :nsol["actions"].shape[1]] = nsol["actions"][mask, :]
            res["actions"][~mask, :sol["actions"].shape[1]] = sol["actions"][~mask, :]

            res["reward"][mask] = nsol["reward"][mask]
            res["reward"][~mask] = sol["reward"][~mask]            

            # Recompose our decomposed state and solution to the original problem sizes
            # To save complexity at the cost of some memory, we use our cached state td
            # sol_copy = res.clone()
            
            sol = self.recompose(td, res, phase=phase)
            if history:
                intermediates.append(sol["reward"])
            # print(sol["reward"].mean())
            td = td_base.copy()
        env.check_solution = True
        env.check_solution_validity(td, sol["actions"])
        # print(f"Final Reward improvement: {sol['reward'].mean() - init_reward}")
        if history:
            return sol, intermediates
        return sol

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
        groups_idx, barycenters = DecompPolicy._assign_routes(td["locs"], sol['actions'])
        # Kmeans cluster into subproblems. locs_idx stores the subproblem of each location
        groups_idx = self.partition(barycenters, groups_idx)
        # Reformat our td to count each new subproblem as its own batch element
        td, sol = DecompPolicy.delegate(td, groups_idx, self.k, sol)

        return td, sol

    def recompose(self,
                  td: TensorDict = None,
                  sol: dict = None,
                  phase: str = "train",
                  log_likelihoods: bool = False
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
        device = td.device
        bs = sol["actions"].shape[0] // self.k
        actions = torch.cat((sol["actions"], torch.zeros((sol["actions"].shape[0], 1), dtype=torch.int64, device=device)), dim=1)
        idx = td["locs_idx"]

        # Recomposed tensor. Importantly, we have not removed padded zeros from them middle of rows yet.
        # The rest of the code is devoted to this
        combined = torch.gather(torch.cat((torch.zeros((sol["actions"].shape[0], 1), dtype=torch.int64, device=device), idx), dim=1), 1, actions)
        combined = combined.reshape(bs, -1)

        # Find the mask of all the nodes in the original subproblems we care about
        pad_indices = torch.argmax(torch.cumsum(actions, dim=1), dim=1)
        rows = torch.arange(actions.shape[1], device=device).unsqueeze(0).expand(actions.shape[0], -1)
        mask = rows <= pad_indices.unsqueeze(1) + 1
        mask = mask.reshape(bs, -1)
        del rows, pad_indices

        # Find the row lengths of each subproblem and the row length needed for the final tensor
        row_lengths = mask.sum(dim=1)
        max_length = torch.max(row_lengths)

        # Make the mask_idx of the places we want to fill in in our resulting tensor
        res = torch.zeros((bs, max_length), dtype=torch.int64, device=device)
        idx = torch.arange(max_length, device=device).expand(bs, max_length)
        mask_idx = (idx < row_lengths.unsqueeze(1))
        del idx

        # Fill in our resulting tensor!
        res[mask_idx] = combined[mask]

        # Wrap up in output dictionary
        out = {}
        out["actions"] = res
        # Sum the rewards and the log likelihoods
        out["reward"] = sol["reward"].view((bs, self.k)).sum(dim=1)
        if phase == "train":
            out["loss"] = sol["loss"]
        # if log_likelihoods:
        #     out["log_likelihood"] = sol["log_likelihood"].view((bs, self.k)).sum(dim=1)

        return out
   
    @staticmethod
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
            del tmp
            
            # Assign group numbers to each of the ordered nodes
            groups_idx = torch.gather(grps, 1, indices)
            # Mask out the depot nodes at the end of each row and remove padding
            groups_idx = groups_idx * (srted != mx)
            groups_idx = groups_idx[:, :td_locs.shape[1] - 1]
            del indices

            # Construct barycenters
            # Find all the group ids and make a mask that divides the nodes by group ids
            grp_names = torch.unique(grps)
            grps = grps * (actions != 0)
            msks = grps.unsqueeze(2) == grp_names
            del grps, grp_names
            
            # Find barycenters of x and y coordinates
            # Gather the x/y coordinates to their node positions and calculate their group means
            tmp = gather_by_index(td_locs, actions)
            b = tmp.unsqueeze(2) * msks.unsqueeze(3)
            barycenters = b.sum(dim=1) / (b != 0).sum(dim=1)
            del tmp

            return groups_idx, barycenters
    
    @staticmethod
    def delegate(
        td: TensorDict = None,
        idx: torch.tensor = None,
        k: int = None,
        sol: torch.tensor = None,
        env: RL4COEnvBase = None
    ) -> TensorDict:
        """VECTORIZED
        Divide the passed in tensordict state by passed in labels
        Args:
            td: Tensordict state
            idx: Tensor map from problem nodes to their clusters
            k: Number of clusters
        Returns:
            td_new: Transformed tensordict with shape k * td.shape[0]
        """
        device = td.device
        locs = td["locs"]
        # Set batch and subproblem sizes, as well as maximum node number for zero padding (used later)
        bs, sps, mx = idx.shape[0], idx.shape[1], idx.shape[1] + 1
        # Subproblem divided *map* will map nodes to their original problem indices
        res = torch.zeros((k * bs, sps), dtype=torch.int64, device=device)
        # Take map of nodes in the original problems to their subproblem rows,
        # Indices of each of the nodes in their original subproblems
        # Scatter them onto a map from subproblems to their original problem indices
        map_ps = torch.arange(0, k * bs, step=k, device=device).unsqueeze(1) + idx
        problem_indices = torch.arange(1, sps + 1, device=device).expand(bs, -1)
        res = res.scatter(0, map_ps, problem_indices)

        # Remove zero pads
        res[res == 0] = mx
        res, _ = torch.sort(res, dim=1)
        res = res * (res != mx)
        maxsps = torch.max(torch.nonzero(res, as_tuple=True)[-1]) + 1
        idx_sp = res[:, :maxsps]

        # Locations repeated to be same shape as subproblems tensor
        rep = torch.repeat_interleave(locs, k, dim=0)
        # Construct x and y coordinate location tensors
        locs = gather_by_index(rep, idx_sp)

        # Problem mask is just where the locations are not pads
        problem_mask = idx_sp != 0

        # Gather location demands
        rep = torch.cat((torch.zeros((bs * k, 1), device=device), torch.repeat_interleave(td["demand"], k, dim=0)), dim=1)
        demand = torch.gather(rep, 1, idx_sp)

        # Subdivide the solution actions
        rep = torch.cat((torch.zeros((bs, 1), device=device), idx + 1), dim=1)
        actions = torch.gather(rep, 1, sol["actions"])
        del rep

        # By using a shift tensor we label the connecting depot nodes as their associated subproblems
        mask = actions == 0
        shift = torch.full_like(actions, 0, device=device)
        shift[:, 1:] = actions[:, :-1]
        actions[mask] = shift[mask]
        
        # Delegate the nodes into their respective subproblem rows
        mask = actions != 0
        res = torch.zeros((k * bs + 1, actions.shape[1]), dtype=torch.int64, device=device)
        map_ps = torch.arange(0, k * bs, step=k, dtype=torch.int64, device=device).unsqueeze(1) + actions.to(torch.int64)
        map_ps = map_ps * mask
        res = res.scatter(0, map_ps, sol["actions"])[1:]
        del map_ps

        # By using a shift tensor we shift the unnecessary pads (non depot zeros) to the end of their rows
        shift = torch.full_like(res, 0, device=device)
        shift[:, 1:] = res[:, :-1]
        mask = shift != res
        res = DecompPolicy._shift_pad(bs * k, res, mask)

        # Use two argsorts to get the correct intersubproblem indices of the nodes
        mask = (res != 0)
        res[res == 0] = mx
        res = torch.argsort(torch.argsort(res, dim=1), dim=1) + 1
        res = res * mask

        # Before reset, state td must have these key variable pairs
        td = TensorDict({"locs": locs,
                        "demand": demand,
                        "problem_mask": problem_mask,
                        "depot": torch.repeat_interleave(td["depot"], k, 0),
                        "capacity": torch.repeat_interleave(td["capacity"], k, 0),
                        "locs_idx": idx_sp},
                        batch_size=bs * k,
                        device=device)

        return td, res
    
    @staticmethod
    def _shift_pad(bs, t, mask):
        """Small helper function that moves all zero pads to the back of each row in tensor t
        Args:
            bs: number of rows in tensor t
            t: tensor
            mask: mask of all the nonzero elements in t
        Returns:
            td_new: Transformed tensordict with shape k * td.shape[0]
        """
        device = t.device
        row_lengths = mask.sum(dim=1)
        max_length = torch.max(row_lengths)

        res = torch.zeros((bs, max_length), dtype=torch.int64, device=device)
        idx = torch.arange(max_length, device=device).expand(bs, max_length)
        mask_idx = (idx < row_lengths.unsqueeze(1))
        res[mask_idx] = t[mask]

        return res
