from typing import Any, Union
import torch
from tensordict.tensordict import TensorDict

from rl4co.utils.lightning import get_lightning_device
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.common import ImprovementPolicy, ConstructivePolicy
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.decomposition.naive import BaryCenterDecomp


class DECOMP(REINFORCE):
    """DECOMP is a decomposition based framework for neural combinatorial optimization.
    Built on top of the REINFORCE gradient approximation framework, DECOMP is a model
    focused on decomposing the CO problem into independent and homogenous subproblems.
    Written by Tao et al. with Prof. Kevin Tierney's lab @ Bielefeld University.

    Args:
        env: TorchRL Environment
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        decomp_pol: Policy to decompose/recompose the environment states
        baseline: Baseline to use for the algorithm.
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: ImprovementPolicy = None,
        policy_kwargs={},
        baseline: str = "shared",
        **kwargs,
    ):
        
        # Set default decomposition policy to barycenter clustering technique from 
        # Santini et al. (Decomposition strategies for vehicle routing heuristics)
        if policy is None:
            policy_kwargs_with_defaults = {
                "num_subproblems": 4
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            policy = BaryCenterDecomp(env_name=env.name, **policy_kwargs_with_defaults)
        else:
            policy = policy
        super().__init__(env, policy, baseline, **kwargs)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self._forward_pass(td, self.env, phase=phase)

        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)

        return {"loss": out.get("loss", None), **metrics}
        
    def _forward_pass(
        self,
        td: TensorDict = None,
        env: Union[str, RL4COEnvBase] = None,
        phase: str = "train",
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
            its: Number of iterations of the improvement policy
            temp: Probability of accepting worse solution in simulated annealing (starting temperature)
            r: Decay rate
            history: return list of iteration rewards?
        Returns:
            out: Dictionary containing actions, reward
        """
        # Cache copy of our state dictionary to assist recomposition.
        td_base = td.copy()
        # Generate initial solution using our naive policy
        # Refrain from checking solution validity until the end of the evaluation
        env.check_solution = False
        sol = self.policy.naive_policy(td, env, return_actions=True)

        # Decompose our state td and sol tensor into its k subproblems
        # Returned td is reset to prepare for policy
        td, actions = self.policy.decompose(td, env, sol)
        td = env.reset(td)
        td["action_mask"] = torch.concat((torch.ones((td.batch_size[0], 1), dtype=bool), td["problem_mask"]), dim=1)
                    
        # Apply our constructive policy to each subproblem using parallelization
        nsol = self.policy.policy(td, env, problem_mask=True, phase=phase, return_actions=True, **decoding_kwargs)
        nsol = self.policy.recompose(td, nsol, log_likelihoods=True)
        return nsol

    # def post_setup_hook(self, stage="fit"):
    #     # Make baseline taking model itself and train_dataloader from model as input
    #     self.baseline.setup(
    #         self.policy.policy,
    #         self.env,
    #         batch_size=self.val_batch_size,
    #         device=get_lightning_device(self),
    #         dataset_size=self.data_cfg["val_data_size"],
    #     )

    # def on_train_batch_start(self, batch=None, batch_idx=None, dataloader_idx=None):
    #     print("Hit em with the Hook")
    #     pass