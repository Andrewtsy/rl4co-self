from typing import Any, Union
import torch

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
        policy: ConstructivePolicy = None,
        policy_kwargs={},
        baseline: str = "shared",
        **kwargs,
    ):
        
        # Set default decomposition policy to barycenter clustering technique from 
        # Santini et al. (Decomposition strategies for vehicle routing heuristics)
        if policy is None:
            decomp_policy_kwargs_with_defaults = {
                "n_clusters": 16,
            }
            decomp_policy_kwargs_with_defaults.update(policy_kwargs)
            policy = BaryCenterDecomp(env_name=env.name, **decomp_policy_kwargs_with_defaults)

        super().__init__(env, policy, baseline, **kwargs)
        self.policy = policy

    def shared_step(
        self, 
        batch: Any, 
        batch_idx: int, 
        phase: str,
        its: int,
        dataloader_idx: int = None
    ):
        """Step method.
        This overwritten method alters environment states into decomposed states before applying the policy.
        """
        td = self.env.reset(batch)
        
        out = self.policy(td=td, env=self.env, its=its)

        if phase == "train":
            out = self.calculate_loss(td, batch, out)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
        