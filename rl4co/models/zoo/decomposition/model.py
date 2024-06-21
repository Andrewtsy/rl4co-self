from typing import Any, Union
import torch

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
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
        policy: torch.nn.Module = None,
        policy_kwargs={},
        decomp_pol: torch.nn.Module = None,
        decomp_pol_kwargs={},
        naive_pol: torch.nn.Module = None,
        baseline: str = "shared",
        **kwargs,
    ):
        
        # Set default policy to attention model from 
        # Kool et al. (Attention learn to solve routing problems!)
        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs_with_defaults)
        
        # Set default decomposition policy to barycenter clustering technique from 
        # Santini et al. (Decomposition strategies for vehicle routing heuristics)
        if decomp_pol is None:
            decomp_policy_kwargs_with_defaults = {
                "n_clusters": 16,
            }
            decomp_policy_kwargs_with_defaults.update(decomp_pol_kwargs)
            decomp_pol = BaryCenterDecomp(env_name=env.name, **decomp_policy_kwargs_with_defaults)

        if naive_pol is None:
            naive_pol = policy

        super().__init__(env, policy, baseline, **kwargs)
        self.decomp_pol = decomp_pol
        self.naive_pol = naive_pol

    def shared_step(
        self, 
        batch: Any, 
        batch_idx: int, 
        phase: str,
        its: int,
        dataloader_idx: int = None
    ):
        """Primary method of the model.
        This overwritten method alters environment states into decomposed states before applying the policy.
        """
        td = self.env.reset(batch)
        out = self.naive_pol(td, self.env, return_actions=True)

        

        