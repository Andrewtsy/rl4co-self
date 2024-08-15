from typing import Any, Union
import torch
from torch.optim.optimizer import Optimizer
from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.common import ImprovementPolicy, ConstructivePolicy
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.decomposition.naive import BaryCenterDecomp
from rl4co.utils.lightning import get_lightning_device


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
        its: Number of iterations of the improvement policy
        temp: Probability of accepting worse solution in simulated annealing (starting temperature)
        r: SA temperature decay rate
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: ImprovementPolicy = None,
        policy_kwargs={},
        baseline: str = "mean",
        its: int = 1,
        temp: int = 0,
        r: int = 0,
        **kwargs,
    ):
        self.its, self.temp, self.r = its, temp, r

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
        
    @property
    def automatic_optimization(self):
        # Manual custom optimization
        return False

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        device = td.device

        if phase == "train":
            # Store the optimizers
            opt = self.optimizers()

            td_base = td.copy()

            # Run through a full pass of multi iteration simulated annealing
            # See forward pass of decomp policy for more info/comments
            sol = self.policy.naive_policy(td, self.env, return_actions=True)
            opt.zero_grad()
            for _ in range(self.its):

                self.env.check_solution = False
                td, _ = self.policy.decompose(td, self.env, sol)
                td = self.env.reset(td)
                td_bl = td.copy()
                td["action_mask"] = torch.concat((torch.ones((td.batch_size[0], 1), dtype=bool, device=device), td["problem_mask"]), dim=1)
                sol = self.policy.policy(td, self.env, problem_mask=True, phase=phase, return_actions=True)
                
                # Calculate loss
                sol = self.calculate_loss(td_bl, batch, sol, use_extra=False)

                # Manual backward pass and clip
                self.manual_backward(sol["loss"])

                sol = self.policy.recompose(td, sol, phase=phase, log_likelihoods=True)
                td = td_base.copy()
            
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
            self.env.check_solution = True
        else:
            # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
            sol = self.policy(td, self.env, phase=phase, select_best=phase != "train")

        metrics = self.log_metrics(sol, phase, dataloader_idx=dataloader_idx)
        return {"loss": sol.get("loss", None), **metrics}
    
    def post_setup_hook(self, stage="fit"):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy.policy,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            dataset_size=self.data_cfg["val_data_size"],
        )

    # def call_setup_hook(self):
    #     self.baseline.setup(
    #         self.policy.policy,
    #         self.env,
    #         batch_size=self.val_batch_size,
    #         device=get_lightning_device(self),
    #         dataset_size=self.data_cfg["val_data_size"],
    #     )

    def on_train_epoch_end(self):
        """Callback for end of training epoch: we evaluate the baseline"""
        self.baseline.epoch_callback(
            self.policy.policy,
            env=self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.data_cfg["val_data_size"],
        )
        # Need to call super() for the dataset to be reset
        super(REINFORCE, self).on_train_epoch_end()