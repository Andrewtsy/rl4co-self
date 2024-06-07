from sklearn.cluster import KMeans
from tensordict.tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.am import AttentionModel
import rl4co.models.zoo.decomposition

def shout():
    print("Hello World")

class KmeansDecomp():
    """Baseline K-means clustering for naive decomposition.
    Input a routing environment and outputs an environment split into k subproblems.

    Args:
        k: Number of clusters
        state: Tensordictionary problem state
        pol: Policy used for decomposition, scikit-learn's kmeans clustering
    """

    def __init__(
        self,
        k: int = None,
        env: RL4COEnvBase = None,
    ):
        self.k = k
        self.env = env
        self.pol = KMeans(n_clusters=k)

    def forward(
        self,
        td: TensorDict = None
    ) -> RL4COEnvBase:
        """Kmeans split the environment
        Args:
            td: Tensordict state
        Returns:
            env: Transformed environment
        """
        batch_size = td["locs"].shape[0] * self.k
        td_new = TensorDict({}, batch_size=batch_size)

        locs = td["locs"][:, 1:]
        labels = torch.vstack([torch.tensor(self.pol.fit_predict(ex)) for ex in locs])

        nlocs_shape = (batch_size,) + td["locs"].shape[1:]
        n_shape_wd = (batch_size,) + td["demand"].shape[1:]
        n_shape_wod = (batch_size,) + td["visited"].shape[1:]

        locs_new = torch.zeros(*nlocs_shape)
        demand_new = torch.zeros(*n_shape_wd)
        visited_new = torch.ones(*n_shape_wod)
        action_new = torch.zeros(*n_shape_wod, dtype=torch.bool)

        for row in range(len(locs_new)):
            i = row // self.k
            l = row % self.k

            idx = torch.where(labels[i] == l)[0]

            locs_new[row, 0] = td["locs"][i, 0]
            locs_new[row, 1:len(idx) + 1, :] = locs[i, idx, :]

            demand_new[row, :len(idx)] = td["demand"][i, idx]
            visited_new[row, 1:len(idx) + 1] = torch.zeros(len(idx))
            action_new[row, :len(idx) + 1] = torch.ones(len(idx) + 1, dtype=torch.bool)

        td_new["locs"], td_new["demand"], td_new["visited"], td_new["action_mask"] = locs_new, demand_new, visited_new, action_new
        for key in td.keys():
            if key not in ["locs", "demand", "visited", "action_mask"]:
                td_new[key] = td[key].repeat(1, self.k).reshape(batch_size, -1)
        return td_new


if __name__ == "__main__":
    shout()

    import torch
    from rl4co.envs import CVRPEnv
    from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
    from rl4co.utils.trainer import RL4COTrainer

    # RL4CO env based on TorchRL
    env = CVRPEnv(generator_params={'num_loc': 20})

    # Model: default is AM with REINFORCE and greedy rollout baseline
    model = AttentionModel(env,
                    baseline='rollout',
                    train_data_size=1_000, # really small size for demo
                    val_data_size=100)

    td = env.reset(batch_size=[2])

    decomp = KmeansDecomp(4)
    td_new = decomp.forward(td)

    input("wait...")
    pass