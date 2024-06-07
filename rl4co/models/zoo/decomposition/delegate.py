from tensordict.tensordict import TensorDict
import torch

def shout():
    print("Hello World")

class Delegate():
    """Light class to split up a tensordict state into groups based on cluster labels

    Args:
    """

    def __init__(
        self
    ):
        pass

    def forward(
        td: TensorDict = None,
        labels: torch.tensor = None,
        k: int = 0
    ) -> TensorDict:
        """Divide the passed in tensordict state by passed in labels
        Args:
            td: Tensordict state
            labels: Labels of the clusters
            k: Number of clusters
        Returns:
            td_new: Transformed tensordict with shape k * td.shape[0]
        """
        batch_size = td["locs"].shape[0] * k
        td_new = TensorDict({}, batch_size=batch_size)

        locs = td["locs"][:, 1:]

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