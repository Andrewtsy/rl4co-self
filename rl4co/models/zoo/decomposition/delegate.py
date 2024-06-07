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
        td_new = TensorDict({"locs": torch.zeros(*(batch_size,) + td["locs"].shape[1:]),
                             "demand": torch.zeros(*(batch_size,) + td["demand"].shape[1:]),
                             "visited": torch.ones(*(batch_size,) + td["visited"].shape[1:]),
                             "action_mask": torch.zeros(*(batch_size,) + td["action_mask"].shape[1:], dtype=torch.bool),}, 
                             batch_size=batch_size)
        locs = td["locs"][:, 1:]

        for row in range(len(td_new["locs"])):
            i = row // k
            l = row % k

            idx = torch.where(labels[i] == l)[0]

            td_new["locs"][row, 0] = td["locs"][i, 0]
            td_new["locs"][row, 1:len(idx) + 1, :] = locs[i, idx, :]

            td_new["demand"][row, :len(idx)] = td["demand"][i, idx]
            td_new["visited"][row, 1:len(idx) + 1] = torch.zeros(len(idx))
            td_new["action_mask"][row, :len(idx) + 1] = torch.ones(len(idx) + 1, dtype=torch.bool)

        for key in td.keys() - td_new.keys():
            td_new[key] = td[key].repeat(1, k).reshape(batch_size, -1)

        return td_new