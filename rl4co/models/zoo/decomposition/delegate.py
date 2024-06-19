from tensordict.tensordict import TensorDict
import torch

class Delegate():
    """Light class to split up a tensordict state into groups based on cluster labels
    """

    def __init__(
        self
    ):
        pass

    @staticmethod
    def delegate(
        td: TensorDict = None,
        loc_idx: torch.tensor = None,
        k: int = None
    ) -> TensorDict:
        """Divide the passed in tensordict state by passed in labels
        Args:
            td: Tensordict state
            labels: Tensor labels of the clusters
            k: Number of clusters
        Returns:
            td_new: Transformed tensordict with shape k * td.shape[0]
        """
        batch_size = td["locs"].shape[0] * k
        subproblem_size = Delegate._find_sps(loc_idx)

        td_new = TensorDict({"locs": torch.zeros(*(batch_size, subproblem_size + 1, 2)),
                             "demand": torch.zeros(*(batch_size, subproblem_size)),
                             "visited": torch.ones(*(batch_size, subproblem_size + 1)),
                             "action_mask": torch.zeros(*(batch_size, subproblem_size + 1), dtype=torch.bool)}, 
                             batch_size=batch_size)

        for row in range(len(td_new["locs"])):
            i = row // k
            l = row % k

            idx = torch.where(loc_idx[i] == l)[0]

            td_new["locs"][row, 0] = td["locs"][i, 0]
            td_new["locs"][row, 1:len(idx) + 1, :] = td["locs"][:, 1:][i, idx, :]

            td_new["demand"][row, :len(idx)] = td["demand"][i, idx]
            td_new["visited"][row, 1:len(idx) + 1] = torch.zeros(len(idx))
            td_new["action_mask"][row, :len(idx) + 1] = torch.ones(len(idx) + 1, dtype=torch.bool)

        for key in td.keys() - td_new.keys():
            td_new[key] = td[key].repeat(1, k).reshape(batch_size, -1)

        return td_new
    
    @staticmethod
    def del_sp(loc_idx, labels):
        locsp_idx = torch.zeros_like(loc_idx)
        for i in range(locsp_idx.shape[0]):
            for j in range(locsp_idx.shape[1]):
                locsp_idx[i, j] = labels[i, loc_idx[i, j]]
        return locsp_idx

    
    @staticmethod
    def _find_sps(
        loc_idx: torch.Tensor = None
    ) -> int:
        """Small helper method to find the maximum subproblem size
        """
        row_f = lambda row: max(torch.bincount(row))
        max_sps = 0
        for row in loc_idx:
            max_sps = max(row_f(row), max_sps)
        return max_sps