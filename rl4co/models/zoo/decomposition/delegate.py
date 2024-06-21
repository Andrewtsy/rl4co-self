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

        td_new = TensorDict({"locs": torch.zeros(*(batch_size, subproblem_size, 2)),
                             "demand": torch.zeros(*(batch_size, subproblem_size)),
                             "problem_mask": torch.zeros(*(batch_size, subproblem_size), dtype=torch.bool),
                             "depot": td["depot"].repeat(1, k).reshape(batch_size, -1),
                             "capacity": td["capacity"].repeat(1, k).reshape(batch_size, -1)},
                             batch_size=batch_size)

        for row in range(len(td_new["locs"])):
            i, l = row // k, row % k

            idx = torch.where(loc_idx[i] == l)[0]

            td_new["locs"][row, :len(idx), :] = td["locs"][i, 1:][idx, :]

            td_new["demand"][row, :len(idx)] = td["demand"][i, idx]
            td_new["problem_mask"][row, :len(idx)] = torch.ones(len(idx), dtype=torch.bool)

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