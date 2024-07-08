from tensordict.tensordict import TensorDict
import torch

class Delegate():
    """Light class to split up a tensordict state into groups based on cluster labels.
    Like a Utils class. Will merge into other classes later.
    """

    def __init__(
        self
    ):
        pass

    @staticmethod
    def delegate(
        td: TensorDict = None,
        locs_idx: torch.tensor = None,
        k: int = None
    ) -> TensorDict:
        """Divide the passed in tensordict state by passed in labels
        Args:
            td: Tensordict state
            loc_idx: Tensor labels of the clusters
            k: Number of clusters
        Returns:
            td_new: Transformed tensordict with shape k * td.shape[0]
        """
        batch_size = td["locs"].shape[0] * k
        # Find greatest subproblem size to standardize new td size
        subproblem_size = Delegate._find_gs(locs_idx[:, 1:])

        # Before reset, state td must have these key variable pairs
        td_new = TensorDict({"locs": torch.zeros(*(batch_size, subproblem_size, 2)),
                             "demand": torch.zeros(*(batch_size, subproblem_size)),
                             "problem_mask": torch.zeros(*(batch_size, subproblem_size), dtype=torch.bool),
                             "depot": td["depot"].repeat(1, k).reshape(batch_size, -1),
                             "capacity": td["capacity"].repeat(1, k).reshape(batch_size, -1),
                             "locs_idx": torch.zeros(*(batch_size, (subproblem_size + 1)))},
                             batch_size=batch_size)

        # Iterate through new batches, calculating the old batch i and cluster l of each
        for row in range(len(td_new["locs"])):
            i, l = row // k, row % k
            # Set td_new location nodes to their respective clusters
            idx = torch.where(locs_idx[i, 1:] == l)[0]
            td_new["locs"][row, :len(idx), :] = td["locs"][i, 1:][idx, :]

            # Assign new demand and problem masks based on the new clusters
            td_new["demand"][row, :len(idx)] = td["demand"][i, idx]
            td_new["problem_mask"][row, :len(idx)] = torch.ones(len(idx), dtype=torch.bool)

            # Mappings from new locations to old locations for recomposition
            td_new["locs_idx"][row, 1:len(idx) + 1] = idx + 1

        return td_new
    
    @staticmethod
    def del_sp(loc_idx, labels):
        locsp_idx = torch.zeros_like(loc_idx)
        for i in range(locsp_idx.shape[0]):
            for j in range(locsp_idx.shape[1]):
                locsp_idx[i, j] = labels[i, loc_idx[i, j]]
        return locsp_idx

    @staticmethod
    def no_pad(row, pad=torch.tensor([0], dtype=torch.int)):
        i = len(row) - 1
        while i > 0 and torch.all(row[i] == pad):
            i -= 1
        return row[:i + 1]

    
    @staticmethod
    def _find_gs(
        loc_idx: torch.Tensor = None
    ) -> int:
        """Small helper method to find the maximum group size
        Takes as input tensor 
        """
        row_f = lambda row: max(torch.bincount(row))
        max_gs = 0
        for row in loc_idx:
            max_gs = max(row_f(row), max_gs)
        return max_gs