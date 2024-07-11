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
        idx: torch.tensor = None,
        k: int = None
    ) -> TensorDict:
        """VECTORIZED
        Divide the passed in tensordict state by passed in labels
        Args:
            td: Tensordict state
            idx: Tensor map from problem nodes to their clusters
            k: Number of clusters
        Returns:
            td_new: Transformed tensordict with shape k * td.shape[0]
        """
        # Remove depot node
        locs = td["locs"]
        # Set batch and subproblem sizes, as well as maximum node number for zero padding (used later)
        bs, sps, mx = idx.shape[0], idx.shape[1], idx.shape[1] + 1
        # Subproblem divided *map* will map nodes to their original problem indices
        res = torch.zeros((k * bs, sps), dtype=torch.int64)
        # Take map of nodes in the original problems to their subproblem rows,
        # Indices of each of the nodes in their original subproblems
        # Scatter them onto a map from subproblems to their original problem indices
        map_ps = torch.arange(0, k * bs, step=k).unsqueeze(1) + idx
        problem_indices = torch.arange(sps).expand(bs, -1) + 1
        res = res.scatter(0, map_ps, problem_indices)

        # Remove zero pads
        res[res == 0] = mx
        res, _ = torch.sort(res, dim=1)
        res = res * (res != mx)
        maxsps = torch.max(torch.nonzero(res, as_tuple=True)[-1]) + 1
        idx = res[:, :maxsps]

        # Locations repeated to be same shape as subproblems tensor
        rep = torch.repeat_interleave(locs, 2, dim=0)
        # Independently construct x and y coordinate location tensors
        xy = []
        for i in range(2):
            # Gather coordinates by idx
            xy.append(torch.gather(rep[:, :, i], 1, idx))
        locs = torch.stack(xy, dim=-1)

        problem_mask = idx != 0

        # Before reset, state td must have these key variable pairs
        td = TensorDict({"locs": locs,
                        "demand": torch.zeros(*(bs * k, maxsps)),
                        "problem_mask": problem_mask,
                        "depot": torch.repeat_interleave(td["depot"], k, 0),
                        "capacity": torch.repeat_interleave(td["capacity"], k, 0),
                        "locs_idx": idx},
                        batch_size=bs * k)

        return td
    
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
    def no_nan(row):
        msk = torch.isnan(row)
        return row[:torch.nonzero(msk)[0][0]] if torch.nonzero(msk).numel() > 0 else row

    
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