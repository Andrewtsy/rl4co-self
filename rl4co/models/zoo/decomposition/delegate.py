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

    # @staticmethod
    # def delegate(
    #     td: TensorDict = None,
    #     locs_idx: torch.tensor = None,
    #     k: int = None
    # ) -> TensorDict:
    #     """Divide the passed in tensordict state by passed in labels
    #     Args:
    #         td: Tensordict state
    #         loc_idx: Tensor labels of the clusters
    #         k: Number of clusters
    #     Returns:
    #         td_new: Transformed tensordict with shape k * td.shape[0]
    #     """
    #     bs = td["locs"].shape[0] * k
    #     # Find greatest subproblem size to standardize new td size
    #     sps = Delegate._find_gs(locs_idx[:, 1:])

    #     # Before reset, state td must have these key variable pairs
    #     td_new = TensorDict({"locs": torch.zeros(*(bs, sps, 2)),
    #                          "demand": torch.zeros(*(bs, sps)),
    #                          "problem_mask": torch.zeros(*(bs, sps), dtype=torch.bool),
    #                          "depot": td["depot"].repeat(1, k).reshape(bs, -1),
    #                          "capacity": td["capacity"].repeat(1, k).reshape(bs, -1),
    #                          "locs_idx": torch.zeros(*(bs, (sps + 1)))},
    #                          bs=bs)

    #     # Iterate through new batches, calculating the old batch i and cluster l of each
    #     for row in range(len(td_new["locs"])):
    #         i, l = row // k, row % k
    #         # Set td_new location nodes to their respective clusters
    #         idx = torch.where(locs_idx[i, 1:] == l)[0]
    #         td_new["locs"][row, :len(idx), :] = td["locs"][i, 1:][idx, :]

    #         # Assign new demand and problem masks based on the new clusters
    #         td_new["demand"][row, :len(idx)] = td["demand"][i, idx]
    #         td_new["problem_mask"][row, :len(idx)] = torch.ones(len(idx), dtype=torch.bool)

    #         # Mappings from new locations to old locations for recomposition
    #         td_new["locs_idx"][row, 1:len(idx) + 1] = idx + 1

    #     return td_new

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
        problem_indices = torch.arange(sps).expand(bs, -1)
        res = res.scatter(0, map_ps, problem_indices)

        # Remove zero pads
        res[res == 0] = mx
        res, _ = torch.sort(res, dim=1)
        res = res * (res != mx)
        maxsps = torch.max(torch.nonzero(res, as_tuple=True)[-1])
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
                        "demand": torch.zeros(*(bs * k, sps)),
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