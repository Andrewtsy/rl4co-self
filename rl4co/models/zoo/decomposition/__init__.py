from .model import DECOMP
from .decomp import DecompPolicy
from .naive import BaryCenterDecomp

# other = gather_by_index(td_locs, actions) 
# b = (other.unsqueeze(3) * msks.unsqueeze(2))
# other = b.sum(dim=1) / (b != 0).sum(dim=1)