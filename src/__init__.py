from .models.EdgeFace import EdgeFace
from .models.KANFace import KANFace

def get_model(name, **kwargs):
    num_features = kwargs.get('num_features')
    grid_size = kwargs.get('grid_size')
    rank_ratio = kwargs.get("rank_ratio")
    if name == "EdgeFace":
        return EdgeFace(num_features=num_features)
    elif name == "KANFace":
        return KANFace(rank_ratio=rank_ratio, grid_size=grid_size, neuron_fun="mean", num_features=num_features)
    else:
        raise ValueError()