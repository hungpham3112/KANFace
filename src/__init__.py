from .models.ConvKAN import ConvKAN
from .models.EdgeFace import EdgeFace
from .models.EdgeFaceKAN import EdgeFaceKAN

def get_model(name, **kwargs):
    num_features = kwargs.get('num_features')
    grid_size = kwargs.get('grid_size')
    rank_ratio = kwargs.get("rank_ratio")
    # Conv_KAN
    if name == "ConvKAN_sum":
        return ConvKAN(neuron_fun="sum", num_features=num_features)
    elif name == "ConvKAN_mean":
        return ConvKAN(neuron_fun="mean", num_features=num_features)
    elif name == "EdgeFace":
        return EdgeFace(num_features=num_features)
    elif name == "EdgeFaceKAN_sum":
        return EdgeFaceKAN(rank_ratio=rank_ratio, grid_size=grid_size, neuron_fun="sum", num_features=num_features)
    elif name == "EdgeFaceKAN_mean":
        return EdgeFaceKAN(rank_ratio=rank_ratio, grid_size=grid_size, neuron_fun="mean", num_features=num_features)
    else:
        raise ValueError()