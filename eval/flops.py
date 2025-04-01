import torch
import torch.nn as nn
import sys
sys.path.append('..')
from ptflops import get_model_complexity_info
from src import get_model
import argparse
from src.layers.KANLinear import KANLinear
from copy import deepcopy


FLOPs_MAP = {
    "zero": 0,
    "identity": 0,
    "relu": 1,
    'square_relu': 2,
    "sigmoid":4,
    "silu":5,
    "tanh":6,
    "gelu": 14,
    "polynomial2": 1+2+3-1,
    "polynomial3": 1+2+3+4-1,
    "polynomial5": 1+2+3+4+5-1,
}

class CustomIdentity(nn.Module):
    def __init__(self, in_dim=None, out_dim=None):
        super(CustomIdentity, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        return x  # Pass through the input without modification

class ModelWithoutKANLinear(nn.Module):
    def __init__(self, model):
        super(ModelWithoutKANLinear, self).__init__()
        # Create a deepcopy of the model
        self.model = deepcopy(model)
        # Replace all KANLinear layers with CustomIdentity
        for name, module in list(self.model.named_modules()):  # Use list() to avoid size modification issues
            if isinstance(module, KANLinear):
                # Create a CustomIdentity layer with the same in_features and out_features
                custom_identity = CustomIdentity(in_dim=module.in_dim, out_dim=module.out_dim)
                # Locate the parent module and replace the KANLinear layer
                parent_module, attr_name = self._find_parent_module(name)
                if parent_module is not None:
                    setattr(parent_module, attr_name, custom_identity)

    def _find_parent_module(self, module_name):
        """
        Find the parent module and the attribute name corresponding to `module_name`.
        """
        names = module_name.split('.')
        parent = self.model
        for name in names[:-1]:
            parent = getattr(parent, name, None)
            if parent is None:
                return None, None
        return parent, names[-1]

    def forward(self, x):
        return self.model(x)


def custom_kan_linear_hook(module, input, output):
    # Extract input and output dimensions of the KANLinear layer
    din = input[0].shape[1]  # Input size (features in the input tensor)
    dout = output.shape[1]  # Output size (features in the output tensor)
    grid_size = module.num  # Grid size defined in KANLinear
    spline_order = module.k  # Spline order defined in KANLinear

    print(f"din: {din}, dout: {dout}, grid_size: {grid_size}, spline_order: {spline_order}")

    # Automatically calculate FLOPs and Parameters for KANLinear
    custom_flops = layer_flops(din, dout, shortcut_name="silu", grid=grid_size, k=spline_order)
    custom_params = layer_parameters(din, dout, shortcut_name="silu", grid=grid_size, k=spline_order)
    # Store custom FLOPs and Parameters as an attribute on the module
    module.__custom_flops__ = custom_flops
    module.__custom_params__ = custom_params

    # We return the output without modifying it
    return output

def register_kan_linear_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            module.register_forward_hook(custom_kan_linear_hook)

def layer_flops(din, dout, shortcut_name="silu", grid=5, k=3):
    """
    Custom FLOPs calculation for KANLinear.
    Args:
        din (int): Input dimensions.
        dout (int): Output dimensions.
        shortcut_name (str): Name of the shortcut activation. Default is "silu".
        grid (int): Grid size parameter.
        k (int): Spline order parameter.
    Returns:
        int: Calculated FLOPs for KANLinear.
    """
    flops = (din * dout) * (9 * k * (grid + 1.5 * k) + 2 * grid - 2.5 * k + 1)
    
    # Shortcut FLOPs
    if shortcut_name == "zero":
        shortcut_flops = 0
    else:
        shortcut_flops = FLOPs_MAP[shortcut_name] * din + 2 * din * dout
    
    return flops + shortcut_flops

def layer_parameters(din, dout, shortcut_name="silu", grid=5, k=3):
    """
    Custom Parameters calculation for KANLinear.
    Args:
        din (int): Input dimensions.
        dout (int): Output dimensions.
        shortcut_name (str): Name of the shortcut activation. Default is "silu".
        grid (int): Grid size parameter.
        k (int): Spline order parameter.
    Returns:
        int: Calculated Parameters for KANLinear.
    """
    parameters = din * dout * (grid + k + 2) + dout
    if shortcut_name == "zero":
        shortcut_parameters = 0
    else:
        shortcut_parameters = din * dout
    return parameters + shortcut_parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, default="KANFace")
    parser.add_argument('--rank_ratio', type=float, default=0.6)
    parser.add_argument('--num_features', type=int, default=512)
    parser.add_argument('--grid_size', type=int, default=25)


    args = parser.parse_args()

    # Load the model
    net = get_model(args.name, rank_ratio=args.rank_ratio, num_features=args.num_features, grid_size=args.grid_size)

    # Step 1: Compute Base FLOPs (excluding KANLinear)
    net_without_kan = ModelWithoutKANLinear(net)
    macs, params = get_model_complexity_info(
        net_without_kan, (3, 112, 112), backend='pytorch', as_strings=False,
        print_per_layer_stat=False, verbose=True
    )
    base_flops = int(macs) * 2
    base_params = int(params)

    # Step 2: Register the custom hooks to KANLinear layers
    register_kan_linear_hooks(net)

    # Step 3: Calculate Total FLOPs (including custom KANLinear FLOPs)
    macs, params = get_model_complexity_info(
        net, (3, 112, 112), backend='pytorch', as_strings=False,
        print_per_layer_stat=False, verbose=True
    )

    total_flops = base_flops
    total_params = base_params
    for name, module in net.named_modules():
        if isinstance(module, KANLinear):
            total_flops += getattr(module, "__custom_flops__", 0)  # Add custom FLOPs from hook
            total_params += getattr(module, "__custom_params__", 0)  # Add custom Parameters from hook

    # Step 4: Print Results
    # Convert to MFLOPs and MParams
    base_flops_m = base_flops / 1e6
    kan_flops_m = (total_flops - base_flops) / 1e6
    total_flops_m = total_flops / 1e6

    base_params_m = base_params / 1e6
    kan_params_m = (total_params - base_params) / 1e6
    total_params_m = total_params / 1e6

    # Print formatted results
    print(f"Base FLOPs (excluding KANLinear): {base_flops_m:.2f} ({base_flops}) MFLOPs")
    print(f"KANLinear FLOPs: {kan_flops_m:.2f} ({total_flops - base_flops}) MFLOPs")
    print(f"Total FLOPs (with custom KANLinear FLOPs): {total_flops_m:.2f} ({int(total_flops)}) MFLOPs")

    print(f"Base Parameters (excluding KANLinear): {base_params_m:.2f} ({base_params}) MParams")
    print(f"KANLinear Parameters: {kan_params_m:.2f} ({total_params - base_params}) MParams")
    print(f"Total Parameters (with custom KANLinear Parameters): {total_params_m:.2f} ({total_params}) MParams")

