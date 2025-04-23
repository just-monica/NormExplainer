import torch
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch.nn import Parameter
from typing import Dict, Union, List
def set_edge_masks(
    model: torch.nn.Module,
    mask: Union[Tensor, Parameter],
    edge_index: Tensor,
    apply_sigmoid: bool = False,
    layers: List = None
):
    r"""Apply mask to every graph layer in the :obj:`model`."""
    loop_mask = edge_index[0] != edge_index[1]

    # Loop over layers and set masks on MessagePassing layers:
    if layers is None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                # Skip layers that have been explicitly set to `False`:
                if module.explain is False:
                    continue

                # Convert mask to a param if it was previously registered as one.
                # This is a workaround for the fact that PyTorch does not allow
                # assignments of pure tensors to parameter attributes:
                if (not isinstance(mask, Parameter)):
                    mask = Parameter(mask)

                module.explain = True
                module._edge_mask = mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = apply_sigmoid
    else:
        i = 0
        for module in model.modules():
            if isinstance(module, MessagePassing):
                # Skip layers that have been explicitly set to `False`:
                if module.explain is False:
                    continue

                # Convert mask to a param if it was previously registered as one.
                # This is a workaround for the fact that PyTorch does not allow
                # assignments of pure tensors to parameter attributes:
                if (not isinstance(mask, Parameter)):
                    mask = Parameter(mask)

                if i in layers:
                    module.explain = True
                    module._edge_mask = mask
                    module._loop_mask = loop_mask
                    module._apply_sigmoid = apply_sigmoid
                i += 1

def clear_edge_masks(model: torch.nn.Module):
    r"""Clear all masks from the :obj:`model`."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.explain = False
            module._edge_mask = None
            module._loop_mask = None
            module._apply_sigmoid = False