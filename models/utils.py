import torch
from torch import Tensor
from typing import Optional

def first_node_pooling(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    r"""For each graph in the batch, filter out its first node feature.
    This can be used when we are aiming to predict some feature associated with the first node,
    rather than at graph level.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    assert(batch is not None)
    size = int(batch.max().item() + 1) if size is None else size
    batch_right_1 = torch.cat([batch[-1:], batch[0:-1]])
    batch_right_1[0] = -1
    return x[(batch - batch_right_1) == 1]



def first_and_last_node_pooling(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    r"""For each graph in the batch, filter out its first and last node features.
    This can be used when we are aiming to predict some feature associated with the first and last nodes,
    rather than at graph level.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    assert(batch is not None)
    size = int(batch.max().item() + 1) if size is None else size
    batch_right_1 = torch.cat([batch[-1:], batch[0:-1]])
    batch_right_1[0] = -1
    batch_left_1 = torch.cat([batch[1:], batch[:1]])
    batch_left_1[-1] = batch[-1] + 1
    # return the first and last node across batches; 
    # when using this pooling layer, do not need to double out_dim just because prediction label doubles
    first_and_last_mask = ((batch - batch_right_1) == 1) | ((batch_left_1 - batch) == 1)
    out = x[first_and_last_mask]
    return out