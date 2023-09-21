import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from models.utils import first_node_pooling, first_and_last_node_pooling
from typing import Callable

# from models.layers.egnn_layer import EGNNLayer

# class EGNNModel(torch.nn.Module):
#     """
#     E-GNN model from "E(n) Equivariant Graph Neural Networks".
#     """
#     def __init__(
#         self,
#         num_layers: int = 5,
#         emb_dim: int = 128,
#         in_dim: int = 1,
#         out_dim: int = 1,
#         activation: str = "relu",
#         norm: str = "layer",
#         aggr: str = "sum",
#         pool: str = "sum",
#         residual: bool = True,
#         equivariant_pred: bool = False
#     ):
#         """
#         Initializes an instance of the EGNNModel class with the provided parameters.

#         Parameters:
#         - num_layers (int): Number of layers in the model (default: 5)
#         - emb_dim (int): Dimension of the node embeddings (default: 128)
#         - in_dim (int): Input dimension of the model (default: 1)
#         - out_dim (int): Output dimension of the model (default: 1)
#         - activation (str): Activation function to be used (default: "relu")
#         - norm (str): Normalization method to be used (default: "layer")
#         - aggr (str): Aggregation method to be used (default: "sum")
#         - pool (str): Global pooling method to be used (default: "sum")
#         - residual (bool): Whether to use residual connections (default: True)
#         - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
#         """
#         super().__init__()
#         self.equivariant_pred = equivariant_pred
#         self.residual = residual

#         # Embedding lookup for initial node features
#         self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

#         # Stack of GNN layers
#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr))

#         # Global pooling/readout function
#         self.pool = {"mean": global_mean_pool, "sum": global_add_pool,
#                     "first": first_node_pooling, "first_and_last": first_and_last_node_pooling,
#                     "none": lambda x,y: x}[pool]

#         if self.equivariant_pred:
#             # Linear predictor for equivariant tasks using geometric features
#             self.pred = torch.nn.Linear(emb_dim + 3, out_dim)
#         else:
#             # MLP predictor for invariant tasks using only scalar features
#             self.pred = torch.nn.Sequential(
#                 torch.nn.Linear(emb_dim, emb_dim),
#                 torch.nn.ReLU(),
#                 torch.nn.Linear(emb_dim, out_dim)
#             )

#     def forward(self, batch):
#         h = self.emb_in(batch.atoms)  # (n,) -> (n, d), where n = |node| * batchsize, d = emb_dim
#         pos = batch.pos  # (n, 3)

#         for conv in self.convs:
#             # Message passing layer
#             h_update, pos_update = conv(h, pos, batch.edge_index)

#             # Update node features (n, d) -> (n, d)
#             h = h + h_update if self.residual else h_update 

#             # Update node coordinates (no residual) (n, 3) -> (n, 3)
#             pos = pos_update

#         if not self.equivariant_pred:
#             # Select only scalars for invariant prediction
#             out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d); if pool==first_and_last then (2*batch_size, d); if pool=none then (n, d)
#         else:
#             out = self.pool(torch.cat([h, pos], dim=-1), batch.batch)
            
#         return self.pred(out)  # (batch_size, out_dim); if pool==first_and_last then (2*batch_size, out_dim); if pool=none then (n, d)




from models.EGNN_codebase.egnn_clean import EGNN

class EGNNModel(EGNN):
    """
    E-GNN model from "E(n) Equivariant Graph Neural Networks".
    """
    def __init__(
        self,
        num_layers: int = 5,
        emb_dim: int = 128,
        in_dim: int = 1,
        out_dim: int = 1,
        activation: Callable[[Tensor], Tensor] = torch.nn.SiLU(),
        pool: str = "sum",
        residual: bool = True,
        equivariant_pred: bool = False
    ):  
        """
        Initializes an instance of the EGNNModel class with the provided parameters.

        Parameters:
        - num_layers (int): Number of layers in the model (default: 5)
        - emb_dim (int): Dimension of the node embeddings (default: 128)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - activation (Tensor -> Tensor): Activation function to be used (default: torch.nn.SiLU())
        - pool (str): Global pooling method to be used (default: "sum")
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
        """
        super().__init__(
            in_node_nf = in_dim,
            hidden_nf = emb_dim,
            out_node_nf = out_dim,
            act_fn = activation,
            n_layers = num_layers,
            residual = residual
        )
        self.equivariant_pred = equivariant_pred
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool,
                    "first": first_node_pooling, "first_and_last": first_and_last_node_pooling,
                    "none": lambda x,y: x}[pool]

    def forward(self, batch):
        # torch.nn.Embedding is equivalent to a linear layer with one hot input
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d), where n = |node| * batchsize, d = emb_dim
        pos = batch.pos  # (n, 3)
        edges = batch.edge_index
        
        for i in range(0, self.n_layers):
            h, pos, _ = self._modules["gcl_%d" % i](h, edges, pos, edge_attr=None)
        h = self.embedding_out(h)  # (n, out_dim)

        if not self.equivariant_pred:
            # Select only scalars for invariant prediction
            out = self.pool(h, batch.batch)  # (n, out_dim) -> (batch_size, out_dim); if pool==first_and_last then (2*batch_size, out_dim); if pool=none then (n, out_dim)
        else:
            out = self.pool(torch.cat([h, pos], dim=-1), batch.batch)
            
        return out