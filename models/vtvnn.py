import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from typing import Callable

from models.utils import first_node_pooling, first_and_last_node_pooling

from models.VTVNN_codebase.vtvnn_clean import VTV_GCL, VTV_GCL_Dense, VTVNN
from models.VTVNN_codebase.permutation_equivariant import get_indice, get_indice_dense, indice_vectorize_3dim

class VTVNNModel(VTVNN):
    """
    VTVNN model.
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
        dense: bool = True
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
            residual = residual,
            dense = dense,
            node_prediction = True
        )
        self.dense = dense
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool,
                    "first": first_node_pooling, "first_and_last": first_and_last_node_pooling,
                    "none": lambda x,y: x}[pool]

    def forward(self, batch, print_info=False):
        if print_info:
            print("--------- start ---------")
        # torch.nn.Embedding is equivalent to a linear layer with one hot input
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d), where n = |node| * batchsize, d = emb_dim
        pos = batch.pos  # (n, 3)
        rows, cols = batch.edge_index

        nb_rows, nb_cols, nb_num_nodes = get_indice(rows)
        nb_edge = [nb_rows, nb_cols]

        if self.dense:
            indice_w, indice_x, indice_y = get_indice_dense(nb_num_nodes)
            indice_vectorize = indice_vectorize_3dim(indice_w, indice_x, indice_y, nb_num_nodes.max().long().item())
        
        for i in range(0, self.n_layers):
            if self.dense:
                if print_info:
                    print("gcl_%d" % i)
                h, pos, _ = self._modules["gcl_%d" % i](h, pos, batch.edge_index, nb_edge, None, nb_num_nodes, indice_vectorize, batch.batch, print_info=print_info)
            else:
                h, pos, _ = self._modules["gcl_%d" % i](h, pos, batch.edge_index, nb_edge, None, nb_num_nodes, batch.batch)
        
        h = self.embedding_out(h)  # (n, out_dim)

        out = self.pool(h, batch.batch)  # (n, out_dim) -> (batch_size, out_dim); if pool==first_and_last then (2*batch_size, out_dim); if pool=none then (n, out_dim)

        if print_info:
            print("--------------- end ---------------")
        
        return out
