from torch import nn
import torch

from models.VTVNN_codebase.permutation_equivariant import PPGN_layer, IGN_2to1, MLP, PPGN_layer_Dense, IGN_2to1_Dense
from models.VTVNN_codebase.permutation_equivariant import get_indice, get_indice_dense, VTV, block_diag_to_dense_from_indice, indice_vectorize_3dim
from models.VTVNN_codebase.egnn_adapted import unsorted_segment_sum, unsorted_segment_mean

import torch_scatter



class VTV_GCL(nn.Module):
    """
    VTV_GCL: VTV Graph Convolution Layer
    
    Activation:
    1. Compute the coord weight based on (1).VTV and (2).node_to_neighbour_graph
    2. Update coord
    3. Update node_feature
    """
    def __init__(self, in_node_nf, out_node_nf, hidden_nf, in_edge_nf, VTV_coord_nf, 
                 act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, tanh=False) -> None:
        super(VTV_GCL, self).__init__()
        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.hidden_nf = hidden_nf
        self.in_edge_nf = in_edge_nf
        self.act_fn = act_fn
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.VTV_coord_nf = VTV_coord_nf
        self.coords_agg = 'mean'  # assume using mean aggregation
        self.pe = True  # permutation equivariant

        self.edge_encoder = MLP(2*self.in_node_nf, self.in_node_nf, [self.hidden_nf])
        self.PPGN_layer = PPGN_layer(VTV_coord_nf + self.in_node_nf, self.in_node_nf)
        self.IGN_2to1 = IGN_2to1(self.in_node_nf, self.in_node_nf)

        self.nb_x_nf = 1

        self.node_dec = MLP(in_node_nf, in_node_nf, [self.hidden_nf])

        # coord model
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(in_node_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter %s' % self.coords_agg)
        coord = coord + agg
        return coord

    def getPositionEncoding_torch(self, a, d, n=10000, a_scale=10):
        pos_enc = torch.zeros((a.shape[0], d), device=a.device)
        dividers = torch.arange(d//2, device=a.device).float()
        div_term = torch.exp(torch.log(torch.tensor(n, device=a.device)) * (2 * dividers / d))
        pos_enc[:, 0::2] = torch.sin(a * a_scale / div_term)
        pos_enc[:, 1::2] = torch.cos(a * a_scale / div_term)
        return pos_enc

    def forward(self, h, x, edges, nb_edge, edge_attr, nb_num_nodes, batch=None):
        """
        :param h: node_feature
        :param x: coord (n, 3)
        :param nb_x: coord_diff
        :param nb_edge: edge_index
        :param edge_attr: edge_attr
        :param batch: batch
        """
        rows, cols = edges
        coord_diff = x[rows] - x[cols]
        nb_rows, nb_cols = nb_edge

        nb_x = VTV(nb_rows, nb_cols, coord_diff)

        if not self.pe:
            nb_x = nb_x.unsqueeze(-1).repeat(1, self.VTV_coord_nf)
        else:
            nb_x = self.getPositionEncoding_torch(nb_x.unsqueeze(-1), self.VTV_coord_nf, a_scale=self.VTV_coord_nf/2) # 2 is a hyperparameter, needs to be change for datasets with different scales
        
        nb_num_nodes = nb_num_nodes.long()

        # edge from node
        edge_from_node = torch.cat([h[rows], h[cols]], dim=1)
        edge_from_node = self.edge_encoder(edge_from_node)

        nb_x_from_node = edge_from_node[nb_edge[0]] * edge_from_node[nb_edge[1]] #VTV(nb_edge[0], nb_edge[1], edge_from_node)

        # edge from coord (nb_x)
        nb_x = torch.cat([nb_x, nb_x_from_node], dim=1)
        
        nb_x = self.PPGN_layer(nb_x, nb_edge, num_nodes=nb_num_nodes)
        nb_x_tensor1 = self.IGN_2to1(nb_x, None, nb_num_nodes) # TODO: Degree scaler
        
        # update coord
        x = self.coord_model(x, edges, coord_diff, nb_x_tensor1)

        # update node_feature
        nb_x_tensor0 = torch_scatter.scatter_add(nb_x_tensor1, rows, dim=0, dim_size=x.shape[0]) #* torch.log(nb_num_nodes + 1).unsqueeze(-1)
        h = h + self.node_dec(nb_x_tensor0)

        return h, x, edge_attr



class VTV_GCL_Dense(nn.Module):
    """
    VTV_GCL: VTV Graph Convolution Layer
    
    Activation:
    1. Compute the coord weight based on (1)VTV and (2)node_to_neighbour_graph
    2. Update coord
    3. Update node_feature
    """
    def __init__(self, in_node_nf, out_node_nf, hidden_nf, in_edge_nf, VTV_coord_nf, 
                 act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, tanh=False) -> None:
        super(VTV_GCL_Dense, self).__init__()
        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.hidden_nf = hidden_nf
        self.in_edge_nf = in_edge_nf
        self.act_fn = act_fn
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.VTV_coord_nf = VTV_coord_nf
        self.coords_agg = 'mean'
        self.pe = True

        self.edge_encoder = MLP(2*self.in_node_nf, self.in_node_nf, [self.hidden_nf])
        self.PPGN_layer = PPGN_layer_Dense(VTV_coord_nf + self.in_node_nf, self.in_node_nf)
        self.IGN_2to1 = IGN_2to1_Dense(self.in_node_nf, self.in_node_nf)

        self.nb_x_nf = 1

        self.node_dec = MLP(in_node_nf, in_node_nf, [self.hidden_nf])

        # coord model
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(in_node_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def getPositionEncoding_torch(self, a, d, n=10000, a_scale=10):
        pos_enc = torch.zeros((a.shape[0], d), device=a.device)
        dividers = torch.arange(d//2, device=a.device).float()
        div_term = torch.exp(torch.log(torch.tensor(n, device=a.device)) * (2 * dividers / d))
        pos_enc[:, 0::2] = torch.sin(a * a_scale / div_term)
        pos_enc[:, 1::2] = torch.cos(a * a_scale / div_term)
        return pos_enc

    def forward(self, h, x, edges, nb_edge, edge_attr, nb_num_nodes, indice_vectorized, batch=None, print_info=False):
        """
        :param h: node_feature
        :param x: coord (n, 3)
        :param nb_x: coord_diff
        :param nb_edge: edge_index
        :param edge_attr: edge_attr
        :param batch: batch
        """
        rows, cols = edges
        coord_diff = x[rows] - x[cols]
        nb_rows, nb_cols = nb_edge

        nb_x = VTV(nb_rows, nb_cols, coord_diff)  # neighbor graph, sparse
        
        if print_info:
            print("nb_x 0 pe -->", nb_x.abs().mean())
        if not self.pe:
            nb_x = nb_x.unsqueeze(-1).repeat(1, self.VTV_coord_nf)
        else:
            nb_x = self.getPositionEncoding_torch(nb_x.unsqueeze(-1), self.VTV_coord_nf, a_scale=self.VTV_coord_nf/2) # 2 is a hyperparameter, needs to be change for datasets with different scales
        if print_info:
            print("nb_x 1 pe -->", nb_x.abs().mean()) 
        
        nb_num_nodes = nb_num_nodes.long()
        max_block = nb_num_nodes.max().long().item()

        # edge from node
        edge_from_node = torch.cat([h[rows], h[cols]], dim=1)
        edge_from_node = self.edge_encoder(edge_from_node)
        
        if print_info:
            print("nb_x_from_node 0 pe -->", edge_from_node.abs().mean())
        nb_x_from_node = edge_from_node[nb_edge[0]] * edge_from_node[nb_edge[1]] #VTV(nb_edge[0], nb_edge[1], edge_from_node)
        nb_x_from_node = torch.sqrt(torch.nn.functional.relu(nb_x_from_node)) - torch.sqrt(torch.nn.functional.relu(-nb_x_from_node)) # sqrt relu
        if print_info:
            print("nb_x_from_node 1 pe -->", nb_x_from_node.abs().mean())

        # edge from coord (nb_x)
        nb_x = torch.cat([nb_x, nb_x_from_node], dim=1)

        # to dense
        nb_x, _mask = block_diag_to_dense_from_indice(nb_x, indice_vectorized, (nb_num_nodes.shape[0], max_block, max_block, nb_x.shape[-1])) 


        if print_info:
            print("nb_x 0 PPGN -->", nb_x.abs().mean()) 
        nb_x = self.PPGN_layer(nb_x, _mask, print_info)  # DEBUG INFO: graident explosion e+15
        if print_info:
            print("nb_x 1 PPGN -->", nb_x.abs().mean())

        nb_x_tensor1 = self.IGN_2to1(nb_x, _mask)  # TODO: Degree scalar
        nb_x_tensor0 = torch.sum(nb_x_tensor1, dim=1)

        # to sparse  # NOTE not sure if this is correct
        nb_x_tensor1 = nb_x_tensor1.masked_select(_mask.sum(1) > 0).view(-1, nb_x_tensor1.shape[-1])

        # update coord
        if print_info:
            print("x 0 coord_model -->", x.abs().mean()) 
            print("nb_x_tensor 0 coord_model -->", nb_x_tensor1.abs().mean())

        x = self.coord_model(x, edges, coord_diff, nb_x_tensor1)
        if print_info:
            print("x 1 coord_model -->", x.abs().mean())

        # update node_feature
        h = h + self.node_dec(nb_x_tensor0)

        return h, x, edge_attr



class VTVNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(), device='cpu', dense=True,
                 n_layers=4, residual=True, normalize=False, attention=False, tanh=False, node_prediction=False) -> None:
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''
        super(VTVNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.node_prediction = node_prediction
        self.device = device
        self.VTV_coord_nf = 64
        self.pe = True
        self.dense = dense


        for i in range(0, n_layers):
            if self.dense:
                self.add_module("gcl_%d" % i, VTV_GCL_Dense(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_edge_nf=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh, VTV_coord_nf=self.VTV_coord_nf))
            else:
                self.add_module("gcl_%d" % i, VTV_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_edge_nf=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh, VTV_coord_nf=self.VTV_coord_nf))

        if node_prediction:
            self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        else:
            self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                        act_fn,
                                        nn.Linear(self.hidden_nf, self.hidden_nf))

            self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                        act_fn,
                                        nn.Linear(self.hidden_nf, 1))

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr, batch=None, print_info=False):
        if print_info:
            print("---- start -----")
        h = self.embedding_in(h)
        rows, cols = edges
        nb_rows, nb_cols, nb_num_nodes = get_indice(rows)
        # indice_x, indice_y = indice_x.to(self.device), indice_y.to(self.device)
        
        nb_edge = [nb_rows, nb_cols]

        if self.dense:
            indice_w, indice_x, indice_y = get_indice_dense(nb_num_nodes)
            indice_vectorize = indice_vectorize_3dim(indice_w, indice_x, indice_y, nb_num_nodes.max().long().item())

        for i in range(0, self.n_layers):
            if self.dense:
                if print_info:
                    print("gcl_%d" % i)
                h, x, _ = self._modules["gcl_%d" % i](h, x, edges, nb_edge, edge_attr, nb_num_nodes, indice_vectorize, batch, print_info)
            else:
                h, x, _ = self._modules["gcl_%d" % i](h, x, edges, nb_edge, edge_attr, nb_num_nodes, batch)
        
        if self.node_prediction:
            h = self.embedding_out(h)
        else:
            h = self.node_dec(h)
            h = torch_scatter.scatter_mean(h, batch, dim=0)
            h = self.graph_dec(h)
        
        if print_info:
            print("------------------------- end ----------------------------")
        return h, x



def sparse_rate(tensor):
    return tensor.count_nonzero() / tensor.numel()
