from torch import nn
import torch
from torch_geometric import utils
import torch_scatter

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX = 10
MESH_GRIDS_X = {i: torch.meshgrid(torch.arange(0, i).to(device), torch.arange(0, i).to(device))[0].flatten() for i in range(1, MAX)}
MESH_GRIDS_Y = {i: torch.meshgrid(torch.arange(0, i).to(device), torch.arange(0, i).to(device))[1].flatten() for i in range(1, MAX)}

class IGN_2to2(nn.Module):
    """
    This is the implementation of the block diagnal IGN-2to2 layer.
    """
    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0) -> None:
        super(IGN_2to2, self).__init__()
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val

        self.basis_dimension = 15

        # initialize values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True)

        # bias
        self.diag_bias = torch.nn.Parameter(torch.zeros(1, self.output_depth))
        self.all_bias = torch.nn.Parameter(torch.zeros(1, self.output_depth))

    def forward(self, x, edge_index, num_nodes):
        indice_x, indice_y = edge_index
        ops_out = sparse_contractions_2to2_blockdiag(x, num_nodes, normalization = self.normalization, normalization_val = self.normalization_val)  # Sum(m_i x m_i) x D x n_Basis
        output = torch.einsum('dsb,ndb->ns', self.coeffs, ops_out)  # N x S x m x m

        # bias
        mat_diag_bias = (indice_x == indice_y).float().detach().unsqueeze(dim=-1) * self.diag_bias
        output = output + self.all_bias + mat_diag_bias
        return output
    


class IGN_2to2_Dense(nn.Module):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m x m tensor
    '''

    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0, device = 'cpu'):
        super(IGN_2to1_Dense, self).__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device

        self.basis_dimension = 15

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True).to(device = self.device)

        # bias
        self.diag_bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1, 1)).to(device = self.device)
        self.all_bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1, 1)).to(device = self.device)

        # params
        # self.params = torch.nn.ParameterList([self.coeffs, self.diag_bias, self.all_bias])

    def forward(self, inputs, mask):
        # mask = mask.permute(0, 3, 1, 2)     # N x m x m x D -> N x D x m x m
        inputs = inputs.permute(0, 3, 1, 2) # N x m x m x D -> N x D x m x m
        m = inputs.size(3)  # extract dimension

        ops_out = contractions_2_to_2(inputs, m, normalization = self.normalization)
        ops_out = torch.stack(ops_out, dim = 2)

        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, ops_out)  # N x S x m x m

        # bias
        mat_diag_bias = torch.eye(inputs.size(3)).unsqueeze(dim = 0).unsqueeze(dim = 0).to(device = self.device) * self.diag_bias
        output = output + self.all_bias + mat_diag_bias
        output = output.permute(0, 2, 3, 1) # N x S x m x m -> N x m x m x S
        return output



class IGN_2to1(nn.Module):
    """
    This is the implementation of the IGN-2to1 layer.
    """
    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0) -> None:
        super(IGN_2to1, self).__init__()
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val

        self.basis_dimension = 5
        
        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth))

    def forward(self, x, edges_index, num_nodes):
        ops_out = sparse_contractions_2to1_blockdiag(x, num_nodes, normalization = self.normalization) # Sum(m_i) x D x n_Basis
        output = torch.einsum('dsb,ndb->ns', self.coeffs, ops_out)  # Sum(m_i) x D
        
        # bias
        output = output + self.bias
        return output



class IGN_2to1_Dense(nn.Module):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m tensor
    '''

    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0, device = 'cpu'):
        super(IGN_2to1_Dense, self).__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device

        self.basis_dimension = 5

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True).to(device = self.device)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1)).to(device = self.device)

        # params
        # self.params = torch.nn.ParameterList([self.coeffs, self.bias])

    def forward(self, inputs, mask):
        N, m, _, D = inputs.shape  # extract dimension
        inputs = inputs.permute(0, 3, 1, 2) # N x m x m x D -> N x D x m x m

        # if inputs.abs().mean().item() > 5:
        ops_out = contractions_2_to_1(inputs, m, normalization = self.normalization)
        ops_out = torch.stack(ops_out, dim = 2)  # N x D x B x m

        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)  # N x S x m

        # scaler = torch.sum(mask, dim=(1,2), keepdim=True).sqrt()

        output = output / D # normalization

        # bias
        output = output + self.bias
        
        output = output.permute(0, 2, 1) # N x S x m -> N x m x S

        output = output * (mask.sum(dim = 1) > 0)
        
        return output



class Polynormial_2to2(nn.Module):
    """
    This is the implementation of the Polynormial-2to2 layer.
    """
    def __init__(self) -> None:
        super(Polynormial_2to2, self).__init__()
        pass

    def forward(self, x, edges_index, num_nodes):
        pass

class PPGN_layer(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(PPGN_layer, self).__init__()
        self.MLP1 = MLP(input_dim, output_dim, hidden_dims=[128, 128])
        self.MLP2 = MLP(input_dim, output_dim, hidden_dims=[128, 128])
        self.agg = torch.nn.Linear(input_dim + output_dim, output_dim)
    
    def forward(self, x, edges_index, num_nodes):
        x1 = self.MLP1(x)
        x2 = self.MLP2(x)
        _, out = sparse_bmm_blockdiag(edges_index, x1, edges_index, x2, num_nodes)
        out = torch.sqrt(torch.nn.functional.relu(out)) - torch.sqrt(torch.nn.functional.relu(-out)) # signed sqrt
        out = torch.cat([x, out], dim=1)
        out = self.agg(out)
        return out



class PPGN_layer_Dense(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(PPGN_layer_Dense, self).__init__()
        self.MLP1 = MLP(input_dim, output_dim, hidden_dims=[128, 128])
        self.MLP2 = MLP(input_dim, output_dim, hidden_dims=[128, 128])
        self.agg = torch.nn.Linear(input_dim + output_dim, output_dim)
    
    def forward(self, x, mask, print_info):
        N, m, _, D = x.shape
        if print_info:
            _ = [print(torch.mean(torch.abs(p)), n) for n, p in self.named_parameters() if (p.grad is not None)]
        x1 = self.MLP1(x, mask)
        x2 = self.MLP2(x, mask)
        # print('before mm')
        # print((x1 + x2)[mask.repeat(1,1,1,128)].abs().mean())
        out = torch.einsum('nijd,njkd->nikd', x1, x2)
        scaler = torch.sum(mask, dim=(1,2), keepdim=True).sqrt()
        out = out / scaler
        # print('after mm')
        # print(out[mask.repeat(1,1,1,128)].abs().mean())
        out = torch.sqrt(torch.nn.functional.relu(out)) - torch.sqrt(torch.nn.functional.relu(-out)) # signed sqrt
        # print('after sqrt')
        # print(out[mask.repeat(1,1,1,128)].abs().mean())
        out = torch.cat([x, out], dim=-1)
        out = self.agg(out) * mask
        # print('after agg')
        # print(out[mask.repeat(1,1,1,128)].abs().mean())
        
        return out



class MLP(nn.Module):
    """
    MLP operating on (N, D) node features.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128]):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LayerNorm(input_dim)) # NOTE: might be problematic due to the sparsity
        hidden_dims = [input_dim] + hidden_dims

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.LayerNorm(hidden_dims[i+1])) # NOTE: might be problematic due to the sparsity
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x)  # NOTE: might wnat to use a mask here: "x = layer(x) * mask if mask is not None else layer(x)"
        return x



def sparse_bmm_blockdiag(a_indices, a_values, b_indices, b_values, numnodes):
    """
    batched matrix multiplication for coo matrix
    let n be non zero entries
    args
    a_value: n x d
    b_value: n x d
    a_index: 2 x n
    b_index: 2 x n 
    numnodes: torch tensor of number of nodes each block
    return:
    out_value
    out_index

    note: nodenums = batch.ptr[1:] - batch.ptr[:-1]
    """
    numnodes2_list = (numnodes**2).cpu().numpy().tolist()
    numnodes_list = numnodes.cpu().numpy().tolist()
    blocks_a = torch.split(a_values, numnodes2_list)
    blocks_b = torch.split(b_values, numnodes2_list)
    l_out = []
    for blocks_a, block_b, num in zip(blocks_a, blocks_b, numnodes_list):
        shape_a = blocks_a.size()
        # shape_b = block_b.size()
        out = blocks_a.view(num, num, -1).permute(-1, 0, 1) @ block_b.view(num, num, -1).permute(-1, 0, 1)
        l_out.append(out.permute(1, 2, 0).view(shape_a))
    out_indices = a_indices
    out_values = torch.cat(l_out, dim=0)
    return out_indices, out_values



def sparse_bmm_convert(a_indices, a_values, b_indices, b_values):
    """
    batched matrix multiplication for coo matrix
    let n be non zero entries
    args
    a_value: n x d
    b_value: n x d
    a_index: 2 x n
    b_index: 2 x n 
    return:
    out_value
    out_index
    """

    a_dense = torch.sparse_coo_tensor(a_indices, a_values).to_dense()
    b_dense = torch.sparse_coo_tensor(b_indices, b_values).to_dense()

    out_dense = a_dense.permute(-1, 0, 1) @ b_dense.permute(-1, 0, 1)

    out_sparse = out_dense.permute(1, 2, 0).to_sparse(2)

    out_indices = out_sparse.indices()
    out_values = out_sparse.values()

    return out_indices, out_values



def sparse_mm_forloop_batch(a_indices, a_values, b_indices, b_values):
    # NOTE: not finished (TODO)
    a_sparse = torch.sparse_coo_tensor(a_indices, a_values)
    b_sparse = torch.sparse_coo_tensor(b_indices, b_values)
    n, d = a_values.size()
    out = []
    for i in range(d):
        out_sparse = torch.sparse.mm(a_sparse[:, :, i], b_sparse[:, :, i])
        out.append(out_sparse.values)
    return torch.stack(out, dim=1)

    

def sparse_contractions_2to1_blockdiag(a_values, numnodes, normalization = 'inf', normalization_val = 1.0):
    numnodes2_list = (numnodes**2).cpu().numpy().tolist()
    numnodes_list = numnodes.cpu().numpy().tolist()
    blocks_a = torch.split(a_values, numnodes2_list)
    l_out = []
    for blocks_a, num in zip(blocks_a, numnodes_list):
        n, d = blocks_a.size()
        n_biasis = 5
        inputs = blocks_a.view(1, num, num, -1).permute(0, -1, 1, 2)
        out = contractions_2_to_1(inputs, num, normalization = normalization, normalization_val = normalization_val) 
        out = torch.stack(out, dim=3) # N x D x m x n_Basis
        l_out.append(out.permute(0, 2, 1, 3).view(num, d, n_biasis)) # m x D x n_Basis
    out_values = torch.cat(l_out, dim=0) # Sum(m_i) x D x n_Basis
    return out_values 



def sparse_contractions_2to2_blockdiag(a_values, numnodes, normalization = 'inf', normalization_val = 1.0):
    numnodes2_list = (numnodes**2).cpu().numpy().tolist()
    numnodes_list = numnodes.cpu().numpy().tolist()
    blocks_a = torch.split(a_values, numnodes2_list)
    l_out = []
    for blocks_a, num in zip(blocks_a, numnodes_list):
        n, d = blocks_a.size()
        n_biasis = 15
        inputs = blocks_a.view(1, num, num, -1).permute(0, -1, 1, 2)
        out = contractions_2_to_2(inputs, num, normalization = normalization, normalization_val = normalization_val) 
        out = torch.stack(out, dim=4) # N x D x m x m x n_Basis
        l_out.append(out.permute(0, 2, 3, 1, 4).view(n, d, n_biasis)) # m x m x D x n_Basis
    out_values = torch.cat(l_out, dim=0) # Sum(m_i x m_i) x D x n_Basis
    return out_values 



# op2_2_to_2
def contractions_2_to_2(inputs, dim, normalization = 'inf', normalization_val = 1.0):  # N x D x m x m
    diag_part = torch.diagonal(inputs, dim1 = 2, dim2 = 3)   # N x D x m
    sum_diag_part = torch.sum(diag_part, dim = 2).unsqueeze(dim = 2)  # N x D x 1
    sum_of_rows = torch.sum(inputs, dim = 3)  # N x D x m
    sum_of_cols = torch.sum(inputs, dim = 2)  # N x D x m
    sum_all = torch.sum(sum_of_rows, dim = 2)  # N x D

    # op1 - (1234) - extract diag
    op1 = torch.diag_embed(diag_part)  # N x D x m x m

    # op2 - (1234) + (12)(34) - place sum of diag on diag
    op2 = torch.diag_embed(torch.cat([sum_diag_part for d in range(dim)], dim = 2))  # N x D x m x m

    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    op3 = torch.diag_embed(sum_of_rows)  # N x D x m x m

    # op4 - (1234) + (124)(3) - place sum of col i on diag ii
    op4 = torch.diag_embed(sum_of_cols)  # N x D x m x m

    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    op5 = torch.diag_embed(torch.cat([sum_all.unsqueeze(dim = 2) for d in range(dim)], dim = 2))  # N x D x m x m

    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    op6 = torch.cat([sum_of_cols.unsqueeze(dim = 3) for d in range(dim)], dim = 3)  # N x D x m x m

    # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
    op7 = torch.cat([sum_of_rows.unsqueeze(dim = 3) for d in range(dim)], dim = 3)  # N x D x m x m

    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op8 = torch.cat([sum_of_cols.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m x m

    # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
    op9 = torch.cat([sum_of_rows.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m x m

    # op10 - (1234) + (14)(23) - identity
    op10 = inputs  # N x D x m x m

    # op11 - (1234) + (13)(24) - transpose
    op11 = inputs.transpose(3, 2)  # N x D x m x m

    # op12 - (1234) + (234)(1) - place ii element in row i
    op12 = torch.cat([diag_part.unsqueeze(dim = 3) for d in range(dim)], dim = 3)  # N x D x m x m

    # op13 - (1234) + (134)(2) - place ii element in col i
    op13 = torch.cat([diag_part.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m x m

    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    op14 = torch.cat([sum_diag_part for d in range(dim)], dim = 2)
    op14 = torch.cat([op14.unsqueeze(dim = 3) for d in range(dim)], dim = 3) # N x D x m x m

    # op15 - sum of all ops - place sum of all entries in all entries
    op15 = torch.cat([sum_all.unsqueeze(dim = 2) for d in range(dim)], dim = 2)
    op15 = torch.cat([op15.unsqueeze(dim = 3) for d in range(dim)], dim = 3) # N x D x m x m
    
    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim
            op3 = op3 / dim
            op4 = op4 / dim
            op5 = op5 / (dim ** 2)
            op6 = op6 / dim
            op7 = op7 / dim
            op8 = op8 / dim
            op9 = op9 / dim
            op14 = op14 / dim
            op15 = op15 / (dim ** 2)

    return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]



# ops_2_to_1
def contractions_2_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    diag_part = torch.diagonal(inputs, dim1 = 2, dim2 = 3)  # N x D x m

    sum_diag_part = torch.sum(diag_part, dim = 2).unsqueeze(dim = 2)  # N x D x 1
    sum_of_rows = torch.sum(inputs, dim = 3)  # N x D x m
    sum_of_cols = torch.sum(inputs, dim = 2)  # N x D x m
    sum_all = torch.sum(inputs, dim = (2, 3))  # N x D

    # op1 - (123) - extract diag
    op1 = diag_part  # N x D x m

    # op2 - (123) + (12)(3) - tile sum of diag part
    op2 = torch.cat([sum_diag_part for d in range(dim)], dim = 2)  # N x D x m

    # op3 - (123) + (13)(2) - place sum of row i in element i
    op3 = sum_of_rows  # N x D x m

    # op4 - (123) + (23)(1) - place sum of col i in element i
    op4 = sum_of_cols  # N x D x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    op5 = torch.cat([sum_all.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m

    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim
            op3 = op3 / dim
            op4 = op4 / dim
            op5 = op5 / (dim ** 2)

    return [op1, op2, op3, op4, op5]



# following functions are just used while testing:
def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges



def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr



def get_indice(rows): # sparse block diagnal indice
    degrees = utils.degree(rows)
    degrees_cumsum = torch.cat([torch.zeros(1, device=degrees.device), torch.cumsum(degrees, dim=0)]).long()
    indice_x = torch.cat([MESH_GRIDS_X[degree] + degrees_cumsum[idx] for idx, degree in enumerate(list(degrees.cpu().numpy()))])
    indice_y = torch.cat([MESH_GRIDS_Y[degree] + degrees_cumsum[idx] for idx, degree in enumerate(list(degrees.cpu().numpy()))])
    return indice_x, indice_y, degrees



def get_indice_dense(degrees): # dense indice
    degrees = degrees.long()
    max_degree = torch.max(degrees)
    indice_w = torch.cat([idx * torch.ones(degree**2, device=degrees.device, dtype=torch.long) for idx, degree in enumerate(list(degrees.cpu().numpy()))])
    indice_x = torch.cat([MESH_GRIDS_X[degree] for degree in list(degrees.cpu().numpy())])
    indice_y = torch.cat([MESH_GRIDS_Y[degree] for degree in list(degrees.cpu().numpy())])
    return indice_w, indice_x, indice_y



def indice_vectorize_3dim(indice_w, indice_x, indice_y, max_degree):
    return indice_w * max_degree**2 + indice_x * max_degree + indice_y



def block_diag_to_dense(values, blocks):
    """
    values: (N, D)
    """
    N, D = values.shape
    num_blocks = blocks.shape[0]
    max_block = torch.max(blocks).long()
    ph = torch.zeros((num_blocks*max_block*max_block,D), device=values.device)
    mask = torch.zeros((num_blocks*max_block*max_block,1), dtype=torch.bool, device=values.device)

    indice_w, indice_x, indice_y = get_indice_dense(blocks)
    indice_vectorize = indice_vectorize_3dim(indice_w, indice_x, indice_y, max_degree=max_block)
    ph[indice_vectorize] = values
    mask[indice_vectorize] = True

    return ph.view(num_blocks, max_block, max_block, -1), mask(num_blocks, max_block, max_block, -1), indice_vectorize



def block_diag_to_dense_from_indice(values, indice_vectorized, dense_shape):
    N, W, H, D = dense_shape
    ph = torch.zeros((N*W*H, D), device=values.device)
    mask = torch.zeros((N*W*H, 1), dtype=torch.bool, device=values.device)
    ph[indice_vectorized] = values
    mask[indice_vectorized] = True
    return ph.view(N, W, H, -1), mask.view(N, W, H, -1)
    


def VTV(indice_x, indice_y, coord_diff):
    """
    input
    indice_x: (sparse) indice of repeated meshgrid_x for neighbours
    indice_y: (sparse) indice of repeated meshgrid_y for neighbours
    
    return
    VTV: (sparse) block diag N_nodes^2 
    """
    VTV = torch.einsum("nd,nd->n", coord_diff[indice_x], coord_diff[indice_y])
    VTV = torch.sqrt(torch.nn.functional.relu(VTV)) - torch.sqrt(torch.nn.functional.relu(-VTV)) # sqrt relu
    return VTV
