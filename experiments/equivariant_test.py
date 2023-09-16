import random
import math
import torch
import torch_geometric
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

import e3nn
from functools import partial
import itertools


def generate_invariant_dataset(num=5, fold=3, dim=2, target="max", seed = 0):
    """Generate random rotationally equivalent star graphs.
    
    ----------
    Parameters:
    - num (int): number of star graphs to generate
    - fold (list of int): number of spokes to be considered when generating star graphs
    - dim (int): range of random angles (2D or 3D)
    - target (str): metric to be used in pooling angles
    - seed (int): random seed
    """

    assert dim == 2 or dim == 3, "dimension must be 2 or 3."
    assert target in ["max", "mean"], "regression target must be 'max' or 'mean' of angles"

    torch.manual_seed(seed)
    random.seed(seed)

    dataset = []

    # atoms representing central and non-central nodes
    atoms = torch.LongTensor([0,] + [0,] * fold)
    # edges read vertically; star graph: 0 -> 1, 0 -> 2 ...
    edge_index = torch.LongTensor( [ [0,] * fold, list(range(1, fold + 1)) ] )
    # origin and first spoke
    x = torch.Tensor([1, 0, 0])
    pos = [torch.Tensor([0, 0, 0]), x]

    if dim == 2:
        for count in range(1, fold):
            # random angle between 0 and 2*pi
            random_angle = random.uniform(0, 2 * math.pi)
            new_point = torch.Tensor([math.cos(random_angle), math.sin(random_angle), 0])
            pos.append(new_point)

    elif dim == 3:
        for count in range(1, fold):
            theta = random.uniform(0, 2 * math.pi)  # Random angle around z-axis
            phi = random.uniform(0, math.pi)  # Random angle from z-axis (polar angle)
            new_point = torch.Tensor([
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi)
            ])
            pos.append(new_point)

    # scale the pos:
    avg_vec = sum(pos)
    alpha = random.uniform(-1, 2)
    pos1 = [p + alpha * avg_vec for p in pos[1:]]
    pos = pos[:1] + pos1
    

    # compute all possible angles
    spoke_positions = pos[1:]
    angles = []
    for combo in itertools.combinations(spoke_positions, 2):
        v1, v2 = combo
        angle = torch.acos(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2)))
        angles.append(angle)
    
    if target == "max":
        y = torch.Tensor([ max(angles) ])
    elif target == "mean":
        y = torch.Tensor([ sum(angles)/len(angles) ])

    pos = torch.stack(pos)
    data = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data.edge_index = to_undirected(data.edge_index)
    dataset.append(data)

    for _ in range(num-1):
        R = e3nn.o3.rand_matrix()
        pos2 = [x @ R.T for x in pos]
        data = Data(atoms=atoms, edge_index=edge_index, pos=torch.stack(pos2), y=y)
        data.edge_index = to_undirected(data.edge_index)
        dataset.append(data)

    return dataset