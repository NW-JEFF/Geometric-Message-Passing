
import sys
sys.path.append('../')
import pandas as pd
import os

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

import argparse



#from experiments.utils.plot_utils import plot_2d, plot_3d
from experiments.utils.train_utils import run_experiment_reg
from models import SchNetModel, DimeNetPPModel, SphereNetModel, EGNNModel, GVPGNNModel, TFNModel, MACEModel

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# parse args
parser = argparse.ArgumentParser(description="A simple script to demonstrate argparse usage.")
    
parser.add_argument("--model", type=str, required=True, help="which model to test")
parser.add_argument("--max_corr", type=int, required=False, default=3, help="max correlation")
parser.add_argument("--max_ell", type=int, required=False, default=3, help="max ell")
parser.add_argument("--n_epochs", type=int, required=False, default=600, help="epochs to train")
parser.add_argument("--n_layers", type=int, required=False, default=2, help="number of layers to train")
parser.add_argument("--n_data", type=int, required=False, default=1000, help="number of datapoints")
# parser.add_argument("--n_spoke", type=int, required=False, default=5, help="number of spokes in data")
parser.add_argument("--fold", type=int, nargs='+', help="List of integer values which is the number of spoke could be sampled in the dataset")
parser.add_argument("--cosine", action="store_true", help="Enable cosine lr decay.")
    
args = parser.parse_args()

print("PyTorch version {}".format(torch.__version__))
print("PyG version {}".format(torch_geometric.__version__))
print("e3nn version {}".format(e3nn.__version__))


def create_star_graphs(num=5, fold=[3,], dim=2, target="max", seed = 0):
    """Generate star graphs with unit-length spokes where angles are randomly assigned."""

    assert dim == 2 or dim == 3, "dimension must be 2 or 3."
    assert target in ["max", "mean"], "regression target must be 'max' or 'mean' of angles"

    torch.manual_seed(seed)
    random.seed(seed)

    dataset = []

    for _ in range(num):
        n_spoke = random.choice(fold)
        # atoms representing central and non-central nodes
        atoms = torch.LongTensor([0,] + [0,] * n_spoke)
        # edges read vertically; star graph: 0 -> 1, 0 -> 2 ...
        edge_index = torch.LongTensor( [ [0,] * n_spoke, list(range(1, n_spoke + 1)) ] )
        # origin and first spoke
        x = torch.Tensor([1, 0, 0])
        pos = [torch.Tensor([0, 0, 0]), x]  

        if dim == 2:
            for count in range(1, n_spoke):
                # random angle between 0 and 2*pi
                random_angle = random.uniform(0, 2 * math.pi)
                new_point = torch.Tensor([math.cos(random_angle), math.sin(random_angle), 0])
                pos.append(new_point)

        elif dim == 3:
            for count in range(1, n_spoke):
                theta = random.uniform(0, 2 * math.pi)  # Random angle around z-axis
                phi = random.uniform(0, math.pi)  # Random angle from z-axis (polar angle)
                new_point = torch.Tensor([
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi)
                ])
                pos.append(new_point)

        # scale the pos
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
        
        print(y[:20])

        pos = torch.stack(pos)
        data = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
        data.edge_index = to_undirected(data.edge_index)
        dataset.append(data)

    return dataset


dataset = create_star_graphs(args.n_data, args.fold, 3)

train_ratio = 0.5
val_ratio = 0.2
test_ratio = 0.3

num_samples = len(dataset)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(0)
)

batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set parameters
model_name = args.model
correlation = args.max_corr
max_ell = args.max_ell

model_dict = {
    "schnet": SchNetModel,
    "dimenet": DimeNetPPModel,
    "spherenet": SphereNetModel,
    "egnn": partial(EGNNModel, equivariant_pred=True),
    "gvp": partial(GVPGNNModel, equivariant_pred=True),
    "tfn": partial(TFNModel, max_ell=max_ell, equivariant_pred=True),
    "mace": partial(MACEModel, max_ell=max_ell, correlation=correlation, equivariant_pred=True),
}

assert(args.model in model_dict.keys())
model = model_dict[model_name](num_layers=args.n_layers, in_dim=1, out_dim=1)

# regression task
best_val_acc, test_acc, train_time, result_string = run_experiment_reg(
    model, 
    train_loader,
    val_loader, 
    test_loader,
    n_epochs=args.n_epochs,
    n_times=5,
    device=device,
    verbose=True,
    cosine=args.cosine
)


# log_data


args_dict = vars(args)
args['best_val_acc'] = best_val_acc
args['test_acc'] = test_acc
args['train_time'] = train_time
args['result_string'] = result_string

# Read existing CSV file or create a new DataFrame if the file doesn't exist
csv_file_path = 'exp_history.csv'
if os.path.isfile(csv_file_path):
    df = pd.read_csv(csv_file_path)
else:
    df = pd.DataFrame()

# Append new column with current args
df[len(df.columns)] = pd.Series(args_dict)

# Save DataFrame back to CSV
df.to_csv(csv_file_path, index=False)
