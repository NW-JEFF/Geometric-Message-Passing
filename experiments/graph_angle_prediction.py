import sys
sys.path.append('/mnt/workspace/linchen/nanxiang/Geometric-Message-Passing')
import os
import json
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

from experiments.utils.train_utils import run_experiment_reg
from models import SchNetModel, DimeNetPPModel, SphereNetModel, EGNNModel, GVPGNNModel, TFNModel, MACEModel
from models.vtvnn import VTVNNModel
from experiments.utils.create_graphs import create_star_graphs, create_paired_star_graphs, \
                                            create_paired_star_graphs_with_two_centers, create_paired_complete_graphs, \
                                            create_paired_complete_graphs_with_full_centers, create_paired_radius_graphs_with_full_centers




# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parse args
parser = argparse.ArgumentParser(description="Start testing geometric bottleneck in graph networks!")
# general parameters
parser.add_argument("--model", type=str, required=True, help="which model to test")
parser.add_argument("--dataset", type=str, required=True, help="which type of dataset to use")
parser.add_argument("--pool", type=str, default="mean", help="type of pooling layers")
parser.add_argument("--max_corr", type=int, required=False, default=3, help="max correlation")
parser.add_argument("--max_ell", type=int, required=False, default=3, help="max ell")
parser.add_argument("--dim", type=int, required=False, default=3, help="dimension of angles")
parser.add_argument("--n_epochs", type=int, required=False, default=600, help="epochs to train")
parser.add_argument("--n_layers", type=int, required=False, default=2, help="number of layers to train")
parser.add_argument("--n_data", type=int, required=False, default=1000, help="number of datapoints to be generated")
parser.add_argument("--lr", type=float, required=False, default=1e-4, help="the initial learning rate")
parser.add_argument("--cosine", action="store_true", help="enable cosine learning rate decay")
parser.add_argument("--equivariant", action="store_true", help="equivariant prediction or not")
# needed for star graphs
parser.add_argument("--fold", type=int, nargs='+', help="list of numbers of spokes that could occur in star graph datasets")
# needed for complete and radius graphs
parser.add_argument("--n_nodes", type=int, nargs='+', help="list of numbers of nodes that could occur in complete graph datasets")
# needed for graphs whose regression targets are angles of specified pairs
parser.add_argument("--n_pairs", type=int, help="number of pairs of nodes to be considered when computing target angles")
# relavant for star graphs with two centers
parser.add_argument("--single_center", action="store_true", help="only use a single center when the model is paired_star2")
parser.add_argument("--loss_mask", action="store_true", help="only compute loss with respect to part of the predictions")
# relavant for radius graphs
parser.add_argument("--connection", type=str, default="min", help="the level of connection to establish when constructing radius graphs")
# relavant for VTVNN
parser.add_argument("--dense", type=bool, default="True", help="whether or not use dense layers in VTVNN") 

args = parser.parse_args()

print("PyTorch version {}".format(torch.__version__))
print("PyG version {}".format(torch_geometric.__version__))
print("e3nn version {}".format(e3nn.__version__))




# Model setup
model_dict = {
    "schnet": partial(SchNetModel, pool=args.pool),
    "dimenet": partial(DimeNetPPModel, pool=args.pool),
    "spherenet": partial(SphereNetModel, pool=args.pool),
    "egnn": partial(EGNNModel, equivariant_pred=args.equivariant, pool=args.pool),
    "gvp": partial(GVPGNNModel, equivariant_pred=args.equivariant, pool=args.pool),
    "tfn": partial(TFNModel, max_ell=args.max_ell, equivariant_pred=args.equivariant, pool=args.pool),
    "mace": partial(MACEModel, max_ell=args.max_ell, correlation=args.max_corr, equivariant_pred=args.equivariant, pool=args.pool),
    "vtvnn": partial(VTVNNModel, dense=args.dense, pool=args.pool)
}

assert(args.model in model_dict.keys())
model_func = model_dict[args.model]




# Generate graph dataset
dataset_dict = {
    "star": create_star_graphs,
    "paired_star": create_paired_star_graphs,
    "paired_star2": create_paired_star_graphs_with_two_centers,
    "paired_complete": create_paired_complete_graphs,
    "complete_full": create_paired_complete_graphs_with_full_centers,
    "radius_full": create_paired_radius_graphs_with_full_centers
}

assert(args.dataset in dataset_dict.keys())
dataset_func = dataset_dict[args.dataset]

# in_dim: number of distinct atom types of the generated dataset.
# out_dim: output dimension of the model; need to choose the pooling layer properly to match the dimension of true labels in train_utils.train_reg;

if args.dataset == "star":
    assert args.fold is not None
    dataset = dataset_func(num=args.n_data, fold=args.fold, dim=args.dim, target="max")
    model_args = {'num_layers' : args.n_layers, 'in_dim' : 1, 'out_dim' : 1}

elif args.dataset == "paired_star":
    assert args.n_pairs is not None
    assert args.fold is not None
    dataset = dataset_func(num=args.n_data, fold=args.fold, dim=args.dim, n_pairs=args.n_pairs)
    model_args = {'num_layers' : args.n_layers, 'in_dim' : args.n_pairs + 2, 'out_dim' : args.n_pairs}

elif args.dataset == "paired_star2":
    # if pool is not first_and_last, then out_dim should be args.n_pairs * 2 to match dimension of true labels, 
    # but that is not sensible because true labels come from two nodes in each graph and each node feature has dim=n_pairs,
    # so we should adhere to pool = first_and_last in this case; if single_center==True, then should use pool = first
    assert args.pool == "first_and_last" or args.pool == "first"
    assert args.n_pairs is not None
    assert args.fold is not None
    dataset = dataset_func(num=args.n_data, fold=args.fold, dim=args.dim, n_pairs=args.n_pairs, single_center=args.single_center)
    model_args = {'num_layers' : args.n_layers, 'in_dim' : args.n_pairs + 2, 'out_dim' : args.n_pairs}

elif args.dataset == "paired_complete":
    assert args.n_pairs is not None
    assert args.n_nodes is not None
    dataset = dataset_func(num=args.n_data, n_nodes=args.n_nodes, dim=args.dim, n_pairs=args.n_pairs)
    model_args = {'num_layers' : args.n_layers, 'in_dim' : args.n_pairs + 2, 'out_dim' : args.n_pairs}

elif args.dataset == "complete_full":
    assert args.pool == "none"
    assert args.n_pairs is not None
    assert args.n_nodes is not None
    dataset = dataset_func(num=args.n_data, n_nodes=args.n_nodes, dim=args.dim, n_pairs=args.n_pairs)
    model_args = {'num_layers' : args.n_layers, 'in_dim' : args.n_pairs + 1, 'out_dim' : args.n_pairs}

elif args.dataset == "radius_full":
    assert args.pool == "none"
    assert args.n_pairs is not None
    assert args.n_nodes is not None
    dataset = dataset_func(num=args.n_data, n_nodes=args.n_nodes, dim=args.dim, connection=args.connection, n_pairs=args.n_pairs)
    model_args = {'num_layers' : args.n_layers, 'in_dim' : args.n_pairs + 1, 'out_dim' : args.n_pairs}




# Training setup
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




# Regression task
loss_mask = False
if args.loss_mask:
    if args.dataset == "paired_star2":
        loss_mask = True
    else:
        print(f"loss mask is not supported for dataset: {args.dataset}.")

best_val_acc, test_acc, train_time, mean, std = run_experiment_reg(
    model_func,
    model_args, 
    train_loader,
    val_loader, 
    test_loader,
    n_epochs=args.n_epochs,
    n_times=1,
    device=device,
    verbose=True,
    cosine=args.cosine,
    lr=args.lr,
    loss_mask=loss_mask
)




# Log data
args_dict = vars(args)
args_dict['best_val_acc'] = best_val_acc
args_dict['test_acc'] = test_acc
args_dict['train_time'] = train_time
args_dict['mean'] = mean
args_dict['std'] = std

# File path to save the results
results_file_path = 'exp_history.json'

# Read existing JSON file or create a new list if the file doesn't exist
if os.path.isfile(results_file_path):
    with open(results_file_path, 'r') as file:
        results_list = json.load(file)
else:
    results_list = []

# Append new args_dict to the results list
results_list.append(args_dict)

# Save the results list back to the JSON file
with open(results_file_path, 'w') as file:
    json.dump(results_list, file, indent=4)
