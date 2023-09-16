import sys
sys.path.append('../')
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


#from experiments.utils.plot_utils import plot_2d, plot_3d
from experiments.utils.train_utils import run_experiment_reg
from models import SchNetModel, DimeNetPPModel, SphereNetModel, EGNNModel, GVPGNNModel, TFNModel, MACEModel

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

