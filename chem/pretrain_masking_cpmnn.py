import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loader import MoleculeDataset, mol_to_graph_data_obj_simple

from util import ExtractSubstructureContextPair, MaskAtom
from dataloader import DataLoaderSubstructContext, DataLoaderMasking
from torch_geometric.loader import DataLoader

import argparse
from argparse import ArgumentParser, Namespace

from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.utils import create_logger
from chemprop.train import make_predictions

from chemprop.models import build_model

from chemprop.train.run_training import run_training
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.utils import create_logger
from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions


from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph

from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR

import timeit

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train(args, model_list, loader, optimizer_list, criterion, device):
    model, linear_pred_atoms  = model_list
    optimizer_model, optimizer_linear_pred_atoms = optimizer_list

    model.train()
    linear_pred_atoms.train()

    loss_accum = 0
    acc_node_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)
        batch_smile_masked = batch.smile_masked 
        _, node_rep_masked = model.encoder(batch_smile_masked)

        ## loss for nodes
        pred_node = linear_pred_atoms(node_rep_masked[batch.masked_atom_indices])
        loss = criterion(pred_node.double(), batch.mask_node_label[:,0].to(device))

        acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0].to(device))
        acc_node_accum += acc_node

        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_atoms.step()

        loss_accum += float(loss.cpu().item())
        if step == 10:
            break

    return loss_accum/step, acc_node_accum/step

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'd_new_smiles', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = 'pretrain_model_masking', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    print('Loading dataset ...')
    dataset = MoleculeDataset("data/dataset/" + args.dataset , dataset=args.dataset, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = 0.15, mask_edge=0))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    #set up model
    args_cpmnn = parse_train_args()
    modify_train_args(args_cpmnn)

    args_cpmnn.emb_dim = args.emb_dim

    args_cpmnn.dataset_type = 'classification'
    args_cpmnn.metric = 'auc'

    args_cpmnn.data_path = 'data/S_dataset_modify.csv'

    debug = print
    logger = None

    debug('Loading model for downstream task')
    args_cpmnn.task_names = get_task_names(args_cpmnn.data_path)
    data = get_data(path=args_cpmnn.data_path, args=args_cpmnn, logger=logger)
    args_cpmnn.num_tasks = data.num_tasks()
    args_cpmnn.features_size = data.features_size()
    debug(f'Number of labels in the downstream task = {args_cpmnn.num_tasks}')

    model = build_model(args_cpmnn)
    model = model.to(device)
    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    model_list = [model, linear_pred_atoms]

    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms]

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, criterion, device)
        print(train_loss, train_acc_atom, train_acc_bond)

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")

if __name__ == "__main__":
    main()