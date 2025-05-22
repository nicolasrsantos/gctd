import torch
import numpy as np
import random
import tensorly as tl
from pathlib import Path
from torch.utils.data import DataLoader
from os import path as osp

from models.other_arcs import *
from .data_handling import COODataset

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    tl.check_random_state(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_configs(args):
    if args.dataset.lower() == "cora":
        args.x_dim, args.out_dim, args.N = 1433, 7, 2708
    elif args.dataset.lower() == "citeseer":
        args.x_dim, args.out_dim, args.N = 3703, 6, 3327
    elif args.dataset.lower() == "pubmed":
        args.x_dim, args.out_dim, args.N = 500, 3, 19717
    elif args.dataset.lower() == "ogbn-arxiv":
        args.x_dim, args.out_dim, args.N = 128, 40, 169343 
    elif args.dataset.lower() == "flickr":
        args.x_dim, args.out_dim, args.N = 500, 7, 44625
    elif args.dataset.lower() == "reddit":
        args.x_dim, args.out_dim, args.N = 602, 210, 153932
    else:
        raise ValueError("dataset not supported")

    return args


def save_graph(graph, args):
    root_dir = "saved_ours/"
    dataset_saved_dir = osp.join(root_dir, args.dataset)
    
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    Path(dataset_saved_dir).mkdir(parents=True, exist_ok=True)
    
    filename = f"{dataset_saved_dir}/{args.dataset}_{args.reduction_rate}_R{args.R}"
    if args.use_kmeans == 1:
        filename = f"{dataset_saved_dir}/{args.dataset}_{args.reduction_rate}_R{args.R}"

    edge_index = graph.edge_index.detach().cpu().numpy()
    y = graph.y.detach().cpu().numpy()
    x = graph.x.detach().cpu().numpy()

    torch.save(y, f"{filename}_y.pt")
    torch.save(edge_index, f"{filename}_edge_index.pt")
    torch.save(x, f"{filename}_x.pt")


def get_gnn(args):
    if args.gnn.lower() == "gcn":
        gnn = MyGCN(args.x_dim, args.hidden_dim, args.out_dim)
    elif args.gnn.lower() == "sgc":
        gnn = MySGC(args.x_dim, args.out_dim, 2)
    elif args.gnn.lower() == "appnp":
        gnn = MyAPPNP(args.x_dim, args.hidden_dim, args.out_dim, K=2)
    elif args.gnn.lower() == "sage":
        gnn = MyGraphSAGE(args.x_dim, args.out_dim, args.hidden_dim, args)
    elif args.gnn.lower() == "cheby":
        gnn = MyChebyNet(args.x_dim, args.hidden_dim, args.out_dim, K=2)

    return gnn


def get_loaders(multi_view, vals, args):
    rec_loaders = []
    for i in range(len(multi_view)):
        tmp = COODataset(multi_view[i], vals[i])
        loader = DataLoader(tmp, batch_size=args.bsz, num_workers=16, shuffle=True)
        rec_loaders.append(loader)

    return rec_loaders
