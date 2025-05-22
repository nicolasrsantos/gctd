import torch
import gzip
import json
import scipy.sparse as sp
import numpy as np
import pickle as pkl
from torch_geometric.utils import (
    to_dense_adj,
    dropout_edge,
    add_random_edge,
    from_scipy_sparse_matrix,
    negative_sampling,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid, WikipediaNetwork
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch.utils.data import Dataset
from os import path as osp
from pathlib import Path
import time
from ogb.nodeproppred import PygNodePropPredDataset

class DataHandler():
    def __init__(self, dataset):
        self.dataset = dataset
        
        self.edge_index = self.dataset["edge_index"]
        self.feat_full = self.dataset["x"]
        self.x = self.dataset["x"]
        self.y_full = self.dataset["y"]
        self.nclass_full = len(torch.unique(self.y_full))
        self.adj_full = to_dense_adj(self.edge_index).squeeze()

        # train nodes
        self.train_idx = self.dataset["train_mask"]
        # this can consume a lot of memory and can be solved by using torch
        # geometric's subgraph function
        self.adj_train = self.adj_full[np.ix_(self.train_idx, self.train_idx)]
        self.feat_train = self.feat_full[self.train_idx]
        self.y_train = self.y_full[self.train_idx]
        self.nclass_train = len(set(self.y_train))

        # val nodes
        self.val_idx = self.dataset["val_mask"]
        self.adj_val = self.adj_full[np.ix_(self.val_idx, self.val_idx)]
        self.feat_val = self.feat_full[self.val_idx]
        self.y_val = self.y_full[self.val_idx]
        self.nclass_val = len(set(self.y_val))

        # test_nodes
        self.test_idx = self.dataset["test_mask"]
        self.adj_test = self.adj_full[np.ix_(self.test_idx, self.test_idx)]
        self.feat_test = self.feat_full[self.test_idx]
        self.y_test = self.y_full[self.test_idx]
        self.nclass_test = len(set(self.y_test))
        self.mask_list = self.get_mask_list()

    def get_mask_list(self):
        mask_list = torch.zeros(len(self.y_full), dtype=torch.long)
        mask_list[self.val_idx] = 1
        mask_list[self.test_idx] = 2

        return mask_list


class COODataset(Dataset):
    def __init__(self, idxs, vals):
        assert idxs.size()[1] == len(vals)
        self.idxs = idxs
        self.vals = vals
    
    def __len__(self):
        return self.idxs.size()[1]

    def __getitem__(self, idx):
        return self.idxs[0, idx], self.idxs[1, idx], self.vals[idx]


def load_graphsaint(data_dir, dataset_name):
    dataset_path = osp.join(data_dir, dataset_name)
    
    adj = sp.load_npz(dataset_path + "/adj_full.npz")
    splits = json.load(open(dataset_path + "/role.json", "r"))
    x = np.load(dataset_path + "/feats.npy")
    class_map = json.load(open(dataset_path + "/class_map.json", "r"))

    return adj, splits, x, class_map


def process_labels(class_map, num_vertices):
    """
    setup vertex property map for output classests
    code adapted from https://github.com/ChandlerBang/GCond
    """
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[int(k)] = v
    else:
        class_arr = np.zeros(num_vertices, dtype=int)
        for k, v in class_map.items():
            class_arr[int(k)] = v
        class_arr = class_arr - class_arr.min()
    return class_arr


def prepare_data(data_dir, dataset_name):
    planetoid = ["cora", "citeseer", "pubmed"]
    graphsaint = ["ogbn-arxiv", "flickr", "reddit"]

    if dataset_name.lower() in planetoid:
        dataset = Planetoid(data_dir, dataset_name.lower())[0]
        return DataHandler(dataset)
    elif dataset_name.lower() in graphsaint:
        adj, splits, x, class_map = load_graphsaint(data_dir, dataset_name)
        
        n_nodes = adj.shape[0]
        labels = process_labels(class_map, n_nodes)
        labels = torch.tensor(labels, dtype=torch.long)
        
        if dataset_name == 'ogbn-arxiv':
            adj = adj + adj.T
            adj[adj > 1] = 1
        edge_index, _ = from_scipy_sparse_matrix(adj)

        x_train = x[splits["tr"]]
        scaler = StandardScaler()
        scaler.fit(x_train)
        x = scaler.transform(x)
        x = torch.tensor(x, dtype=torch.float)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[splits["tr"]] = 1
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask[splits["va"]] = 1
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask[splits["te"]] = 1
        
        dataset = Data()
        dataset.edge_index = edge_index
        dataset.x = x
        dataset.y = labels
        dataset.train_mask = train_mask
        dataset.val_mask = val_mask
        dataset.test_mask = test_mask

        return DataHandler(dataset)
    else:
        raise Exception(
            f"dataset not supported. expected {planetoid, graphsaint},"
            f"but got {dataset_name.lower()}"
        )


def load_data(args):
    cache_path = osp.join(osp.join(args.data_dir, args.dataset), "cache/")
    Path(cache_path).mkdir(exist_ok=True, parents=True)

    dump_path = osp.join(cache_path, args.dataset + f"_{args.R}")
    multi_str = f"_{args.add_ratio}_{args.drop_ratio}.multi_view"
    vals_str = f"_{args.add_ratio}_{args.drop_ratio}.vals"
    
    data = prepare_data(args.data_dir, args.dataset)
    
    if osp.exists(dump_path + multi_str) and osp.exists(dump_path + vals_str):
        print("loading cached files")
        
        with gzip.open(dump_path + multi_str, "rb") as f:
            multi_view = pkl.load(f)
        with gzip.open(dump_path + vals_str, "rb") as f:
            vals = pkl.load(f)
    else:
        print("cached files not found. generating files from scratch.")
        if args.dataset in ["flickr", "reddit"]:
            multi_view = augment_graph_sparse(data, args, False)
        else:
            multi_view = augment_graph_sparse(data, args, True)

        vals = []
        for i in range(args.R):
            print(f"generating view {i}")
            true_vals = torch.ones(multi_view[i].shape[1])
            
            negative_edges = negative_sampling(multi_view[i], force_undirected=True)
            multi_view[i] = torch.cat((multi_view[i], negative_edges), dim=1)
            negative_vals = torch.zeros(negative_edges.size()[1])

            tmp = []
            tmp.extend(true_vals)
            tmp.extend(negative_vals)
            tmp = torch.tensor(tmp, dtype=torch.float)
            vals.append(tmp)

        with gzip.open(dump_path + multi_str, "wb") as f:
            pkl.dump(multi_view, f)
        with gzip.open(dump_path + vals_str, "wb") as f:
            pkl.dump(vals, f)

    del data.edge_index

    return data, multi_view, vals


def augment_graph_sparse(data, args, transductive):
    if transductive:
        print("transductive")
        adj = data.edge_index    
    else:
        print("inductive")
        adj = data.adj_train.argwhere().t()
    adjs = [adj]

    for _ in range(args.R - 1):
        augmented_graph = dropout_edge(adj, p=args.drop_ratio, force_undirected=True)[0]
        augmented_graph = add_random_edge(augmented_graph, p=args.add_ratio, force_undirected=True)[0]
        adjs.append(augmented_graph)

    return adjs


def load_cached_data(args):
    root_dir = "saved_ours/"
    dataset_saved_dir = osp.join(root_dir, args.dataset)

    filename = f"{dataset_saved_dir}/{args.dataset}_{args.reduction_rate}_argmax"
    if args.use_kmeans == 1:
        filename = f"{dataset_saved_dir}/{args.dataset}_{args.reduction_rate}_kmeans"
    
    cached_graphs = []
    for idx in range(5):
        cur_graph = Data()
        cur_graph.x = torch.load(f"{filename}_feats_{idx}.pt")
        cur_graph.y = torch.load(f"{filename}_y_{idx}.pt")
        cur_graph.edge_index = torch.load(f"{filename}_edge_index_{idx}.pt")
        cur_graph.edge_weight = torch.load(f"{filename}_edge_weight_{idx}.pt")
        cur_graph.train_mask = torch.load(f"{filename}_train_mask_{idx}.pt")
        cur_graph.val_mask = torch.load(f"{filename}_val_mask_{idx}.pt")
        cur_graph.test_mask = torch.load(f"{filename}_test_mask_{idx}.pt")
        cached_graphs.append(cur_graph)

    return cached_graphs