import numpy as np
import torch
from torch.nn.init import uniform_, kaiming_uniform_
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from collections import defaultdict, OrderedDict
import tensorly as tl
import faiss
from torch_geometric.utils import is_undirected, to_undirected
import wandb
from dotmap import DotMap
from collections import Counter

class GCTD(torch.nn.Module):
    def __init__(self, data, device, args):
        super(GCTD, self).__init__()
        self.data = data
        self.N = args.N # real graph adj is NxN
        self.N_prime = int(self.N * args.reduction_rate) # syn graph adj is N'xN'
        self.R = args.R # number of augmentations of adj + 1
        self.args = args
        self.device = device
        self.factor, self.core = self.init_parameters()

    def forward(self, slice_idx, batch):
        A = self.factor.weight # N x N_prime
        G = self.core[slice_idx].weight # N_prime x N_prime
        A = F.relu(A)            
        G = F.relu(G)                

        A_indexed = A[batch[0]].reshape(-1, 1, self.N_prime) # batch x 1 x N_prime
        A_indexed_transpose = A[batch[1]].reshape(-1, self.N_prime, 1) # batch x N_prime x 1

        rec = torch.matmul(A_indexed, G)
        rec = torch.matmul(rec, A_indexed_transpose)

        return rec.reshape(-1)


    def init_parameters(self):
        factor = torch.nn.Embedding(self.N, self.N_prime, padding_idx=0)
        core = nn.ModuleList([
            nn.Embedding(self.N_prime, self.N_prime) for _ in range(self.R)
        ])

        uniform_(factor.weight.data)
        for core_slice in core:
            uniform_(core_slice.weight.data) 
        
        factor.weight.data = tl.abs(factor.weight.data)
        for i in range(self.R):
            core[i].weight.data = tl.abs(core[i].weight.data)
            
        return factor, core


    def to_edge_index(self):
        slices = torch.tensor(
            np.array([g.weight.detach().cpu().numpy() for g in self.core])
        )

        new_adj = torch.mean(slices, dim=0, dtype=torch.float)
        assert new_adj.shape[0] == len(self.core[0].weight[0])
        assert new_adj.shape[1] == len(self.core[0].weight[0])

        edge_index = new_adj >= 0.05
        edge_index = edge_index.argwhere().t()
        edge_weight = new_adj[edge_index[0], edge_index[1]]
        assert edge_index[0].shape == edge_weight.shape
        assert edge_index[0].shape == edge_index[1].shape

        if not is_undirected(edge_index):
            edge_index, edge_weight = to_undirected(edge_index, edge_weight)

        self.synthetic_edges = edge_index.size()[1]
        wandb.log({"edge_index_size": edge_index.size()})

        return edge_index, edge_weight


    def class_split_distribution(self):
        y_count = torch.bincount(self.data.y_full)
        original_dist = y_count/y_count.sum()
        supernode_dist = torch.round(self.N_prime * original_dist)

        train_total = self.data.train_idx.sum()
        val_total = self.data.val_idx.sum()
        test_total = self.data.test_idx.sum()

        train_reduced_dist = self.N_prime * train_total // self.N
        val_reduced_dist = self.N_prime * val_total // self.N
        test_reduced_dist = self.N_prime * test_total // self.N

        if train_reduced_dist == 0:
            train_reduced_dist += 1
        if val_reduced_dist == 0:
            val_reduced_dist += 1
        if test_reduced_dist == 0:
            test_reduced_dist += 1

        graphsaint = ["reddit", "flickr"]
        if self.args.dataset in graphsaint:
            return supernode_dist, [train_reduced_dist, 0, 0]    
        return supernode_dist, [train_reduced_dist, val_reduced_dist, test_reduced_dist]


    def compute_supernode_info(self, candidate_nodes, supernode_y_count, supernode_mask_count):
        supernode_dist, supernode_mask_dist = self.class_split_distribution()

        if isinstance(candidate_nodes, tuple):
            col_masks = self.data.mask_list[candidate_nodes[0]]
        elif isinstance(candidate_nodes, torch.Tensor):
            col_masks = self.data.mask_list[candidate_nodes]
        
        mask_count = col_masks.bincount()
        count_nonzero = mask_count.nonzero().squeeze()
        sorted_masks = mask_count[count_nonzero].argsort()
        mask = None
        if count_nonzero.dim() == 0:
            mask = count_nonzero
        else:
            mask_candidates = count_nonzero[sorted_masks]
            for mask_candidate in mask_candidates:
                if supernode_mask_count[mask_candidate.item()] < supernode_mask_dist[mask_candidate.item()]:
                    mask = mask_candidate.item()
                    supernode_mask_count[mask_candidate.item()] += 1
                    break
            if mask is None:
                rand_idx = np.random.randint(0, len(mask_candidates))
                mask = mask_candidates[rand_idx]
        chosen_mask_idx = torch.argwhere(col_masks == mask).squeeze()
        y_chosen_mask = self.data.y_full[chosen_mask_idx]

        chosen_y = None
        if y_chosen_mask.numel() > 1:
            y_chosen_maskcount = y_chosen_mask.bincount()
            nz = y_chosen_maskcount.nonzero().squeeze()
            sorted_classes = torch.argsort(y_chosen_maskcount[nz])
            if nz.dim() == 0:
                chosen_y = nz
            else:
                y_candidates = nz[sorted_classes]

                for y_candidate in y_candidates:
                    if supernode_y_count[y_candidate.item()] < supernode_dist[y_candidate.item()]:
                        chosen_y = y_candidate.item()
                        supernode_y_count[y_candidate.item()] += 1
                        break
                if chosen_y is None:
                    rand_y_idx = np.random.randint(0, len(y_candidates))
                    chosen_y = y_candidates[rand_y_idx]
        elif y_chosen_mask.numel() == 1:
            # since there's only one class among the elements that belong to
            # the same class, we assign it directly
            chosen_y = y_chosen_mask
        else:
            raise Exception(f"there is no class to assign to supernode {i}")

        merged_nodes = torch.argwhere(
            (self.data.y_full == chosen_y) & (self.data.mask_list == mask)
        ).squeeze()

        if len(merged_nodes) == 0:
            raise Exception(f"no merged nodes for supernode {i}")

        col_embs = self.data.feat_full[merged_nodes]
        avg_emb = torch.mean(col_embs, axis=0).squeeze()
        
        return chosen_y, mask, avg_emb


    def compute_assignments(self):
        supernode_x, supernode_y, supernode_masks = [], [], []
        supernode_y_count, supernode_mask_count = defaultdict(int), defaultdict(int)

        if self.args.use_kmeans == 0:
            for i in range(len(self.factor.weight[0])):
                col = self.factor.weight[:, i].cpu().detach()
                candidate_nodes = torch.where(col > 0)
                if candidate_nodes[0].numel() == 0:
                    print(f"no available nodes to merge for supernode {i}")
                    continue
                
                y, mask, emb = self.compute_supernode_info(
                    candidate_nodes,
                    supernode_y_count,
                    supernode_mask_count
                )

                supernode_y.append(y)
                supernode_masks.append(mask)
                supernode_x.append(emb)
        else:
            kmeans = faiss.Kmeans(
                self.N_prime,
                self.N_prime,
                niter=self.args.kmeans_iter,
                nredo=self.args.kmeans_redo,
                verbose=False,
                seed=42,
                min_points_per_centroid=1,
                max_points_per_centroid=self.N,
                gpu=False # change this to gpu id to run KMeans on GPU
            )
            kmeans.train(self.factor.weight.detach().cpu())
            _, membership = kmeans.index.search(self.factor.weight.detach().cpu(), 1)

            supernodes = defaultdict(list)
            for idx, assignment in enumerate(membership):
                supernode_idx = assignment[0]
                supernodes[supernode_idx].append(idx)
            
            supernodes = OrderedDict(sorted(supernodes.items()))
            for candidate_nodes in supernodes.values():
                if len(candidate_nodes) == 0:
                    print(f"no available nodes to merge for supernode {i}")
                    continue
                
                candidate_nodes = torch.tensor(candidate_nodes, dtype=torch.long)
                y, mask, emb = self.compute_supernode_info(
                    candidate_nodes,
                    supernode_y_count,
                    supernode_mask_count
                )

                supernode_y.append(y)
                supernode_masks.append(mask)
                supernode_x.append(emb)

        return np.array(supernode_x), np.array(supernode_y), np.array(supernode_masks)


    def to_data_obj(self):
        self.factor.weight.data = F.relu(self.factor.weight.data)
        for i in range(self.R):
            self.core[i].weight.data = F.relu(self.core[i].weight.data)

        edge_index, edge_weight = self.to_edge_index()
        feats, y, masks = self.compute_assignments()
        feats = torch.tensor(feats, dtype=torch.float)
            
        data = Data()
        data.x = feats
        data.y = torch.tensor(y, dtype=torch.long)
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        n_nodes = len(data.y)
        data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_idx = np.argwhere(masks == 0)
        data.train_mask[train_idx.squeeze()] = 1

        data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_idx = np.argwhere(masks == 1)
        data.val_mask[val_idx.squeeze()] = 1

        data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_idx = np.argwhere(masks == 2)
        data.test_mask[test_idx.squeeze()] = 1

        existing_edges = data.edge_index.size()[1]
        possible_edges = n_nodes * n_nodes
        wandb.log({
            "n_nodes": n_nodes,
            "possible edges": possible_edges,
            "existing edges": existing_edges,
            "graph density": existing_edges / possible_edges,
        })

        return data


    def get_loaders(self, data):
        # we train the gnn with the reduced training nodes, but we validate
        # and test using the original validation and test nodes
        train_bsz = sum(data.train_mask).item()

        if self.args.dataset in ["reddit"]:
            # if your server has enough memory to load the full val and test sets
            # just removed this condition and use the code inside the else block
            val_bsz = test_bsz = 64
        else:
            val_bsz = sum(self.data.dataset["val_mask"]).item()
            test_bsz = sum(self.data.dataset["test_mask"]).item()
        print(f"batch sizes {train_bsz},{val_bsz},{test_bsz}")

        train_loader = NeighborLoader(
            data,
            num_neighbors=[-1] * 2,
            batch_size=train_bsz,
            input_nodes=data.train_mask # reduced training nodes
        )
        val_loader = NeighborLoader(
            self.data.dataset,
            num_neighbors=[-1] * 2,
            batch_size=val_bsz,
            input_nodes=self.data.dataset["val_mask"] # original val nodes
        )
        test_loader = NeighborLoader(
            self.data.dataset,
            num_neighbors=[-1] * 2,
            batch_size=test_bsz,
            input_nodes=self.data.dataset["test_mask"] # original test nodes
        )

        loaders = DotMap()
        loaders.train_loader = train_loader
        loaders.val_loader = val_loader
        loaders.test_loader = test_loader
        
        return loaders
