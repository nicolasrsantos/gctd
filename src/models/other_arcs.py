import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GCNConv,
    Linear,
    SAGEConv,
    APPNP,
    ChebConv,
    SGConv
)

class MyGCN(torch.nn.Module):
    def __init__(self, x_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(x_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
            
    def forward(self, x, edge_index, edge_weight, args):
        if args.weighted == 1:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = self.conv2(x, edge_index)

        return x
    

class MyGraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, args):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight, args):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=args.dropout, training=self.training) # we disable dropout during evaluation
        x = self.conv2(x, edge_index)

        return x


class MyAPPNP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super(MyAPPNP, self).__init__()
        self.conv1 = Linear(in_channels, hidden_channels)
        self.conv2 = Linear(hidden_channels, out_channels)
        self.appnp = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index, edge_weight, args):
        if args.weighted == 1:
            x = self.conv1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=args.dropout) # we disable dropout during evaluation
            x = self.conv2(x)
            x = self.appnp(x, edge_index, edge_weight)
        else:
            x = self.conv1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=args.dropout) # we disable dropout during evaluation
            x = self.conv2(x)
            x = self.appnp(x, edge_index)
        
        return x
    

class MyChebyNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K):
        super(MyChebyNet, self).__init__()
        self.conv1 = ChebConv(in_channels=in_channels, out_channels=hidden_channels, K=K)  # K is the chebyshev filter size
        self.conv2 = ChebConv(in_channels=hidden_channels, out_channels=out_channels, K=K)
    
    def forward(self, x, edge_index, edge_weight, args):
        if args.weighted == 1:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=args.dropout) # we disable dropout during evaluation
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=args.dropout) # we disable dropout during evaluation
            x = self.conv2(x, edge_index)

        return x
    

class MySGC(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(MySGC, self).__init__()
        self.conv = SGConv(in_channels=in_channels, out_channels=out_channels, K=K) 

    def forward(self, x, edge_index, edge_weight, args):
        if args.weighted == 1:
            x = self.conv(x, edge_index, edge_weight)
        else:
            x = self.conv(x, edge_index)
        return x