from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True, 
        help="Datasets implemented: cora, citeseer, pubmed, ogbn-arxiv, flickr, and reddit."
    )
    parser.add_argument("--reduction_rate", type=float, required=True)
    parser.add_argument("--rec_epochs", type=int, default=200)
    parser.add_argument("--gnn_epochs", type=int, default=600)
    parser.add_argument("--verbose", action="store_true", default="store_true")
    parser.add_argument("--cuda", action="store_true", default="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gnn", type=str, default="gcn")
    parser.add_argument("--bsz", type=int, default=1024)
    #parser.add_argument("--rec_bsz", type=int, default=256)
    parser.add_argument("--add_ratio", type=float, default=0.1)
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument("--lr_rec", type=float, default=0.001)
    parser.add_argument("--lr_gnn", type=float, default=0.001)
    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=0.001)
    parser.add_argument("--wd_gnn", type=float, default=0.001)
    parser.add_argument("--weighted", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--use_kmeans", type=int, default=1)
    parser.add_argument("--kmeans_redo", type=int, default=1)
    parser.add_argument("--kmeans_iter", type=int, default=20)
    parser.add_argument("--use_cached", action="store_true", default="store_true")
    
    return parser.parse_args()