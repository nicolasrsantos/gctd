# Multi-view Graph Condensation via Tensor Decomposition

This repository is the official PyTorch implementation of **"Multi-view Graph Condensation via Tensor Decomposition"**, published in the **Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining (WSDM 2026)**.

Paper Link: https://dl.acm.org/doi/10.1145/3773966.3777968  

## Requirements

This code was implemented using Python 3.11.5 and the following packages:

- `dotmap==1.3.30`
- `faiss==1.9.0`
- `gdown==5.2.0`
- `ogb==1.3.6`
- `scikit-learn==1.6.1`
- `scipy==1.13.1`
- `tensorly==0.9.0`
- `torch==2.1.2`
- `torch-geometric==2.6.1`
- `wandb==0.19.6`

## Datasets

The following datasets are supported:

- `cora`
- `citeseer`
- `pubmed`
- `ogbn-arxiv`
- `flickr`
- `reddit`

Cora, Citeseer, Pubmed, and Ogbn-arxiv are automatically downloaded when running the code.

For Flickr and Reddit, we use the versions provided by GraphSAINT. They are available at:

- https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz
- https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg with code `f1ao`

After downloading Flickr and Reddit, place them in the appropriate dataset directory used by the repository.

## How to run the code

The main script is:

```bash
python src/train.py --dataset <dataset_name> --reduction_rate <reduction_rate>
```

For example, to run GCTD on Cora with a 1.3% condensation ratio:

```bash
python src/train.py --dataset cora --reduction_rate 0.013
```

You can also manually set the main hyperparameters:

```bash
python src/train.py \
  --dataset cora \
  --reduction_rate 0.013 \
  --R 5 \
  --add_ratio 0.1 \
  --drop_ratio 0.1 \
  --rec_epochs 200 \
  --gnn_epochs 600 \
  --hidden_dim 256 \
  --gnn gcn \
  --lr_rec 0.001 \
  --lr_gnn 0.001 \
  --wd 0.001 \
  --wd_gnn 0.001 \
  --bsz 1024 \
  --use_kmeans 1 \
  --cuda
```

## Hyperparameter search with Weights & Biases

We used Weights & Biases (`wandb`) to perform hyperparameter search. An example sweep configuration is provided in:

```bash
wandb_example.yml
```

To launch a sweep, run:

```bash
wandb sweep wandb_example.yml
```

Then start an agent using the command printed by W&B, for example:

```bash
wandb agent <entity>/<project>/<sweep_id>
```

The user can either use the provided sweep file, create their own sweep file, or manually set any parameter through the command line.

## Main arguments

The following arguments can be modified when running `src/train.py`.

### Dataset and reduction

- `--dataset`

  Dataset used in the experiment. Available options are `cora`, `citeseer`, `pubmed`, `ogbn-arxiv`, `flickr`, and `reddit`.

- `--reduction_rate`

  Condensation ratio used to define the size of the synthetic graph.

- `--rec_epochs`

  Number of epochs used to optimize the tensor decomposition.

  Default: `200`

- `--R`

  Number of graph views/tensor slices. When `R=1`, the method reduces to the single-view matrix tri-factorization setting.

  Default: `5`

- `--add_ratio`

  Percentage of random edges added when generating augmented graph views.

  Default: `0.1`

- `--drop_ratio`

  Percentage of edges randomly removed when generating augmented graph views.

  Default: `0.1`

- `--lr_rec`

  Learning rate used during the reconstruction/decomposition step.

  Default: `0.001`

- `--wd`

  Weight decay used during the reconstruction/decomposition step.

  Default: `0.001`

- `--bsz`

  Batch size used during the decomposition step.

  Default: `1024`

- `--atol`

  Absolute tolerance used as a convergence criterion for the reconstruction loss.

  Default: `1e-7`

- `--weighted`

  Determines whether the values from the learned core tensor are used as edge weights in the condensed graph.

  Default: `1`

- `--use_kmeans`

  Uses K-Means on the learned factor matrix to assign original nodes to synthetic nodes.

  Default: `1`

- `--kmeans_redo`

  Number of K-Means restarts.

  Default: `1`

- `--kmeans_iter`

  Number of K-Means iterations.

  Default: `20`

### GNN evaluation parameters

- `--gnn_epochs`

  Number of epochs used to train the GNN on the condensed graph.

  Default: `600`

- `--gnn`

  GNN architecture used for evaluation.

  Default: `gcn`

- `--hidden_dim`

  Hidden dimension of the GNN.

  Default: `256`

- `--dropout`

  Dropout used during GNN training.

  Default: `0.0`

- `--lr_gnn`

  Learning rate used during GNN training.

  Default: `0.001`

- `--wd_gnn`

  Weight decay used during GNN training.

  Default: `0.001`

### Runtime options

- `--cuda`

  Enables GPU training when available.

  Default: enabled

- `--gpu_id`

  GPU ID used for training.

  Default: `0`

- `--verbose`

  Prints additional information during execution.

  Default: enabled

- `--use_cached`

  Uses cached files (e.g., multi-view tensors) when available.

  Default: enabled

## Reference

If you use this code, please cite our paper:

```bibtex
@inproceedings{10.1145/3773966.3777968,
author = {Roque dos Santos, Nicolas and Ahn, Dawon and Minatel, Diego and de Andrade Lopes, Alneu and Papalexakis, Evangelos},
title = {Multi-view Graph Condensation via Tensor Decomposition},
year = {2026},
isbn = {9798400722929},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3773966.3777968},
doi = {10.1145/3773966.3777968},
booktitle = {Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
pages = {563–573},
numpages = {11},
keywords = {graph condensation, tensor decomposition, graph neural networks, matrix tri-factorization},
location = {USA},
series = {WSDM '26}
}
```
