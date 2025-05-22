
import numpy as np
import torch
import time
import wandb
import gc
import os

from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch.nn import functional as F

from utils.data_handling import *
from utils.utils import set_seed, get_gnn, load_configs, get_loaders
from utils.args import get_args
from models.gctd import GCTD
from utils.losses import squared_diff, tensor_squared

os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "10000"

def train(train_loader, gnn, gnn_opt, device, args):
    gnn = gnn.to(device)
    gnn.train()

    y_pred, y_true = [], []
    total_examples = total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        batch_size = batch.batch_size

        out = gnn(batch.x, batch.edge_index, batch.edge_weight, args)[:batch_size]
        
        loss = F.cross_entropy(out, batch.y[:batch_size])
        gnn_opt.zero_grad()
        loss.backward()
        gnn_opt.step()

        preds = F.softmax(out, dim=-1)
        preds = preds.argmax(dim=-1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(batch.y[:batch_size].cpu().numpy())

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        total_loss = total_loss / total_examples
    train_acc = accuracy_score(y_true, y_pred)

    return total_loss, train_acc


@torch.no_grad()
def eval(loader, gnn, device, args):
    gnn = gnn.to(device)
    gnn.eval()

    y_pred, y_true = [], []
    total_examples = total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.batch_size
        
        out = gnn(batch.x, batch.edge_index, batch.edge_weight, args)[:batch_size]
        loss = F.cross_entropy(out, batch.y[:batch_size])
        preds = F.softmax(out, dim=-1)
        preds = preds.argmax(dim=-1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(batch.y[:batch_size].cpu().numpy())

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        total_loss = total_loss / total_examples
    eval_acc = accuracy_score(y_true, y_pred)

    return total_loss, eval_acc


def gnn_training(loaders, gnn, device, args):
    gnn_opt = torch.optim.Adam(
        list(gnn.parameters()),
        lr=args.lr_gnn,
        weight_decay=args.wd_gnn,
    )

    best_model = deepcopy(gnn.state_dict())
    best_loss = 10**10

    for epoch in range(1, args.gnn_epochs + 1):
        gnn_loss, acc = train(loaders.train_loader, gnn, gnn_opt, device, args)
        val_loss, val_acc = eval(loaders.val_loader, gnn, device, args)
        
        print(
            f"epoch {epoch} | gnn loss {gnn_loss:.6f} | acc {acc*100:.2f} | "
            f"val loss {val_loss:.6f} | val_acc {val_acc*100:.2f}"
        )
        
        wandb.log({
            "loss": gnn_loss,
            "train_acc": acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_loss:
            best_loss = val_loss
            best_val_acc = val_acc
            best_model = deepcopy(gnn.state_dict())

    del loaders.train_loader
    del loaders.val_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    # testing trained on original test set
    gnn.load_state_dict(best_model)
    test_loss, test_acc = eval(loaders.test_loader, gnn, device, args)
    
    return test_loss, test_acc, best_val_acc


def condensation(rec_loaders, device, gctd, norm_factor, args):
    loss_fn = torch.nn.MSELoss()

    rec_opt = torch.optim.Adam(
        list(gctd.parameters()),
        lr=args.lr_rec,
        weight_decay=args.wd,
    )

    stop = 0
    best_model = deepcopy(gctd.state_dict())
    best_rec_err = prev_rec_err = 10**10

    elapsed_times = []
    for epoch in range(1, args.rec_epochs + 1):
        epoch_rec_err = 0.0
        start_time = time.time_ns()
        for slice_idx, loader in enumerate(rec_loaders):
            for batch in loader:
                batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                vals = batch[2].to(device)

                reconstructed_tensor = gctd.forward(slice_idx, batch)
                rec_err = loss_fn(reconstructed_tensor, vals) / norm_factor
                epoch_rec_err += rec_err

                rec_opt.zero_grad()
                rec_err.backward()
                rec_opt.step()
        elapsed = time.time_ns() - start_time
        elapsed_times.append(elapsed)

        print(f'epoch: {epoch}, rec error: {epoch_rec_err:.15f}')
        wandb.log({
            "epoch rec err": epoch_rec_err,
        })

        if epoch_rec_err < best_rec_err:
            stop = 0
            best_model = deepcopy(gctd.state_dict())
            best_rec_err = epoch_rec_err
        else:
            stop += 1

        abs_diff = (abs(prev_rec_err-epoch_rec_err) < args.atol)
        if stop == 10 or abs_diff:
            print(f"rec early stopping epoch {epoch}")
            break
        
        prev_rec_err = epoch_rec_err

    # supernode assignments and dataloader preparation
    gctd.load_state_dict(best_model)
    reduced_data = gctd.to_data_obj()
    loaders = gctd.get_loaders(reduced_data)

    condensation_time = sum(elapsed_times)/1e9
    print(f"condensation time (s): {condensation_time}")
    wandb.log({
        "best_log_err": best_rec_err
    })

    return reduced_data, loaders, condensation_time

def experiment(rec_loaders, data, gnn, device, norm_factor, args):
    # condensation
    print("condensing graph")
    gctd = GCTD(data, device, args)
    gctd.train()
    gctd = gctd.to(device)
    _, loaders, condensation_time = condensation(rec_loaders, device, gctd, norm_factor, args)

    # gnn training
    print()
    print("training gnn")
    test_loss, test_acc, best_val_acc = gnn_training(loaders, gnn, device, args)
    print(f"test loss {test_loss} | test acc{test_acc*100:.2f}")

    return test_acc, best_val_acc, condensation_time

    
def run(args):
    data, multi_view, vals = load_data(args) 
    gnn = get_gnn(args)
    rec_loaders = get_loaders(multi_view, vals, args)

    wandb.login()
    wandb.init(config={"args": args})

    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        if args.gpu_id is None:
            args.gpu_id = 0
        device = torch.device(f"cuda:{args.gpu_id}")
    
    norm_factor = tensor_squared(vals)
    test_acc, best_val_acc, condensation_time = experiment(
        rec_loaders, data, gnn, device, norm_factor, args
    )
        
    wandb.log({
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "condensation_time": condensation_time
    })
    wandb.finish()

if __name__ == "__main__":
    set_seed(42)
    args = get_args()
    args = load_configs(args)
    assert 0 < args.reduction_rate <= 1, "reduction rate must be 0 < reduction rate <= 1."
    print(args)

    run(args)
