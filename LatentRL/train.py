import os
import json
import math
import argparse
import datetime
import warnings
from typing import Tuple
from rdkit import RDLogger

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from model.dgvae import DGVAE
from model.beta_scheduler import BetaScheduler
from data import CustomDataset, ShuffledIterableDataset
from utils import seed_everything
from tools.script.generate import generate

warnings.filterwarnings("ignore", category=UserWarning)
RDLogger.DisableLog("rdApp.*")

def build_datasets(data_dir: str) -> Tuple[list, list, dict, dict, dict]:
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    vocab_cache_path = os.path.join(data_dir, "vocab_cache.json")

    train_set = ShuffledIterableDataset(path=train_path, vocab_cache_path=vocab_cache_path)
    test_set = CustomDataset(path=test_path, vocab_cache_path=vocab_cache_path)
    edge_vocab, node_vocab, smiles_vocab, max_len = (
        train_set._edge_vocab,
        train_set._node_vocab,
        train_set._smiles_vocab,
        train_set._max_len
    )
    return train_set, test_set, edge_vocab, node_vocab, smiles_vocab, max_len

def make_loaders(train_set, test_set, args):
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    return train_loader, test_loader

def get_graph_data(data): 
    x = data.x.to("cuda", non_blocking=True)
    edge_index = data.edge_index.to("cuda", non_blocking=True)
    edge_attr = data.edge_attr.to("cuda", non_blocking=True)
    batch = data.batch.to("cuda", non_blocking=True)
    return x, edge_index, edge_attr, batch

def get_sequence_data(data):
    smiles = data.smiles.to("cuda", non_blocking=True)
    input_seq, tgt_seq = smiles[:, :-1], smiles[:, 1:]
    return input_seq, tgt_seq

def intialize(args, max_len, edge_vocab, node_vocab, smiles_vocab): 
    model = DGVAE(
        dim_encoder=args.dim_encoder,
        dim_decoder=args.dim_decoder,
        dim_latent=args.dim_latent,
        dim_encoder_ff=args.dim_encoder_ff,
        dim_decoder_ff=args.dim_decoder_ff,
        num_encoder_layer=args.num_encoder_layer,
        num_decoder_layer=args.num_decoder_layer,
        num_encoder_head=args.num_encoder_head,
        num_decoder_head=args.num_decoder_head,
        dropout=args.dropout,
        pool=args.pool,
        max_len=max_len,
        edge_vocab=edge_vocab,
        node_vocab=node_vocab,
        smiles_vocab=smiles_vocab,
    ).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = BetaScheduler(
        total_epochs=args.epochs,
        schedule=args.schedule,
        start=args.beta_start,
        end=args.beta_end,
        n_cycles=args.num_cycles,
        cycle_schedule=args.cycle_schedule,
        warmup_fraction=args.warmup_fraction,
    )

    writer = SummaryWriter(log_dir=args.log_dir)
    return model, optimizer, scheduler, writer

def evaluate(model, test_loader, smiles_vocab, beta, writer, step, N=30_000, active_threshold=1e-2):
    model.eval()

    # Metrics accumulators
    n_batches, total_loss, total_recon, total_kl, all_mu = 0, 0.0, 0.0, 0.0, []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", position=1, leave=False):
            # Graph-level
            x, edge_index, edge_attr, batch = get_graph_data(data)

            # Sequence-level
            input_seq, tgt_seq = get_sequence_data(data)

            # Forward pass
            pred, mu, logvar = model(x, edge_index, edge_attr, batch, input_seq)

            # Collect mus 
            all_mu.append(mu.detach().cpu()) 

            # Losses
            recon = F.nll_loss(
                pred.reshape(-1, len(smiles_vocab)),
                tgt_seq.reshape(-1),
                ignore_index=smiles_vocab["[pad]"],
            )
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Accumulate metrics
            total_loss += (recon + beta * kl).item(); total_recon += recon.item(); total_kl += kl.item(); n_batches += 1

        # Generate SMILES
        _, _, validity, uniqueness = generate(model, N)

        # Compute averages
        avg_loss, avg_recon, avg_kl = total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

        # Print on one line
        print(f"[METRICS] Loss: {avg_loss:.3f} | Recon: {avg_recon:.3f} | KL: {avg_kl:.3f} | Validity: {validity:.3f} | Uniqueness: {uniqueness:.3f}")

    # Active dimension threshold (you can tune this)
    mu_var = torch.cat(all_mu, dim=0).var(dim=0, unbiased=False)  # [latent_dim]
    active_mask = mu_var > active_threshold
    num_active = active_mask.sum().item()

    # Log to TensorBoard
    writer.add_scalar("Test/Loss_total", avg_loss, step)
    writer.add_scalar("Test/Loss_reconstruction", avg_recon, step)
    writer.add_scalar("Test/Loss_KL", avg_kl, step)
    writer.add_scalar("Metric/Validity", validity, step)
    writer.add_scalar("Metric/Uniqueness", uniqueness, step)
    writer.add_scalar("Latent/Test_ActiveDims", num_active, step)

    return avg_loss

def main(args):
    # Set seed
    seed_everything(109)

    # Training variables
    step, best_loss = 0, float("inf")

    # Build datasets and loaders
    train_set, test_set, edge_vocab, node_vocab, smiles_vocab, max_len = build_datasets(args.data_dir)
    train_loader, test_loader = make_loaders(train_set, test_set, args)
    
    # Initialize model, optimizer, scheduler, writer
    model, optimizer, scheduler, writer = intialize(args, max_len, edge_vocab, node_vocab, smiles_vocab)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        beta = scheduler.get_beta(epoch)

        # Training loop
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", position=0)
        for _, data in enumerate(loop):
            # Graph-level data
            x, edge_index, edge_attr, batch = get_graph_data(data)

            # Sequence data
            input_seq, tgt_seq = get_sequence_data(data)

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            pred, mu, logvar = model(x, edge_index, edge_attr, batch, input_seq)

            # Reconstruction loss
            recon_loss = F.nll_loss(
                pred.reshape(-1, len(smiles_vocab)),
                tgt_seq.reshape(-1),
                ignore_index=smiles_vocab["[pad]"],
            )

            # KL divergence with free bits
            mu_safe, logvar_safe = mu.clamp(min=-30.0, max=30.0), logvar.clamp(min=-30.0, max=20.0)
            kl_elementwise = -0.5 * (1 + logvar_safe - mu_safe.pow(2) - logvar_safe.exp())
            free_nats = args.free_bits * math.log(2.0) 
            kl_per_dim = kl_elementwise.mean(dim=0)  # [latent_dim]
            kl_per_dim_adjusted = (kl_per_dim - free_nats).clamp_min(0.0)
            kl_loss = kl_per_dim_adjusted.sum()

            # Total loss
            loss = recon_loss + beta * kl_loss

            # Backward
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track active dimensions in the batch (for monitoring)
            active_threshold = 1e-2
            mu_var_batch = mu_safe.var(dim=0, unbiased=False).detach()
            active_dims_batch = (mu_var_batch > active_threshold).sum().item()

            # Progress bar
            loop.set_postfix(
                loss=float(loss.item()),
                recon=float(recon_loss.item()),
                kl=float(kl_loss.item()),
                beta=float(beta),
                grad_norm=float(grad_norm),
                active_dim=float(active_dims_batch)
            )

            # Periodic eval + checkpoint
            if step > 0 and step % args.eval_every == 0:
                avg_loss = evaluate(model, test_loader, smiles_vocab, beta, writer, step)
                model.save(f"{args.log_dir}/step_{step}.pt")

                if avg_loss < best_loss and epoch > args.warmup_fraction * args.epochs: 
                    best_loss = avg_loss 
                    model.save(f"{args.log_dir}/best.pt")

            # Log to TensorBoard
            writer.add_scalar("Train/Loss_total", loss.item(), step)
            writer.add_scalar("Train/Loss_reconstruction", recon_loss.item(), step)
            writer.add_scalar("Train/Loss_KL", kl_loss.item(), step)
            writer.add_scalar("Train/GradNorm", grad_norm, step)
            writer.add_scalar("Train/Beta", beta, step)
            writer.add_scalar("Latent/Train_ActiveDims_batch", active_dims_batch, step)
            writer.add_histogram("Latent/Train_KL_per_dim", kl_per_dim, step)
            writer.add_histogram("Latent/Train_mu_var_batch", mu_var_batch, step)
            step += 1

        # End-of-training eval
        evaluate(model, test_loader, smiles_vocab, beta, writer, step)
        model.save(f"{args.log_dir}/step_{step}.pt")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TGVAE model with configurable hyperparameters")

    # General settings
    parser.add_argument("--name", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--data_dir", type=str, default="data/pretrain/zinc22_MBC")
    parser.add_argument("--log_dir", type=str, default=None)

    # Model hyperparameters
    parser.add_argument("--dim_encoder", type=int, default=128)
    parser.add_argument("--dim_decoder", type=int, default=256)
    parser.add_argument("--dim_latent", type=int, default=128)
    parser.add_argument("--dim_encoder_ff", type=int, default=256)
    parser.add_argument("--dim_decoder_ff", type=int, default=1024)
    parser.add_argument("--num_encoder_layer", type=int, default=16)
    parser.add_argument("--num_decoder_layer", type=int, default=4)
    parser.add_argument("--num_encoder_head", type=int, default=4)
    parser.add_argument("--num_decoder_head", type=int, default=8)
    parser.add_argument("--pool", type=str, default="add", choices=["add", "mean"])
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--free_bits", type=float, default=0.5)
    parser.add_argument("--eval_every", type=int, default=30000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--pin_memory", action="store_true", default=True)

    # VAE scheduling hyperparameters
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "sigmoid", "step", "cyclical"])
    parser.add_argument("--cycle_schedule", type=str, default="linear", choices=["linear", "sigmoid", "step"])
    parser.add_argument("--num_cycles", type=int, default=5)
    parser.add_argument("--beta_start", type=float, default=0.00001)
    parser.add_argument("--beta_end", type=float, default=1.0)
    parser.add_argument("--warmup_fraction", type=float, default=0.5)

    args = parser.parse_args()

    # Set up logging directory and save config
    args.log_dir = f"checkpoint/{args.name}" if args.log_dir is None else args.log_dir
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    main(args)
