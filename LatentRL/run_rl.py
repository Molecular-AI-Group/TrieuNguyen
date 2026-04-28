import os 
import json
import argparse
import datetime
from collections import Counter
import matplotlib
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from data import CustomDataset
from model.dgvae import DGVAE
from model.policy import DeeperPolicy
from torch.utils.tensorboard import SummaryWriter
from tools.script.analysis import *
from utils import (
    vae_encode,
    vae_decode,
    read_input,
    get_mol,
    get_mw,
    get_logp,
    get_sa,
    get_sim,
    get_num_heavy_atoms,
    has_substructure_match,
)
MBC_NOTAIL = 'c1c(c(c2c(c1)c(cc(=O)o2)O)C)O[C@H]1[C@@H]([C@@H]([C@H](C(O1)(C)C)OC)OC(=O)c1[nH]c(cc1)C)O'

matplotlib.use("Agg")  
RDLogger.DisableLog("rdApp.*")

def parse_args():
    parser = argparse.ArgumentParser(description='DGVAE Molecular Optimization')

    # Input/output paths
    parser.add_argument('--input', '-i', type=str, default='data/rl/input/MBC.txt')
    parser.add_argument('--output', '-o', type=str, default='data/rl/output')
    parser.add_argument('--name', '-n', type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Model and optimization parameters
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--lr', '-lr', type=float, default=3e-4)
    parser.add_argument('--ckpt', '-ckpt', type=str, default='checkpoint/zinc22_MBC/best.pt')
    parser.add_argument('--num_steps', '-s', type=int, default=1000)
    parser.add_argument('--max_occurrences', '-mo', type=int, default=3)
    parser.add_argument('--vocab_cache_path', '-vocab', type=str, default='data/pretrain/zinc22_MBC/vocab_cache.json')

    # Reward component weights and normalization ranges
    parser.add_argument('--w_logp', type=float, default=1.0)
    parser.add_argument('--w_mw', type=float, default=1.0)
    parser.add_argument('--w_sim', type=float, default=1.0)
    parser.add_argument('--w_sa', type=float, default=1.0)
    parser.add_argument('--w_hva', type=float, default=1.0,
                        help='Weight for the heavy atom count score component')
    parser.add_argument('--min_logp', type=float, default=2.56)
    parser.add_argument('--max_logp', type=float, default=5.03)
    parser.add_argument('--min_mw', type=float, default=473.0)
    parser.add_argument('--max_mw', type=float, default=676.0)
    parser.add_argument('--substructure_match', type=str, default=MBC_NOTAIL,
                        help='SMILES string or path to a .txt file containing the substructure SMILES')

    
    # Entropy regularization and learnable sigma (Advanced configuration)
    parser.add_argument('--use_entropy', action='store_true', default=False, help='Use entropy regularization for exploration')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Coefficient for entropy regularization')
    parser.add_argument('--learnable_sigma', action='store_true', default=False)
    parser.add_argument('--min_sigma', type=float, default=0.01)
    parser.add_argument('--max_sigma', type=float, default=1.0)

    args = parser.parse_args()
    args.output_folder = f"{args.output}/{args.name}"
    args.input_smiles = read_input(args.input)
    args.input_logp = get_logp(args.input_smiles)
    args.input_mw = get_mw(args.input_smiles)
    args.input_sa = get_sa(args.input_smiles)
    args.input_num_heavy_atoms = get_num_heavy_atoms(args.input_smiles)
    args.date = datetime.datetime.now().strftime("%m/%d/%Y")

    # Resolve substructure SMILES from file or direct string
    if os.path.isfile(args.substructure_match):
        with open(args.substructure_match, 'r') as f:
            args.substructure_match_smiles = f.read().strip().splitlines()[0]
    else:
        args.substructure_match_smiles = args.substructure_match

    os.makedirs(args.output_folder, exist_ok=True)

    with open(f'{args.output_folder}/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    return args

def initialize(args): 
    # Load pretrained model
    model = DGVAE.load(args.ckpt).to("cuda")
    model.eval()

    # Initialize policy network and optimizer
    policy = DeeperPolicy(dim=args.dimension, N=args.num_samples, sigma=args.sigma, learnable_sigma=args.learnable_sigma, min_sigma=args.min_sigma, max_sigma=args.max_sigma).to("cuda")
    opt = Adam(policy.parameters(), lr=args.lr)

    # Initialize data loader (Only 1 SMILES)
    loader = DataLoader(CustomDataset(args.input, vocab_cache_path=args.vocab_cache_path, latent_rl=True))

    return model, policy, opt, loader

def get_penalty(smiles, counter, max_occurences): 
    count = counter[smiles]
    if count > max_occurences:
        return 0.0 
    else: 
        return 1.0 - (count - 1) / max_occurences

def normalize(x, min_val, max_val, higher_is_better=True, strict=False):
    if strict and (x < min_val or x > max_val):
        return 0.0
    if x <= min_val:
        norm = 0.0
    elif x >= max_val:
        norm = 1.0
    else:
        norm = (x - min_val) / (max_val - min_val)
    return norm if higher_is_better else 1.0 - norm

def average(*args):
    return sum(args) / len(args)


def main():
    args = parse_args()

    # TensorBoard writer for logging
    writer = SummaryWriter(log_dir=args.output_folder)

    # Initialize SMILES counter
    results, actions, smiles_counter = [], [], Counter()
    
    # Set up model, policy, optimizer, and data loader
    model, policy, opt, loader = initialize(args)

    # Track best score across all steps
    best_score = 0.0

    pbar = tqdm(range(args.num_steps), desc="Optimizing")
    for step in pbar:
        for data in loader:
            # Sample actions in latent space
            action, log_prob, entropy = policy()

            # Save action to list
            actions.append(action.detach().cpu().numpy())

            # Encode input, perturb latent, and decode to SMILES
            z0 = vae_encode(model, data).repeat(args.num_samples, 1)
            gen_smiles_list = vae_decode(model, z0 + action)

            # Update SMILES counter
            smiles_counter.update(gen_smiles_list)

            # Track metrics for this batch
            scores, valid_smiles, unique_smiles_in_batch, num_valid, num_substructure_match = [], [], set(), 0, 0
            
            # Compute raw scores
            for _, smiles in enumerate(gen_smiles_list): 
                mol = get_mol(smiles)

                # If invalid molecule, score = 0
                if mol is None: 
                    scores.append(0.0)
                    results.append([step, smiles, None, None, None, None, None, 0.0, 0.0])
                    continue

                # Track valid molecules
                num_valid += 1
                valid_smiles.append(smiles)
                unique_smiles_in_batch.add(smiles)

                # Compute properties
                sa, mw, logp, hva, sim, has_substructure = get_sa(mol), get_mw(mol), get_logp(mol), get_num_heavy_atoms(mol), get_sim(mol, args.input_smiles), has_substructure_match(mol, args.substructure_match_smiles)

                # Track substructure matches
                if has_substructure:
                    num_substructure_match += 1

                # Get penalty for occurrences
                penalty = get_penalty(smiles, smiles_counter, args.max_occurrences)

                # Normalize individual components
                logp_score = normalize(logp, min_val=args.min_logp, max_val=args.max_logp, higher_is_better=False, strict=True)
                mw_score = normalize(mw, min_val=args.min_mw, max_val=args.max_mw, higher_is_better=False, strict=True)
                sa_score = normalize(sa, min_val=1.0, max_val=10.0, higher_is_better=False, strict=False)
                hva_score = 1.0 if 34 <= hva <= 49 else 0.0

                # Composite score - only include components with non-zero weights
                score_components = []
                total_weight = 0.0
                if args.w_logp > 0:
                    score_components.append(args.w_logp * logp_score)
                    total_weight += args.w_logp
                if args.w_sim > 0:
                    score_components.append(args.w_sim * sim)
                    total_weight += args.w_sim
                if args.w_mw > 0:
                    score_components.append(args.w_mw * mw_score)
                    total_weight += args.w_mw
                if args.w_sa > 0:
                    score_components.append(args.w_sa * sa_score)
                    total_weight += args.w_sa
                if args.w_hva > 0:
                    score_components.append(args.w_hva * hva_score)
                    total_weight += args.w_hva
                
                score = sum(score_components) / total_weight if score_components and has_substructure else 0.0

                # Add to scores for policy gradient update
                scores.append(score * penalty)

                # Save results for this molecule
                results.append([step, smiles, has_substructure, logp, hva, mw, sa, score, score * penalty])

            # Policy gradient update (REINFORCE with baseline)
            r = torch.tensor(scores, dtype=torch.float32, device="cuda")
            r_mean = r.mean().item()
            
            # Update best score
            current_best = r.max().item()
            if current_best > best_score:
                best_score = current_best
            
            # Update running mean baseline
            baseline = r_mean if step == 0 else 0.9 * baseline + 0.1 * r_mean
            adv = r - baseline
            
            # Policy loss with optional entropy regularization
            policy_loss = -(adv * log_prob).mean()
            if args.use_entropy:
                entropy_bonus = entropy.mean()
                loss = policy_loss - args.entropy_coef * entropy_bonus
            else:
                loss = policy_loss
                entropy_bonus = torch.tensor(0.0)

            # Perform optimization step
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            opt.step()

            # Calculate metrics for logging
            validity_pct = (num_valid / len(gen_smiles_list)) * 100
            uniqueness_in_batch_pct = (len(unique_smiles_in_batch) / len(gen_smiles_list)) * 100
            uniqueness_overall_pct = (len(set(valid_smiles)) / max(num_valid, 1)) * 100
            substructure_match_pct = (num_substructure_match / len(gen_smiles_list)) * 100
            action_length = torch.norm(action, p=2, dim=1).mean().item()

            # Update tqdm progress bar
            pbar.set_postfix({
                'B': f'{best_score:.3f}', 
                'A': f'{r_mean:.3f}',
                'V': f'{validity_pct:.1f}%',
                'U': f'{uniqueness_in_batch_pct:.1f}%',
                'SSM': f'{substructure_match_pct:.1f}%',
                'AL': f'{action_length:.3f}'
            })

            # TensorBoard logging 
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/policy_loss", policy_loss.item(), step)
            writer.add_scalar("train/entropy", entropy.mean().item(), step)
            if args.use_entropy:
                writer.add_scalar("train/entropy_bonus", entropy_bonus.item(), step)
            writer.add_scalar("train/reward", r_mean, step)
            writer.add_scalar("train/best_score", best_score, step)
            writer.add_scalar("train/advantage", adv.mean().item(), step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/action_length", action_length, step)
            writer.add_scalar("molecules/validity_pct", validity_pct, step)
            writer.add_scalar("molecules/uniqueness_in_batch_pct", uniqueness_in_batch_pct, step)
            writer.add_scalar("molecules/uniqueness_overall_pct", uniqueness_overall_pct, step)
            writer.add_scalar("molecules/substructure_match_pct", substructure_match_pct, step)
            writer.add_scalar("molecules/num_valid", num_valid, step)
            writer.add_scalar("molecules/num_unique_in_batch", len(unique_smiles_in_batch), step)
            writer.add_scalar("molecules/num_substructure_match", num_substructure_match, step)



    # Create DataFrame at the end
    df = pd.DataFrame(results, columns=["Step", "SMILES", "HasSubstructure", "LogP", "NumHeavyAtoms", "MW", "SA", "Score", "Penalized Score"])
    
    # Save results (embedding will be saved as a column with arrays)
    df.to_csv(f"{args.output_folder}/results.csv", index=False)  
    np.save(f"{args.output_folder}/actions.npy", np.concatenate(actions, axis=0))
    
    # Close TensorBoard writer
    writer.close()

    # Post-training analysis and visualizations
    plot_best_smiles_progression(df, f"{args.output_folder}/best_smiles_progression.png")
    plot_property_distributions(df, vars(args), f"{args.output_folder}/property_distributions.png")
    plot_action_distribution(f"{args.output_folder}/actions.npy", f"{args.output_folder}/action_distribution.png", num_steps=args.num_steps, dimension=args.dimension)
    plot_umap_molecules(df, f"{args.output_folder}/umap_molecules.png")

if __name__ == "__main__":
    main()