import os
import sys
import csv
import json
import time
import argparse
import datetime
import subprocess
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import CustomDataset
from model.dgvae import DGVAE
from model.policy import Policy, DeeperPolicy
from torch.utils.tensorboard import SummaryWriter
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
from tools.script.analysis import *

# ── Add tools/script to path so job_autodock helpers are importable ───────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "script"))
from job_autodock import write_chunk_csv, write_sbatch_script, combine_results

# ── Docking defaults ──────────────────────────────────────────────────────────
PYTHONSH       = "/share/lab_karolak/Team/LatentRL/tools/AutoDock/mgltools_x86_64Linux2_1.5.7/bin/pythonsh"
PREPARE_LIGAND = "/share/lab_karolak/Team/LatentRL/tools/AutoDock/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
VINA_BIN       = "/share/lab_karolak/Team/LatentRL/tools/AutoDock/autodock_vina_1_1_2_linux_x86/bin/vina"
DEFAULT_RECEPTOR = "/share/lab_karolak/Team/LatentRL/tools/AutoDock/mols/protein_no_MG.pdbqt"
DEFAULT_CONFIG   = "/share/lab_karolak/Team/LatentRL/tools/AutoDock/mols/config_ori_25.txt"

TOTAL_CPUS = 480   # total CPUs available across all SLURM jobs
MAX_CPUS_PER_JOB = 4   # cap CPUs allocated to a single SLURM job

# Docking score normalization anchors:
DOCK_SCORE_BEST  = -10.5   # kcal/mol  → normalized 1.0
DOCK_SCORE_WORST =  -1.0   # kcal/mol  → normalized 0.0

MBC_NOTAIL = 'c1c(c(c2c(c1)c(cc(=O)o2)O)C)O[C@H]1[C@@H]([C@@H]([C@H](C(O1)(C)C)OC)OC(=O)c1[nH]c(cc1)C)O'

matplotlib.use("Agg")
RDLogger.DisableLog("rdApp.*")


# ─────────────────────────────────────────────────────────────────────────────
# Docking helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_docking(affinity, k=2.0):
    lin = (affinity - DOCK_SCORE_WORST) / (DOCK_SCORE_BEST - DOCK_SCORE_WORST)
    lin = np.clip(lin, 0.0, 1.0)

    exp_val = np.exp(k * lin)
    return float(np.clip((exp_val - 1) / (np.exp(k) - 1), 0.0, 1.0))

def dock_smiles_batch(
    smiles_list: list[str],
    dock_dir: str,
    receptor: str = DEFAULT_RECEPTOR,
    config: str   = DEFAULT_CONFIG,
    num_workers: int = 4,
) -> dict[str, dict]:
    """
    Run docking for a batch of SMILES by submitting SLURM jobs via job_autodock.py
    helpers, waiting for all jobs to finish, then reading the combined results.

    Returns a dict keyed by SMILES:
        {
            "affinity"    : float | None,   # best-mode affinity (kcal/mol)
            "dock_score"  : float | None,   # normalized to [0, 1]
            "all_modes"   : dict | None,    # full Vina output (all modes)
        }
    """
    os.makedirs(dock_dir, exist_ok=True)

    # ── Write all SMILES to a single CSV ──────────────────────────────────────
    input_csv = os.path.join(dock_dir, "input.csv")
    write_chunk_csv(smiles_list, input_csv)

    # ── Split into chunks (one per worker) and submit SLURM jobs ─────────────
    num_jobs   = min(num_workers, len(smiles_list))
    chunks_dir = os.path.join(dock_dir, "chunks")
    logs_dir   = os.path.join(dock_dir, "slurm_logs")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    from job_autodock import split_smiles
    chunks   = split_smiles(smiles_list, num_jobs)
    job_ids: list[str] = []

    python_exec = sys.executable

    # Distribute CPUs as evenly as possible across jobs
    cpus_per_job = min(MAX_CPUS_PER_JOB, max(1, TOTAL_CPUS // max(len(chunks), 1)))
    mem_per_job  = max(8, cpus_per_job * 10)   # ~10 GB per CPU, min 8 GB

    for job_idx, chunk in enumerate(chunks):
        chunk_csv   = os.path.join(chunks_dir, f"chunk_{job_idx:04d}.csv")
        job_out_dir = os.path.join(dock_dir, f"job_{job_idx:04d}")
        os.makedirs(job_out_dir, exist_ok=True)

        write_chunk_csv(chunk, chunk_csv)

        script_path = write_sbatch_script(
            job_id       = job_idx,
            input_csv    = chunk_csv,
            job_out_dir  = job_out_dir,
            receptor     = receptor,
            config       = config,
            log_dir      = logs_dir,
            python_exec  = python_exec,
            slurm_cpus   = cpus_per_job,
            slurm_mem_gb = mem_per_job,
        )

        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            slurm_job_id = result.stdout.strip().split()[-1]
            job_ids.append(slurm_job_id)
            print(f"[dock] Submitted SLURM job {slurm_job_id} (chunk {job_idx}: {len(chunk)} SMILES)")
        else:
            print(f"[dock] sbatch failed for chunk {job_idx}:\n"
                  f"  stdout: {result.stdout.strip()}\n"
                  f"  stderr: {result.stderr.strip()}")

    # ── Wait for all submitted jobs to finish ─────────────────────────────────
    if job_ids:
        print(f"[dock] Waiting for {len(job_ids)} SLURM job(s): {job_ids}")
        _wait_for_slurm_jobs(job_ids, poll_interval=30)
        print(f"[dock] All SLURM jobs finished.")

    # ── Combine per-molecule JSON files from all job output dirs ──────────────
    combine_results(dock_dir)

    # ── Parse combined_results.json → return dict keyed by SMILES ────────────
    combined_json = os.path.join(dock_dir, "combined_results.json")
    return _parse_combined_results(combined_json)


def _wait_for_slurm_jobs(job_ids: list[str], poll_interval: int = 30) -> None:
    """
    Block until every SLURM job in *job_ids* is no longer in the queue
    (i.e. not visible in `squeue`).
    """
    pending = set(job_ids)
    while pending:
        result = subprocess.run(
            ["squeue", "--jobs", ",".join(pending), "--noheader", "-o", "%i"],
            capture_output=True, text=True,
        )
        still_running = set(result.stdout.strip().split()) if result.stdout.strip() else set()
        pending = pending & still_running
        if pending:
            print(f"[dock] Still running: {sorted(pending)} — checking again in {poll_interval}s")
            time.sleep(poll_interval)


def _parse_combined_results(json_path: str) -> dict[str, dict]:
    """
    Read combined_results.json written by combine_results() and return a
    dict keyed by SMILES with the same schema as the old dock_smiles_batch():
        {
            "affinity"   : float | None,
            "dock_score" : float | None,
            "all_modes"  : dict | None,
        }
    """
    results: dict[str, dict] = {}
    if not os.path.exists(json_path):
        return results

    try:
        with open(json_path) as fh:
            records = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[dock] Warning: could not read '{json_path}': {exc}")
        return results

    for rec in records:
        smi = rec.get("SMILES")
        if not smi:
            continue

        # Collect all integer mode keys (stored as strings in JSON)
        mode_data: dict[int, dict] = {}
        for k, v in rec.items():
            try:
                mode_data[int(k)] = v
            except (ValueError, TypeError):
                pass

        if not mode_data:
            results[smi] = {"affinity": None, "dock_score": None, "all_modes": None}
            continue

        best_mode    = mode_data[min(mode_data)]          # mode 1 = best pose
        affinity     = best_mode.get("affinity", None)
        dock_score   = normalize_docking(affinity) if affinity is not None else None

        results[smi] = {
            "affinity"  : affinity,
            "dock_score": dock_score,
            "all_modes" : {k: v for k, v in mode_data.items()},
        }

    return results

def has_IH_group(smiles: str) -> bool:
    """
    Returns True if the SMILES contains an [IH] group.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ih_pattern = Chem.MolFromSmarts("[IH]")
    return mol.HasSubstructMatch(ih_pattern)

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DGVAE Molecular Optimization with Docking Reward")

    # Input / output
    parser.add_argument("--input",  "-i",    type=str, default="data/rl/input/MBC.txt")
    parser.add_argument("--output", "-o",    type=str, default="data/rl/output")
    parser.add_argument("--name",   "-n",    type=str,
                        default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Model / optimization
    parser.add_argument("--num_samples",    type=int,   default=128)
    parser.add_argument("--sigma",          type=float, default=0.5)
    parser.add_argument("--dimension",      type=int,   default=128)
    parser.add_argument("--lr",    "-lr",   type=float, default=3e-4)
    parser.add_argument("--ckpt",  "-ckpt", type=str,   default="checkpoint/zinc22_MBC/best.pt")
    parser.add_argument("--num_steps", "-s",type=int,   default=1000)
    parser.add_argument("--max_occurrences", "-mo", type=int, default=3)
    parser.add_argument("--vocab_cache_path", "-vocab", type=str,
                        default="data/pretrain/zinc22_MBC/vocab_cache.json")

    # Reward weights
    parser.add_argument("--w_logp",   type=float, default=0.0)
    parser.add_argument("--w_mw",     type=float, default=0.0)
    parser.add_argument("--w_sim",    type=float, default=0.0)
    parser.add_argument("--w_sa",     type=float, default=0.0)
    parser.add_argument("--w_hva",    type=float, default=0.0,
                        help="Weight for heavy-atom-count component (default: 0.0).")
    parser.add_argument("--w_dock",   type=float, default=1.0,
                        help="Weight for docking score component (default: 1.0).")

    # Normalization ranges
    parser.add_argument("--min_logp", type=float, default=2.56)
    parser.add_argument("--max_logp", type=float, default=5.03)
    parser.add_argument("--min_mw",   type=float, default=473.0)
    parser.add_argument("--max_mw",   type=float, default=676.0)

    # Docking
    parser.add_argument("--receptor", type=str, default=DEFAULT_RECEPTOR,
                        help="Path to receptor PDBQT file.")
    parser.add_argument("--config",   type=str, default=DEFAULT_CONFIG,
                        help="Path to Vina config file.")
    parser.add_argument("--dock_workers", type=int, default=120,
                        help="Parallel workers for Steps 1 & 2 of each docking batch.")

    # Substructure match
    parser.add_argument("--substruct", type=str, default=MBC_NOTAIL,
                        help="SMILES string or path to a .txt file containing the substructure "
                             "to require in generated molecules. Defaults to MBC_NOTAIL.")


    # Entropy regularization and learnable sigma (Advanced arguments)
    parser.add_argument("--use_entropy",  action="store_true", default=False)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--learnable_sigma", action="store_true", default=False)
    parser.add_argument("--min_sigma", type=float, default=0.01)
    parser.add_argument("--max_sigma", type=float, default=1.0)

    args = parser.parse_args()

    # ── Resolve substructure SMILES ───────────────────────────────────────────
    substruct_val = args.substruct
    if os.path.isfile(substruct_val):
        with open(substruct_val) as fh:
            substruct_val = fh.read().strip().splitlines()[0].strip()
        print(f"[substruct] Loaded substructure SMILES from file: {substruct_val!r}")
    else:
        print(f"[substruct] Using substructure SMILES: {substruct_val!r}")
    args.substruct_smiles = substruct_val

    args.output_folder  = f"{args.output}/{args.name}"
    args.input_smiles   = read_input(args.input)
    args.input_logp     = get_logp(args.input_smiles)
    args.input_mw       = get_mw(args.input_smiles)
    args.input_sa       = get_sa(args.input_smiles)
    args.input_num_heavy_atoms = get_num_heavy_atoms(args.input_smiles)
    args.date           = datetime.datetime.now().strftime("%m/%d/%Y")

    os.makedirs(args.output_folder, exist_ok=True)
    # Dedicated directory for all docking files
    args.docking_root = os.path.join(args.output_folder, "docking")
    os.makedirs(args.docking_root, exist_ok=True)

    with open(f"{args.output_folder}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize(args):
    model  = DGVAE.load(args.ckpt).to("cuda")
    model.eval()

    policy = DeeperPolicy(
        dim=args.dimension, N=args.num_samples,
        sigma=args.sigma,
        learnable_sigma=args.learnable_sigma,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
    ).to("cuda")
    opt    = Adam(policy.parameters(), lr=args.lr)
    loader = DataLoader(
        CustomDataset(args.input, vocab_cache_path=args.vocab_cache_path, latent_rl=True)
    )
    return model, policy, opt, loader


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers (same as run_rl.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_penalty(smiles, counter, max_occurrences):
    count = counter[smiles]
    if count > max_occurrences:
        return 0.0
    return 1.0 - (count - 1) / max_occurrences


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


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    writer = SummaryWriter(log_dir=args.output_folder)

    results, actions, smiles_counter = [], [], Counter()
    model, policy, opt, loader = initialize(args)

    best_score = 0.0
    baseline   = None

    pbar = tqdm(range(args.num_steps), desc="Optimizing")
    for step in pbar:
        for data in loader:
            # ── Sample actions in latent space ────────────────────────────────
            action, log_prob, entropy = policy()
            actions.append(action.detach().cpu().numpy())

            # ── Encode → perturb → decode ─────────────────────────────────────
            z0 = vae_encode(model, data).repeat(args.num_samples, 1)
            gen_smiles_list = vae_decode(model, z0 + action)

            # ── Update occurrence counter ─────────────────────────────────────
            smiles_counter.update(gen_smiles_list)

            # ── Collect valid SMILES for docking ──────────────────────────────
            mol_map: dict[str, object] = {}   # smiles → RDKit mol (valid only)
            for smi in gen_smiles_list:
                mol = get_mol(smi)
                if mol is not None and smi not in mol_map:
                    if has_IH_group(smi):  # skip molecules with [IH] group
                        continue
                    if has_substructure_match(mol, args.substruct_smiles):  # only dock if substructure matches
                        if 34 <= get_num_heavy_atoms(mol) <= 48:  # only dock if HVA in range
                            mol_map[smi] = mol

            # ── Run docking on unique valid SMILES ────────────────────────────
            step_dock_dir = os.path.join(args.docking_root, f"step_{step:05d}")
            dock_results: dict[str, dict] = {}
            if mol_map:
                unique_to_dock = list(mol_map.keys())

                # Check global cache first
                cached_results, still_to_dock = _get_cached_results(unique_to_dock)
                dock_results.update(cached_results)

                if still_to_dock:
                    new_dock_results = dock_smiles_batch(
                        still_to_dock,
                        step_dock_dir,
                        receptor    = args.receptor,
                        config      = args.config,
                        num_workers = args.dock_workers,
                    )
                    dock_results.update(new_dock_results)
                    _append_to_global_cache(new_dock_results, run_name=args.name, step=step)
                else:
                    print(f"[cache] Step {step}: all {len(unique_to_dock)} SMILES served from cache.")

            # ── Compute scores ────────────────────────────────────────────────
            scores: list[float] = []
            num_valid = 0
            num_substructure_match = 0
            valid_smiles: list[str] = []
            unique_smiles_in_batch: set[str] = set()

            # Collect docking metrics for logging
            dock_scores_this_step: list[float] = []
            affinities_this_step:  list[float] = []

            for smi in gen_smiles_list:
                mol = get_mol(smi)

                if mol is None:
                    scores.append(0.0)
                    results.append([
                        step, smi, None, None, None, None, None,
                        None, None, None, 0.0, 0.0,
                    ])
                    continue

                # ── Skip molecules with [IH] group ────────────────────────────
                if has_IH_group(smi):
                    scores.append(0.0)
                    results.append([
                        step, smi, False, None, None, None, None, None,
                        None, None, 0.0, 0.0,
                    ])
                    continue

                num_valid += 1
                valid_smiles.append(smi)
                unique_smiles_in_batch.add(smi)

                # ── Molecular properties ──────────────────────────────────────
                sa = get_sa(mol)
                mw = get_mw(mol)
                logp = get_logp(mol)
                sim = get_sim(mol, args.input_smiles)
                hva           = get_num_heavy_atoms(mol)
                has_substruct = has_substructure_match(mol, args.substruct_smiles)

                if has_substruct:
                    num_substructure_match += 1

                # ── Docking score ─────────────────────────────────────────────
                d_info     = dock_results.get(smi, {})
                affinity   = d_info.get("affinity",   None)
                dock_score = d_info.get("dock_score", None)

                if affinity is not None:
                    affinities_this_step.append(affinity)
                if dock_score is not None:
                    dock_scores_this_step.append(dock_score)

                # ── Penalty for repeated molecules ────────────────────────────
                penalty = get_penalty(smi, smiles_counter, args.max_occurrences)

                # ── Heavy atom count gate ─────────────────────────────────────
                hva_valid = 34 <= hva <= 48

                # ── Normalize individual components ───────────────────────────
                logp_score = normalize(logp, min_val=args.min_logp, max_val=args.max_logp,
                                       higher_is_better=False, strict=True)
                mw_score   = normalize(mw,   min_val=args.min_mw,   max_val=args.max_mw,
                                       higher_is_better=False, strict=True)
                sa_score   = normalize(sa,   min_val=1.0, max_val=10.0,
                                       higher_is_better=False, strict=False)
                # heavy atom score: 1.0 inside [34,48], else 0.0
                hva_score  = 1.0 if hva_valid else 0.0
                dock_norm  = dock_score if dock_score is not None else 0.0

                # ── Composite score ───────────────────────────────────────────
                score_components: list[float] = []
                active_weights: list[float] = []
                if args.w_logp > 0:
                    score_components.append(args.w_logp * logp_score)
                    active_weights.append(args.w_logp)
                if args.w_sim > 0:
                    score_components.append(args.w_sim * sim)
                    active_weights.append(args.w_sim)
                if args.w_mw > 0:
                    score_components.append(args.w_mw * mw_score)
                    active_weights.append(args.w_mw)
                if args.w_sa > 0:
                    score_components.append(args.w_sa * sa_score)
                    active_weights.append(args.w_sa)
                if args.w_hva > 0:
                    score_components.append(args.w_hva * hva_score)
                    active_weights.append(args.w_hva)
                if args.w_dock > 0:
                    score_components.append(args.w_dock * dock_norm)
                    active_weights.append(args.w_dock)

                total_weight = sum(active_weights)
                score = (
                    sum(score_components) / total_weight
                    if score_components and total_weight > 0 and has_substruct and hva_valid
                    else 0.0
                )

                scores.append(score * penalty)

                results.append([
                    step, smi,
                    has_substruct,
                    logp, hva, mw, sa, sim,
                    affinity, dock_norm,
                    score, score * penalty,
                ])

            # ── Policy gradient update (REINFORCE with baseline) ──────────────
            r       = torch.tensor(scores, dtype=torch.float32, device="cuda")
            r_mean  = r.mean().item()

            current_best = r.max().item()
            if current_best > best_score:
                best_score = current_best

            baseline = r_mean if baseline is None else 0.9 * baseline + 0.1 * r_mean
            adv      = r - baseline

            policy_loss = -(adv * log_prob).mean()
            if args.use_entropy:
                entropy_bonus = entropy.mean()
                loss = policy_loss - args.entropy_coef * entropy_bonus
            else:
                loss          = policy_loss
                entropy_bonus = torch.tensor(0.0)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=1.0
            )
            opt.step()


            # ── Metrics ───────────────────────────────────────────────────────
            total_gen          = len(gen_smiles_list)
            validity_pct       = (num_valid / total_gen) * 100
            uniqueness_batch   = (len(unique_smiles_in_batch) / total_gen) * 100
            uniqueness_overall = (len(set(valid_smiles)) / max(num_valid, 1)) * 100
            substruct_pct      = (num_substructure_match / total_gen) * 100
            action_length      = torch.norm(action, p=2, dim=1).mean().item()

            avg_dock_score  = float(np.mean(dock_scores_this_step))  if dock_scores_this_step  else 0.0
            best_dock_score = float(np.max(dock_scores_this_step))   if dock_scores_this_step  else 0.0
            avg_affinity    = float(np.mean(affinities_this_step))   if affinities_this_step   else 0.0
            best_affinity   = float(np.min(affinities_this_step))    if affinities_this_step   else 0.0   # min = most negative = best
            dock_success    = len(dock_scores_this_step)

            pbar.set_postfix({
                "B"     : f"{best_score:.3f}",
                "A"     : f"{r_mean:.3f}",
                "BAff" : f"{best_affinity:.2f}",
                "V"     : f"{validity_pct:.1f}%",
                "SSM"   : f"{substruct_pct:.1f}%",
            })

            # ── TensorBoard ───────────────────────────────────────────────────
            writer.add_scalar("train/loss",          loss.item(),          step)
            writer.add_scalar("train/policy_loss",   policy_loss.item(),   step)
            writer.add_scalar("train/entropy",       entropy.mean().item(),step)
            if args.use_entropy:
                writer.add_scalar("train/entropy_bonus", entropy_bonus.item(), step)
            writer.add_scalar("train/reward",        r_mean,               step)
            writer.add_scalar("train/best_score",    best_score,           step)
            writer.add_scalar("train/advantage",     adv.mean().item(),    step)
            writer.add_scalar("train/grad_norm",     grad_norm if isinstance(grad_norm, float) else grad_norm.item(), step)
            writer.add_scalar("train/action_length", action_length,        step)

            writer.add_scalar("molecules/validity_pct",          validity_pct,       step)
            writer.add_scalar("molecules/uniqueness_batch_pct",  uniqueness_batch,   step)
            writer.add_scalar("molecules/uniqueness_overall_pct",uniqueness_overall, step)
            writer.add_scalar("molecules/substructure_match_pct",substruct_pct,      step)
            writer.add_scalar("molecules/num_valid",             num_valid,          step)
            writer.add_scalar("molecules/num_unique_in_batch",   len(unique_smiles_in_batch), step)
            writer.add_scalar("molecules/num_substructure_match",num_substructure_match,      step)

            # Docking-specific metrics
            writer.add_scalar("docking/avg_dock_score",  avg_dock_score,  step)
            writer.add_scalar("docking/best_dock_score", best_dock_score, step)
            writer.add_scalar("docking/avg_affinity",    avg_affinity,    step)
            writer.add_scalar("docking/best_affinity",   best_affinity,   step)
            writer.add_scalar("docking/num_docked",      dock_success,    step)

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(
        results,
        columns=[
            "Step", "SMILES", "HasSubstructure",
            "LogP", "NumHeavyAtoms", "MW", "SA", "Similarity",
            "DockingAffinity", "DockScore",
            "Score", "PenalizedScore",
        ],
    )
    df.to_csv(f"{args.output_folder}/results.csv", index=False)
    np.save(f"{args.output_folder}/actions.npy", np.concatenate(actions, axis=0))

    # ── Save combined docking JSON / CSV ──────────────────────────────────────
    _save_docking_summary(df, args.output_folder)

    writer.close()

    # ── Post-training analysis ────────────────────────────────────────────────
    plot_best_smiles_progression(df, f"{args.output_folder}/best_smiles_progression.png")
    plot_property_distributions(df, vars(args), f"{args.output_folder}/property_distributions.png")
    plot_action_distribution(
        f"{args.output_folder}/actions.npy",
        f"{args.output_folder}/action_distribution.png",
        num_steps=args.num_steps,
        dimension=args.dimension,
    )
    plot_umap_molecules(df, f"{args.output_folder}/umap_molecules.png")


# ─────────────────────────────────────────────────────────────────────────────
# Post-run docking summary
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_DOCK_CACHE = "/share/lab_karolak/Team/LatentRL/data/rl/global_dock_cache.csv"
GLOBAL_DOCK_CACHE_LOCK = GLOBAL_DOCK_CACHE + ".lock"

# Build column names for all 10 modes
_MODE_COLS = []
for _m in range(1, 11):
    _MODE_COLS += [f"mode{_m}_affinity", f"mode{_m}_rmsd_lb", f"mode{_m}_rmsd_ub"]

GLOBAL_CACHE_COLUMNS = ["SMILES"] + _MODE_COLS + ["run_name", "step", "timestamp", "SMILES_key"]


import fcntl
import tempfile



def _load_global_cache() -> pd.DataFrame:
    """
    Load the global docking cache CSV.
    Acquires a shared (read) lock so concurrent readers don't block each other,
    but a writer will block until all readers finish.
    Returns an empty DataFrame if the file doesn't exist or is unreadable.
    """
    if not os.path.exists(GLOBAL_DOCK_CACHE):
        return pd.DataFrame(columns=GLOBAL_CACHE_COLUMNS)

    lock_path = GLOBAL_DOCK_CACHE_LOCK
    with open(lock_path, "a") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_SH)   # shared / read lock
            try:
                return pd.read_csv(GLOBAL_DOCK_CACHE)
            except Exception as exc:
                print(f"[cache] Warning: could not read global cache '{GLOBAL_DOCK_CACHE}': {exc}")
                return pd.DataFrame(columns=GLOBAL_CACHE_COLUMNS)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _save_global_cache(df: pd.DataFrame) -> None:
    """
    Save the global docking cache CSV atomically.
    Acquires an exclusive (write) lock, writes to a temp file in the same
    directory, then atomically replaces the target — so readers always see
    a complete file and a crash mid-write never corrupts the cache.
    """
    cache_dir = os.path.dirname(GLOBAL_DOCK_CACHE)
    os.makedirs(cache_dir, exist_ok=True)

    lock_path = GLOBAL_DOCK_CACHE_LOCK
    with open(lock_path, "a") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)   # exclusive / write lock

            # Write to a temp file in the same directory so os.replace is
            # guaranteed to be atomic (same filesystem, no cross-device move).
            tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w") as tmp_fh:
                    df.to_csv(tmp_fh, index=False)
                os.replace(tmp_path, GLOBAL_DOCK_CACHE)   # atomic on POSIX
            except Exception:
                # Clean up the temp file if anything goes wrong before replace
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

def _make_smiles_key(smi: str, existing_keys: set) -> str:
    """
    Return a unique key for this SMILES.
    First occurrence → bare SMILES.
    Subsequent dockings of the same molecule → SMILES_2, SMILES_3, …
    """
    if smi not in existing_keys:
        return smi
    i = 2
    while f"{smi}_{i}" in existing_keys:
        i += 1
    return f"{smi}_{i}"


def _append_to_global_cache(
    dock_results: dict,
    run_name: str,
    step: int,
) -> None:
    """
    Append new docking results to the global cache CSV.
    All 10 modes are stored as separate columns.
    Each SMILES gets a unique key (_2, _3, …) if it was docked before.
    """
    cache_df = _load_global_cache()
    existing_keys = set(cache_df["SMILES_key"].tolist()) if not cache_df.empty else set()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []

    for smi, info in dock_results.items():
        key = _make_smiles_key(smi, existing_keys)
        existing_keys.add(key)   # prevent duplicates within the same batch

        row: dict = {
            "SMILES_key": key,
            "SMILES"    : smi,
            "run_name"  : run_name,
            "step"      : step,
            "timestamp" : timestamp,
        }

        all_modes = info.get("all_modes") or {}
        for m in range(1, 11):
            mode_info = all_modes.get(m, {})
            row[f"mode{m}_affinity"] = mode_info.get("affinity", None)
            row[f"mode{m}_rmsd_lb"]  = mode_info.get("rmsd_lb",  None)
            row[f"mode{m}_rmsd_ub"]  = mode_info.get("rmsd_ub",  None)

        new_rows.append(row)

    if new_rows:
        updated = pd.concat(
            [cache_df, pd.DataFrame(new_rows, columns=GLOBAL_CACHE_COLUMNS)],
            ignore_index=True,
        )
        _save_global_cache(updated)
        print(f"[cache] Appended {len(new_rows)} entries → {GLOBAL_DOCK_CACHE}")


def _save_docking_summary(df: pd.DataFrame, output_folder: str) -> None:
    """
    From the full results DataFrame, write:
      - docking_summary.csv  – one row per unique SMILES with best affinity
      - docking_summary.json – same data as a JSON array
    Sorted by DockingAffinity (most negative first = best binder).
    """
    docked = df.dropna(subset=["DockingAffinity"]).copy()
    if docked.empty:
        print("[summary] No docked molecules to summarise.")
        return

    # Keep the best-affinity result per unique SMILES
    best_per_smiles = (
        docked.sort_values("DockingAffinity")
              .groupby("SMILES", as_index=False)
              .first()
              .sort_values("DockingAffinity")
    )

    summary_csv = os.path.join(output_folder, "docking_summary.csv")
    best_per_smiles.to_csv(summary_csv, index=False)
    print(f"[summary] Docking CSV  → {summary_csv}  ({len(best_per_smiles)} unique SMILES)")

    summary_json = os.path.join(output_folder, "docking_summary.json")
    best_per_smiles.to_json(summary_json, orient="records", indent=2)
    print(f"[summary] Docking JSON → {summary_json}")


def _get_cached_results(smiles_list: list[str]) -> tuple[dict[str, dict], list[str]]:
    """
    For each SMILES in smiles_list, check the global cache.
    If a SMILES has >= 3 cached entries, get the min affinity/dock_score
    and return it as a cached result (no re-docking needed).

    Returns:
        cached_results : dict[str, dict] – same schema as dock_smiles_batch()
        to_dock        : list[str]       – SMILES that still need docking
    """
    cache_df = _load_global_cache()
    cached_results: dict[str, dict] = {}
    to_dock: list[str] = []

    for smi in smiles_list:
        if cache_df.empty:
            to_dock.append(smi)
            continue

        rows = cache_df[cache_df["SMILES"] == smi]
        if len(rows) >= 3:
        # if len(rows) >= 5:
            # Average mode1 affinity across all cached runs
            affinities = rows["mode1_affinity"].dropna().tolist()
            if affinities:
                # avg_affinity = float(np.mean(affinities))
                avg_affinity = float(np.min(affinities))  # take the best (most negative) affinity
                avg_dock_score = normalize_docking(avg_affinity)
                cached_results[smi] = {
                    "affinity"  : avg_affinity,
                    "dock_score": avg_dock_score,
                    "all_modes" : None,   # averaged – no single mode breakdown
                }
                print(f"[cache] Using cached min for {smi!r} "
                      f"({len(rows)} runs, min affinity={avg_affinity:.2f})")
            else:
                # Rows exist but no valid affinity – re-dock
                to_dock.append(smi)
        else:
            to_dock.append(smi)

    return cached_results, to_dock


if __name__ == "__main__":
    main()
