# LatentRL

Latent-space molecular generation and optimization using a pretrained DGVAE model plus policy-gradient reinforcement learning.

This repository supports three end-to-end workflows:

1. **Prepare pretraining data** from core scaffolds + building blocks (`get_pretrain_data.py`)
2. **Pretrain DGVAE** on SMILES (`train.py`)
3. **Optimize molecules with RL** using either property rewards (`run_rl.py`) or docking rewards (`run_rl_docking.py`)

---

## What is in this repository

- `get_pretrain_data.py` — builds train/test pretraining sets and vocabulary cache
- `train.py` — pretrains DGVAE and writes checkpoints
- `run_rl.py` — latent RL with property-based reward
- `run_rl_docking.py` — latent RL with docking score reward (plus optional property terms)
- `data.py`, `utils.py`, `model/` — data pipeline, helper utilities, and model definitions
- `tools/script/` — analysis/docking helpers used by training/RL pipelines
- `data/` — raw inputs and generated datasets
- `checkpoint/` — training and TensorBoard artifacts

---

## Requirements

- Linux/macOS (Linux is most tested)
- Python environment with dependencies from `requirements.txt`
- CUDA-capable GPU recommended for training/RL
- For docking workflow: AutoDock Vina binaries, receptor/config files, and a SLURM environment (`sbatch`, `squeue`)

---

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Workflow 1: Prepare pretraining data

Use `get_pretrain_data.py` to create a dataset under `data/pretrain/<name>/`.

### Download source datasets

Before running `get_pretrain_data.py`, download:

- base pretraining molecules from ZINC22: https://cartblanche.docking.org/
- building block libraries from Enamine: https://enamine.net/

### Minimal example

```bash
python get_pretrain_data.py \
  --core "CCCC" \
  --pretrain data/raw/zinc22/zinc22_heavy_atoms_30to49_canonical.txt \
  --building-blocks data/raw/enamine_building_blocks_jan2026/Enamine_AlkylHalides_19893cmpds_20260108.sdf \
  --name my_dataset
```

### Key behavior

- If `--core-atom-idx` is omitted, the script creates `core_index.png` and prompts for an atom index interactively.
- Two independent train/test splits are performed:
  - base pretrain set: `--pretrain-train-ratio` / `--pretrain-test-ratio`
  - combinatorial set: `--bb-train-ratio` / `--bb-test-ratio`
- Optional property filtering is available:
  - `--min-logp`, `--max-logp`
  - `--min-mw`, `--max-mw`
  - `--min-num-heavy-atoms`, `--max-num-heavy-atoms`

### Output files

`data/pretrain/<name>/` includes:

- `train.txt`
- `test.txt`
- `vocab_cache.json`
- intermediate files such as `core_loaded.smi`, `combinatorial_generated.smi`, `merged_all.smi`
- `args.json` (run arguments)

---

## Workflow 2: Pretrain DGVAE

Train with the dataset produced above.

### Minimal example

```bash
python train.py \
  --name my_pretrain_run \
  --data_dir data/pretrain/my_dataset
```

### Commonly tuned options

- `--batch_size` (default: `128`)
- `--epochs` (default: `10`)
- `--lr` (default: `5e-4`)
- `--eval_every` (default: `30000`)
- latent/model dimensions such as `--dim_latent`, `--dim_encoder`, `--dim_decoder`

### Output files

`checkpoint/<run_name>/` contains:

- `config.json`
- `step_*.pt` checkpoints
- `best.pt` (best validation checkpoint)
- `events.out.tfevents.*` (TensorBoard logs)

To monitor logs:

```bash
tensorboard --logdir checkpoint
```

---

## Workflow 3A: RL optimization with property reward

Use `run_rl.py` to optimize molecules in latent space with REINFORCE.

### Minimal example

```bash
python run_rl.py \
  --input data/rl/input/MBC.txt \
  --output data/rl/output \
  --name rl_property_run \
  --ckpt checkpoint/zinc22_MBC/best.pt
```

### Important arguments

- `--num_steps` (default: `1000`)
- `--num_samples` (default: `128`)
- `--sigma` (default: `1.0`)
- `--vocab_cache_path` (default: `data/pretrain/zinc22_MBC/vocab_cache.json`)
- reward weights:
  - `--w_logp`, `--w_mw`, `--w_sim`, `--w_sa`, `--w_hva`
- substructure constraint:
  - `--substructure_match` accepts either a SMILES string or a path to a text file
- exploration controls:
  - `--use_entropy`, `--entropy_coef`
  - `--learnable_sigma`, `--min_sigma`, `--max_sigma`

### Output files

`<output>/<name>/` includes:

- `results.csv`
- `actions.npy`
- `config.json`
- `best_smiles_progression.png`
- `property_distributions.png`
- `action_distribution.png`
- `umap_molecules.png`
- TensorBoard logs (`events.out.tfevents.*`)

---

## Workflow 3B: RL optimization with docking reward

Use `run_rl_docking.py` to include docking affinity in the reward.

### Minimal example

```bash
python run_rl_docking.py \
  --input data/rl/input/MBC.txt \
  --output data/rl/output \
  --name rl_docking_run \
  --ckpt checkpoint/zinc22_MBC/best.pt \
  --receptor tools/AutoDock/mols/protein_no_MG.pdbqt \
  --config tools/AutoDock/mols/config_ori_25.txt
```

### Important arguments

- `--w_dock` (default: `1.0`)
- `--dock_workers` (default: `120`)
- docking inputs:
  - `--receptor` (PDBQT)
  - `--config` (Vina config)
- property weights also available (`--w_logp`, `--w_mw`, `--w_sim`, `--w_sa`, `--w_hva`)
- substructure constraint via `--substruct`

### Output files

`<output>/<name>/` includes property-RL outputs plus:

- `docking/` (per-step docking artifacts)
- `docking_summary.csv`
- `docking_summary.json`

A cross-run cache file is also used by default (`data/rl/global_dock_cache.csv`) to avoid redocking repeated molecules.

> Note: in the current code, the global docking cache constant is set to an absolute path in `run_rl_docking.py` (`GLOBAL_DOCK_CACHE`). If you run this repository in a different location, update that constant to a path valid in your environment.

---

## Input data format

RL input files (for `--input`) should be plain text with a seed SMILES string (or first line used as seed). Example:

```text
CCO
```