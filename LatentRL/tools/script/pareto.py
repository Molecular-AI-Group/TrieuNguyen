import os 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from tqdm import tqdm
from rdkit import Chem, RDLogger
from multiprocessing import Pool, cpu_count
RDLogger.DisableLog('rdApp.*')
sns.set_style("whitegrid")
os.chdir("/home/80027464/LatentRL/tools/notebook/")

data_dir = "/home/80027464/LatentRL/data/" 
figure_dir = "/home/80027464/LatentRL/figure/"

color_rl = '#2E86AB' 
color_comb = '#A23B72'  
color_rl_docking = '#E07B39'

def read_input(path):
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines()]

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


# ------------------------------------------------------------------ #
# Shared memory helpers                                               #
# ------------------------------------------------------------------ #

# Global references used by worker processes — set once per pool via
# pool initializer so the large ref array is never pickled per-task.
_shm_vals   = None   # full active_vals array (read-only)
_shm_ref    = None   # current Pareto frontier (read-only)


def _pool_init(vals_shape, vals_dtype, vals_data,
               ref_shape,  ref_dtype,  ref_data):
    """Initializer: deserialise the two shared arrays into globals."""
    global _shm_vals, _shm_ref
    _shm_vals = np.frombuffer(vals_data, dtype=vals_dtype).reshape(vals_shape)
    _shm_ref  = np.frombuffer(ref_data,  dtype=ref_dtype ).reshape(ref_shape)


def _dominated_chunk_shm(args):
    """
    Worker: check which rows of _shm_vals[indices] are dominated by _shm_ref.
    Only indices (an int array) are pickled — not the data.
    """
    indices = args
    points  = _shm_vals[indices]
    ref     = _shm_ref
    out = np.empty(len(points), dtype=bool)
    for j, p in enumerate(points):
        out[j] = np.any(
            np.all(ref <= p, axis=1) & np.any(ref < p, axis=1)
        )
    return out


def _dominated_by_any(args):
    """
    Fallback worker (no shared memory): used for the crowding step and
    any call where the ref array is small enough to pickle cheaply.
    """
    chunk_vals, ref_vals = args
    out = np.empty(len(chunk_vals), dtype=bool)
    for j, p in enumerate(chunk_vals):
        out[j] = np.any(
            np.all(ref_vals <= p, axis=1) & np.any(ref_vals < p, axis=1)
        )
    return out


def _crowding_distance_col(args):
    """Crowding distance contribution for one objective column."""
    col_vals, n = args
    crowding = np.zeros(n)
    order = np.argsort(col_vals)
    crowding[order[0]] = np.inf
    crowding[order[-1]] = np.inf
    for k in range(1, len(order) - 1):
        crowding[order[k]] += col_vals[order[k + 1]] - col_vals[order[k - 1]]
    return crowding


# ------------------------------------------------------------------ #
# Core helpers                                                        #
# ------------------------------------------------------------------ #

def _extract_one_front(values: np.ndarray,
                       active_idx: np.ndarray,
                       chunk_size: int,
                       n_workers: int) -> np.ndarray:
    """
    Extract one non-dominated front from values[active_idx].

    Uses a fresh Pool per front whose workers hold `active_vals` in their
    process memory via the initializer — only index arrays are pickled
    per task, eliminating the dominant IPC bottleneck.

    Strategy
    --------
    1.  Start with all active points as candidates.
    2.  Sweep through chunks:
          a. Check each live point in the chunk against the current
             frontier; mark dominated ones False.
          b. Re-check earlier survivors against the new chunk survivors.
    3.  Between steps a and b the shared ref array is updated by
        re-initialising the pool — cheap because active_vals never changes.

    Returns the global indices of the non-dominated points.
    """
    active_vals     = values[active_idx]          # local contiguous copy
    n_active        = len(active_vals)
    is_pareto_local = np.ones(n_active, dtype=bool)

    # Pre-serialise active_vals once as raw bytes for the pool initializer
    vals_bytes = active_vals.tobytes()
    vals_shape = active_vals.shape
    vals_dtype = active_vals.dtype

    def _make_pool(ref_vals: np.ndarray) -> mp.Pool:
        """Spawn a pool whose workers share active_vals and ref_vals."""
        return mp.Pool(
            n_workers,
            initializer=_pool_init,
            initargs=(vals_shape, vals_dtype, vals_bytes,
                      ref_vals.shape, ref_vals.dtype, ref_vals.tobytes()),
        )

    chunks = [(i, min(i + chunk_size, n_active))
              for i in range(0, n_active, chunk_size)]

    for start, end in chunks:
        live_in_chunk = is_pareto_local[start:end]
        if not np.any(live_in_chunk):
            continue

        current_ref  = active_vals[is_pareto_local]
        live_indices = np.where(live_in_chunk)[0]                 # local to chunk
        chunk_global = (start + live_indices)                     # indices into active_vals

        # --- step a: dominated check for this chunk ---
        sub_size = max(1, -(-len(chunk_global) // n_workers))     # ceiling div
        tasks    = [chunk_global[i: i + sub_size]
                    for i in range(0, len(chunk_global), sub_size)]

        with _make_pool(current_ref) as pool:
            dom_flags = np.concatenate(pool.map(_dominated_chunk_shm, tasks))

        is_pareto_local[chunk_global[dom_flags]] = False

        # --- step b: re-check earlier survivors against new chunk survivors ---
        new_survivors = active_vals[start:end][is_pareto_local[start:end]]
        if len(new_survivors) == 0:
            continue

        earlier_mask        = is_pareto_local.copy()
        earlier_mask[start:end] = False
        earlier_idx         = np.where(earlier_mask)[0]
        if len(earlier_idx) == 0:
            continue

        sub_size2 = max(1, -(-len(earlier_idx) // n_workers))
        tasks2    = [earlier_idx[i: i + sub_size2]
                     for i in range(0, len(earlier_idx), sub_size2)]

        with _make_pool(new_survivors) as pool:
            dom_flags2 = np.concatenate(pool.map(_dominated_chunk_shm, tasks2))

        is_pareto_local[earlier_idx[dom_flags2]] = False

    return active_idx[is_pareto_local]


def _crowding_sample(sub_df: pd.DataFrame,
                     columns: list,
                     n_needed: int,
                     n_workers: int,
                     pool: mp.Pool) -> pd.DataFrame:
    """Return up to n_needed rows from sub_df, selected by crowding distance."""
    if len(sub_df) <= n_needed:
        return sub_df

    vals   = sub_df[columns].values.astype(np.float64)
    mins   = vals.min(axis=0)
    maxs   = vals.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    normed = (vals - mins) / ranges

    col_tasks     = [(normed[:, c], len(sub_df)) for c in range(normed.shape[1])]
    col_crowdings = list(pool.imap(_crowding_distance_col, col_tasks))
    crowding      = np.sum(col_crowdings, axis=0)
    top_indices   = np.argsort(crowding)[::-1][:n_needed]
    return sub_df.iloc[top_indices].reset_index(drop=True)


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def extract_pareto_front_sample(
    df: pd.DataFrame,
    columns: list,
    n_sample: int = 1000,
    minimize: bool = True,
    chunk_size: int = 5000,
    n_workers: int = None,
    first_front_only: bool = False,
) -> pd.DataFrame:
    """
    Iteratively extract non-dominated fronts (NSGA-II rank assignment)
    until n_sample points are collected.

    For each front that fits entirely within the remaining budget it is
    kept whole.  The last front that would exceed the budget is trimmed
    by crowding distance so that exactly n_sample points are returned.

    A ``pareto_rank`` column (1 = best front) is added to the result.

    Args:
        df:          Input DataFrame.
        columns:     Objective columns to optimise.
        n_sample:    Target number of points to return.
        minimize:    True -> minimise all objectives; False -> maximise.
        chunk_size:  Rows per dominated-check task (tune for memory/speed).
        n_workers:   CPU workers (None = all cores).

    Returns:
        DataFrame with up to n_sample points, sorted by pareto_rank.
    """
    df = df.reset_index(drop=True).copy()
    if n_workers is None:
        n_workers = mp.cpu_count()

    print(f"Using {n_workers} workers | chunk_size={chunk_size} | target={n_sample}")

    values = df[columns].values.astype(np.float64)
    if not minimize:
        values = -values

    collected: list = []
    remaining_idx   = np.arange(len(df))
    front_rank      = 1
    total_so_far    = 0

    pbar = tqdm(desc="Fronts extracted", unit="front")
    while total_so_far < n_sample and len(remaining_idx) > 0:
        n_needed = n_sample - total_so_far
        print(f"\n-- Front {front_rank} | pool={len(remaining_idx)} "
              f"| still need={n_needed}")

        front_idx  = _extract_one_front(
            values, remaining_idx, chunk_size, n_workers
        )
        front_size = len(front_idx)
        print(f"   Front {front_rank} size: {front_size}")

        front_df                = df.iloc[front_idx].copy()
        front_df["pareto_rank"] = front_rank

        if front_size <= n_needed:
            collected.append(front_df.reset_index(drop=True))
            total_so_far  += front_size
            remaining_idx  = remaining_idx[~np.isin(remaining_idx, front_idx)]
            print(f"   Kept all {front_size}  (total: {total_so_far})")
            if first_front_only:
                print("   Stopping after first front (first_front_only=True).")
                break
        else:
            print(f"   Trimming to {n_needed} via crowding distance ...")
            with mp.Pool(n_workers) as pool:
                trimmed = _crowding_sample(
                    front_df, columns, n_needed, n_workers, pool
                )
            collected.append(trimmed)
            total_so_far += len(trimmed)
            print(f"   Kept {len(trimmed)}  (total: {total_so_far})")
            break

        front_rank += 1
        pbar.update(1)
    pbar.close()

    result = pd.concat(collected, ignore_index=True)
    print(f"\nDone. {len(result)} points across "
          f"{result['pareto_rank'].nunique()} front(s).")
    return result


MIN_LOGP, MAX_LOGP = 2.56, 5.03
MIN_MW, MAX_MW = 473, 676
MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS = 34, 48


all_docking = pd.read_csv(data_dir + "docking/all_canonicalized.csv")
smiles_to_affinity = all_docking.set_index('SMILES')['lowest_mode1_affinity']

all_comb = pd.read_csv(data_dir + "combinatorial/all_canonicalized.csv") 
all_comb = all_comb.drop_duplicates(subset="SMILES", keep='first').reset_index(drop=True)
all_comb = all_comb[~all_comb['SMILES'].str.contains('IH')].copy()
all_comb.rename(columns={'logP': 'LogP', 'num_heavy_atoms': 'NumHeavyAtoms'}, inplace=True)

all_comb_hva = all_comb[all_comb['NumHeavyAtoms'].between(MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS)].copy()
all_comb_filtered = all_comb[
    (all_comb['NumHeavyAtoms'].between(MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS)) &
    (all_comb['MW'].between(MIN_MW, MAX_MW)) &
    (all_comb['LogP'].between(MIN_LOGP, MAX_LOGP))
].copy()

rl1 = pd.read_csv(data_dir + "latent_rl/output/rl1/results_canonicalized.csv")
rl2 = pd.read_csv(data_dir + "latent_rl/output/rl2/results_canonicalized.csv")
rl3 = pd.read_csv(data_dir + "latent_rl/output/rl3/results_canonicalized.csv")
all_rl = pd.concat([rl1, rl2, rl3], ignore_index=True).drop_duplicates(subset="SMILES", keep='first').reset_index(drop=True)
all_rl = all_rl[~all_rl['SMILES'].str.contains('IH')].copy()
all_rl = all_rl.dropna()

all_rl_hassubstructure = all_rl[all_rl['HasSubstructure'] == True].copy()
all_rl_hva = all_rl_hassubstructure[all_rl_hassubstructure['NumHeavyAtoms'].between(MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS)].copy()

all_rl_filtered = all_rl[
    (all_rl['HasSubstructure'] == True) &
    (all_rl['NumHeavyAtoms'].between(MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS)) &
    (all_rl['MW'].between(MIN_MW, MAX_MW)) &
    (all_rl['LogP'].between(MIN_LOGP, MAX_LOGP))
].copy()

rl_docking1 = pd.read_csv(data_dir + "latent_rl/output/rl_docking1/results_canonicalized.csv")
rl_docking2 = pd.read_csv(data_dir + "latent_rl/output/rl_docking2/results_canonicalized.csv")
rl_docking3 = pd.read_csv(data_dir + "latent_rl/output/rl_docking3/results_canonicalized.csv")
all_rl_docking = pd.concat([rl_docking1, rl_docking2, rl_docking3], ignore_index=True).drop_duplicates(subset="SMILES", keep='first').reset_index(drop=True)
all_rl_docking = all_rl_docking[~all_rl_docking['SMILES'].str.contains('IH')].copy()
all_rl_docking = all_rl_docking.dropna()

all_rl_docking_hassubstructure = all_rl_docking[all_rl_docking['HasSubstructure'] == True].copy()
all_rl_docking_hva = all_rl_docking_hassubstructure[all_rl_docking_hassubstructure['NumHeavyAtoms'].between(MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS)].copy()

all_rl_docking_filtered = all_rl_docking[
    (all_rl_docking['HasSubstructure'] == True) &
    (all_rl_docking['NumHeavyAtoms'].between(MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS)) &
    (all_rl_docking['MW'].between(MIN_MW, MAX_MW)) &
    (all_rl_docking['LogP'].between(MIN_LOGP, MAX_LOGP))
].copy()

top1k_comb = all_comb_filtered.assign(score=all_comb_filtered['LogP'] + all_comb_filtered['SA']).sort_values(by='score').head(1000)
top1k_rl = all_rl_filtered.assign(score=all_rl_filtered['LogP'] + all_rl_filtered['SA']).sort_values(by='score').head(1000)

all_comb['lowest_mode1_affinity'] = all_comb['SMILES'].map(smiles_to_affinity)
all_comb_filtered['lowest_mode1_affinity'] = all_comb_filtered['SMILES'].map(smiles_to_affinity)
all_comb_hva['lowest_mode1_affinity'] = all_comb_hva['SMILES'].map(smiles_to_affinity)
all_rl['lowest_mode1_affinity'] = all_rl['SMILES'].map(smiles_to_affinity)
all_rl_hassubstructure['lowest_mode1_affinity'] = all_rl_hassubstructure['SMILES'].map(smiles_to_affinity)
all_rl_hva['lowest_mode1_affinity'] = all_rl_hva['SMILES'].map(smiles_to_affinity)
all_rl_filtered['lowest_mode1_affinity'] = all_rl_filtered['SMILES'].map(smiles_to_affinity)
all_rl_docking['lowest_mode1_affinity'] = all_rl_docking['SMILES'].map(smiles_to_affinity)
all_rl_docking_hassubstructure['lowest_mode1_affinity'] = all_rl_docking_hassubstructure['SMILES'].map(smiles_to_affinity)
all_rl_docking_hva['lowest_mode1_affinity'] = all_rl_docking_hva['SMILES'].map(smiles_to_affinity)
all_rl_docking_filtered['lowest_mode1_affinity'] = all_rl_docking_filtered['SMILES'].map(smiles_to_affinity)
top1k_comb['lowest_mode1_affinity'] = top1k_comb['SMILES'].map(smiles_to_affinity)
top1k_rl['lowest_mode1_affinity'] = top1k_rl['SMILES'].map(smiles_to_affinity)

all_comb = all_comb.dropna(subset=['lowest_mode1_affinity'])
all_comb_filtered = all_comb_filtered.dropna(subset=['lowest_mode1_affinity'])
all_comb_hva = all_comb_hva.dropna(subset=['lowest_mode1_affinity'])
all_rl = all_rl.dropna(subset=['lowest_mode1_affinity'])
all_rl_hassubstructure = all_rl_hassubstructure.dropna(subset=['lowest_mode1_affinity'])
all_rl_hva = all_rl_hva.dropna(subset=['lowest_mode1_affinity'])
all_rl_filtered = all_rl_filtered.dropna(subset=['lowest_mode1_affinity'])
all_rl_docking = all_rl_docking.dropna(subset=['lowest_mode1_affinity'])
all_rl_docking_hassubstructure = all_rl_docking_hassubstructure.dropna(subset=['lowest_mode1_affinity'])
all_rl_docking_hva = all_rl_docking_hva.dropna(subset=['lowest_mode1_affinity'])
all_rl_docking_filtered = all_rl_docking_filtered.dropna(subset=['lowest_mode1_affinity'])
top1k_comb = top1k_comb.dropna(subset=['lowest_mode1_affinity'])
top1k_rl = top1k_rl.dropna(subset=['lowest_mode1_affinity'])

print(f"all_comb: {len(all_comb):,}")
print(f"all_comb_filtered: {len(all_comb_filtered):,}")
print(f"all_comb_hva: {len(all_comb_hva):,}\n")

print(f"all_rl: {len(all_rl):,}")
print(f"all_rl_filtered: {len(all_rl_filtered):,}")
print(f"all_rl_hassubstructure: {len(all_rl_hassubstructure):,}")
print(f"all_rl_hva: {len(all_rl_hva):,}\n")

print(f"all_rl_docking: {len(all_rl_docking):,}")
print(f"all_rl_docking_filtered: {len(all_rl_docking_filtered):,}")
print(f"all_rl_docking_hassubstructure: {len(all_rl_docking_hassubstructure):,}")
print(f"all_rl_docking_hva: {len(all_rl_docking_hva):,}\n")


all_comb_pareto_nodock = extract_pareto_front_sample(all_comb_hva, columns=["LogP", "MW", "SA"], n_sample=1000, minimize=True)
all_rl_pareto_nodock = extract_pareto_front_sample(all_rl_hva, columns=["LogP", "MW", "SA"], n_sample=1000, minimize=True)

all_comb_pareto_nodock.to_csv(data_dir + "pareto/pareto_comb_1k_nodock.csv", index=False)
all_rl_pareto_nodock.to_csv(data_dir + "pareto/pareto_rl_1k_nodock.csv", index=False)

all_comb_pareto_first_front_docking = extract_pareto_front_sample(all_comb_hva, columns=["LogP", "MW", "SA", "lowest_mode1_affinity"], minimize=True, first_front_only=True)
all_rl_pareto_first_front_docking = extract_pareto_front_sample(all_rl_hva, columns=["LogP", "MW", "SA", "lowest_mode1_affinity"], minimize=True, first_front_only=True)
all_rl_docking_pareto_first_front_docking = extract_pareto_front_sample(all_rl_docking_hva, columns=["LogP", "MW", "SA", "lowest_mode1_affinity"], minimize=True, first_front_only=True)

all_comb_pareto_first_front_docking.to_csv(data_dir + "pareto/pareto_comb_1stfront_docking.csv", index=False)
all_rl_pareto_first_front_docking.to_csv(data_dir + "pareto/pareto_rl_1stfront_docking.csv", index=False)
all_rl_docking_pareto_first_front_docking.to_csv(data_dir + "pareto/pareto_rl_docking_1stfront_docking.csv", index=False)