import argparse
import copy
import json
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Optional
import sys
import subprocess

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.rdmolops import CombineMols
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# -----------------------------
# Property filter helpers
# -----------------------------
def _property_worker(
    smiles: str,
    min_logp: Optional[float],
    max_logp: Optional[float],
    min_mw: Optional[float],
    max_mw: Optional[float],
    min_num_heavy_atoms: Optional[int],
    max_num_heavy_atoms: Optional[int],
) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Compute only required properties
    if (min_logp is not None) or (max_logp is not None):
        logp = Crippen.MolLogP(mol)
        if (min_logp is not None and logp < min_logp) or (max_logp is not None and logp > max_logp):
            return None

    if (min_mw is not None) or (max_mw is not None):
        mw = Descriptors.MolWt(mol)
        if (min_mw is not None and mw < min_mw) or (max_mw is not None and mw > max_mw):
            return None

    if (min_num_heavy_atoms is not None) or (max_num_heavy_atoms is not None):
        n_heavy = mol.GetNumHeavyAtoms()
        if (min_num_heavy_atoms is not None and n_heavy < min_num_heavy_atoms) or (
            max_num_heavy_atoms is not None and n_heavy > max_num_heavy_atoms
        ):
            return None

    return Chem.MolToSmiles(mol, canonical=True)


def _property_worker_from_tuple(args) -> Optional[str]:
    return _property_worker(*args)


def _apply_property_filters_parallel(
    smiles_list: List[str],
    min_logp: Optional[float],
    max_logp: Optional[float],
    min_mw: Optional[float],
    max_mw: Optional[float],
    min_num_heavy_atoms: Optional[int],
    max_num_heavy_atoms: Optional[int],
    n_jobs: int,
    chunksize: int = 500,
) -> List[str]:
    if not smiles_list:
        return []

    args_iter = [
        (
            s,
            min_logp,
            max_logp,
            min_mw,
            max_mw,
            min_num_heavy_atoms,
            max_num_heavy_atoms,
        )
        for s in smiles_list
    ]

    if n_jobs == 1:
        kept = [
            _property_worker(*a)
            for a in tqdm(args_iter, total=len(args_iter), desc="Property filter", unit="mol")
        ]
    else:
        kept = []
        with Pool(processes=n_jobs) as pool:
            for result in tqdm(
                pool.imap_unordered(_property_worker_from_tuple, args_iter, chunksize=chunksize),
                total=len(args_iter),
                desc="Property filter (parallel)",
                unit="mol",
            ):
                kept.append(result)

    out = [s for s in kept if s is not None]
    # keep deterministic unique order
    return list(dict.fromkeys(out))


def _has_any_property_filter(args) -> bool:
    return any(
        v is not None
        for v in [
            args.min_logp,
            args.max_logp,
            args.min_mw,
            args.max_mw,
            args.min_num_heavy_atoms,
            args.max_num_heavy_atoms,
        ]
    )

# -----------------------------
# Visualization helpers
# -----------------------------
def idx_annotate(x):
    if isinstance(x, str):
        x = Chem.MolFromSmiles(x)
    if x is None:
        return None
    mol = copy.deepcopy(x)
    for idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(idx).SetProp("molAtomMapNumber", str(idx))
    return mol


def save_core_index_image(core_smiles: str, out_path: Path = Path("core_index.png")) -> Path:
    mol = idx_annotate(core_smiles)
    if mol is None:
        raise ValueError("Invalid core SMILES for visualization.")
    img = Draw.MolToImage(mol, size=(900, 700))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return out_path


# -----------------------------
# I/O helpers
# -----------------------------
def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_smiles_from_sdf(path: Path) -> List[str]:
    supplier = Chem.SDMolSupplier(str(path))
    smiles = []
    for mol in supplier:
        if mol is None:
            continue
        s = Chem.MolToSmiles(mol)
        if s:
            smiles.append(s)
    return smiles


def _normalize_one_smiles(args) -> Optional[str]:
    s, remove_multicomponent = args
    s = s.strip()
    if not s:
        return None
    if "*" in s:
        return None
    if "[IH]" in s.upper():
        return None

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None

    can = Chem.MolToSmiles(mol, canonical=True)
    if remove_multicomponent and "." in can:
        return None
    return can


def _normalize_smiles_list(
    smiles: Iterable[str],
    remove_multicomponent: bool = True,
    n_jobs: int = 1,
    chunksize: int = 1000,
) -> List[str]:
    smiles_list = list(smiles)
    if not smiles_list:
        return []

    args_iter = [(s, remove_multicomponent) for s in smiles_list]

    if n_jobs <= 1:
        normalized = [
            _normalize_one_smiles(a)
            for a in tqdm(args_iter, total=len(args_iter), desc="Normalizing SMILES", unit="mol")
        ]
    else:
        with Pool(processes=n_jobs) as pool:
            normalized = list(
                tqdm(
                    pool.imap(_normalize_one_smiles, args_iter, chunksize=chunksize),
                    total=len(args_iter),
                    desc="Normalizing SMILES (parallel)",
                    unit="mol",
                )
            )

    out = [s for s in normalized if s is not None]
    # deduplicate while preserving order
    return list(dict.fromkeys(out))


def load_smiles_input(input_value: str, normalize_n_jobs: int = 1) -> List[str]:
    """
    Accepts:
      - Raw SMILES string
      - Path to .txt / .smi / .sdf
    """
    p = Path(input_value)
    if p.exists():
        suffix = p.suffix.lower()
        if suffix in {".txt", ".smi"}:
            return _normalize_smiles_list(_read_lines(p), n_jobs=normalize_n_jobs)
        if suffix == ".sdf":
            return _normalize_smiles_list(_load_smiles_from_sdf(p), n_jobs=normalize_n_jobs)
        raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .smi, or .sdf")

    # otherwise treat as raw SMILES
    return _normalize_smiles_list([input_value], n_jobs=normalize_n_jobs)


def load_many_sources(values: List[str], normalize_n_jobs: int = 1) -> List[str]:
    all_smiles = []
    for v in values:
        all_smiles.extend(load_smiles_input(v, normalize_n_jobs=normalize_n_jobs))
    # deduplicate while preserving order
    seen = set()
    dedup = []
    for s in all_smiles:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def write_smiles(path: Path, smiles: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in smiles:
            f.write(s + "\n")


# -----------------------------
# Chemistry helpers (from notebook logic)
# -----------------------------
def auto_add(core, block, core_atom_idx: Optional[int] = None) -> List[str]:
    if isinstance(core, str):
        core = Chem.MolFromSmiles(core)
    if isinstance(block, str):
        block = Chem.MolFromSmiles(block)

    if core is None or block is None:
        return []

    combo = CombineMols(core, block)
    output = []

    core_indices = [core_atom_idx] if core_atom_idx is not None else list(range(core.GetNumAtoms()))
    block_start = core.GetNumAtoms()
    block_end = combo.GetNumAtoms()

    for i in core_indices:
        if i < 0 or i >= core.GetNumAtoms():
            continue
        for j in range(block_start, block_end):
            for b in (
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
            ):
                editable = Chem.EditableMol(combo)
                editable.AddBond(i, j, order=b)
                try:
                    m = editable.GetMol()
                    Chem.SanitizeMol(m)
                    output.append(Chem.MolToSmiles(m, canonical=True))
                except Exception:
                    pass

    # deduplicate while preserving order
    return list(dict.fromkeys(output))


def combine_core_with_building_blocks(
    cores: List[str],
    building_blocks: List[str],
    core_atom_idx: Optional[int] = None,
) -> List[str]:
    out: Set[str] = set()

    # Progress over building blocks (more informative when cores is small, often 1)
    for bb in tqdm(building_blocks, total=len(building_blocks), desc="Combining building blocks", unit="bb"):
        for core in cores:
            out.update(auto_add(core, bb, core_atom_idx=core_atom_idx))

    return sorted(out)


# -----------------------------
# Split helpers
# -----------------------------
def split_dataset_by_ratio(
    data: List[str],
    train_ratio: float,
    test_ratio: float,
    seed: Optional[int] = 42,
    label: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Returns: (train, test)
    Randomized split with optional seed.
    """
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train-ratio and test-ratio must be > 0.")
    if abs((train_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError(f"train-ratio + test-ratio must sum to 1 (got {train_ratio} + {test_ratio}).")

    unique = list(dict.fromkeys(data))

    if seed is not None:
        random.seed(seed)
    random.shuffle(unique)

    n = len(unique)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    # absorb any rounding remainder into test
    if n_train + n_test < n:
        n_test += (n - (n_train + n_test))

    train = unique[:n_train]
    test = unique[n_train : n_train + n_test]

    if label:
        print(f"  [{label}] total={n}  train={len(train)} ({train_ratio:.3f})  test={len(test)} ({test_ratio:.3f})")

    return train, test


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create upsampling data from core SMILES and building blocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--core",
        nargs="+",
        required=True,
        help="Core input(s): raw SMILES or .txt/.smi/.sdf path(s). Can pass multiple.",
    )
    p.add_argument(
        "--pretrain",
        nargs="+",
        required=True,
        help="Base pretraining data source(s): .txt/.smi/.sdf or raw SMILES.",
    )
    p.add_argument(
        "--building-blocks",
        nargs="+",
        required=True,
        help="Building block source(s): .txt/.smi/.sdf or raw SMILES. Can pass multiple.",
    )
    p.add_argument(
        "--core-atom-idx",
        type=int,
        default=None,
        help="Optional fixed core atom index for attachment. If omitted, script prompts interactively.",
    )

    # ── Pretrain split ratios ──────────────────────────────────────────────────
    pretrain_grp = p.add_argument_group(
        "Pretrain split ratios",
        "Controls train/test split for the base pretraining set.",
    )
    pretrain_grp.add_argument(
        "--pretrain-train-ratio",
        type=float,
        default=0.98,
        help="Train fraction for the pretrain set (must sum to 1 with --pretrain-test-ratio).",
    )
    pretrain_grp.add_argument(
        "--pretrain-test-ratio",
        type=float,
        default=0.02,
        help="Test fraction for the pretrain set (must sum to 1 with --pretrain-train-ratio).",
    )

    # ── Building-block split ratios ────────────────────────────────────────────
    bb_grp = p.add_argument_group(
        "Building-block split ratios",
        "Controls train/test split for the combinatorial (building-block derived) molecules.",
    )
    bb_grp.add_argument(
        "--bb-train-ratio",
        type=float,
        default=0.8,
        help="Train fraction for the combinatorial set (must sum to 1 with --bb-test-ratio).",
    )
    bb_grp.add_argument(
        "--bb-test-ratio",
        type=float,
        default=0.2,
        help="Test fraction for the combinatorial set (must sum to 1 with --bb-train-ratio).",
    )

    # ── Random seed ───────────────────────────────────────────────────────────
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for both splits. Set to -1 to disable (non-deterministic).",
    )

    p.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name. Outputs will be written to: <outdir>/<name>/",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="data/pretrain",
        help="Base output directory.",
    )

    # ── Property filters ──────────────────────────────────────────────────────
    filt_grp = p.add_argument_group(
        "Property filters",
        "Optional molecular property constraints applied to the merged set before splitting.",
    )
    filt_grp.add_argument("--min-logp", type=float, default=None, help="Minimum logP.")
    filt_grp.add_argument("--max-logp", type=float, default=None, help="Maximum logP.")
    filt_grp.add_argument("--min-mw", type=float, default=None, help="Minimum molecular weight.")
    filt_grp.add_argument("--max-mw", type=float, default=None, help="Maximum molecular weight.")
    filt_grp.add_argument("--min-num-heavy-atoms", type=int, default=None, help="Minimum heavy atom count.")
    filt_grp.add_argument("--max-num-heavy-atoms", type=int, default=None, help="Maximum heavy atom count.")

    # ── Parallelism ───────────────────────────────────────────────────────────
    p.add_argument(
        "--num-workers",
        type=int,
        default=max(cpu_count() - 1, 1),
        help="Parallel workers for optional property filtering.",
    )
    p.add_argument(
        "--normalize-workers",
        type=int,
        default=max(cpu_count() - 1, 1),
        help="Parallel workers for SMILES normalization/canonicalization.",
    )

    # ── Vocab extraction ──────────────────────────────────────────────────────
    p.add_argument(
        "--extract-vocab-tokenizer",
        type=str,
        default="spe",
        help="Tokenizer type for vocabulary extraction.",
    )
    p.add_argument(
        "--extract-vocab-spe-path",
        type=str,
        default="data/raw/vocab/SPE_ChEMBL.txt",
        help="Path to the sentencepiece vocabulary file.",
    )
    p.add_argument(
        "--extract-vocab-chunk-size",
        type=int,
        default=10000,
        help="Chunk size for vocabulary extraction.",
    )
    p.add_argument(
        "--extract-vocab-workers",
        type=int,
        default=None,
        help="Number of workers for vocabulary extraction.",
    )
    return p


def _prompt_core_atom_idx(core_smiles: str, outdir: Path) -> int:
    image_path = save_core_index_image(core_smiles, outdir / "core_index.png")
    print(f"Core index visualization saved to: {image_path.resolve()}")
    print("Open this image, inspect atom map numbers, then choose index.")
    while True:
        user_in = input("Enter core atom index to attach building blocks: ").strip()
        try:
            idx = int(user_in)
            mol = Chem.MolFromSmiles(core_smiles)
            if mol is None:
                raise ValueError("Invalid core SMILES.")
            if 0 <= idx < mol.GetNumAtoms():
                return idx
            print(f"Index out of range. Valid range: 0..{mol.GetNumAtoms()-1}")
        except ValueError:
            print("Please enter a valid integer index.")


def _validate_ratio_pair(train_ratio: float, test_ratio: float, label: str) -> None:
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError(f"[{label}] Both train and test ratios must be > 0.")
    if abs((train_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError(
            f"[{label}] train-ratio ({train_ratio}) + test-ratio ({test_ratio}) must sum to 1.0."
        )


def main() -> None:
    args = build_argparser().parse_args()


    # ── Validate ratio pairs early ─────────────────────────────────────────────
    _validate_ratio_pair(args.pretrain_train_ratio, args.pretrain_test_ratio, "pretrain")
    _validate_ratio_pair(args.bb_train_ratio, args.bb_test_ratio, "building-blocks")

    seed: Optional[int] = None if args.seed == -1 else args.seed

    outdir = Path(args.outdir) / args.name
    outdir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.outdir) / args.name / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    # ── Load core first, then prompt atom index early ──────────────────────────
    cores = load_many_sources(args.core, normalize_n_jobs=max(1, args.normalize_workers))
    if not cores:
        raise ValueError("No valid core SMILES were loaded.")

    core_atom_idx = args.core_atom_idx
    if core_atom_idx is None:
        core_atom_idx = _prompt_core_atom_idx(cores[0], outdir)

    # Validate provided index against first core
    first_core_mol = Chem.MolFromSmiles(cores[0])
    if first_core_mol is None:
        raise ValueError("First core SMILES is invalid after normalization.")
    if not (0 <= core_atom_idx < first_core_mol.GetNumAtoms()):
        raise ValueError(
            f"core_atom_idx={core_atom_idx} out of range for first core "
            f"(0..{first_core_mol.GetNumAtoms()-1})."
        )

    # ── Load heavy inputs ──────────────────────────────────────────────────────
    pretrain = load_many_sources(args.pretrain, normalize_n_jobs=max(1, args.normalize_workers))
    building_blocks = load_many_sources(args.building_blocks, normalize_n_jobs=max(1, args.normalize_workers))

    if not pretrain:
        raise ValueError("No valid pretraining SMILES were loaded.")
    if not building_blocks:
        raise ValueError("No valid building block SMILES were loaded.")

    # ── Apply property filters to pretrain set ─────────────────────────────────
    pretrain_filtered_out = 0
    if _has_any_property_filter(args):
        before_n = len(pretrain)
        pretrain = _apply_property_filters_parallel(
            smiles_list=pretrain,
            min_logp=args.min_logp,
            max_logp=args.max_logp,
            min_mw=args.min_mw,
            max_mw=args.max_mw,
            min_num_heavy_atoms=args.min_num_heavy_atoms,
            max_num_heavy_atoms=args.max_num_heavy_atoms,
            n_jobs=max(1, args.num_workers),
            chunksize=500,
        )
        pretrain_filtered_out = before_n - len(pretrain)

    # ── Generate combinatorial molecules from building blocks ──────────────────
    combinatorial = combine_core_with_building_blocks(
        cores=cores,
        building_blocks=building_blocks,
        core_atom_idx=core_atom_idx,
    )

    # Apply property filters to combinatorial set
    combinatorial_filtered_out = 0
    if _has_any_property_filter(args):
        before_n = len(combinatorial)
        combinatorial = _apply_property_filters_parallel(
            smiles_list=combinatorial,
            min_logp=args.min_logp,
            max_logp=args.max_logp,
            min_mw=args.min_mw,
            max_mw=args.max_mw,
            min_num_heavy_atoms=args.min_num_heavy_atoms,
            max_num_heavy_atoms=args.max_num_heavy_atoms,
            n_jobs=max(1, args.num_workers),
            chunksize=500,
        )
        combinatorial_filtered_out = before_n - len(combinatorial)

    # ── Split pretrain and combinatorial sets independently ────────────────────
    print("\nSplitting datasets independently:")
    pretrain_train, pretrain_test = split_dataset_by_ratio(
        pretrain,
        train_ratio=args.pretrain_train_ratio,
        test_ratio=args.pretrain_test_ratio,
        seed=seed,
        label="pretrain",
    )
    combinatorial_train, combinatorial_test = split_dataset_by_ratio(
        combinatorial,
        train_ratio=args.bb_train_ratio,
        test_ratio=args.bb_test_ratio,
        seed=seed,
        label="combinatorial",
    )

    # ── Merge splits into final train / test sets ──────────────────────────────
    # Deduplicate within each final split while preserving order.
    # Pretrain comes first so combinatorial molecules that appear in pretrain are
    # counted in the pretrain partition rather than the combinatorial one.
    train = list(dict.fromkeys(pretrain_train + combinatorial_train))
    test  = list(dict.fromkeys(pretrain_test  + combinatorial_test))
    random.shuffle(train), random.shuffle(test)
    
    # ── Write outputs ──────────────────────────────────────────────────────────
    write_smiles(outdir / "core_loaded.smi", cores)
    write_smiles(outdir / "pretrain_loaded.smi", pretrain)
    write_smiles(outdir / "building_blocks_loaded.smi", building_blocks)
    write_smiles(outdir / "combinatorial_generated.smi", combinatorial)
    write_smiles(outdir / "pretrain_train.smi", pretrain_train)
    write_smiles(outdir / "pretrain_test.smi", pretrain_test)
    write_smiles(outdir / "combinatorial_train.smi", combinatorial_train)
    write_smiles(outdir / "combinatorial_test.smi", combinatorial_test)
    write_smiles(outdir / "train.txt", train)
    write_smiles(outdir / "test.txt", test)

    # ── Vocabulary extraction ──────────────────────────────────────────────────
    extract_script = Path(__file__).resolve().parent / "tools" / "script" / "extract_vocab.py"
    if extract_script.exists():
        cmd = [
            sys.executable,
            str(extract_script),
            str(outdir),
            "--tokenizer",
            args.extract_vocab_tokenizer,
            "--spe-vocab-path",
            args.extract_vocab_spe_path,
            "--chunk-size",
            str(args.extract_vocab_chunk_size),
        ]
        if args.extract_vocab_workers is not None:
            cmd.extend(["--workers", str(args.extract_vocab_workers)])

        print("\nRunning vocabulary extraction/merge...")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        raise FileNotFoundError(f"extract_vocab.py not found at {extract_script}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Dataset: {args.name}")
    print(f"{'='*60}")
    print(f"  Input cores:                {len(cores)}")
    print(f"  Selected core atom index:   {core_atom_idx}")
    print(f"  Normalization workers:      {max(1, args.normalize_workers)}")
    print()
    print(f"  Pretrain set")
    print(f"    Raw loaded:               {len(pretrain) + pretrain_filtered_out}")
    if _has_any_property_filter(args):
        print(f"    Filtered out:             {pretrain_filtered_out}")
    print(f"    After filtering:          {len(pretrain)}")
    print(f"    Train split:              {len(pretrain_train)}  (ratio={args.pretrain_train_ratio})")
    print(f"    Test split:               {len(pretrain_test)}  (ratio={args.pretrain_test_ratio})")
    print()
    print(f"  Combinatorial (building-block derived)")
    print(f"    Building blocks loaded:   {len(building_blocks)}")
    print(f"    Generated molecules:      {len(combinatorial) + combinatorial_filtered_out}")
    if _has_any_property_filter(args):
        print(f"    Filtered out:             {combinatorial_filtered_out}")
    print(f"    After filtering:          {len(combinatorial)}")
    print(f"    Train split:              {len(combinatorial_train)}  (ratio={args.bb_train_ratio})")
    print(f"    Test split:               {len(combinatorial_test)}  (ratio={args.bb_test_ratio})")
    print()
    print(f"  Final merged sets (after dedup across sources)")
    print(f"    Train:                    {len(train)}")
    print(f"    Test:                     {len(test)}")
    print(f"{'='*60}")
    print(f"  Saved to: {outdir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()