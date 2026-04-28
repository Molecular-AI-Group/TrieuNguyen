import os
import json
import argparse
import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from optimize_pdb import mol_to_3d_pdb, collect_smiles

# ── Paths ────────────────────────────────────────────────────────────────────
PYTHONSH = "/home/80027464/LatentRL/tools/AutoDock/mgltools_x86_64Linux2_1.5.7/bin/pythonsh"
PREPARE_LIGAND = "/home/80027464/LatentRL/tools/AutoDock/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
VINA_BIN = "/home/80027464/LatentRL/tools/AutoDock/autodock_vina_1_1_2_linux_x86/bin/vina"
DEFAULT_RECEPTOR = "/home/80027464/LatentRL/tools/AutoDock/mols/protein_no_MG.pdbqt"
DEFAULT_CONFIG = "/home/80027464/LatentRL/tools/AutoDock/mols/config_ori_25.txt"


# ── Step 1: SMILES → PDB (parallelisable) ───────────────────────────────────
def _step1_worker(args_tuple):
    """Convert a single SMILES to a 3D PDB file."""
    idx, smi, pdb_path = args_tuple
    status = mol_to_3d_pdb(smi, pdb_path)
    return idx, smi, status


# ── Step 2: PDB → PDBQT (parallelisable) ────────────────────────────────────
def _step2_worker(args_tuple):
    """Convert a single PDB file to PDBQT."""
    idx, smi, pdb_path, pdbqt_path = args_tuple
    result = subprocess.run(
        [PYTHONSH, PREPARE_LIGAND, "-l", pdb_path, "-o", pdbqt_path],
        capture_output=True, text=True,
    )
    success = result.returncode == 0
    return idx, smi, success


# ── Step 3: Docking (sequential, all CPUs per molecule) ─────────────────────
def run_vina(ligand_pdbqt: str, output_pdbqt: str, log_file: str,
             receptor: str = DEFAULT_RECEPTOR,
             config: str = DEFAULT_CONFIG,
             num_cpus: int = 1,
             smiles: str = None,
             json_file: str = None) -> dict | None:
    """Run AutoDock Vina with all CPUs and return results keyed by mode."""
    result = subprocess.run(
        [VINA_BIN,
         "--ligand", ligand_pdbqt,
         "--receptor", receptor,
         "--config", config,
         "--out", output_pdbqt,
         "--log", log_file],
    )
    if result.returncode != 0:
        return None

    results = {"SMILES": smiles}
    with open(log_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].isdigit():
                mode = int(parts[0])
                results[mode] = {
                    "affinity": float(parts[1]),
                    "rmsd_lb": float(parts[2]),
                    "rmsd_ub": float(parts[3]),
                }

    if json_file:
        with open(json_file, "w") as jf:
            json.dump(results, jf, indent=2)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: SMILES → 3D PDB → PDBQT → AutoDock Vina docking."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="A SMILES string, a .txt file (one SMILES per line), or a .csv with a SMILES column.",
    )
    parser.add_argument(
        "-o", "--output", default="data/docking",
        help="Output directory (default: data/docking/<timestamp>).",
    )
    parser.add_argument(
        "-r", "--receptor", default=DEFAULT_RECEPTOR,
        help="Path to receptor PDBQT file.",
    )
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG,
        help="Path to Vina config file.",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=os.cpu_count(),
        help="Number of parallel workers for Steps 1 & 2 (default: all CPU cores).",
    )
    args = parser.parse_args()

    num_cpus = args.jobs or os.cpu_count()

    # Collect SMILES
    smiles_list = collect_smiles(args.input)
    if not smiles_list:
        print("No valid SMILES found in the input.")
        return

    # Prepare output directory
    out_dir = os.path.join(args.output, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    total = len(smiles_list)
    pad = len(str(total - 1))

    # Build path maps
    paths = {}
    for idx, smi in enumerate(smiles_list):
        prefix = str(idx).zfill(pad)
        paths[idx] = {
            "smi": smi,
            "pdb": os.path.join(out_dir, f"{prefix}.pdb"),
            "pdbqt": os.path.join(out_dir, f"{prefix}.pdbqt"),
            "dock_out": os.path.join(out_dir, f"{prefix}_out.pdbqt"),
            "log": os.path.join(out_dir, f"{prefix}.log"),
            "json": os.path.join(out_dir, f"{prefix}.json"),
        }

    # ── Step 1: SMILES → PDB (parallel) ─────────────────────────────────────
    print(f"Step 1/3: Converting {total} SMILES → PDB with {num_cpus} workers…")
    step1_ok = set()
    step1_tasks = [(idx, smi, paths[idx]["pdb"]) for idx, smi in enumerate(smiles_list)]

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(_step1_worker, t): t for t in step1_tasks}
        for future in as_completed(futures):
            idx, smi, status = future.result()
            if status == "ok":
                step1_ok.add(idx)
                print(f"  [{idx:>{pad}}] PDB OK       {smi}")
            else:
                print(f"  [{idx:>{pad}}] PDB FAILED ({status})  {smi}")

    print(f"  → {len(step1_ok)}/{total} succeeded\n")

    if not step1_ok:
        print("No molecules passed Step 1. Exiting.")
        return

    # ── Step 2: PDB → PDBQT (parallel) ──────────────────────────────────────
    print(f"Step 2/3: Converting {len(step1_ok)} PDB → PDBQT with {num_cpus} workers…")
    step2_ok = set()
    step2_tasks = [
        (idx, paths[idx]["smi"], paths[idx]["pdb"], paths[idx]["pdbqt"])
        for idx in sorted(step1_ok)
    ]

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(_step2_worker, t): t for t in step2_tasks}
        for future in as_completed(futures):
            idx, smi, success = future.result()
            if success:
                step2_ok.add(idx)
                print(f"  [{idx:>{pad}}] PDBQT OK     {smi}")
            else:
                print(f"  [{idx:>{pad}}] PDBQT FAILED {smi}")

    print(f"  → {len(step2_ok)}/{len(step1_ok)} succeeded\n")

    if not step2_ok:
        print("No molecules passed Step 2. Exiting.")
        return

    # ── Step 3: Docking (sequential, all CPUs per molecule) ──────────────────
    print(f"Step 3/3: Docking {len(step2_ok)} molecule(s) sequentially (using {num_cpus} CPUs each)…")
    ok_count = 0
    failed_count = 0

    for idx in sorted(step2_ok):
        p = paths[idx]
        smi = p["smi"]
        dock_results = run_vina(
            p["pdbqt"], p["dock_out"], p["log"],
            receptor=args.receptor, config=args.config,
            num_cpus=num_cpus,
            smiles=smi,
            json_file=p["json"],
        )
        if dock_results:
            ok_count += 1
            best = dock_results[1]["affinity"]
            print(f"  [{idx:>{pad}}] DOCK OK  (best: {best:>7.2f} kcal/mol)  {smi}")
        else:
            failed_count += 1
            print(f"  [{idx:>{pad}}] DOCK FAILED  {smi}")

    print(f"\nDone. {ok_count}/{total} molecules docked successfully.")
    print(f"Results saved to '{out_dir}/'.")


if __name__ == "__main__":
    main()