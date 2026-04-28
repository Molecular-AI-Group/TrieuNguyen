import os
import argparse
import datetime
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem


# 3D embedding parameters (good defaults)
params = AllChem.ETKDGv3()
params.randomSeed = 42
params.useSmallRingTorsions = True
params.useBasicKnowledge = True


def mol_to_3d_pdb(smiles: str, out_path: str) -> str:
    """Generates a 3D-optimised PDB file for *smiles* and writes it to *out_path*.
    Returns a status string: 'ok', 'bad_smiles', or 'embed_failed'.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "bad_smiles"

    mol = Chem.AddHs(mol)

    # Embed 3D
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        # try again with random coords
        cid = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if cid < 0:
            return "embed_failed"

    # Optimize geometry
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        # still write the embedded structure if optimization fails
        pass

    pdb_block = Chem.MolToPDBBlock(mol)
    with open(out_path, "w") as f:
        f.write(pdb_block)
    return "ok"


def collect_smiles(input_arg: str) -> list[str]:
    """Parse *input_arg* and return a list of unique, valid canonical SMILES.

    Accepts:
      - a bare SMILES string
      - a path to a .txt file (one SMILES per line)
      - a path to a .csv file with a column named 'SMILES', 'smiles', or 'Smiles'
    """
    raw: list[str] = []

    if os.path.isfile(input_arg):
        ext = os.path.splitext(input_arg)[1].lower()
        if ext == ".csv":
            with open(input_arg, newline="") as fh:
                reader = csv.DictReader(fh)
                col = None
                for candidate in ("SMILES", "smiles", "Smiles"):
                    if candidate in (reader.fieldnames or []):
                        col = candidate
                        break
                if col is None:
                    raise ValueError(
                        f"CSV file '{input_arg}' has no column named "
                        "'SMILES', 'smiles', or 'Smiles'."
                    )
                for row in reader:
                    raw.append(row[col].strip())
        else:
            # treat as plain text, one SMILES per line
            with open(input_arg) as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        raw.append(s)
    else:
        # treat the argument itself as a single SMILES string
        raw.append(input_arg.strip())

    # deduplicate by canonical SMILES, keeping only valid molecules
    seen: set[str] = set()
    unique_valid: list[str] = []
    for smi in raw:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            unique_valid.append(canon)

    return unique_valid


def _worker(args_tuple):
    """Top-level function required for pickling with ProcessPoolExecutor."""
    idx, smi, filename, pad = args_tuple
    status = mol_to_3d_pdb(smi, filename)
    return idx, smi, filename, status, pad


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D-optimised PDB files from SMILES."
    )
    parser.add_argument(
        "input",
        help=(
            "A SMILES string, a path to a .txt file (one SMILES per line), "
            "or a path to a .csv file with a 'SMILES'/'smiles'/'Smiles' column."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default="pdb_output",
        help="Name of the output folder where PDB files will be saved (default: timestamp).",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: all CPU cores).",
    )
    args = parser.parse_args()

    smiles_list = collect_smiles(args.input)
    if not smiles_list:
        print("No valid SMILES found in the input.")
        return

    args.output = f"{args.output}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.output, exist_ok=True)

    total = len(smiles_list)
    pad = len(str(total - 1))
    ok_count = failed_count = 0

    tasks = [
        (idx, smi, os.path.join(args.output, f"{str(idx).zfill(pad)}.pdb"), pad)
        for idx, smi in enumerate(smiles_list)
    ]

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            idx, smi, filename, status, pad = future.result()
            if status == "ok":
                ok_count += 1
                print(f"[{idx:>{pad}}] OK       -> {filename}")
            else:
                failed_count += 1
                print(f"[{idx:>{pad}}] FAILED ({status}): {smi}")

    print(f"\nDone. {ok_count}/{total} PDB files written to '{args.output}/'.")


if __name__ == "__main__":
    main()
