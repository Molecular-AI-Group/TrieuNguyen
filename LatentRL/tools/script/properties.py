import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed
from tqdm import tqdm

# SA score requires this import (part of rdkit contrib)
from rdkit.Chem import RDConfig
import sys
import os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_CSV   = "/home/80027464/LatentRL/data/latent_rl/output/MBC_docking/global_dock_cache.csv"          # <-- change to your CSV path
OUTPUT_CSV  = "/home/80027464/LatentRL/data/latent_rl/output/MBC_docking/global_dock_cache_properties.csv"
SMILES_COL  = "SMILES"                 # <-- change to your SMILES column name
# ────────────────────────────────────────────────────────────────────────────

def compute_properties(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "MW": None, "logP": None, "HBD": None,
            "HBA": None, "HeavyAtoms": None, "SA": None,
            "AromaticRings": None, "SaturatedRings": None, "RotatableBonds": None,
        }
    return {
        "MW":             round(Descriptors.MolWt(mol), 3),
        "logP":           round(Descriptors.MolLogP(mol), 3),
        "HBD":            rdMolDescriptors.CalcNumHBD(mol),
        "HBA":            rdMolDescriptors.CalcNumHBA(mol),
        "HeavyAtoms":     mol.GetNumHeavyAtoms(),
        "SA":             round(sascorer.calculateScore(mol), 3),
        "AromaticRings":  rdMolDescriptors.CalcNumAromaticRings(mol),
        "SaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from '{INPUT_CSV}'")

    if SMILES_COL not in df.columns:
        raise ValueError(f"Column '{SMILES_COL}' not found. Available: {df.columns.tolist()}")

    tqdm.pandas(desc="Computing properties")
    props = df[SMILES_COL].progress_apply(compute_properties).apply(pd.Series)

    smiles_idx = df.columns.get_loc(SMILES_COL) + 1
    for i, col in enumerate(props.columns):
        df.insert(smiles_idx + i, col, props[col])

    invalid = df["MW"].isna().sum()
    if invalid:
        print(f"Warning: {invalid} SMILES could not be parsed (properties set to None)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()