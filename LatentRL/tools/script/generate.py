import random
import numpy as np
import argparse
from tqdm import tqdm
from rdkit import Chem, RDLogger
from model.dgvae import DGVAE
from utils import get_sa, get_logp, get_mw, get_num_heavy_atoms
import seaborn as sns
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

def generate(model, N=10_000, batch_size=1_000):
    gen_smiles = []

    model.eval()
    for _ in tqdm(range(N // batch_size), desc=f"Generating", position=1, leave=False):
        gen_smiles += model.generate(N=batch_size)

    valid_smiles = [x for x in gen_smiles if Chem.MolFromSmiles(x) != None]
    unique_smiles = list(set(gen_smiles))
    validity, uniqueness = (
        len(valid_smiles) / len(gen_smiles),
        len(unique_smiles) / len(gen_smiles),
    )
    return gen_smiles, valid_smiles, validity, uniqueness

def compute_props(smiles_list):
    logp, sa, mw, n_heavy = [], [], [], []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue

        logp.append(get_logp(mol))
        sa.append(get_sa(mol))
        mw.append(get_mw(mol))
        n_heavy.append(get_num_heavy_atoms(mol))

    return {
        "logp": np.array(logp),
        "sa": np.array(sa),
        "mw": np.array(mw),
        "n_heavy": np.array(n_heavy),
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument('--training_data', type=str, default=None)
    parser.add_argument("--num_gen", "-N", type=int, default=30_000)
    parser.add_argument("--batch", "-B", type=int, default=1_000)
    args = parser.parse_args()

    model = DGVAE.load(args.path).to("cuda")

    gen_smiles, valid_smiles, validity, uniqueness = generate(model, N=args.num_gen, batch_size=args.batch)
    print(f"validity: {validity}, uniqueness: {uniqueness}")

    # If training data is provided plot distribution comparison 
    if args.training_data is not None:
        props = np.load(args.training_data)
        train_logp, train_sa, train_mw, train_n_heavy = props['logp'], props['sa'], props['mw'], props['n_heavy']

        # Compute generated molecule properties
        gen_props = compute_props(valid_smiles)
        gen_logp, gen_sa, gen_mw, gen_n_heavy = gen_props['logp'], gen_props['sa'], gen_props['mw'], gen_props['n_heavy']

        # Plot 2x2 KDE distribution comparison
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # logP
        ax = axes[0, 0]
        sns.kdeplot(train_logp, ax=ax, label="Train", fill=True, alpha=0.3)
        sns.kdeplot(gen_logp, ax=ax, label="Generated", fill=True, alpha=0.3)
        ax.set_title("logP")
        ax.legend()

        # SA score
        ax = axes[0, 1]
        sns.kdeplot(train_sa, ax=ax, label="Train", fill=True, alpha=0.3)
        sns.kdeplot(gen_sa, ax=ax, label="Generated", fill=True, alpha=0.3)
        ax.set_title("SA Score")
        ax.legend()

        # Molecular Weight
        ax = axes[1, 0]
        sns.kdeplot(train_mw, ax=ax, label="Train", fill=True, alpha=0.3)
        sns.kdeplot(gen_mw, ax=ax, label="Generated", fill=True, alpha=0.3)
        ax.set_title("Molecular Weight")
        ax.legend()

        # Heavy atoms
        ax = axes[1, 1]
        sns.kdeplot(train_n_heavy, ax=ax, label="Train", fill=True, alpha=0.3)
        sns.kdeplot(gen_n_heavy, ax=ax, label="Generated", fill=True, alpha=0.3)
        ax.set_title("Number of Heavy Atoms")
        ax.legend()

        plt.tight_layout()
        plt.savefig("kde_train_vs_generated.png", dpi=300)


