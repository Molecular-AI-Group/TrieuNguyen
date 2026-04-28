import os
import re 
import csv
import json
import torch
import random 
import codecs
import datetime
import numpy as np
from typing import List, Tuple
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, RDConfig
from SmilesPE.tokenizer import SPE_Tokenizer
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

RDLogger.DisableLog("rdApp.*")

TOKEN_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|_|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
TOKEN_REGEX = re.compile(TOKEN_PATTERN)


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def normal_tokenizer(smi: str) -> List[str]:
    tokens = [t for t in TOKEN_REGEX.findall(smi)]
    if smi != "".join(tokens):
        raise ValueError(f"SMILES could not be rejoined: {smi}")
    return tokens

def spe_tokenizer(smi):
    spe_vocab = codecs.open("data/raw/vocab/SPE_ChEMBL.txt")
    spe = SPE_Tokenizer(spe_vocab)
    token = spe.tokenize(smi)
    return token.split(" ")

def mol2graph(mol: Chem.rdchem.Mol) -> Tuple[List[str], List[List[int]], List[str]]:
    x, edge_index, edge_attr = [], [], []
    for atom in mol.GetAtoms():
        x.append(atom.GetSymbol())
    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[b, e], [e, b]]
        edge_attr += [str(bond.GetBondType())] * 2
    return x, edge_index, edge_attr

def seed_everything(seed: int = 42, determinism=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # speed
    if determinism:
        torch.use_deterministic_algorithms(True)


def vae_encode(model, data, device="cuda"):
    x = data.x.to(device, non_blocking=True)
    edge_index = data.edge_index.to(device, non_blocking=True)
    edge_attr = data.edge_attr.to(device, non_blocking=True)
    batch = data.batch.to(device, non_blocking=True)
    pool = model.encoder(x, edge_index, edge_attr, batch)
    z, _, _ = model.latent_model(pool)
    return z

def vae_decode(model, z_batch, device="cuda"):
    with torch.no_grad():
        batch_size = z_batch.shape[0]
        smiles = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
        for _ in range(model.args["max_len"] - 1):
            next_token = model.step(z_batch, smiles)
            smiles = torch.cat([smiles, next_token], dim=-1)
        return model.translate(smiles)


def read_input(path):
    with open(path, "r") as f:
        return f.readline().strip()

def get_mol(smi):
    return Chem.MolFromSmiles(smi)

def get_mw(x):
    if isinstance(x, str):
        x = get_mol(x)
    return Descriptors.MolWt(x) if x is not None else None

def get_logp(x):
    if isinstance(x, str):
        x = get_mol(x)
    return Descriptors.MolLogP(x) if x is not None else None

def get_sa(x): 
    if isinstance(x, str):
        x = get_mol(x)
    return sascorer.calculateScore(x) if x is not None else None

def get_hbd(x):
    if isinstance(x, str):
        x = get_mol(x)
    return Descriptors.NumHDonors(x) if x is not None else None

def get_hba(x):
    if isinstance(x, str):
        x = get_mol(x)
    return Descriptors.NumHAcceptors(x) if x is not None else None

def get_num_heavy_atoms(x):
    if isinstance(x, str):
        x = get_mol(x)
    return x.GetNumHeavyAtoms() if x is not None else None

def get_morgan_fp(x, radius=2, nBits=2048):
    if isinstance(x, str):
        x = get_mol(x)
    return AllChem.GetMorganFingerprintAsBitVect(x, radius, nBits)

def get_sim(x, y):
    if isinstance(x, str):
        x = get_mol(x)
    if isinstance(y, str):
        y = get_mol(y)
    return DataStructs.TanimotoSimilarity(get_morgan_fp(x), get_morgan_fp(y))

def has_substructure_match(x, substruct):
    if isinstance(x, str):
        x = get_mol(x)
    substruct = get_mol(substruct)
    return x.HasSubstructMatch(substruct) 




