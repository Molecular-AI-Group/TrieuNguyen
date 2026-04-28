import re
import pickle
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
from rdkit import Chem
from SmilesPE.tokenizer import SPE_Tokenizer
import codecs

TOKEN_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|_|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
TOKEN_REGEX = re.compile(TOKEN_PATTERN)

# Global tokenizer configuration
_tokenizer_config = None

def init_worker(tokenizer_type, spe_vocab_path):
    """Initialize worker process with tokenizer configuration"""
    global _tokenizer_config
    _tokenizer_config = {
        'type': tokenizer_type,
        'spe_vocab_path': spe_vocab_path,
        'spe_tokenizer': None
    }
    
    if tokenizer_type == 'spe' and spe_vocab_path:
        spe_vocab = codecs.open(spe_vocab_path)
        _tokenizer_config['spe_tokenizer'] = SPE_Tokenizer(spe_vocab)

def tokenize_smiles(smi):
    """Tokenize SMILES using the configured tokenizer"""
    global _tokenizer_config
    
    if _tokenizer_config['type'] == 'normal':
        tokens = [t for t in TOKEN_REGEX.findall(smi)]
        if smi != "".join(tokens):
            raise ValueError(f"SMILES could not be rejoined: {smi}")
        return tokens
    elif _tokenizer_config['type'] == 'spe':
        return _tokenizer_config['spe_tokenizer'].tokenize(smi).split(" ")
    else:
        raise ValueError(f"Unknown tokenizer type: {_tokenizer_config['type']}")

def process_chunk(args):
    lines, chunk_id = args
    local_tokens = set()
    local_atoms = set()
    local_edges = set()
    local_max_len = 0
    valid_count = 0
    invalid_count = 0
    
    for line in lines:
        smi = line.strip()
        if not smi:
            continue
        
        try:
            tokens = tokenize_smiles(smi)
            local_tokens.update(tokens)
            seq_len = len(tokens) + 2
            local_max_len = max(local_max_len, seq_len)
            
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Extract atoms
                for atom in mol.GetAtoms():
                    local_atoms.add(atom.GetSymbol())
                
                # Extract edge types
                for bond in mol.GetBonds():
                    bond_type = str(bond.GetBondType())
                    local_edges.add(bond_type)
            
            valid_count += 1
        except (ValueError, Exception):
            invalid_count += 1
            continue
    
    return local_tokens, local_atoms, local_edges, local_max_len, valid_count, invalid_count

def read_file_in_chunks(filepath: str, chunk_size: int = 10000):
    chunk = []
    chunk_id = 0
    
    with open(filepath, 'r', buffering=8192*16) as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield (chunk, chunk_id)
                chunk = []
                chunk_id += 1
        
        if chunk:
            yield (chunk, chunk_id)

def build_vocab_parallel(filepath: str, tokenizer_type: str, spe_vocab_path: str = None, 
                        num_workers: int = None, chunk_size: int = 10000):
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    base_vocab = {"[start]": 0, "[end]": 1, "[pad]": 2}
    all_tokens = set()
    all_atoms = set()
    all_edges = set()
    global_max_len = 0
    total_valid = 0
    total_invalid = 0
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(tokenizer_type, spe_vocab_path)
    ) as executor:
        chunks = [(chunk[0], chunk[1]) for chunk in read_file_in_chunks(filepath, chunk_size)]
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {Path(filepath).name}"):
            tokens, atoms, edges, max_len, valid, invalid = future.result()
            all_tokens.update(tokens)
            all_atoms.update(atoms)
            all_edges.update(edges)
            global_max_len = max(global_max_len, max_len)
            total_valid += valid
            total_invalid += invalid
    
    vocab = base_vocab.copy()
    for token in sorted(all_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    node_vocab = {}
    for atom in sorted(all_atoms):
        node_vocab[atom] = len(node_vocab)
    node_vocab["UNK"] = len(node_vocab)
    
    edge_vocab = {}
    for edge in sorted(all_edges):
        edge_vocab[edge] = len(edge_vocab)
    
    stats = {
        'total_valid': total_valid,
        'total_invalid': total_invalid,
        'vocab_size': len(vocab),
        'node_vocab_size': len(node_vocab),
        'edge_vocab_size': len(edge_vocab),
        'max_len': global_max_len
    }
    
    return vocab, node_vocab, edge_vocab, global_max_len, stats

def merge_vocabularies(*vocab_dicts):
    """Merge multiple vocabularies maintaining base tokens"""
    base_tokens = {"[start]", "[end]", "[pad]"}
    
    # Merge SMILES vocab
    merged_vocab = {"[start]": 0, "[end]": 1, "[pad]": 2}
    all_tokens = set()
    for vocab_dict in vocab_dicts:
        vocab = vocab_dict['vocab']
        all_tokens.update(k for k in vocab.keys() if k not in base_tokens)
    
    for token in sorted(all_tokens):
        if token not in merged_vocab:
            merged_vocab[token] = len(merged_vocab)
    
    # Merge node vocab
    merged_node_vocab = {}
    all_atoms = set()
    for vocab_dict in vocab_dicts:
        node_vocab = vocab_dict['node_vocab']
        all_atoms.update(k for k in node_vocab.keys() if k != "UNK")
    
    for atom in sorted(all_atoms):
        merged_node_vocab[atom] = len(merged_node_vocab)
    merged_node_vocab["UNK"] = len(merged_node_vocab)
    
    # Merge edge vocab
    merged_edge_vocab = {}
    all_edges = set()
    for vocab_dict in vocab_dicts:
        edge_vocab = vocab_dict['edge_vocab']
        all_edges.update(edge_vocab.keys())
    
    for edge in sorted(all_edges):
        merged_edge_vocab[edge] = len(merged_edge_vocab)
    
    return merged_vocab, merged_node_vocab, merged_edge_vocab

def process_dataset(data_folder: str, tokenizer_type: str, spe_vocab_path: str = None, 
                   num_workers: int = None, chunk_size: int = 10000):
    data_folder = Path(data_folder)
    
    # Find all .txt files in the folder
    txt_files = sorted(data_folder.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_folder}")
    
    print(f"Found {len(txt_files)} .txt file(s): {[f.name for f in txt_files]}")
    
    # Process each file
    file_results = {}
    for txt_file in txt_files:
        print(f"\nProcessing {txt_file.name}...")
        vocab, node_vocab, edge_vocab, max_len, stats = build_vocab_parallel(
            str(txt_file), tokenizer_type, spe_vocab_path, num_workers, chunk_size
        )
        file_results[txt_file.stem] = {
            'vocab': vocab,
            'node_vocab': node_vocab,
            'edge_vocab': edge_vocab,
            'max_len': max_len,
            'stats': stats
        }
    
    # Merge all vocabularies
    print("\nMerging vocabularies from all files...")
    unified_vocab, unified_node_vocab, unified_edge_vocab = merge_vocabularies(*file_results.values())
    unified_max_len = max(result['max_len'] for result in file_results.values())
    
    # Prepare stats for each file
    file_stats = {filename: result['stats'] for filename, result in file_results.items()}
    
    vocab_cache = {
        'smiles_vocab': unified_vocab,
        'node_vocab': unified_node_vocab,
        'edge_vocab': unified_edge_vocab,
        'max_len': unified_max_len,
        'tokenizer_type': tokenizer_type,
        'file_stats': file_stats
    }
    
    json_output_path = data_folder / "vocab_cache.json"
    with open(json_output_path, 'w') as f:
        json.dump(vocab_cache, f, indent=2)
    
    print(f"\nVocabulary saved to {json_output_path}")
    print(f"\nUnified vocabularies:")
    print(f"  SMILES vocab size: {len(unified_vocab)}")
    print(f"  Node vocab size: {len(unified_node_vocab)}")
    print(f"  Edge vocab size: {len(unified_edge_vocab)}")
    print(f"  Edge types found: {sorted(unified_edge_vocab.keys())}")
    print(f"  Max sequence length: {unified_max_len}")
    print(f"\nStatistics per file:")
    for filename, stats in file_stats.items():
        print(f"  {filename}.txt: {stats}")
    
    return unified_vocab, unified_node_vocab, unified_edge_vocab, unified_max_len

def main():
    parser = argparse.ArgumentParser(description='Build vocabulary from all .txt files in a folder in parallel')
    parser.add_argument('data_folder', type=str, help='Path to folder containing .txt files')
    parser.add_argument('--tokenizer', type=str, choices=['normal', 'spe'], default='spe', help='Tokenizer type (default: spe)')
    parser.add_argument('--spe-vocab-path', type=str, default='data/vocab/SPE_ChEMBL.txt', help='Path to SPE vocabulary file (default: data/vocab/SPE_ChEMBL.txt)')
    parser.add_argument('--workers', type=int, default=None, help=f'Number of parallel workers (default: {mp.cpu_count()} CPUs)')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Number of lines to process per chunk (default: 10000)')
    
    args = parser.parse_args()
    
    try:
        process_dataset(args.data_folder, args.tokenizer, args.spe_vocab_path, args.workers, args.chunk_size)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())