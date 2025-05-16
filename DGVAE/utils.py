import re
import os
import json
import torch
import codecs
import random
import argparse
import datetime
import numpy as np
import os.path as op
import torch.nn.functional as F
from tqdm import tqdm 
from rdkit import Chem 
from model.main import TGVAEv2
from argparse import Namespace
from SmilesPE.tokenizer import SPE_Tokenizer 
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence



# Pretraining functions
def get_config() : 
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='experiment_1')

    # Model hyperparameters
    parser.add_argument('-tr', '--train', type=str, default='moses.txt')
    parser.add_argument('-de', '--dim_encoder', type=int, default=512)
    parser.add_argument('-dd', '--dim_decoder', type=int, default=512)
    parser.add_argument('-dl', '--dim_latent', type=int, default=256)
    parser.add_argument('-def', '--dim_encoder_ff', type=int, default=512)
    parser.add_argument('-ddf', '--dim_decoder_ff', type=int, default=512)
    parser.add_argument('-nel', '--num_encoder_layer', type=int, default=4)
    parser.add_argument('-ndl', '--num_decoder_layer', type=int, default=4)
    parser.add_argument('-neh', '--num_encoder_head', type=int, default=8)
    parser.add_argument('-ndh', '--num_decoder_head', type=int, default=16)
    parser.add_argument('-doe', '--dropout_encoder', type=float, default=0.3)
    parser.add_argument('-dog', '--dropout_gat', type=float, default=0.3)
    parser.add_argument('-dod', '--dropout_decoder', type=float, default=0.3)

    # Training hyperparameters
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-m', '--model', type=str, default='tgvaev2')
    parser.add_argument('-tk', '--tokenizer', type=str, default='char')
    parser.add_argument('-se', '--save_every', type=int, default=1)
    parser.add_argument('-ge', '--generate_every', type=int, default=1)
    parser.add_argument('-ss', '--start_save', type=int, default=5)
    parser.add_argument('-sg', '--start_generate', type=int, default=5)

    # Loss function hyperparameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6)
    parser.add_argument('-gc', '--gradient_clipping', type=float, default=5.0)
    
    # Annealing hyperparameters
    parser.add_argument('-aes', '--anneal_epoch_start', type=int, default=0)
    parser.add_argument('-aws', '--anneal_weight_start', type=float, default=0.00005)
    parser.add_argument('-awe', '--anneal_weight_end', type=float, default=1.0)

    config = parser.parse_args()

    assert config.tokenizer in ['char', 'pe'], f'Tokenizer {config.tokenizer} is not supported. Use "char" or "pe".'
    assert config.model in ['tgvaev2', 'simple'], f'Model {config.model} is not supported. Use "tgvaev2" or "simple".'

    config.trained_epoch = 0


    config.script_dir = op.dirname(op.abspath(__file__))
    config.checkpoint_dir = op.join(config.script_dir, 'checkpoint', config.name)
    config.data_dir = op.join(config.script_dir, 'data')
    config.generate_dir = op.join(config.data_dir, 'generate', config.name)


    if op.exists(op.join(config.checkpoint_dir, 'config.json')) : 
        print(f'Resume training of training named {config.name}')
        return Namespace(**load(op.join(config.checkpoint_dir, 'config.json'))) # Load previous config 
    else : 
        return config

def get_dataset(config) : 
    """
    Loads or processes a molecular dataset for training.

    This function attempts to load preprocessed dataset files (SMILES vocabulary, graph vocabulary,
    max token length, and the dataset itself) from a directory based on the training file name and tokenizer.
    If these files are not found, it processes the raw SMILES data from scratch using the specified tokenizer
    (either 'default' for character-level or 'pe' for positional encoding), converts it into a graph-based
    format compatible with PyTorch Geometric, and saves the processed outputs for future use.

    Args:
        config (Namespace): Configuration object with attributes:
            - data_dir (str): Base directory for data.
            - train (str): File name of the raw training data.
            - tokenizer (str): Tokenizer type ('char' or 'pe').

    Returns:
        tuple: (dataset, max_token, vocab_smi, vocab_graph)
            - dataset (list): List of graph-formatted molecular data.
            - max_token (int): Maximum token length found in SMILES.
            - vocab_smi (dict): Vocabulary mapping for SMILES tokens.
            - vocab_graph (dict): Vocabulary mapping for graph components.
    """
    processed_dir = op.join(config.data_dir, 'processed', f'{op.splitext(config.train)[0]}_{config.tokenizer}')
    try : 
        # Load preprocessed data
        vocab_smi = load(op.join(processed_dir, 'vocab_smi.json')); print(f'\nLoaded vocab_smi.json from {processed_dir}')
        vocab_graph = load(op.join(processed_dir, 'vocab_graph.json')); print(f'Loaded vocab_graph.json from {processed_dir}')
        max_token = load(op.join(processed_dir, 'max_token.pt')); print(f'Loaded max_token.pt from {processed_dir}')
        dataset = load(op.join(processed_dir, 'data.pt')); print(f'Loaded data.pt from {processed_dir}\n')

    except (FileNotFoundError, IOError) as e:
        print(f'\nPreprocessed data not found: {e}. \nProcessing raw data...\n') 

        # Load raw SMILES data
        raw_smi = read_smi(op.join(config.data_dir, 'raw', config.train))

        # Process raw data
        smi, node_feature, edge_index, edge_attr, vocab_smi, vocab_graph, max_token = process_data(raw_smi,
                                                                                                   tokenizer=char_tokenizer if config.tokenizer == 'char' else pe_tokenizer)
        
        # Convert to PyTorch Geometric Data format
        dataset = [
            CustomData(x=nf, edge_index=ei, edge_attr=ea, smi=s)
            for nf, ei, ea, s in zip(node_feature, edge_index, edge_attr, smi)
        ]

        # Save processed data

        os.makedirs(processed_dir, exist_ok=True)
        save(dataset, op.join(processed_dir, 'data.pt')); print(f'\nSaved data.pt to {processed_dir}')
        save(max_token, op.join(processed_dir, 'max_token.pt')); print(f'Saved max_token.pt to {processed_dir}')
        save(vocab_smi, op.join(processed_dir, 'vocab_smi.json')); print(f'Saved vocab_smi.json to {processed_dir}')
        save(vocab_graph, op.join(processed_dir, 'vocab_graph.json')); print(f'Saved vocab_graph.json to {processed_dir}\n')

    # Update config for future reference
    config = vars(config) 
    config['vocab_smi'] = vocab_smi
    config['vocab_graph'] = vocab_graph
    config['max_token'] = max_token

    return Namespace(**config), dataset, max_token, vocab_smi, vocab_graph

def get_model(config, device, snapshot=None) :
    model = TGVAEv2(dim_encoder=config.dim_encoder,
                    dim_decoder=config.dim_decoder,
                    dim_encoder_ff=config.dim_encoder_ff,
                    dim_latent=config.dim_latent,
                    dim_decoder_ff=config.dim_decoder_ff,
                    num_encoder_layer=config.num_encoder_layer,
                    num_decoder_layer=config.num_decoder_layer,
                    num_encoder_head=config.num_encoder_head,
                    num_decoder_head=config.num_decoder_head,
                    dropout_encoder=config.dropout_encoder,
                    dropout_gat=config.dropout_gat,
                    dropout_decoder=config.dropout_decoder,
                    size_graph_vocab=len(config.vocab_graph),
                    size_smi_vocab=len(config.vocab_smi),
                    model=config.model,
                    device=device).to(device)

    if snapshot : # For generate.py
        model.load_state_dict(snapshot['MODEL_STATE'])
        return model
    
    if config.trained_epoch == 0 : 
        optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else : 
        snapshot = load(op.join(config.checkpoint_dir, f'snapshot_{config.trained_epoch}.pt'))
        model.load_state_dict(snapshot['MODEL_STATE'])
        optim = torch.optim.Adam(model.parameters())
        optim.load_state_dict(snapshot['OPTIMIZER_STATE'])
        
    annealer = monotonic_annealer(config.epoch, 
                                    config.anneal_epoch_start,
                                    config.anneal_weight_start,
                                    config.anneal_weight_end)

    return model, optim, annealer


# Loss functions
def recon_loss_fn(pred, tgt, vocab) :
    return F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['[PAD]'])

def kl_loss_fn(mu, sigma) : 
    return -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())

def loss_fn(out, tgt, beta, config) :
    pred, mu, sigma = out
    return recon_loss_fn(pred, tgt, config.vocab_smi) + beta * kl_loss_fn(mu, sigma)

def monotonic_annealer(n_epoch, epoch_start, weight_start, weight_end):
    inc = (weight_end - weight_start) / (n_epoch - epoch_start)
    annealing_weights = []
    for i in range(n_epoch):
        k = (i - epoch_start) if i >= epoch_start else 0
        annealing_weights.append(weight_start + k * inc)
    return annealing_weights

# Data processing functions
def process_smi(smi, vocab, tokenizer) : 
    out = []
    for t in tokenizer(smi) : 
        if t not in vocab : vocab[t] = len(vocab)
        out.append(vocab[t])
    out = [vocab['[START]']] + out + [vocab['[END]']]
    return torch.tensor(out, dtype=torch.long)

def process_graph(smi, graph_vocab, edge_vocab) : 
    mol = Chem.MolFromSmiles(smi)
    node_feature, edge_index, edge_attr = [], [], []

    for atom in mol.GetAtoms() : 
        symbol = atom.GetSymbol() 
        if symbol not in graph_vocab : graph_vocab[symbol] = len(graph_vocab)
        node_feature.append(graph_vocab[symbol])

    for bond in mol.GetBonds() : 
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[b, e], [e, b]]
        edge_attr += [edge_vocab[str(bond.GetBondType())]] * 2

    node_feature = torch.tensor(node_feature)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().view(2, -1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    return node_feature, edge_index, edge_attr

def process_data(data, tokenizer) : 
    smi_vocab = {'[START]': 0, '[END]': 1, '[PAD]': 2}
    edge_vocab = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
    graph_vocab = {}

    smi_list = []
    node_feature_list, edge_index_list, edge_attr_list = [], [], []

    for smi in tqdm(data, desc='Processing data') : 

        tokenized_smi = process_smi(smi, smi_vocab, tokenizer)
        node_feature, edge_index, edge_attr = process_graph(smi, graph_vocab, edge_vocab)

        smi_list.append(tokenized_smi)
        node_feature_list.append(node_feature)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)

    smi_list = pad_sequence(smi_list, batch_first=True, padding_value=smi_vocab['[PAD]'])

    return smi_list, node_feature_list, edge_index_list, edge_attr_list, smi_vocab, graph_vocab, smi_list.shape[1]

def decode_data(data, smi_vocab, device=None) : 
    inp_graph, smi = data.to(device), data.smi.to(device) 
    inp_smi, tgt_smi = smi[:, :-1], smi[:, 1:]
    inp_smi_mask = get_mask(inp_smi, smi_vocab)
    return inp_graph, inp_smi, inp_smi_mask.to(device), tgt_smi

def token2smi(token, inv_vocab_smi) : 
    smiles = ''.join([inv_vocab_smi[t] for t in token])
    smiles = re.sub(r"\[START\]|\[PAD\]|\[END\]", "", smiles)
    return smiles 

def convert_token(token, vocab_smi) : 
    inv_vocab_smi = {v:k for k, v in vocab_smi.items()}
    token = token.tolist() 
    smiles = [token2smi(t, inv_vocab_smi) for t in token]
    return smiles

def read_smi(path, delimiter='\t', titleLine=False) : 
    result = [] 
    if path.endswith('.txt'): 
        with open(path, 'r') as f : 
            for smi in tqdm(f.readlines(), desc='Reading SMILES') : 
                if Chem.MolFromSmiles(smi) is not None : 
                    result.append(smi.strip())
    elif path.endswith('.sdf') : 
        supplier = Chem.SDMolSupplier(path)
        for mol in tqdm(supplier, desc='Reading SMILES') : 
            if mol is None : 
                continue 
            result.append(Chem.MolToSmiles(mol))
    elif path.endswith('.smi'): 
        supplier = Chem.SmilesMolSupplier(path, delimiter=delimiter, titleLine=titleLine)
        for mol in tqdm(supplier, desc='Reading SMILES') : 
            if mol is None : 
                continue 
            result.append(Chem.MolToSmiles(mol))
    return result

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def get_mask(target, smi_vocab) :
        mask = (target != smi_vocab['[PAD]']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)

class CustomData(Data) : 
    def __cat_dim__(self, key, value, *args, **kwargs) : 
        if key == 'smi' :
            return None 
        return super().__cat_dim__(key, value, *args, **kwargs) 

# Tokenizer functions
def char_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smi)]
    assert smi == ''.join(tokens), ("{} could not be joined".format(smi))
    return tokens

def pe_tokenizer(smi) : 
    spe_vocab = codecs.open('smilespe.txt') 
    spe = SPE_Tokenizer(spe_vocab) 
    token = spe.tokenize(smi)
    return token.split(' ')


# Other functions
def save(data, path, mode='w') : 
    if path.endswith('.pt') :
        torch.save(data, path)
    elif path.endswith('.json') :
        with open(path, 'w') as f : 
            json.dump(data, f, indent=4)
    elif path.endswith('.txt') : 
        with open(path, mode) as f : 
            for line in data : 
                f.write(line+'\n')

def load(file) : 
    if file.endswith('.pt') : 
        return torch.load(file)
    elif file.endswith('.json') :
        with open(file, 'r') as f : 
            return json.load(f)
    elif file.endswith('.txt') :
        with open(file, 'r') as f : 
            return [line.strip() for line in f.readlines()]

def checkpoint(config, model, optim, epoch) : 
    snapshot = {'MODEL_STATE': model.state_dict(), 'OPTIMIZER_STATE': optim.state_dict()}
    config.trained_epoch = epoch
    save(snapshot, op.join(config.checkpoint_dir, f'snapshot_{epoch}.pt'))
    save(vars(config), op.join(config.checkpoint_dir, 'config.json'))

def set_seed(seed):
    random.seed(seed)  # Set Python's random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # Set PyTorch's CUDA seed (single GPU)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if using multi-GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_molecule(config, model, epoch, num_gen=30000, batch=500) : 
    model.eval()
    with torch.no_grad() :
        for _ in tqdm(range(num_gen // batch), desc=f'Generating epoch {epoch}') : 
            smi_token = model.generate(config, num_gen=batch)
            smi = convert_token(smi_token, config.vocab_smi)
            save(smi, op.join(config.generate_dir, f'epoch_{epoch}.txt'), mode='a')

def print_config(config) :
    print(f'\nName: {config.name}')
    print('=' * 20 + 'MODEL CONFIG' + '=' * 22)
    print('\nEncoder: ')
    print(f'\t dim_encoder: {config.dim_encoder}')
    print(f'\t dim_encoder_ff: {config.dim_encoder_ff}')
    print(f'\t dim_latent: {config.dim_latent}')
    print(f'\t num_encoder_head: {config.num_encoder_head}')
    print(f'\t num_encoder_layer: {config.num_encoder_layer}')
    print(f'\t dropout_encoder: {config.dropout_encoder}')
    print(f'\t dropout_gat: {config.dropout_gat}')
    print('Decoder: ')
    print(f'\t dim_decoder: {config.dim_decoder}')
    print(f'\t dim_decoder_ff: {config.dim_decoder_ff}')
    print(f'\t num_decoder_head: {config.num_decoder_head}')
    print(f'\t num_decoder_layer: {config.num_decoder_layer}')
    print(f'\t dropout_decoder: {config.dropout_decoder}\n')
    print(f'=' * 20 + 'TRAINING CONFIG' + '=' * 20)
    print(f'\n\t batch: {config.batch}')
    print(f'\t epoch: {config.epoch}')
    print(f'\t model: {config.model}')
    print(f'\t tokenizer: {config.tokenizer}')
    print(f'\t start_save: {config.start_save}')
    print(f'\t save_every: {config.save_every}')
    print(f'\t start_generate: {config.start_generate}')
    print(f'\t generate_every: {config.generate_every}\n')
    print(f'=' * 20 + 'LOSS CONFIG' + '=' * 24)
    print(f'\n\t learning_rate: {config.learning_rate}')
    print(f'\t weight_decay: {config.weight_decay}')
    print(f'\t gradient_clipping: {config.gradient_clipping}\n')
    print(f'=' * 20 + 'ANNEALING CONFIG' + '=' * 19)
    print(f'\n\t anneal_epoch_start: {config.anneal_epoch_start}')
    print(f'\t anneal_weight_start: {config.anneal_weight_start}')
    print(f'\t anneal_weight_end: {config.anneal_weight_end}\n')
    print('=' * 50)