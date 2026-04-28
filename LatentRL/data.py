import torch
import random
import torch_geometric
import torch.nn.functional as F
from utils import * 
from collections import deque
from torch.utils.data import Dataset, IterableDataset, get_worker_info

class SMILESGraphData(torch_geometric.data.Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "smiles":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

class CustomDataset(IterableDataset): 
    def __init__(self, path, vocab_cache_path, latent_rl=False): 
        super().__init__() 
        self._path = path 
        self._latent_rl = latent_rl
        self._vocab_cache = load_json(vocab_cache_path)

        self._smiles_vocab = self._vocab_cache['smiles_vocab']
        self._node_vocab = self._vocab_cache['node_vocab']
        self._edge_vocab = self._vocab_cache['edge_vocab'] 
        self._max_len = self._vocab_cache['max_len']


    def _line_stream(self):
        with open(self._path, "r") as f:
            for line in f:
                yield line.rstrip("\n")

    def _encode_graph(self, graph):
        x, edge_index, edge_attr = graph

        x = F.one_hot(
            torch.tensor([self._node_vocab.get(atom, self._node_vocab["UNK"]) for atom in x]),
            num_classes=len(self._node_vocab)
        ).float()

        edge_attr = F.one_hot(
            torch.tensor([self._edge_vocab[b] for b in edge_attr]),
            num_classes=len(self._edge_vocab)
        ).float()

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return x, edge_index, edge_attr

    def _encode_tokens(self, tokens):
        seq = ["[start]"] + tokens + ["[end]"]
        seq += ["[pad]"] * (self._max_len - len(seq))
        ids = [self._smiles_vocab[s] for s in seq]
        return torch.tensor(ids, dtype=torch.long)

    def _transformation(self, smiles: str): 
        mol = get_mol(smiles) 

        if mol is None:
            return None 

        # Tokenize
        if not self._latent_rl:
            tokens = self._encode_tokens(spe_tokenizer(smiles))
        else: 
            tokens = None

        # Extract graph
        graph = mol2graph(mol)
        x, edge_index, edge_attr = self._encode_graph(graph)

        return SMILESGraphData(
            x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=tokens
        )

    def __iter__(self):
        worker = get_worker_info()
        stream = self._line_stream()
        if worker is None:
            it = stream
        else:
            def shard():
                for i, item in enumerate(stream):
                    if i % worker.num_workers == worker.id:
                        yield item

            it = shard()

        for x in it:
            data = self._transformation(x)
            if data is None:
                continue  
            yield data


class ShuffledIterableDataset(CustomDataset):
    def __init__(self, path, vocab_cache_path, buffer_size=50000):
        super().__init__(path, vocab_cache_path)
        self.buffer_size = int(buffer_size)

    def __iter__(self):
        worker = get_worker_info()
        base_stream = self._line_stream()
        if worker is None:
            stream = base_stream
            rng = random.Random()
        else:
            wid, nworkers = worker.id, worker.num_workers

            def shard():
                for i, line in enumerate(base_stream):
                    if (i % nworkers) == wid:
                        yield line

            stream = shard()
            rng = random.Random(12345 + wid)  

        buffer = deque()

        for line in stream:
            data = self._transformation(line)
            if data is None:
                continue
            buffer.append(data)

            if len(buffer) >= self.buffer_size:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = buffer[-1]
                buffer.pop()

        while buffer:
            yield buffer.popleft()