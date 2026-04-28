import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from model.sublayer import *
from torch.nn import LayerNorm, LeakyReLU, BatchNorm1d
from typing import List


class DGVAE_Encoder(nn.Module):
    def __init__(
        self, dim_encoder, size_graph_vocab, dim_edge, num_head, num_layer, dropout, pool="add"
    ):
        super(DGVAE_Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.pool = pool
        for i in range(num_layer):
            if i == 0:
                conv = gnn.GATv2Conv(
                    size_graph_vocab,
                    dim_encoder // num_head,
                    heads=num_head,
                    dropout=dropout,
                    edge_dim=dim_edge,
                )
            else:
                conv = gnn.GATv2Conv(
                    dim_encoder,
                    dim_encoder // num_head,
                    heads=num_head,
                    dropout=dropout,
                    edge_dim=dim_edge,
                )
            norm = nn.BatchNorm1d(dim_encoder)
            act = nn.LeakyReLU(inplace=True)
            layer = gnn.models.DeepGCNLayer(
                conv, norm, act, block="res+", dropout=dropout
            )
            self.layers.append(layer)
        self.norm = BatchNorm1d(dim_encoder)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
        if self.pool == "mean":
            return gnn.global_mean_pool(self.norm(x), batch)
        elif self.pool == "add":
            return gnn.global_add_pool(self.norm(x), batch)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

class LatentModel(nn.Module):
    def __init__(self, dim_encoder, dim_latent):
        super(LatentModel, self).__init__()
        self.mu = nn.Linear(dim_encoder, dim_latent)
        self.sigma = nn.Linear(dim_encoder, dim_latent)

    def forward(self, x):
        mu, sigma = self.mu(x), self.sigma(x)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, sigma


class DecoderLayer(nn.Module):
    def __init__(self, dim, dim_ff, num_head, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_head, dim, dropout)
        self.src_attn = MultiHeadedAttention(num_head, dim, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = clones(SublayerConnection(dim, dropout), 3)

    def forward(self, x, memory, smi_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, smi_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, None))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, dim, dim_latent, dim_ff, num_head, num_layer, dropout):
        super(Decoder, self).__init__()
        self.upsize_layer = Upsize(dim_latent, dim, dropout)
        self.layers = clones(DecoderLayer(dim, dim_ff, num_head, dropout), num_layer)
        self.norm = LayerNorm(dim)

    def forward(self, x, memory, smi_mask):
        memory = self.upsize_layer(memory)
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, smi_mask)
        return self.norm(x)


class DGVAE(nn.Module):
    def __init__(
        self,
        dim_encoder=256,
        dim_decoder=256,
        dim_latent=128,
        dim_encoder_ff=256,
        dim_decoder_ff=256,
        num_encoder_layer=4,
        num_decoder_layer=4,
        num_encoder_head=8,
        num_decoder_head=8,
        dropout=0.1,
        max_len=100,
        edge_vocab=None,
        node_vocab=None,
        smiles_vocab=None,
        pool="add",
    ):
        super(DGVAE, self).__init__()

        self.args = {
            "dim_encoder": dim_encoder,
            "dim_decoder": dim_decoder,
            "dim_latent": dim_latent,
            "dim_encoder_ff": dim_encoder_ff,
            "dim_decoder_ff": dim_decoder_ff,
            "num_encoder_layer": num_encoder_layer,
            "num_decoder_layer": num_decoder_layer,
            "num_encoder_head": num_encoder_head,
            "num_decoder_head": num_decoder_head,
            "dropout": dropout,
            "max_len": max_len,
            "edge_vocab": edge_vocab,
            "node_vocab": node_vocab,
            "smiles_vocab": smiles_vocab,
            "pool": pool,
        }
        self._smiles_inv_vocab = {v: k for k, v in smiles_vocab.items()}

        self.smi_embedding = SmilesEmbedding(len(smiles_vocab), dim_decoder, dropout)

        self.encoder = DGVAE_Encoder(
            dim_encoder=dim_encoder,
            size_graph_vocab=len(node_vocab),
            dim_edge=len(edge_vocab),
            num_head=num_encoder_head,
            num_layer=num_encoder_layer,
            dropout=dropout,
            pool=pool,
        )

        self.latent_model = LatentModel(dim_encoder, dim_latent)
        self.decoder = Decoder(
            dim_decoder,
            dim_latent,
            dim_decoder_ff,
            num_decoder_head,
            num_decoder_layer,
            dropout,
        )

        self.generator = nn.Linear(dim_decoder, len(smiles_vocab))

    def _mask(self, target):
        def _subsequent_mask(size):
            attn_shape = (1, size, size)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
                torch.uint8
            )
            return subsequent_mask == 0

        mask = (target != self.args["smiles_vocab"]["[pad]"]).unsqueeze(-2)
        return mask & _subsequent_mask(target.size(-1)).type_as(mask.data)

    def step(self, z, smiles):
        mask = self._mask(smiles)
        smiles = self.smi_embedding(smiles)
        output = F.log_softmax(self.generator(self.decoder(smiles, z, mask)), dim=-1)
        _, idx = torch.topk(output, 1, dim=-1)
        return idx[:, -1, :]

    def translate(self, smiles_list: torch.Tensor) -> List[str]:
        smiles_list = smiles_list.cpu().numpy()

        smiles_list = [
            "".join(self._smiles_inv_vocab[int(t)] for t in smiles)
            for smiles in smiles_list
        ]

        smiles_list = [
            (smiles.split("[start]")[1]).split("[end]")[0] for smiles in smiles_list
        ]
        return smiles_list

    def generate(self, N=500):
        z = torch.randn(N, self.args["dim_latent"]).to("cuda")
        smiles = torch.zeros(N, 1, dtype=torch.long).to("cuda")  # [START] Token
        for _ in range(self.args["max_len"] - 1):
            next_token = self.step(z, smiles)
            smiles = torch.cat([smiles, next_token], dim=-1)

        return self.translate(smiles_list=smiles)

    def forward(self, x, edge_index, edge_attr, batch, smiles):
        mask = self._mask(smiles)
        smiles = self.smi_embedding(smiles)

        pool = self.encoder(x, edge_index, edge_attr, batch)
        z, mu, logvar = self.latent_model(pool)
        out = self.decoder(smiles, z, mask)

        out = F.log_softmax(self.generator(out), dim=-1)

        return out, mu, logvar

    def save(self, filepath: str):
        package = {
            "class_name": "DGVAE",
            "args": self.args,
            "state_dict": self.state_dict(),
            "torch_version": torch.__version__,
        }
        torch.save(package, filepath)

    @classmethod
    def load(cls, filepath: str, map_location=None, strict: bool = True):
        pkg = torch.load(filepath, map_location=map_location, weights_only=False)
        args = pkg["args"]
        model = cls(**args)
        model.load_state_dict(pkg["state_dict"], strict=strict)
        return model
