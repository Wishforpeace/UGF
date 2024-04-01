import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.subNets.Module import ModuleParallel,LayerNormParallel
import sys

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange,self).__init__()

    def forward(self, x, ln, ln_threshold):
        ln1, ln2 = ln[0].weight.abs(), ln[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
       
        
        x1[:, :,ln1 >= ln_threshold] = x[0][:,:,ln1 >= ln_threshold]
        x1[:, :,ln1 < ln_threshold] = x[1][:,:,ln1 < ln_threshold]
        x2[:, :,ln2 >= ln_threshold] = x[1][:,:, ln2 >= ln_threshold]
        x2[:, :,ln2 < ln_threshold] = x[0][:,:, ln2 < ln_threshold]
        return [x1, x2]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just a way to do matrix multiplication with the last dimension
        # and summing up. Could be done with torch.matmul and summing.
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class PositionwiseFeedforward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embed_size,ln_threshold, heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNormParallel(embed_size,2)
        self.norm2 = LayerNormParallel(embed_size,2)
        self.attn = MultiHeadAttention(embed_size, heads)
        self.ff = PositionwiseFeedforward(embed_size, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.exchange = Exchange()
        self.ln_list = []
        self.ln_threshold = ln_threshold

        for module in self.norm2.modules():
            if isinstance(module, nn.LayerNorm):
                self.ln_list.append(module)

    def forward(self, x, mask):
        attn_output_0 = self.attn(x[0], x[0], x[0], mask)
        attn_output_1 = self.attn(x[1], x[1], x[1], mask)
        x = self.norm1([attn_output_0 + x[0],attn_output_1 + x[1]])
        x[0] = self.dropout(x[0])
        x[1] = self.dropout(x[1])

        ff_output_0 = self.ff(x[0])
        ff_output_1 = self.ff(x[1])

        x = self.norm1([ff_output_0 + x[0],ff_output_1 + x[1]])

        x = self.exchange(x,self.ln_list,self.ln_threshold)

        x[0] = self.dropout(x[0])
        x[1] = self.dropout(x[1])
        return x

class LayerFusionTransformer(nn.Module):
    def __init__(
        self,
        args,
        embed_size,
        num_layers,
        ln_threshold,
        heads,
        ff_dim,
        dropout,
    ):
        super(LayerFusionTransformer, self).__init__()
        self.args = args
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_size=embed_size,
                            ln_threshold=ln_threshold,
                            heads=heads,
                            ff_dim=ff_dim,
                            dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.dropout = ModuleParallel(nn.Dropout(dropout))
        if self.args.is_exchange:
            self.alpha = nn.Parameter(torch.ones(2, requires_grad=True))
            self.register_parameter('alpha', self.alpha)

    def forward(self, x, mask=None):
        for layer in self.layers:
            out = layer(x, mask)


        ens = 0

        alpha_soft = F.softmax(self.alpha,dim=-1)

        for i in range(0,len(x)):
            ens += alpha_soft[i] * out[i].detach()

        return ens