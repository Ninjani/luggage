from collections import defaultdict
import torch
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import hetero_conv
from torch import nn
import math


# Taken from HeteroConv in torch_geometric.nn.conv
# added pos_dict to forward
class HeteroEGNN(HeteroConv):
    def __init__(self, convs, aggr='add', node_dim: int = 12):
        super().__init__(convs, aggr)
        self.node_dim = node_dim

    def forward(
        self,
        x_dict,
        pos_dict,
        edge_index_dict,
        *args_dict,
        **kwargs_dict,
    ):
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                   value_dict.get(dst, None))

            conv = self.convs[str_edge_type]

            if self.node_dim == x_dict[src].shape[1]:
                out = conv(torch.hstack((pos_dict[src], x_dict[src])), edge_index, *args, **kwargs)
            else:
                out = conv(x_dict[src] , edge_index, *args, **kwargs)

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = hetero_conv.group(value, self.aggr)

        return out_dict


class AttentionLayerNorm(nn.Module):
    def __init__(self, attention, embed_dim):
        super().__init__()
        self.attention = attention
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, *args):
        attn_output = self.attention(*args)
        return self.norm(attn_output)

    def pred_with_weights(self, *args):
        attn_output, attn_weights = self.attention.pred_with_weights(*args)
        return self.norm(attn_output), attn_weights
    

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)  # self-attention
        return attn_output
    
    def pred_with_weights(self, x):
        attn_output, attn_weights = self.multihead_attn(x, x, x)  # self-attention
        return attn_output, attn_weights 


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, query, kv):
        attn_output, _ = self.multihead_attn(query, kv, kv)  # cross-attention
        return attn_output
    
    def pred_with_weights(self, query, kv):
        attn_output, attn_weights = self.multihead_attn(query, kv, kv)  # cross-attention
        return attn_output, attn_weights 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
        

class ProteinEmbedding(nn.Module):
    def __init__(self, 
                 residue_feature_dim,
                 protein_embedding_dim,
                 num_heads, num_layers,
                 dropout=0.1, max_prot_len=1000):
        super().__init__()
        self.protein_embedding_dim = protein_embedding_dim
        self.residue_scaler = nn.Sequential(
                nn.Linear(residue_feature_dim, protein_embedding_dim),
                nn.ReLU(),
                nn.Linear(protein_embedding_dim, protein_embedding_dim),
                nn.LayerNorm(protein_embedding_dim))
        self.positional_encoding = PositionalEncoding(self.protein_embedding_dim, max_len=max_prot_len)
        self.self_attn_layers = nn.ModuleList(
            [AttentionLayerNorm(
                SelfAttention(self.protein_embedding_dim, num_heads, dropout=dropout), 
                self.protein_embedding_dim) for _ in range(num_layers)]
            )
        self.cross_attn_layers = nn.ModuleList(
            [AttentionLayerNorm(
                CrossAttention(self.protein_embedding_dim, num_heads, dropout=dropout), 
                self.protein_embedding_dim) for _ in range(num_layers)]
            )

    def forward(self, residue_features):        
        residue_features = self.positional_encoding(self.residue_scaler(residue_features))
        # Apply attention layers
        output_embedding = torch.ones((1, 1, self.protein_embedding_dim)).to(residue_features.device)
        for self_attn_layer, cross_attn_layer in zip(self.self_attn_layers, self.cross_attn_layers):
            residue_features = self_attn_layer(residue_features)
            output_embedding = cross_attn_layer(output_embedding, residue_features)
        return output_embedding.reshape((output_embedding.shape[0], self.protein_embedding_dim))


