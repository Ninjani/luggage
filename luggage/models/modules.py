from collections import defaultdict
import torch
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import hetero_conv


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
