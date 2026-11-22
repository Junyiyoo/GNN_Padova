"""Regression models for square and H3 grids."""
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, RGCNConv

class DirectionalRegressor(nn.Module):
    def __init__(self, name, params, *, n_features, n_directions, edge_index, edge_type, dropout):
        super().__init__()

        # Buffers follow the model to the GPU. Each dataset supplies its own graph.
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_type", edge_type, persistent=False)
        self.dropout = dropout
        self.name = name
        self.activation = {
            "relu": F.relu, "elu": F.elu, "gelu": F.gelu,
        }[params["activation"]]

        in_dim = n_features
        if name == "MLP_ordered":
            # Append one feature block and one presence flag per direction.
            in_dim += n_directions * (n_features + 1)
        hidden = params["hidden_dim"]
        self.layers = nn.ModuleList()

        for _ in range(params["n_layers"]):
            if name in ("MLP_center", "MLP_ordered"):
                layer = nn.Linear(in_dim, hidden)
            elif name == "GCN":
                layer = GCNConv(in_dim, hidden, cached=True)
            elif name == "GraphSAGE":
                layer = SAGEConv(in_dim, hidden, aggr="mean")
            elif name == "GraphConv":
                layer = GraphConv(in_dim, hidden, aggr="add")
            elif name == "RGCN_directional":
                # Each compass direction is a relation, with two shared basis matrices.
                layer = RGCNConv(
                    in_dim, hidden, num_relations=n_directions,
                    num_bases=2, aggr="add", root_weight=True,
                )
            else:
                raise ValueError(name)

            self.layers.append(layer)
            in_dim = hidden

        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        h = data["ordered"] if self.name == "MLP_ordered" else data["center"]

        for layer in self.layers:
            if self.name in ("MLP_center", "MLP_ordered"):
                h = layer(h)
            elif self.name == "RGCN_directional":
                h = layer(h, self.edge_index, self.edge_type)
            else:
                h = layer(h, self.edge_index)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.head(h).squeeze(-1)
