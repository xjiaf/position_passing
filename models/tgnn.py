import torch
from torch import Tensor, LongTensor
from torch_geometric.data import TemporalData
from torch_geometric.nn import MLP, TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from modules.layers import GraphAttentionEmbedding, LinkPredictor


class TGNN(torch.nn.Module):
    def __init__(self, num_nodes: int, raw_msg_dim: int,
                 memory_dim: int, time_dim: int,
                 embedding_dim: int, mlp_hidden_dim: list = None,
                 dropout: float = 0.0, size: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.size = size
        self.memory = TGNMemory(
            num_nodes,
            raw_msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=raw_msg_dim,
            time_enc=self.memory.time_enc,
        )

        if mlp_hidden_dim is None:
            mlp_hidden_dim = [embedding_dim // 2]
        self.mlp = MLP(
            [embedding_dim, *mlp_hidden_dim],
            dropout=dropout,
            act='relu',
            norm='batch_norm',
            bias=True
        )
        self.link_pred = LinkPredictor(in_channels=mlp_hidden_dim[-1])

        # Helper vector to map global node indices to local ones.
        self.assoc = torch.empty(num_nodes, dtype=torch.long)

        # Neighbor loader
        self.loader = LastNeighborLoader(num_nodes=num_nodes, size=size)

    def forward(self, data: TemporalData, src: LongTensor, dst: LongTensor,
                neg_dst: LongTensor, n_id: LongTensor, t: LongTensor,
                msg: Tensor) -> (Tensor, Tensor):

        # Get last subgraph
        n_id, edge_index, e_id = self.loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = self.memory(n_id)
        z = self.gnn(z, last_update, edge_index, data.t[e_id].to(n_id.device),
                     data.msg[e_id].to(n_id.device))
        z = self.mlp(z)

        # Only keep embeddings of target nodes for link prediction.
        z_src, z_dst, z_neg_dst = z[
            self.assoc[src]], z[self.assoc[dst]], z[self.assoc[neg_dst]]

        # Link prediction
        pos_out = self.link_pred(z_src, z_dst)
        neg_out = self.link_pred(z_src, z_neg_dst)

        return pos_out, neg_out

    def update_state(self, src: LongTensor, dst: LongTensor, t: LongTensor,
                     msg: Tensor):
        self.memory.update_state(src, dst, t, msg)
        self.loader.insert(src, dst)

    def reset_state(self):
        self.memory.reset_state()
        self.loader.reset_state()

    def to(self, device):
        # Call the base class to method to handle nn.Module components
        super().to(device)

        # Manually move other tensor attributes
        self.assoc = self.assoc.to(device)

        # Reinitialize the loader with the new device
        self.loader = LastNeighborLoader(num_nodes=self.num_nodes,
                                         size=self.size, device=device)
        return self
