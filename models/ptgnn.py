import torch
from torch import Tensor, LongTensor
from torch_geometric.data import TemporalData
from torch_geometric.nn import MLP
from torch_geometric.nn.models.tgn import (
    LastNeighborLoader,
)

from modules.layers import GraphAttentionEmbedding, LinkPredictor
from modules.ptgn import PositionPassingTGN


class PTGNN(torch.nn.Module):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, pos_embedding_dim: int, embedding_dim: int,
                 mlp_hidden_dim: list = None, step: float = 2.0,
                 dropout: float = 0.0, size: int = 10):
        self.memory = PositionPassingTGN(
            num_nodes,
            pos_embedding_dim,
            raw_msg_dim,
            memory_dim,
            time_dim,
            step=step
        )

        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim*2,
            out_channels=embedding_dim,
            msg_dim=raw_msg_dim,
            time_enc=self.memory.time_enc
        )

        if mlp_hidden_dim is None:
            mlp_hidden_dim = [embedding_dim] // 2

        self.mlp = MLP(
            [embedding_dim, mlp_hidden_dim],
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
        self.assoc[n_id] = torch.arange(n_id.size(0))

        # Get updated memory of all nodes involved in the computation.
        z, pos_z, last_update = self.memory(n_id)
        z = torch.cat([z, pos_z], dim=-1)
        z = self.gnn(z, last_update, edge_index, data.t[e_id].to(n_id.device),
                     data.msg[e_id].to(n_id.device))
        z = self.mlp(z)

        # Only keep embeddings of target nodes for link prediction.
        z_src, z_dst, z_neg_dst = z[
            self.assoc[src]], z[self.assoc[dst]], z[self.assoc[neg_dst]]

        # Link prediction
        pos_out = self.link_pred(z_src, z_dst)
        neg_out = self.link_pred(z_src, z_neg_dst)

        # Update memory and neighbor loader with ground-truth state.
        self.memory.update_state(src, dst, t, msg)
        self.loader.insert(src, dst)

        return pos_out, neg_out

    def reset_state(self):
        self.memory.reset_state()
        self.loader.reset_state()
