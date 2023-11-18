import torch
from torch import nn
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)


class PositionPassingTGN(nn.Module):
    def __init__(self, num_nodes: int, pos_embedding_dim: int, msg_dim: int,
                 memory_dim: int, time_dim: int, step: float = 2.0):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        # pos_memory for position embedding
        self.pos_memory = TGNMemory(
            num_nodes,
            pos_embedding_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(
                pos_embedding_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        self.embedding = nn.Embedding(num_nodes, pos_embedding_dim)
        self.delta = nn.Parameter(torch.rand(1))
        self.step = step

    def forward(self, n_id):
        # get node embedding
        z, last_update = self.memory(n_id)

        # get position embedding
        pos_z, _ = self.pos_memory(n_id)

        return z, pos_z, last_update

    def update_state(self, src, dst, t, msg):
        self.memory.update_state(src, dst, t, msg)

        # get last position message
        last_pos_msg, pos_last_update = self.pos_memory(src)
        pos_emb = self.embedding(src)
        time_diff = t - pos_last_update
        space_time_decay = (
            torch.exp(-torch.sigmoid(self.delta) * time_diff) * self.step
        )
        pos_msg = pos_emb + last_pos_msg * space_time_decay

        # pos_memory update
        self.pos_memory.update_state(src, src, t, pos_msg)

    def reset_memory(self):
        self.memory.reset_state()
        self.pos_memory.reset_state()
