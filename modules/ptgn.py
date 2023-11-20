import torch
from torch import nn, Tensor
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)


class PositionPassingTGN(nn.Module):
    def __init__(self, num_nodes: int, msg_dim: int,
                 memory_dim: int, time_dim: int):
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
            1,
            memory_dim,
            time_dim,
            message_module=PositionMessage(num_nodes, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

    def forward(self, n_id):
        # get node embedding
        z, last_update = self.memory(n_id)

        # get position embedding
        pos_z, _ = self.pos_memory(n_id)

        return z, pos_z, last_update

    def update_state(self, src, dst, t, msg):
        self.memory.update_state(src, dst, t, msg)
        self.pos_memory.update_state(src, dst, t, Tensor(src).unsqueeze(-1))

    def reset_state(self):
        self.memory.reset_state()
        self.pos_memory.reset_state()

    def detach(self):
        self.memory.detach()
        self.pos_memory.detach()


class PositionMessage(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = 3 * memory_dim + time_dim
        self.embedding = nn.Embedding(num_nodes, memory_dim)

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor,
                t_enc: Tensor) -> Tensor:
        pos_msg = self.embedding(raw_msg.int()).reshape(
            z_src.shape[0], z_src.shape[1]) + z_src
        return torch.cat([z_src, z_dst, pos_msg, t_enc], dim=-1)
