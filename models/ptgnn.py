import torch
from torch import nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)


class PTGNN(nn.Module):
    def __init__(self, num_nodes, msg_dim, memory_dim, time_dim, embedding_dim, step):
        super(PTGNN, self).__init__()
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
            embedding_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(
                embedding_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.delta = nn.Parameter(torch.rand(1))
        self.step = step

    def forward(self, src, dst, t, msg, n_id):
        # get node embedding
        z, last_update = self.memory(n_id)
        self.memory.update_state(src, dst, t, msg)

        # get position embedding
        pos_z, _ = self.pos_memory(n_id)

        # get last position message
        last_pos_msg, pos_last_update = self.pos_memory(src)
        pos_emb = self.embedding(src)
        time_diff = t - pos_last_update
        space_time_decay = torch.exp(-self.delta * time_diff) * self.step
        pos_msg = pos_emb + last_pos_msg * space_time_decay

        # pos_memory update
        self.pos_memory.update_state(src, src, t, pos_msg)

        return z, pos_z, last_update


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)
