import torch
from torch import Tensor, LongTensor
from torch_geometric.data import TemporalData
from torch_geometric.nn import MLP
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from modules.layers import GraphAttentionEmbedding, LinkPredictor
from modules.ptgn import PositionPassingTGN


class PTGNN(torch.nn.Module):
    