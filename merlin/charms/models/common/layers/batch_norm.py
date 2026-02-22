import torch
from torch import nn


class GBN(nn.Module):
    """Ghost Batch Normalization (GBN) layer.
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=512):
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim)
    
    def forward(self, x):
        if self.training:
            # Compute the number of splits needed
            n_splits = max(1, x.size(0) // self.virtual_batch_size)
            # Split the input tensor into smaller chunks
            x_split = torch.chunk(x, n_splits, dim=0)
            # Apply batch normalization to each chunk and concatenate the results
            x = torch.cat([self.bn(x_chunk) for x_chunk in x_split], dim=0)
        else:
            x = self.bn(x)
        return x

class BatchNorm1d(nn.Module):
    """Batch Normalization layer with optional Ghost Batch Normalization (GBN).
    
    Args:
        input_dim (int): Number of input features.
        virtual_batch_size (int): Virtual batch size for GBN.
    """

    def __init__(self, num_features, virtual_batch_size=None):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        if self.virtual_batch_size is None:
            self.bn = nn.BatchNorm1d(self.num_features)
        else:
            self.bn = GBN(self.num_features, self.virtual_batch_size)
    
    def forward(self, x):
        return self.bn(x)
