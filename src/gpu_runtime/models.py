"""Small real models used for validation, not pretrained quality benchmarks."""
import torch
from torch import nn


class DenseModel(nn.Module):
    def __init__(self, features=256, hidden=512, outputs=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, outputs))

    def forward(self, x):
        return self.net(x)


class SequenceModel(nn.Module):
    """Masked attention pooling over variable-length item/token histories."""
    def __init__(self, vocabulary=4096, embedding=64, outputs=32):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, embedding)
        self.score = nn.Linear(embedding, 1)
        self.head = nn.Sequential(nn.Linear(embedding, 128), nn.ReLU(),
                                  nn.Linear(128, outputs))

    def forward(self, tokens, lengths):
        encoded = self.embedding(tokens)
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        logits = self.score(encoded).squeeze(-1).masked_fill(~mask, float("-inf"))
        pooled = (encoded * logits.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
        return self.head(pooled)
