from __future__ import annotations

import torch
from torch import nn


class PoseLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        direction_multiplier = 2 if bidirectional else 1
        classifier_width = hidden_size * direction_multiplier
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_width),
            nn.Dropout(dropout),
            nn.Linear(classifier_width, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(features)
        if self.lstm.bidirectional:
            representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            representation = hidden[-1]
        return self.classifier(representation)
