"""
baseline_cnn.py - Baseline B2: 2D Mel-Spectrogram CNN Classifier.
Part of the GNN-BERT Music Context Understanding project (CSE425).

Mandatory Baseline B2 (PDF Section 8 & Table 3):
- Input: 2D Log-Mel Spectrogram (1 x 128 x T_frames)
- No graph structure, no text context.
- Used to benchmark whether GNN structure actually outperforms flat 2D convolutions.
"""

import os
import sys
from typing import Tuple
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MelSpectrogramCNN(nn.Module):
    """
    Standard 4-layer 2D Convolutional Neural Network on Mel-Spectrograms.
    Architecture:
      Conv2d(1 -> 32) -> BatchNorm -> ReLU -> MaxPool2d(2, 2)
      Conv2d(32 -> 64) -> BatchNorm -> ReLU -> MaxPool2d(2, 2)
      Conv2d(64 -> 128) -> BatchNorm -> ReLU -> MaxPool2d(2, 2)
      Conv2d(128 -> 256) -> BatchNorm -> ReLU -> AdaptiveAvgPool2d((1, 1))
      Linear(256 -> num_classes) -> Sigmoid
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, 1, 128, time_frames)
        Returns:
          logits: (batch_size, num_classes)
          features: (batch_size, 256) flattened global feature embedding
        """
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Layer 4
        x = F.relu(self.bn4(self.conv4(x)))

        features = self.global_pool(x).flatten(1)  # (batch_size, 256)
        dropped = self.dropout(features)
        logits = self.fc(dropped)

        return logits, features

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    print("=== Testing Baseline B2: 2D Mel-Spectrogram CNN ===")
    dummy_mel = torch.randn(4, 1, 128, 1292) # 4 tracks, 1 channel, 128 mels, ~30s frames
    model = MelSpectrogramCNN(num_classes=10)
    logits, feat = model(dummy_mel)
    print(f"Input Shape:    {dummy_mel.shape}")
    print(f"Features Shape: {feat.shape} (Expected 4 x 256)")
    print(f"Logits Shape:   {logits.shape} (Expected 4 x 10)")
    assert logits.shape == (4, 10), "Logits shape mismatch!"
    print("SUCCESS: Baseline B2 CNN passes forward verification!")
