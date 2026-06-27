import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, embed_dim, in_channels=3, obs_size=64):
        super().__init__()
        self.stride = 2
        self.embed_dim = embed_dim

        # 4 stride-2 convs divide spatial dims by 16
        flat_size = (obs_size // 16) ** 2 * 256

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(in_features=flat_size, out_features=self.embed_dim)
        )

    def forward(self, x):  # x: (B, C, H, W)
        return self.convs(x)
