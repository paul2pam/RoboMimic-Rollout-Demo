import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, deter_dim, discrete_dim, out_channels=3):
        super().__init__()
        self.total_dim = deter_dim + discrete_dim
        self.stride = 2

        self.convs = nn.Sequential(
            nn.Linear(in_features=self.total_dim, out_features=4096),
            nn.Unflatten(-1, (256, 4, 4)),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=self.stride, padding=1)
        )

    def forward(self, h, z):
        input = torch.cat((h, z), dim=-1)
        return self.convs(input)
