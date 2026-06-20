import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.in_channels = 3                #3 if RGB, 1 if grayscale TODO:un-hardcode
        self.kernel_size = 4                #this and stride are just by default TODO:un-hardcode
        self.stride = 2                     #TODO:un-hardcode
        self.embed_dim = embed_dim  

        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=self.embed_dim)
        )    

    def forward(self, x): #x is the observation, and comes in a raw pixel tensor (B, C, H, W)
        e = self.convs(x)
        return e
    
class Posterior(nn.Module):
    def __init__(self, embed_dim, deter_dim):
        super().__init__()
        self.embed_dim = embed_dim  
        self.deter_dim = deter_dim
        self.total_dim = embed_dim + deter_dim
        self.hidden_dim = 512               #TODO: un-hardcode
        self.stoch_dim = 512                #TODO: un-hardcode

        self.mlp = nn.Sequential(
            nn.Linear(self.total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_dim)
        )
        

    def forward(self, e, h):
        z = torch.cat((e, h), dim=-1)
        z = self.mlp(z)
        return z