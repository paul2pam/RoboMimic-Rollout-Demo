import torch.nn as nn

class Encoder(nn.module):
    def __init__(self, depth, embed_dim):
        super().__init__()
        self.in_channels = 3                #3 if RGB, 1 if grayscale
        self.kernel_size = 4                #this and stride are just by default
        self.stride = 2                     
        self.embed_dim = embed_dim  

        self.convs = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=4, stride = 2)
        )        


        


    def forward(self, x, h): #x is the observation, and comes in a raw pixel tensor (B, C, H, W)
        e = self.convs(x)

        
        return e