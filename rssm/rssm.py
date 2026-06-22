import torch.nn as nn
import torch.nn.functional as F
import torch

from encoder import Encoder
from decoder import Decoder

# Legend: 
#   e = encoded observation             (output of Encoder)
#   h = deterministic recurrent state   (output of GRU)
#   z = stochastic discrete latent state    (output of Dynamics/Representation)
#   a = action vector

# Representation model
class Posterior(nn.Module):
    def __init__(self, embed_dim, deter_dim, discrete_dim, hidden_dim=512):
        super().__init__()
        self.embed_dim = embed_dim  
        self.deter_dim = deter_dim
        self.total_dim = embed_dim + deter_dim
        self.discrete_dim = discrete_dim 
        self.hidden_dim = hidden_dim              

        self.mlp = nn.Sequential(
            nn.Linear(self.total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.discrete_dim)
        )
        

    def forward(self, e, h):
        z = torch.cat((e, h), dim=-1)
        z = self.mlp(z)
        return z
    


# Dynamics predictor
class Prior(nn.Module):
    def __init__(self, deter_dim, discrete_dim, hidden_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.discrete_dim = discrete_dim
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.discrete_dim)
        )

    def forward(self, h):
        z = self.mlp(h)
        return z
        
class SequenceModel(nn.Module):     #Not true to paper, this uses a regular GRUcell, not the block diagonal version
    def __init__(self, action_dim, deter_dim, discrete_dim, hidden_dim=512):
        super().__init__()
        self.deter_dim = deter_dim
        self.discrete_dim = discrete_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(self.discrete_dim + self.action_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.deter_dim)

    def forward(self, h, z, a):
        input = torch.cat((z,a), dim=-1)
        x = self.linear(input)
        h_new = self.gru(x, h)
        return h_new

class RSSM(nn.Module):

    def __init__(self, embed_dim, hidden_dim, deter_dim, discrete_dim, action_dim):
        super().__init__()
        self.sequence        = SequenceModel(action_dim, deter_dim, discrete_dim, hidden_dim)
        self.dynamics        = Prior(deter_dim, discrete_dim, hidden_dim)
        self.representation  = Posterior(embed_dim, deter_dim, discrete_dim, hidden_dim)

        self.encoder         = Encoder(embed_dim)
        self.decoder         = Decoder(deter_dim, discrete_dim)

        self.deter_dim = deter_dim
        self.discrete_dim = discrete_dim
        self.action_dim = action_dim

    def initial(self, batch_size):
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.discrete_dim, device=device)
        return h, z
#testing config
    def observation_step(self, h, z, a, e):
        h_new = self.sequence(h, z, a)
        post_logits = self.representation(e, h_new)  # posterior: uses real obs
        prior_logits = self.dynamics(h_new)           # prior: no obs
        z_new = F.softmax(post_logits, dim=-1)
        return h_new, z_new, prior_logits, post_logits

    def imagination_step(self, h, z, a):
        h_new = self.sequence(h, z, a)
        z_new = self.dynamics(h_new)
        return h_new, z_new
    
    def kl_loss(self, prior_logits, post_logits):
        p = F.softmax(post_logits, dim=-1)
        q = F.softmax(prior_logits, dim=-1)
        return (p * (p.log() - q.log())).sum(-1).mean()

    def decode(self, h, z):
        return self.decoder(h, z)