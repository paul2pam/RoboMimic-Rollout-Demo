import sys
import torch
import numpy as np
import imageio

sys.path.insert(0, "rssm")
from rssm import RSSM
from env_wrapper import CarRacingWrapper

# Must match the dims used in train.py
EMBED_DIM    = 512
HIDDEN_DIM   = 512
DETER_DIM    = 512
DISCRETE_DIM = 512
ACTION_DIM   = 3

IMAGINE_STEPS = 100   # how many frames to dream forward from the seed observation
CHECKPOINT    = "checkpoint.pt"
OUTPUT        = "imagination.gif"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RSSM(EMBED_DIM, HIDDEN_DIM, DETER_DIM, DISCRETE_DIM, ACTION_DIM).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# Seed (h, z) from one real observation
env = CarRacingWrapper()
obs, _ = env.reset()
obs = obs.unsqueeze(0).to(device)   # (1, 3, 64, 64)
action = torch.zeros(1, ACTION_DIM, device=device)

with torch.no_grad():
    h, z = model.initial(1)
    e = model.encoder(obs)
    h, z, _, _ = model.observation_step(h, z, action, e)

    frames = []
    for _ in range(IMAGINE_STEPS):
        h, z = model.imagination_step(h, z, action)
        frame = model.decode(h, z)                          # (1, 3, 64, 64)
        frame = frame.squeeze(0).permute(1, 2, 0)          # (64, 64, 3)
        frame = frame.clamp(0, 1).cpu().numpy()
        frames.append((frame * 255).astype(np.uint8))

imageio.mimsave(OUTPUT, frames, fps=10)
print(f"Saved {OUTPUT}  ({IMAGINE_STEPS} frames)")
