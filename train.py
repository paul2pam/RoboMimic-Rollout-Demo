import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "rssm")
from rssm import RSSM
from env_wrapper import CarRacingWrapper
from replay_buffer import ReplayBuffer

# --- Dims ---
EMBED_DIM   = 512
HIDDEN_DIM  = 512
DETER_DIM   = 512
DISCRETE_DIM = 512
ACTION_DIM  = 3   # CarRacing-v2: [steering, gas, brake]

# --- Training ---
SEQ_LEN    = 50   # consecutive real frames per training sequence (not imagination depth)
BATCH_SIZE = 16
LR          = 1e-4
KL_WEIGHT   = 0.1
FREE_BITS   = 1.0  # minimum KL per step — prevents posterior collapse
TRAIN_STEPS = 500

# --- Data collection ---
INIT_EPISODES   = 10   # random episodes before training starts
COLLECT_EVERY   = 100  # collect one more episode every N train steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model     = RSSM(EMBED_DIM, HIDDEN_DIM, DETER_DIM, DISCRETE_DIM, ACTION_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
buffer    = ReplayBuffer(max_episodes=200)
env       = CarRacingWrapper()


def collect_episode():
    obs, _ = env.reset()
    buffer.start_episode()
    done = False
    steps = 0
    while not done and steps < 1000:
        action = env.action_space.sample()
        buffer.add(obs, action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    buffer.end_episode()


def train_step():
    obs_np, act_np = buffer.sample(BATCH_SIZE, SEQ_LEN)
    # obs_np: (B, T, 3, 64, 64)   act_np: (B, T, action_dim)
    obs = torch.tensor(obs_np).to(device)
    act = torch.tensor(act_np).to(device)

    h, z = model.initial(BATCH_SIZE)
    recon_loss = 0.0
    kl = 0.0

    for t in range(SEQ_LEN):    #loop to calculate total loss
        obs_t = obs[:, t]  # (B, 3, 64, 64)
        act_t = act[:, t]  # (B, action_dim)

        e = model.encoder(obs_t)
        h, z, prior_logits, post_logits = model.observation_step(h, z, act_t, e)

        recon = model.decode(h, z)
        recon_loss += F.mse_loss(recon, obs_t)
        kl += model.kl_loss(prior_logits, post_logits, FREE_BITS)

    recon_loss = recon_loss / SEQ_LEN
    kl = kl / SEQ_LEN
    total_loss = recon_loss + KL_WEIGHT * kl
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    return recon_loss.item(), kl.item()


#collect initial data
print(f"Collecting {INIT_EPISODES} random episodes...")
for i in range(INIT_EPISODES):
    collect_episode()
    print(f"  Episode {i + 1}/{INIT_EPISODES}")

#traiing loop
print(f"\nTraining for {TRAIN_STEPS} steps...")
for step in range(TRAIN_STEPS):
    recon_loss, kl = train_step()

    if step % 50 == 0:
        print(f"Step {step:4d} | recon: {recon_loss:.5f} | kl: {kl:.5f}")

    if step % COLLECT_EVERY == 0:
        collect_episode()

    #saves checkpoints, not necessarily useful
    if step % 500 == 0 and step > 0:
        torch.save(model.state_dict(), f"checkpoint_{step}.pt")
        print(f"  Saved checkpoint_{step}.pt")

torch.save(model.state_dict(), "checkpoint.pt")
print("Saved checkpoint.pt")
print("Done.")
