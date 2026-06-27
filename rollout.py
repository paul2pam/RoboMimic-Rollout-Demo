"""
Autoregressive rollout and visualization for the trained world model.

Usage:
    python rollout.py --checkpoint checkpoint.pt --hdf5 data.hdf5 \
        --demo demo_0 --burn_in 5 --horizon 45

Outputs:
    rollout.mp4   — side-by-side ground truth (left) vs. imagined (right)
    mse_plot.png  — per-frame MSE over the imagination horizon
"""
import sys
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt

sys.path.insert(0, "rssm")
from rssm import RSSM

# Must match train.py
EMBED_DIM    = 512
HIDDEN_DIM   = 512
DETER_DIM    = 512
DISCRETE_DIM = 512
ACTION_DIM   = 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint.pt")
    p.add_argument("--hdf5",       default="test_data.hdf5")
    p.add_argument("--demo",       default="demo_0",
                   help="Group name under data/ in the HDF5 file")
    p.add_argument("--burn_in",    type=int, default=5,
                   help="Frames used to seed the RSSM hidden state")
    p.add_argument("--horizon",    type=int, default=45,
                   help="Frames to imagine after burn-in")
    p.add_argument("--out_video",  default="rollout.mp4")
    p.add_argument("--out_plot",   default="mse_plot.png")
    return p.parse_args()


def load_demo(hdf5_path: str, demo_key: str, device: torch.device):
    """Return (imgs, acts) tensors for the requested demo.

    imgs: (T, 3, 64, 64) float32 in [0, 1]
    acts: (T, 7)          float32
    """
    with h5py.File(hdf5_path, "r") as f:
        demo = f[f"data/{demo_key}"]
        imgs_np = demo["obs/robot0_eye_in_hand_image"][:]  # (T, 84, 84, 3) uint8
        acts_np = demo["actions"][:].astype("float32")     # (T, 7)

    imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
    imgs = F.interpolate(imgs, size=(64, 64), mode="bilinear", align_corners=False)
    acts = torch.from_numpy(acts_np)
    return imgs.to(device), acts.to(device)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] → (H, W, 3) uint8."""
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    model = RSSM(EMBED_DIM, HIDDEN_DIM, DETER_DIM, DISCRETE_DIM, ACTION_DIM).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Load trajectory ──────────────────────────────────────────────────────
    imgs, acts = load_demo(args.hdf5, args.demo, device)
    T = imgs.shape[0]
    need = args.burn_in + args.horizon
    if T < need:
        raise ValueError(f"Demo has {T} frames but burn_in+horizon={need}. Use a longer demo.")
    print(f"Demo '{args.demo}': {T} frames. Burn-in={args.burn_in}, horizon={args.horizon}.")

    # ── Burn-in: seed hidden state with real observations ────────────────────
    with torch.no_grad():
        h, z = model.initial(1)
        for t in range(args.burn_in):
            e = model.encoder(imgs[t].unsqueeze(0))
            h, z, _, _ = model.observation_step(h, z, acts[t].unsqueeze(0), e)

        # ── Imagination: autoregressive rollout conditioned on real actions ──
        imagined = []
        for t in range(args.burn_in, args.burn_in + args.horizon):
            h, z_logits = model.imagination_step(h, z, acts[t].unsqueeze(0))
            z = F.softmax(z_logits, dim=-1)          # normalize before decode + next step
            frame = model.decode(h, z).squeeze(0)    # (3, 64, 64)
            imagined.append(frame.clamp(0, 1))

    gt_frames = imgs[args.burn_in : args.burn_in + args.horizon]  # (H, 3, 64, 64)

    # ── Per-frame MSE ────────────────────────────────────────────────────────
    mse = [F.mse_loss(imagined[i], gt_frames[i]).item() for i in range(args.horizon)]

    # ── Side-by-side video ───────────────────────────────────────────────────
    video_frames = []
    for i in range(args.horizon):
        left  = to_uint8(gt_frames[i])
        right = to_uint8(imagined[i])
        side_by_side = np.concatenate([left, right], axis=1)  # (64, 128, 3)
        video_frames.append(side_by_side)

    imageio.mimwrite(args.out_video, video_frames, fps=10)
    print(f"Saved video: {args.out_video}  ({args.horizon} frames, 64×128)")

    # ── MSE plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(args.horizon), mse, marker="o", markersize=3)
    ax.set_xlabel("Imagination step")
    ax.set_ylabel("MSE vs. ground truth")
    ax.set_title(f"Prediction degradation — {args.demo}, burn-in={args.burn_in}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=150)
    plt.close(fig)
    print(f"Saved MSE plot: {args.out_plot}")
    print(f"Mean MSE: {np.mean(mse):.5f}  |  Final MSE: {mse[-1]:.5f}")


if __name__ == "__main__":
    main()
