"""Generate a small synthetic HDF5 file in robomimic format for testing."""
import numpy as np
import h5py

OUT = "test_data.hdf5"
NUM_DEMOS = 3
T = 20  # timesteps per demo
H, W, C = 84, 84, 3
ACTION_DIM = 7

rng = np.random.default_rng(0)

with h5py.File(OUT, 'w') as f:
    data = f.create_group("data")
    data.attrs["total"] = NUM_DEMOS * T
    data.attrs["env_args"] = '{"env_name": "Lift", "env_kwargs": {}}'

    for i in range(NUM_DEMOS):
        demo = data.create_group(f"demo_{i}")
        demo.attrs["num_samples"] = T

        obs = demo.create_group("obs")
        obs.create_dataset(
            "robot0_eye_in_hand_image",
            data=rng.integers(0, 256, (T, H, W, C), dtype=np.uint8),
        )
        obs.create_dataset(
            "agentview_image",
            data=rng.integers(0, 256, (T, H, W, C), dtype=np.uint8),
        )
        obs.create_dataset(
            "robot0_eef_pos",
            data=rng.uniform(-1, 1, (T, 3)).astype(np.float32),
        )
        obs.create_dataset(
            "robot0_eef_quat",
            data=rng.uniform(-1, 1, (T, 4)).astype(np.float32),
        )
        obs.create_dataset(
            "robot0_gripper_qpos",
            data=rng.uniform(-1, 1, (T, 2)).astype(np.float32),
        )

        demo.create_dataset(
            "actions",
            data=rng.uniform(-1, 1, (T, ACTION_DIM)).astype(np.float32),
        )
        demo.create_dataset(
            "rewards",
            data=rng.uniform(0, 1, (T,)).astype(np.float32),
        )
        demo.create_dataset(
            "dones",
            data=np.zeros(T, dtype=np.float32),
        )

print(f"Wrote {OUT}")
