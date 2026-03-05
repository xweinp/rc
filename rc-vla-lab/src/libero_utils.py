import h5py
import numpy as np
import torch


class LiberoH5Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for Libero HDF5 demonstrations."""

    def __init__(self, h5_path, instruction, dtype=torch.float32, image_keys=None):
        self.h5_path = h5_path
        self.instruction = instruction
        self.dtype = dtype
        self.image_keys = image_keys or [
            "agentview_rgb",
            "agentview_image",
            "frontview_image",
            "image",
        ]
        self.h5_file = h5py.File(self.h5_path, "r")
        self.data_group = self.h5_file["data"] if "data" in self.h5_file else self.h5_file
        self.demo_keys = sorted(list(self.data_group.keys()))
        self.index = []
        for demo_key in self.demo_keys:
            demo = self.data_group[demo_key]
            actions = demo["actions"]
            length = actions.shape[0]
            for t in range(length):
                self.index.append((demo_key, t))

    def __len__(self):
        return len(self.index)

    def _get_obs_image(self, demo, t):
        obs = demo["obs"]
        for key in self.image_keys:
            if key in obs:
                return obs[key][t]
        for key in obs.keys():
            arr = obs[key][t]
            if arr.ndim >= 3 and arr.shape[-1] == 3:
                return arr
        raise ValueError("No image observation found in HDF5 demo")

    def __getitem__(self, idx):
        demo_key, t = self.index[idx]
        demo = self.data_group[demo_key]
        image = self._get_obs_image(demo, t)

        if not isinstance(image, (np.ndarray, torch.Tensor)):
            image = np.array(image)

        obs_tensor = torch.tensor(image, dtype=self.dtype)
        if obs_tensor.ndim == 3 and obs_tensor.shape[2] == 3:
            obs_tensor = obs_tensor.permute(2, 0, 1)

        action = demo["actions"][t]
        action = np.array(action)
        if action.ndim > 1:
            action = action[-1]
        action_tensor = torch.tensor(action, dtype=self.dtype)

        return {
            "observation": obs_tensor,
            "action": action_tensor,
            "instruction": self.instruction,
        }
