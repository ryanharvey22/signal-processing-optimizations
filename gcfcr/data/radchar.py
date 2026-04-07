"""RadChar HDF5 loader (ICASSP 2023). Download from Kaggle: abcxyzi/radchar-icassp-2023."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Integer mapping from RadChar README / paper
SIGNAL_TYPE_NAMES: Dict[int, str] = {
    0: "coherent_pulse_train",
    1: "barker_code",
    2: "polyphase_barker_code",
    3: "frank_code",
    4: "linear_frequency_modulated",
}

NUM_SIGNAL_TYPES = len(SIGNAL_TYPE_NAMES)


class RadCharDataset(Dataset):
    """
    Loads RadChar ``*.h5`` with datasets ``iq`` (N, 512) complex and structured ``labels``.

    For multiprocessing ``DataLoader`` workers, data is read into RAM at init (fine for
    RadChar-Tiny; use ``max_samples`` for larger files during development).
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        *,
        split: str = "train",
        train_fraction: float = 0.9,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(
                f"RadChar HDF5 not found: {self.h5_path}. "
                "Download RadChar-Tiny.h5 (or other variant) from Kaggle: "
                "https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023"
            )

        split = split.lower()
        if split not in ("train", "val", "test", "all"):
            raise ValueError("split must be one of: train, val, test, all")

        with h5py.File(self.h5_path, "r") as f:
            iq = np.asarray(f["iq"][:], dtype=np.complex64)
            labels = np.asarray(f["labels"][:])

        n = iq.shape[0]
        if max_samples is not None:
            n = min(n, max_samples)
            iq = iq[:n]
            labels = labels[:n]

        if split == "all":
            self._indices = np.arange(n, dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            n_train = int(round(train_fraction * n))
            if split == "train":
                self._indices = np.sort(perm[:n_train])
            elif split == "val":
                self._indices = np.sort(perm[n_train:])
            else:  # test — same as val unless you provide a separate file
                self._indices = np.sort(perm[n_train:])

        self._iq = iq
        self._labels = labels

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        i = int(self._indices[idx])
        z = self._iq[i]
        # PyTorch complex tensor (512,)
        iq_tensor = torch.from_numpy(z.astype(np.complex64))
        row = self._labels[i]
        meta: Dict[str, Any] = {
            "signal_type": int(row["signal_type"]),
            "signal_type_name": SIGNAL_TYPE_NAMES[int(row["signal_type"])],
            "number_of_pulses": int(row["number_of_pulses"]),
            "pulse_width": float(row["pulse_width"]),
            "time_delay": float(row["time_delay"]),
            "pulse_repetition_interval": float(row["pulse_repetition_interval"]),
            "snr_db": int(row["signal_to_noise_ratio"]),
            "index": int(row["index"]),
        }
        target = meta["signal_type"]
        return iq_tensor, meta

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Stack IQ and tensorize metadata for training loops."""
        iq_list, metas = zip(*batch)
        iq = torch.stack(iq_list, dim=0)
        keys = metas[0].keys()
        batched: Dict[str, torch.Tensor] = {}
        for k in keys:
            vals = [m[k] for m in metas]
            if k == "signal_type_name":
                continue
            if isinstance(vals[0], str):
                continue
            batched[k] = torch.tensor(vals)
        return iq, batched
