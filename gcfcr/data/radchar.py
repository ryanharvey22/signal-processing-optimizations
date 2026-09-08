"""RadChar HDF5 loader with disjoint train/validation/test assignments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from gcfcr.optimized.data import cap_indices, read_iq_rows, read_radchar_metadata, split_indices

SIGNAL_TYPE_NAMES: Dict[int, str] = {
    0: "coherent_pulse_train", 1: "barker_code", 2: "polyphase_barker_code",
    3: "frank_code", 4: "linear_frequency_modulated",
}
NUM_SIGNAL_TYPES = len(SIGNAL_TYPE_NAMES)


class RadCharDataset(Dataset):
    """Read selected IQ rows into RAM; metadata defines splits before sample caps.

    Default fractions are 80/10/10. Validation and test never share rows. Known
    group identifiers, if present, are kept together. Without them, this loader
    guarantees row separation only. RadChar release variants overlap: use one
    file for all splits, not Tiny for training and Small for testing.
    """
    def __init__(self, h5_path: Union[str, Path], *, split: str = "train",
                 train_fraction: float = 0.8, val_fraction: float = 0.1,
                 seed: int = 42, max_samples: Optional[int] = None) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"RadChar HDF5 not found: {self.h5_path}. Download from "
                "https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023")
        split = split.lower()
        if split not in ("train", "val", "test", "all"):
            raise ValueError("split must be one of: train, val, test, all")
        with h5py.File(self.h5_path, "r") as handle:
            labels, groups = read_radchar_metadata(handle)
            assignments = split_indices(len(labels), seed=seed,
                train_fraction=train_fraction, val_fraction=val_fraction, groups=groups)
            indices = np.arange(len(labels), dtype=np.int64) if split == "all" else assignments[split]
            self._indices = cap_indices(indices, max_samples, seed=seed)
            self._iq = read_iq_rows(handle["iq"], self._indices)
            self._labels = labels[self._indices]
            self.group_ids = groups[self._indices] if groups is not None else None

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        row = self._labels[idx]
        label = int(row["signal_type"])
        meta: Dict[str, Any] = {"signal_type": label,
            "signal_type_name": SIGNAL_TYPE_NAMES[label],
            "snr_db": float(row["signal_to_noise_ratio"]), "index": int(row["index"]),
            "row_id": int(self._indices[idx])}
        for field, convert in (("number_of_pulses", int), ("pulse_width", float),
                               ("time_delay", float), ("pulse_repetition_interval", float)):
            if field in self._labels.dtype.names:
                meta[field] = convert(row[field])
        if self.group_ids is not None:
            meta["group_id"] = int(self.group_ids[idx])
        return torch.from_numpy(self._iq[idx].copy()), meta

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]
                   ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        iq_list, metas = zip(*batch)
        return torch.stack(iq_list), {key: torch.tensor([m[key] for m in metas])
            for key in metas[0] if not isinstance(metas[0][key], str)}
