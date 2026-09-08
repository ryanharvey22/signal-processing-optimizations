"""Factory to build RadChar or MNIST datasets (primary: latent autoencoder training)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Union

from torch.utils.data import Dataset

from gcfcr.data.radchar import RadCharDataset

DatasetName = Literal["radchar", "mnist"]


def build_dataset(
    name: DatasetName,
    *,
    data_dir: Union[str, Path],
    split: str = "train",
    radchar_filename: str = "RadChar-Tiny.h5",
    radchar_h5: Optional[Union[str, Path]] = None,
    radchar_train_fraction: float = 0.8,
    radchar_val_fraction: float = 0.1,
    radchar_seed: int = 42,
    radchar_max_samples: Optional[int] = None,
    mnist_download: bool = True,
) -> Dataset:
    """
    Parameters
    ----------
    name
        ``"radchar"`` or ``"mnist"``.
    data_dir
        Root directory for cached data (MNIST downloads here; RadChar default path is
        ``data_dir / "radchar" / radchar_filename`` unless ``radchar_h5`` is set).
    split
        ``train`` | ``val`` | ``test``. RadChar uses disjoint 80/10/10 splits.
    radchar_h5
        Explicit path to ``*.h5``. If omitted, uses ``RADCHAR_H5`` env var, then
        ``data_dir/radchar/radchar_filename``.
    radchar_max_samples
        Cap the selected split after assigning the full population to train/val/test.

    Returns
    -------
    ``torch.utils.data.Dataset``

    **Item shapes**

    - **radchar:** ``(iq, meta)`` where ``iq`` is ``torch.complex64`` shape ``(512,)``,
      ``meta`` is a dict including ``signal_type`` (0--4) and ``snr_db``.
    - **mnist:** ``(image, target)`` where ``image`` is ``float32`` ``(1, 28, 28)``,
      ``target`` is int 0--9. Flatten to 784 for ``RadCharIQAutoencoder(input_dim=784)``.
    """
    data_dir = Path(data_dir)

    if name == "radchar":
        path = radchar_h5 or os.environ.get("RADCHAR_H5")
        if path is None:
            path = data_dir / "radchar" / radchar_filename
        return RadCharDataset(
            path,
            split=split,
            train_fraction=radchar_train_fraction,
            val_fraction=radchar_val_fraction,
            seed=radchar_seed,
            max_samples=radchar_max_samples,
        )

    if name == "mnist":
        from gcfcr.data.mnist_data import MNISTDataset

        return MNISTDataset(
            data_dir / "mnist",
            split=split,
            download=mnist_download,
        )

    raise ValueError(f"Unknown dataset name: {name!r}")
