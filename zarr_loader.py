"""TS-Zarr input for CAISR: read a BDSP primary store like a raw EDF.

A BDSP primary TS-Zarr store (``<s>_task-psg_eeg.zarr``, the BDSP default
recording format) holds calibrated physical signals at their native rates.
``load_zarr_for_caisr`` returns exactly what ``load_edf_for_caisr`` returns
for an EDF: ``(channel_data, channel_fs)`` with lower-case labels, voltage
channels in volts, channel names standardized with ``channel_table.csv`` and
the same bipolar derivations. Each task script then resamples and prepares the
signals exactly as it does for EDF input.

Requirements: ``pip install bdsp`` (the shared BDSP reader; it installs
``zarr`` and needs Python >= 3.11) in the task environment. Status: implemented
without end-to-end testing against the CAISR models.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

_TO_VOLTS = {"uv": 1e-6, "µv": 1e-6, "μv": 1e-6, "mv": 1e-3, "v": 1.0}


def is_zarr_input(path: str) -> bool:
    """Whether ``path`` names a TS-Zarr store (a ``*.zarr`` directory)."""

    return path.rstrip("/\\").lower().endswith(".zarr") and os.path.isdir(path)


def load_zarr_to_nparrays(
    store_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Read every signal of a primary store as float64 physical arrays.

    Voltage channels (declared uV, mV or V) are returned in volts, like the EDF
    loader; other channels keep their declared unit. Signals that start after
    recording zero are rejected rather than silently shifted.
    """

    try:
        from bdsp.bids_zarr import open_primary
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TS-Zarr input needs the BDSP reader: run `pip install bdsp` inside "
            "this CAISR environment (Python >= 3.11)."
        ) from exc

    channel_dict: Dict[str, np.ndarray] = {}
    fs_dict: Dict[str, float] = {}
    with open_primary(store_path) as record:
        for name in sorted(record.signals):
            channel = record.signals[name]
            if float(channel.offset_s) != 0.0:
                raise ValueError(
                    f"Signal {name} starts {channel.offset_s} s after recording "
                    "zero; CAISR input requires aligned signals"
                )
            data = np.asarray(channel.samples[:], dtype=np.float64)
            unit = str(channel.metadata.get("unit") or "").strip().lower()
            scale = _TO_VOLTS.get(unit)
            if scale is not None and scale != 1.0:
                data = data * scale
            label = name.lower().strip()
            channel_dict[label] = data
            fs_dict[label] = float(channel.rate_hz)
    return channel_dict, fs_dict


def load_zarr_for_caisr(
    store_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """``load_edf_for_caisr`` for a TS-Zarr primary store."""

    try:
        import edf_loader as rules
    except ImportError:  # edf_loader needs edfio; the pyedflib copy is identical
        import edf_loader_pyedflib as rules

    physiological_data, physiological_fs = load_zarr_to_nparrays(store_path)
    rename_rules = rules.load_rename_rules(os.path.abspath(rules.DEFAULT_CSV_PATH))
    rename_map, cols_to_drop = rules.standardize_channel_names_rename_only(
        list(physiological_data), rename_rules
    )
    processed_channels: Dict[str, np.ndarray] = {}
    processed_fs: Dict[str, float] = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        processed_fs[new_label] = physiological_fs.get(old_label, 200.0)

    bipolar_configs = [
        ("f3-m2", "f3", ["m2"]), ("f4-m1", "f4", ["m1"]),
        ("c3-m2", "c3", ["m2"]), ("c4-m1", "c4", ["m1"]),
        ("o1-m2", "o1", ["m2"]), ("o2-m1", "o2", ["m1"]),
        ("e1-m2", "e1", ["m2"]), ("e2-m1", "e2", ["m1"]),
        ("chin1-chin2", "chin 1", ["chin 2"]),
        ("lat", "lleg+", ["lleg-"]), ("rat", "rleg+", ["rleg-"]),
    ]
    for target, positive, negatives in bipolar_configs:
        if target in processed_channels or positive not in processed_channels:
            continue
        if not all(name in processed_channels for name in negatives):
            continue
        involved = [positive, *negatives]
        rates = [processed_fs[name] for name in involved]
        if len(set(rates)) > 1:
            raise ValueError(
                f"Sampling rate mismatch for {target}: {dict(zip(involved, rates))}"
            )
        reference = (
            processed_channels[negatives[0]]
            if len(negatives) == 1
            else tuple(processed_channels[name] for name in negatives)
        )
        derived = rules.derive_bipolar_signal(processed_channels[positive], reference)
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[positive]
    return processed_channels, processed_fs


def load_raw_for_caisr(path: str, edf_loader):
    """Dispatch raw-recording input: a TS-Zarr store or an EDF file."""

    if is_zarr_input(path):
        return load_zarr_for_caisr(path)
    if edf_loader is None:
        raise ImportError(
            "EDF input detected but the EDF reader is not installed in this "
            "environment (see the README)."
        )
    return edf_loader(path)
