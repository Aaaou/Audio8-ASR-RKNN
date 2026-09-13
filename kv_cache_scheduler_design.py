"""Reference CPU-owned KV scheduler for the eventual RKNN decoder graphs.

This module intentionally contains no RKNN graph mutation. It documents and
implements the state transition that the deployment loop must use once the
read-only decoder graph is available.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class CpuOwnedKVCache:
    keys: list[np.ndarray]
    values: list[np.ndarray]

    @property
    def seq_len(self) -> int:
        return int(self.keys[0].shape[-2])

    @property
    def bytes(self) -> int:
        return int(sum(x.nbytes for x in self.keys) + sum(x.nbytes for x in self.values))

    def append(self, key_delta: list[np.ndarray], value_delta: list[np.ndarray]) -> None:
        if len(key_delta) != len(self.keys) or len(value_delta) != len(self.values):
            raise ValueError("cache layer count mismatch")
        self.keys = [np.concatenate((old, new), axis=-2) for old, new in zip(self.keys, key_delta)]
        self.values = [np.concatenate((old, new), axis=-2) for old, new in zip(self.values, value_delta)]

    def as_inputs(self) -> list[np.ndarray]:
        values: list[np.ndarray] = []
        for key, value in zip(self.keys, self.values):
            values.extend((key, value))
        return values


def cache_from_legacy(layers) -> CpuOwnedKVCache:
    return CpuOwnedKVCache(
        keys=[np.asarray(layer.keys) for layer in layers],
        values=[np.asarray(layer.values) for layer in layers],
    )
