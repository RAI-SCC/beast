"""Provides a Tensor class that allows to access the first dimension with keys."""

from typing import Any, Dict, KeysView

from torch import Tensor


class KeyedTensor(Tensor):
    """
    A Tensor class that allows to access the first dimension with keys.

    Parameters
    ----------
    data: Tensor
        The contained data. Shape: (K, *)
    keys: Dict[str, int]
        A dictionary that maps keys to indices. Must map exactly one key to each integer
        between and including 0 and K-1.
    """

    @staticmethod
    def __new__(
        cls, data: Tensor, keys: Dict[str, int], *args: Any, **kwargs: Any
    ) -> "KeyedTensor":
        """Create a new KeyedTensor instance."""
        return super().__new__(cls, data, *args, **kwargs)

    def __init__(self, data: Tensor, keys: Dict[str, int]) -> None:
        super().__init__()
        assert len(keys) == len(data)
        assert sorted(keys.values()) == list(range(len(data)))
        self._keys = keys

    def keys(self) -> KeysView:
        """Return the available keys."""
        return self._keys.keys()

    def __getitem__(self, key: Any) -> Tensor:
        """Return the tensor value at given a key or index."""
        if isinstance(key, str):
            key = self._keys[key]
        return super().__getitem__(key)
