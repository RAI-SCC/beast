from random import randint
from typing import Sequence

import pytest
import torch

from beast.evaluation import KeyedTensor


class TestKeyedTensor:
    def test_key_enforcement(self) -> None:
        dims = tuple(torch.randint(2, 10, [randint(2, 5)]))
        data = torch.randn(dims)

        # Check incorrect key number
        big_key_dict = {}
        for n in range(dims[0] + 1):
            big_key_dict[f"key_{n}"] = n

        with pytest.raises(
            AssertionError, match="keys and data must have the same length"
        ):
            KeyedTensor(data, big_key_dict)

        # Check non-unique mapping
        random_key_dict = {}
        for n in range(dims[0]):
            random_key_dict[f"key_{n}"] = randint(0, dims[0] - 1)

        with pytest.raises(
            AssertionError, match="keys must map uniquely to all indices"
        ):
            KeyedTensor(data, random_key_dict)

        # Check shuffled dict
        shuffled_key_dict = {}
        for n in range(dims[0]):
            shuffled_key_dict[f"key_{n}"] = n
        shuffled_key_dict["key_0"] = 1
        shuffled_key_dict["key_1"] = 0

        KeyedTensor(data, shuffled_key_dict)

    @pytest.mark.parametrize(
        "keys,data_shape",
        [
            [["a", "b", "c", "d"], [3, 6]],
            [["Rüdiger", "Beate", "Jürgen"], [6, 9, 4, 2]],
            [["k1"], []],
        ],
    )
    def test_access(self, keys: Sequence[str], data_shape: Sequence[int]) -> None:
        key_dict = {key: i for i, key in enumerate(keys)}
        data = torch.randn([len(keys), *data_shape])
        tensor = KeyedTensor(data, key_dict)

        assert list(tensor.keys()) == keys
        for index, key in enumerate(keys):
            assert torch.equal(tensor[key], tensor[index])
            assert torch.equal(tensor[key], data[index])
