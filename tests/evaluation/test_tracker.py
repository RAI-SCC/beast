from typing import Any, Dict, List, Tuple

import pytest
import torch

from beast.evaluation import Tracker


class TestTracker:
    @pytest.fixture(scope="class")
    def dummy(self) -> Tuple[Tracker, Tuple[str, ...], Tuple[List[Any], ...]]:
        attrs: Dict[str, List[Any]] = {
            "bool": [True, False, False],
            "int": [1, 2, 3, 4, 5],
            "float": [1.1, 2.2, 3.3, 4.4, 5.5],
            "list": [["a", "b", "c"], ["d", "e", "f"]],
            "tensor": [torch.randn(3, 4), torch.ones(3, 4), torch.zeros(3, 4)],
            "single": [
                42,
            ],
            "pair": ["a", "b"],
        }
        tracker = Tracker()

        for key, val in attrs.items():
            for v in val:
                tracker[key] = v
        ref_keys = tuple(attrs.keys())
        ref_values = tuple(attrs.values())
        return tracker, ref_keys, ref_values

    def test_getter(self) -> None:
        tracker = Tracker()
        with pytest.raises(AttributeError, match="Attribute missing not found"):
            a = tracker.missing  # noqa: F841

    def test_init(
        self, dummy: Tuple[Tracker, Tuple[str, ...], Tuple[List[Any], ...]]
    ) -> None:
        tracker, keys, values = dummy
        for key, val in zip(keys, values):
            assert tracker[key] == val
            assert hasattr(tracker, "_" + key)

    def test_repr(self) -> None:
        tracker = Tracker()
        tracker.print = 1
        tracker.print = 2

        out = tracker.__repr__()
        ref = "Tracker(\n\tprint: [1, 2],\n)"

        assert out == ref

    def test_index_getter(
        self, dummy: Tuple[Tracker, Tuple[str, ...], Tuple[List[Any], ...]]
    ) -> None:
        tracker, keys, values = dummy
        for val in values:
            extra = None
            if len(val) == 1:
                extra = val[0]
            while len(val) < 3:
                val.append(extra)

        ref_dict1 = {key: value[1] for key, value in zip(keys, values)}
        ref_dict2 = {key: value[2] for key, value in zip(keys, values)}

        assert tracker[1] == ref_dict1
        assert tracker[2] == ref_dict2

    def test_update(
        self, dummy: Tuple[Tracker, Tuple[str, ...], Tuple[List[Any], ...]]
    ) -> None:
        tracker, keys, values = dummy

        tracker.update(int=None)
        assert tracker.int is None

        tracker.update({"int": [1, 2, 3]})
        assert tracker.int == [1, 2, 3]

        tracker.update(hard=False, int=4, float=6.6, other=None)
        assert tracker.int == [1, 2, 3, 4]
        assert tracker.float == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
        assert tracker.other == [None]
