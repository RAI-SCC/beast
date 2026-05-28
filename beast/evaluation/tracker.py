"""A tracking utility class."""

from typing import Any, Dict, List, Mapping, cast, overload


class Tracker(object):
    """
    A flexible metric tracker.

    Assign a value to any attribute name. Additional assignments to the same name append
    the value to a list. These attributes can be accessed like a dictionary too.
    Accessing via inter index returns a dictionary of the specified sequence point for
    all attributes.
    """

    _keys: List[str] = []

    def __init__(self) -> None:
        self._keys: List[str] = []

    def keys(self) -> List[str]:
        """Return a list of the tracked attributes."""
        return self._keys

    def _append_attr(self, key: str, value: Any) -> None:
        attr = getattr(self, key)
        assert type(attr[0]) is type(value), (
            f"Value must match attribute type {type(value[0])}"
        )
        attr.append(value)

    def _init_attr(self, key: str, value: Any) -> None:
        self._keys.append(key)
        super().__setattr__(
            "_" + key,
            [
                value,
            ],
        )

    def __getattr__(self, key: str) -> Any:
        """Provide access to tracking attributes."""
        if key in self.keys():
            return super().__getattribute__("_" + key)
        else:
            raise AttributeError(f"Attribute {key} not found")

    def __setattr__(self, key: str, value: Any) -> None:
        """Append writes to tracking attributes."""
        if key in self.keys():
            self._append_attr("_" + key, value)
            return

        if hasattr(self, key) or hasattr(self, "_" + key):
            super().__setattr__(key, value)
        else:
            self._init_attr(key, value)

    @overload
    def update(self, /, hard: bool = False, **kwargs: Any) -> None: ...
    @overload
    def update(self, mapping: Mapping[str, Any], /, hard: bool = True) -> None: ...
    def update(self, mapping=None, /, hard=True, **kwargs) -> None:  # type: ignore [no-untyped-def,misc]
        """
        Update one or more tracking attributes.

        Updates attributes based on a mapping or keyword arguments. If ``hard`` is
        ``True`` (default), this will overwrite the respective attributes. Else the
        provided values will be appended to the tracking lists.
        """
        mapping = mapping or kwargs

        for key, value in mapping.items():
            if key in self.keys():
                if hard:
                    super().__setattr__("_" + key, value)
                else:
                    self._append_attr(key, value)
            else:
                self._init_attr(key, value)

    @overload
    def __getitem__(self, item: str) -> List: ...
    @overload
    def __getitem__(self, item: int) -> Dict: ...
    def __getitem__(self, item: str | int) -> List | Dict:
        """
        Get tracking attribute by name or sequence index.

        If item is a string the respective attribute is returned. If item is an int a
        dictionary is returned. Its keys are the attribute names, the values are taken
        from the list based on the index. If the attribute contains only one value it
        is used. If the provided index is larger than the length of the attribute list,
        the associated value is set as None.

        Parameters
        ----------
        item: str | int
            Attribute name or sequence index

        Returns
        -------
        List | Dict
            Tracked attribute or sequence point dictionary
        """
        if isinstance(item, str):
            return cast(List, self.__getattr__(item))

        # Handle int indices
        out_dict = dict()
        for key in self.keys():
            value = getattr(self, key)
            if len(value) == 1:
                value = value[0]
            elif len(value) < item + 1:
                value = None
            else:
                value = value[item]
            out_dict[key] = value
        return out_dict

    def __setitem__(self, key: str, value: Any) -> None:
        """Index setting falls back to attribute setting."""
        self.__setattr__(key, value)

    def __repr__(self) -> str:
        """Class representation."""
        attr = [f"\t{key}: {getattr(self, key)}," for key in self.keys()]
        return "\n".join([f"{self.__class__.__name__}(", *attr, ")"])
