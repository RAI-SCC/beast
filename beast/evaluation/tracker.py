"""A tracking utility class."""

from typing import Any, Dict, List, cast


class Tracker(object):
    """
    A flexible metric tracker.

    Assign a value to any attribute name. Additional assignments to the same name append
    the value to a list. These attributes can be accessed like a dictionary too.
    Accessing via inter index returns a dictionary of the specified sequence point for
    all attributes.
    """

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
        assert key not in self.keys(), f"Attribute {key} already exists"
        assert not hasattr(self, key), f"Attribute {key} already exists"
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

        try:
            self._init_attr(key, value)
        except AssertionError:
            super().__setattr__(key, value)

    def overwrite(self, key: str, value: Any) -> None:
        """
        Overwrite a tracking attribute.

        Hard sets a tracking attribute instead of appending. Type compatibility of the
        new attribute with the old one is enforced. If the attribute does not exist it
        is initialized.

        Parameters
        ----------
        key: str
            Attribute name
        value: Any
            Attribute value
        """
        if hasattr(self, key):
            raise ValueError(f"Standard attribute {key} cannot be overwritten")
        if key in self.keys():
            super().__setattr__("_" + key, value)
        else:
            self._init_attr(key, value)

    def store(self, data: Dict[str, Any]) -> None:
        """
        Store dictinary values to attributes named by the keys.

        Parameters
        ----------
        data: Dict[str, Any]
            Data dictionary to store
        """
        for key, value in data.items():
            setattr(self, key, value)

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
            elif len(value) < item:
                value = None
            else:
                value = value[item]
            out_dict[key] = value
        return out_dict

    def __repr__(self) -> str:
        """Class representation."""
        attr = [f"\t{key}: {getattr(self, key)}," for key in self.keys()]
        return "\n".join([f"{self.__class__.__name__}(", *attr, ")"])
