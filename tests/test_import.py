"""
Smoke tests to verify that the beast library and its components can be imported properly.
"""


class TestBeastImport:
    """Test basic import of the beast package."""

    def test_import_beast_package(self):
        """Test that the beast package can be imported."""
        import beast

        assert beast is not None

    def test_beast_package_is_module(self):
        """Test that beast is a proper Python module."""
        import beast

        assert hasattr(beast, "__file__")
        assert hasattr(beast, "__path__")
