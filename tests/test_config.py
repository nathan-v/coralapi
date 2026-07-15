import pytest
from pydantic import ValidationError

from coralapi.config import Settings


@pytest.mark.parametrize("value", ["auto", "usb", "pci", "pci:0", "pci:1", "usb:0", ":1"])
def test_valid_device_selectors(value, tmp_path):
    assert Settings(model_dir=tmp_path, device=value).device == value


@pytest.mark.parametrize("value", ["gpu", "pci:", "pci:x", "auto:0", "pcie:0", ":", "1"])
def test_invalid_device_selectors_rejected(value, tmp_path):
    with pytest.raises(ValidationError):
        Settings(model_dir=tmp_path, device=value)
