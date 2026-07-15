"""Best-effort discovery of attached Edge TPU hardware.

This inspects the filesystem directly rather than the TPU runtime, so it works
even when libedgetpu is not installed and is safe to call on any platform.
Actual device binding happens in the delegate (see tflite.py).
"""

from __future__ import annotations

import glob
from pathlib import Path

from coralapi.schemas import DeviceInfo

# USB IDs for the Coral USB Accelerator. 1a6e:089a is the DFU-mode bootloader
# (before the runtime uploads firmware); 18d1:9302 is the running device.
USB_IDS = {("1a6e", "089a"), ("18d1", "9302")}

SYSFS_USB = Path("/sys/bus/usb/devices")


def discover_pcie() -> list[DeviceInfo]:
    """PCIe/M.2 modules appear as /dev/apex_N via the gasket/apex driver."""
    return [DeviceInfo(type="pcie", path=p) for p in sorted(glob.glob("/dev/apex_*"))]


def discover_usb() -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    if not SYSFS_USB.is_dir():
        return devices
    for entry in sorted(SYSFS_USB.iterdir()):
        try:
            vendor = (entry / "idVendor").read_text().strip()
            product = (entry / "idProduct").read_text().strip()
        except OSError:
            continue
        if (vendor, product) in USB_IDS:
            devices.append(DeviceInfo(type="usb", path=str(entry)))
    return devices


def discover_all() -> list[DeviceInfo]:
    return discover_pcie() + discover_usb()
