import importlib
import inspect
from pathlib import Path

import pufferlib.ocean.torch as drive_torch


def _torch_module_source() -> str:
    try:
        src = inspect.getsource(drive_torch)
        if len(src) > 10_000:
            raise ValueError("torch module source too large for getsource")
        return src
    except (OSError, TypeError, ValueError):
        return Path(drive_torch.__file__).read_text(encoding="utf-8")


def test_default_torch_import_does_not_require_conservative_package(monkeypatch):
    # Importing ocean.torch must succeed even if conservative were absent;
    # we approximate by asserting DriveSteerConstrained is not defined in
    # torch.py source and is only present after register().
    src = _torch_module_source()
    assert "class DriveSteerConstrained" not in src


def test_drive_ini_has_no_partner_mode_keys():
    text = Path("pufferlib/config/ocean/drive.ini").read_text(encoding="utf-8")
    assert "partner_mode" not in text
    assert "DriveSteerConstrained" not in text


def test_puffer_drive_creator_still_drive():
    from pufferlib.ocean.environment import MAKE_FUNCTIONS

    assert MAKE_FUNCTIONS["drive"] == "Drive"
