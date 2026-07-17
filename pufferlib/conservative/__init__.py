"""Conservative partner mix training (opt-in via puffer_drive_conservative_mix)."""

__all__ = [
    "ConservativeMixDrive",
]


def __getattr__(name):
    if name == "ConservativeMixDrive":
        from pufferlib.conservative.env import ConservativeMixDrive

        return ConservativeMixDrive
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
