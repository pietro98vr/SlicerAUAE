"""Backward-compatibility shim.

The CUDA/torch setup was replaced by the Slicer-compliant Dependencies module, which
delegates torch installation to the PyTorch extension via the NNUNet extension's
InstallLogic instead of hand-rolling a torch install. This shim re-exports Dependencies
so any old import keeps working.
"""

from .Dependencies import (  # noqa: F401
    MIN_SLICER_VERSION,
    EXTRA_PACKAGES,
    report,
    ensure,
    torchStatus,
    nnunetStatus,
    isNNUNetExtensionAvailable,
    isPyTorchExtensionAvailable,
    parameterDeviceKwargs,
)
