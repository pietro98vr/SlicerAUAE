"""Autonomous, Slicer-compliant dependency preflight.

Added by this project. Slicer extensions must not fail to import when optional heavy packages are
missing, and should install what they need on demand through the official channels. This
module centralises that:

  * torch (with the correct CUDA build) and nnU-Net are installed by the *NNUNet* Slicer
    extension's own ``InstallLogic.setupPythonRequirements()``, which in turn installs the
    *PyTorch* Slicer extension (PyTorchUtils) and picks the best torch wheel for the machine.
    We do NOT hand-roll a torch install (that would fight the PyTorch extension).
  * our two extra pure-Python needs (scipy, PyGithub) are installed with
    ``slicer.util.pip_install`` only if missing.
  * ``report()`` produces a required-vs-installed table and a clear CUDA verdict so the user
    knows, before running, whether inference will use the GPU or fall back to (slow) CPU.

Everything degrades gracefully: if the NNUNet extension is absent the module still loads and
tells the user how to get it (it is declared as an extension dependency, so the Extensions
Manager installs it automatically when this extension is installed).
"""

import importlib.util

import slicer

# Minimum 3D Slicer version (upstream targets the 5.9.0+ preview line).
MIN_SLICER_VERSION = (5, 9, 0)

# Extra pure-Python packages this project needs, as {distribution_name: import_name}.
# torch / torchvision / nnunetv2 are intentionally absent: the NNUNet + PyTorch extensions own them.
EXTRA_PACKAGES = {"scipy": "scipy", "PyGithub": "github"}


def isNNUNetExtensionAvailable():
    """True when the NNUNet Slicer extension (SlicerNNUNetLib) is importable."""
    return importlib.util.find_spec("SlicerNNUNetLib") is not None


def isPyTorchExtensionAvailable():
    """True when the PyTorch Slicer extension (PyTorchUtils) is importable."""
    return importlib.util.find_spec("PyTorchUtils") is not None


def _distVersion(distName):
    try:
        import importlib.metadata as md
        return md.version(distName)
    except Exception:  # noqa: BLE001
        return None


def slicerVersionTuple():
    try:
        parts = slicer.app.applicationVersion.split("-")[0].split(".")
        return tuple(int(p) for p in (parts + ["0", "0", "0"])[:3])
    except Exception:  # noqa: BLE001
        return (0, 0, 0)


def torchStatus():
    """Return a dict describing the installed torch build and CUDA usability."""
    status = {"installed": False, "version": None, "cudaBuild": None, "cudaAvailable": False,
              "device": None, "vramGb": None}
    try:
        import torch
        status["installed"] = True
        status["version"] = torch.__version__
        status["cudaBuild"] = torch.version.cuda
        status["cudaAvailable"] = bool(torch.cuda.is_available())
        if status["cudaAvailable"]:
            status["device"] = torch.cuda.get_device_name(0)
            try:
                free, total = torch.cuda.mem_get_info(0)
                status["vramGb"] = round(total / (1024 ** 3), 1)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return status


def nnunetStatus():
    """Return (installedVersionString, compatibleBool) using the NNUNet extension's own logic."""
    if not isNNUNetExtensionAvailable():
        return (None, False)
    try:
        from SlicerNNUNetLib import InstallLogic
        logic = InstallLogic()
        version = logic.getInstalledNNUnetVersion()
        compatible = bool(logic.isPackageInstalledAndCompatible("nnunetv2"))
        return (str(version) if version is not None else None, compatible)
    except Exception:  # noqa: BLE001
        return (None, False)


def report(log):
    """Print a required-vs-installed dependency report and return True when ready to run.

    'Ready' means: Slicer new enough, NNUNet extension present, nnU-Net compatible, and the
    extra packages present. CUDA is reported but never required (CPU is a valid, slower path).
    """
    ready = True
    log("=== Dependency check ===")

    sv = slicerVersionTuple()
    okSlicer = sv >= MIN_SLICER_VERSION
    log("3D Slicer:            installed %s | required >= %s | %s"
        % (".".join(map(str, sv)), ".".join(map(str, MIN_SLICER_VERSION)), "OK" if okSlicer else "TOO OLD"))
    ready = ready and okSlicer

    hasNN = isNNUNetExtensionAvailable()
    log("NNUNet extension:     %s" % ("installed" if hasNN else "MISSING (install 'NNUNet' from the Extensions Manager)"))
    ready = ready and hasNN

    hasPT = isPyTorchExtensionAvailable()
    log("PyTorch extension:    %s" % ("installed" if hasPT else "not yet (installed automatically with dependencies)"))

    nnVer, nnCompat = nnunetStatus()
    if nnVer:
        log("nnunetv2:             installed %s | compatible: %s" % (nnVer, "yes" if nnCompat else "NO (will be updated)"))
        ready = ready and nnCompat
    else:
        log("nnunetv2:             not installed (will be installed by dependencies)")
        ready = False

    ts = torchStatus()
    if ts["installed"]:
        log("torch:                %s | CUDA build: %s" % (ts["version"], ts["cudaBuild"]))
    else:
        log("torch:                not installed (installed by the PyTorch extension via dependencies)")
        ready = False

    for dist, imp in EXTRA_PACKAGES.items():
        present = importlib.util.find_spec(imp) is not None
        log("%-20s  %s" % (dist + ":", ("installed " + str(_distVersion(dist))) if present else "MISSING (installed by dependencies)"))
        ready = ready and present

    # CUDA verdict
    log("--- CUDA verdict ---")
    if ts["installed"] and ts["cudaAvailable"]:
        vram = (" (%.1f GB VRAM)" % ts["vramGb"]) if ts["vramGb"] else ""
        log("GPU inference AVAILABLE: %s%s. Inference will use CUDA." % (ts["device"], vram))
    elif ts["installed"] and ts["cudaBuild"] is None:
        log("CPU-only torch build: inference will run on CPU (slow, up to ~1 h). "
            "To enable GPU, ensure an NVIDIA GPU + driver, then reinstall dependencies so the "
            "PyTorch extension fetches a CUDA build.")
    elif ts["installed"]:
        log("torch has a CUDA build but no GPU is currently usable (driver/GPU not detected). "
            "Inference will run on CPU.")
    else:
        log("CUDA status unknown until torch is installed. Run 'Install / update dependencies'.")

    log("=== %s ===" % ("READY to run" if ready else "NOT ready - run 'Install / update dependencies'"))
    return ready


def ensure(log, askConfirmation=True):
    """Install everything needed to run, the compliant way. Returns (ok, needsRestart).

    Delegates torch + nnU-Net to the NNUNet extension's InstallLogic (which installs the
    PyTorch extension and the best CUDA torch build), then installs the extra pure-Python
    packages. If the PyTorch extension had to be installed, Slicer must be restarted before
    inference; that is surfaced as needsRestart=True.
    """
    if not isNNUNetExtensionAvailable():
        log("The 'NNUNet' Slicer extension is required and was not found. "
            "Install it from the Extensions Manager (it is a declared dependency of this "
            "extension) and restart Slicer, then try again.")
        return (False, False)

    from SlicerNNUNetLib import InstallLogic
    installLogic = InstallLogic()
    try:
        installLogic.doAskConfirmation = bool(askConfirmation)
    except Exception:  # noqa: BLE001
        pass
    try:
        installLogic.progressInfo.connect(log)
    except Exception:  # noqa: BLE001
        pass

    log("Installing/validating torch + nnU-Net via the NNUNet extension (CUDA-aware)...")
    ok = bool(installLogic.setupPythonRequirements())
    needsRestart = bool(getattr(installLogic, "needsRestart", False))
    if needsRestart:
        log("The PyTorch extension was installed. Please RESTART Slicer, then run again to finish.")
        return (ok, True)

    # Extra pure-Python packages we use directly.
    for dist, imp in EXTRA_PACKAGES.items():
        if importlib.util.find_spec(imp) is None:
            log("Installing " + dist + " ...")
            try:
                slicer.util.pip_install(dist)
            except Exception as exc:  # noqa: BLE001
                log("Could not install " + dist + ": " + str(exc))
                ok = False

    log("Dependency setup %s." % ("completed" if ok else "did not fully complete"))
    return (ok, False)


def parameterDeviceKwargs(deviceChoice):
    """Map a UI device choice to Parameter kwargs. 'Auto' lets nnU-Net decide."""
    choice = (deviceChoice or "Auto").strip().lower()
    if choice == "cuda":
        return {"device": "cuda"}
    if choice == "cpu":
        return {"device": "cpu"}
    return {}
