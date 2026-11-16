"""Dependency preflight and one-click installation.

Added by this project. Slicer extensions must not fail to import when heavy packages are
missing, and should install what they need on demand through the official channels. This
module does that in one call, ``ensure()``:

  * the *PyTorch* Slicer extension (PyTorchUtils) is installed when missing;
  * the machine is inspected (NVIDIA GPU generation and driver through ``nvidia-smi``, Apple
    Silicon, or neither) and the matching PyTorch build is chosen from a small table, see
    ``recommendedBuild``. The automatic choice made by light-the-torch only looks at the
    driver version, which is how a new GPU ends up with a build that has no kernels for it;
  * a PyTorch that cannot work on this machine (CPU-only build next to a usable GPU, a build
    without kernels for the GPU, torch and torchvision from different builds, a broken
    import) is removed cleanly, leftovers included, and replaced. A working install is never
    touched;
  * PyTorch is installed with light-the-torch, the installer of the PyTorch extension, and
    nnU-Net through the *NNUNet* extension's ``InstallLogic``, so nothing fights the two
    extensions. light-the-torch runs in its own process: loading it (and pip with it) inside
    Slicer breaks the next ``import torch`` of the session, see ``repairTypingHooks``;
  * the result is verified in a separate Python process (a small convolution on the GPU), so
    PyTorch is not loaded into Slicer just to be checked and stays replaceable.

``report()`` prints the required-versus-installed table and the verdict for the device.
Everything degrades gracefully: if the NNUNet extension is absent the module still loads and
tells the user how to get it.
"""

import importlib
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import traceback

import slicer

# Minimum 3D Slicer version (upstream targets the 5.9.0+ preview line).
MIN_SLICER_VERSION = (5, 9, 0)

# Extra pure-Python packages this project needs, as {distribution_name: import_name}.
# torch / torchvision / nnunetv2 are intentionally absent: they are handled separately below.
EXTRA_PACKAGES = {"scipy": "scipy", "PyGithub": "github"}

# --- PyTorch build table ---------------------------------------------------------------
# Compute capability levels (major * 10 + minor) that split the NVIDIA generations.
SM_OLDEST_SUPPORTED = 50   # Maxwell (GTX 900). Kepler and older have no current build.
SM_TURING = 75             # GTX 16 / RTX 20. CUDA 12.8+ and 13.x builds start here.
SM_BLACKWELL = 100         # RTX 50 (sm_120). Needs CUDA 12.8 or newer.

# The builds this module installs. Both CUDA 12 indexes are frozen upstream, so the choice
# stays valid: cu126 is the last one with kernels for sm_50 to sm_90 (GTX 900 up to RTX 40),
# cu128 is the first one with sm_100 / sm_120 (RTX 50). cu118 is only for old drivers.
BACKEND_BLACKWELL = "cu128"
BACKEND_STANDARD = "cu126"
BACKEND_OLD_DRIVER = "cu118"

# Oldest NVIDIA driver each CUDA runtime line works with (NVIDIA CUDA release notes).
MIN_DRIVER = {
    11: {"Windows": (452, 39), "Linux": (450, 80, 2)},
    12: {"Windows": (528, 33), "Linux": (525, 60, 13)},
    13: {"Windows": (580,), "Linux": (580,)},
}

# nnunetv2 asks for at least this torch.
MIN_TORCH = "2.1.2"
# torch 2.13 ships license files nested so deep that a default Slicer install on Windows goes
# past the path-length limit ([WinError 206]) and is left half extracted. 2.12 is the newest
# release verified inside Slicer on Windows; raise the cap once a later one is confirmed.
WINDOWS_TORCH_CAP = "<2.13"
WINDOWS_TORCHVISION_CAP = "<0.28"

# Folders a PyTorch install owns in site-packages. pip renames them with a leading '~' while
# it replaces them and leaves those copies behind when a file is locked; torchgen then fails
# to import on the next install.
TORCH_FOLDERS = ("torch", "torchgen", "torchvision", "torchaudio", "functorch")

RESUME_SETTINGS_KEY = "AUAE/resumeDependencyInstall"
PROBE_SETTINGS_KEY = "AUAE/torchProbe"

_hardwareCache = None
_otherGpuCache = None
_probeCache = {}


# --- Small helpers ---------------------------------------------------------------------
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


def _versionTuple(text):
    """'621.24' -> (621, 24). Returns None when no number is found."""
    numbers = re.findall(r"\d+", str(text or ""))
    return tuple(int(n) for n in numbers) if numbers else None


def slicerVersionTuple():
    try:
        parts = slicer.app.applicationVersion.split("-")[0].split(".")
        return tuple(int(p) for p in (parts + ["0", "0", "0"])[:3])
    except Exception:  # noqa: BLE001
        return (0, 0, 0)


def _run(args, timeout=10):
    """Run a system command quietly. Returns its stdout, or None when missing or failing.

    The command gets the environment Slicer was started from, so system tools do not pick up
    the libraries bundled with Slicer.
    """
    try:
        env = slicer.util.startupEnvironment()
    except Exception:  # noqa: BLE001
        env = None
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:  # noqa: BLE001
        return None
    return completed.stdout if completed.returncode == 0 else None


# --- Hardware --------------------------------------------------------------------------
# Used only when an old nvidia-smi cannot report the compute capability itself.
_NAME_TO_CAPABILITY = (
    (r"RTX\s*50\d\d", (12, 0)),
    (r"RTX\s*40\d\d", (8, 9)),
    (r"RTX\s*30\d\d", (8, 6)),
    (r"RTX\s*20\d\d|GTX\s*16\d\d|TITAN RTX", (7, 5)),
    (r"GTX\s*10\d\d|TITAN Xp", (6, 1)),
    (r"GTX\s*9\d\d|GTX\s*750", (5, 2)),
    (r"GTX\s*[67]\d\d|GT\s*[67]\d\d", (3, 5)),
)


def _capabilityFromName(name):
    for pattern, capability in _NAME_TO_CAPABILITY:
        if re.search(pattern, name or "", re.IGNORECASE):
            return capability
    return None


def parseNvidiaSmi(text, hasCapability=True):
    """Parse 'name, driver[, compute_cap], memory' CSV lines from nvidia-smi into GPU dicts."""
    gpus = []
    for line in (text or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < (4 if hasCapability else 3) or not fields[0]:
            continue
        capability = _versionTuple(fields[2]) if hasCapability else None
        if capability is None or len(capability) < 2:
            capability = _capabilityFromName(fields[0])
        memory = _versionTuple(fields[-1])
        gpus.append({
            "name": fields[0],
            "driver": fields[1],
            "computeCapability": tuple(capability[:2]) if capability else None,
            "vramMb": memory[0] if memory else None,
        })
    return gpus


def _queryNvidiaSmi():
    query = "--query-gpu=name,driver_version,compute_cap,memory.total"
    out = _run(["nvidia-smi", query, "--format=csv,noheader,nounits"])
    if out is not None:
        return parseNvidiaSmi(out, hasCapability=True)
    # Older drivers do not know the compute_cap field.
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    return parseNvidiaSmi(out, hasCapability=False) if out is not None else []


def detectHardware(refresh=False):
    """Describe the machine: OS, NVIDIA GPUs (name, driver, compute capability), Apple Silicon.

    Works before PyTorch is installed, which is the point: the build is chosen from this.
    """
    global _hardwareCache
    if _hardwareCache is None or refresh:
        system = platform.system()
        _hardwareCache = {
            "system": system,
            "machine": platform.machine(),
            "nvidia": _queryNvidiaSmi(),
            "appleSilicon": system == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"),
        }
    return _hardwareCache


def otherGpuNames():
    """Names of the display adapters, for the report on machines without an NVIDIA GPU."""
    global _otherGpuCache
    if _otherGpuCache is None:
        names = []
        system = platform.system()
        if system == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_VideoController).Name"], timeout=15)
            names = [line.strip() for line in (out or "").splitlines() if line.strip()]
        elif system == "Linux":
            out = _run(["lspci"])
            names = [line.split(": ", 1)[-1].strip() for line in (out or "").splitlines()
                     if re.search(r"VGA|3D controller|Display", line)]
        _otherGpuCache = names
    return _otherGpuCache


def _smLevel(capability):
    return capability[0] * 10 + capability[1] if capability else None


def _driverSupports(driver, cudaMajor, system):
    """True when the driver is new enough for a CUDA runtime line (or cannot be compared)."""
    minimum = MIN_DRIVER.get(cudaMajor, {}).get(system)
    installed = _versionTuple(driver)
    return minimum is None or installed is None or installed >= minimum


def _gpuLabel(gpu):
    sm = _smLevel(gpu.get("computeCapability"))
    return "%s (%s, driver %s)" % (gpu.get("name"), "sm_%d" % sm if sm is not None else "capability unknown",
                                   gpu.get("driver") or "unknown")


def recommendedBuild(hardware):
    """Pick the PyTorch build for a machine. Pure function of detectHardware()'s result.

    Returns {"backend", "device", "reason"}. backend is a light-the-torch computation backend
    ('cu126', 'cu128', 'cu118', 'cpu'), or None to let the installer take the platform
    default (macOS). device is what inference will run on: 'cuda', 'mps' or 'cpu'.
    """
    system = hardware.get("system")
    nvidia = hardware.get("nvidia") or []
    if nvidia:
        gpu = nvidia[0]  # nnU-Net runs on the first CUDA device
        label = _gpuLabel(gpu)
        sm = _smLevel(gpu.get("computeCapability"))
        if sm is not None and sm < SM_OLDEST_SUPPORTED:
            return {"backend": "cpu", "device": "cpu",
                    "reason": label + " is older than every GPU generation current PyTorch builds support."}
        if not _driverSupports(gpu.get("driver"), 12, system):
            if (sm is None or sm < SM_BLACKWELL) and _driverSupports(gpu.get("driver"), 11, system):
                return {"backend": BACKEND_OLD_DRIVER, "device": "cuda",
                        "reason": label + " has a driver too old for CUDA 12, so the CUDA 11.8 build is used. "
                                          "Update the NVIDIA driver to get a newer build."}
            return {"backend": "cpu", "device": "cpu",
                    "reason": label + " has a driver too old for any usable CUDA build. "
                                      "Update the NVIDIA driver, then install the dependencies again."}
        if sm is not None and sm >= SM_BLACKWELL:
            return {"backend": BACKEND_BLACKWELL, "device": "cuda",
                    "reason": label + " needs a CUDA 12.8 build (the first with kernels for this generation)."}
        return {"backend": BACKEND_STANDARD, "device": "cuda",
                "reason": label + " is covered by the CUDA 12.6 build."}
    if hardware.get("appleSilicon"):
        return {"backend": None, "device": "mps", "reason": "Apple Silicon: the standard build runs on the GPU through MPS."}
    if system == "Darwin":
        return {"backend": None, "device": "cpu", "reason": "Intel Mac: the standard build runs on the CPU."}
    return {"backend": "cpu", "device": "cpu",
            "reason": "No NVIDIA GPU found. PyTorch cannot run nnU-Net on AMD or Intel graphics here, "
                      "so the CPU build is used."}


# --- Installed PyTorch, inspected without importing it ---------------------------------
def _sitePackagesDirs():
    dirs = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path and os.path.isdir(path) and path not in dirs:
            dirs.append(path)
    return dirs


def _cudaFromVersionFile():
    """CUDA version recorded in torch/version.py, for wheels whose version has no '+cuXXX' tag."""
    for site in _sitePackagesDirs():
        try:
            with open(os.path.join(site, "torch", "version.py"), "r", encoding="utf-8") as handle:
                match = re.search(r"^cuda\b[^=\n]*=\s*['\"]([\d.]+)['\"]", handle.read(), re.MULTILINE)
        except Exception:  # noqa: BLE001
            continue
        if match:
            return _versionTuple(match.group(1))[:2]
    return None


def _cudaFromVersion(version):
    """'2.12.0+cu126' -> (12, 6); '2.12.0+cpu' -> None; no tag -> 'unknown'."""
    local = str(version or "").partition("+")[2].lower()
    match = re.match(r"cu(\d+)(\d)$", local)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None if local == "cpu" else "unknown"


def installedTorch():
    """Describe the installed PyTorch from its files, without importing it.

    Importing torch loads its DLLs into Slicer, after which it cannot be replaced until a
    restart, so the decision whether to replace it must not need an import. Returns None when
    torch is not installed, else {"version", "cuda", "torchvision", "loaded"}; cuda is a
    (major, minor) tuple or None for a CPU-only build.
    """
    version = _distVersion("torch")
    if version is None:
        return None
    cuda = _cudaFromVersion(version)
    if cuda == "unknown":
        cuda = _cudaFromVersionFile()
    return {"version": version, "cuda": cuda, "torchvision": _distVersion("torchvision"),
            "loaded": "torch" in sys.modules}


def pairMismatch(torchVersion, torchvisionVersion):
    """True when torch and torchvision cannot work together (different build or release)."""
    if not torchVersion or not torchvisionVersion:
        return False
    if torchVersion.partition("+")[2] != torchvisionVersion.partition("+")[2]:
        return True
    torch = _versionTuple(torchVersion.partition("+")[0])
    vision = _versionTuple(torchvisionVersion.partition("+")[0])
    # Since torch 2.0 the releases are paired as torch 2.N with torchvision 0.(N + 15).
    if torch and vision and len(torch) > 1 and len(vision) > 1 and torch[0] == 2 and vision[0] == 0:
        return vision[1] != torch[1] + 15
    return False


def torchAction(installed, hardware, recommended):
    """Decide what to do with PyTorch: ('none' | 'install' | 'replace', reason).

    A working install is left alone even when it is not the build this module would pick.
    'replace' is returned only for an install that cannot work on this machine.
    """
    if installed is None:
        return ("install", "PyTorch is not installed.")
    if pairMismatch(installed["version"], installed.get("torchvision")):
        return ("replace", "torch %s and torchvision %s do not belong together."
                % (installed["version"], installed.get("torchvision")))
    if recommended.get("device") != "cuda":
        return ("none", "")

    gpu = (hardware.get("nvidia") or [{}])[0]
    sm = _smLevel(gpu.get("computeCapability"))
    cuda = installed.get("cuda")
    if cuda is None:
        return ("replace", "torch %s is a CPU-only build, but this machine has a usable NVIDIA GPU."
                % installed["version"])
    cudaText = "%d.%d" % (cuda[0], cuda[1])
    if sm is not None and sm >= SM_BLACKWELL and cuda < (12, 8):
        return ("replace", "torch %s is built for CUDA %s, which has no kernels for this GPU (sm_%d)."
                % (installed["version"], cudaText, sm))
    if sm is not None and sm < SM_TURING and cuda > (12, 6):
        return ("replace", "torch %s is built for CUDA %s, which dropped this GPU generation (sm_%d)."
                % (installed["version"], cudaText, sm))
    if not _driverSupports(gpu.get("driver"), cuda[0], hardware.get("system")):
        return ("replace", "torch %s is built for CUDA %s, which needs a newer NVIDIA driver than %s."
                % (installed["version"], cudaText, gpu.get("driver")))
    return ("none", "")


# --- PyTorch probe in a separate process -----------------------------------------------
_PROBE_SCRIPT = r'''
import json
out = {"installed": False}
try:
    import torch
    out.update(installed=True, version=torch.__version__, cudaBuild=torch.version.cuda,
               cudaAvailable=bool(torch.cuda.is_available()))
    try:
        out["archList"] = list(torch.cuda.get_arch_list())
    except Exception:
        out["archList"] = None
    try:
        out["mpsAvailable"] = bool(torch.backends.mps.is_available())
    except Exception:
        out["mpsAvailable"] = False
    if torch.cuda.device_count() > 0:
        cc = torch.cuda.get_device_capability(0)
        out["computeCapability"] = "%d.%d" % (cc[0], cc[1])
        out["device"] = torch.cuda.get_device_name(0)
    if out["cudaAvailable"]:
        free, total = torch.cuda.mem_get_info(0)
        out["vramGb"] = round(total / 2 ** 30, 1)
        try:
            x = torch.ones((1, 1, 8, 8, 8), device="cuda")
            y = torch.nn.functional.conv3d(x, torch.ones((1, 1, 3, 3, 3), device="cuda"))
            out["cudaSmokeTest"] = "ok" if abs(float(y.sum().item()) - 27.0 * 216) < 1e-3 else "wrong result"
        except Exception as exc:
            out["cudaSmokeTest"] = str(exc)[:300]
except Exception as exc:
    out["error"] = str(exc)[:300]
print("AUAE_PROBE=" + json.dumps(out))
'''


def _probeSignature():
    """Identifies the torch install and the GPU setup a probe result belongs to."""
    gpu = (detectHardware().get("nvidia") or [{}])[0]
    return "|".join(str(part) for part in (
        _distVersion("torch"), _distVersion("torchvision"), gpu.get("name"), gpu.get("driver")))


def _runProbe():
    """Import torch in a separate Slicer Python process and return what it reports."""
    executable = shutil.which("PythonSlicer") or sys.executable
    try:
        process = slicer.util.launchConsoleProcess([executable, "-c", _PROBE_SCRIPT], useStartupEnvironment=False)
        output, _ = process.communicate(timeout=300)
    except Exception as exc:  # noqa: BLE001
        return {"installed": True, "error": "could not start the check process: %s" % exc}
    for line in (output or "").splitlines():
        if line.startswith("AUAE_PROBE="):
            try:
                return json.loads(line[len("AUAE_PROBE="):])
            except ValueError:
                break
    tail = " ".join((output or "").split())[-300:]
    return {"installed": True, "error": "the check process gave no result. " + tail}


def _archCompatibility(cc, archList):
    """Compare the GPU compute capability with the torch build's compiled architectures.

    Returns (status, message) where status is 'ok', 'warn', or 'unknown'. 'warn' means the GPU
    is newer than every architecture the torch build was compiled for, so CUDA kernels will
    fail to launch and the user needs a newer torch build (e.g. a cu128 wheel for sm_120).
    """
    if not archList:
        return ("unknown", None)
    gpu = cc[0] * 10 + cc[1]
    smLevels = []
    for arch in archList:
        if arch.startswith("sm_") and arch[3:].isdigit():
            smLevels.append(int(arch[3:]))  # sm_90 -> 90, sm_120 -> 120
    exact = ("sm_%d" % gpu) in archList
    maxSm = max(smLevels) if smLevels else 0
    if exact or gpu <= maxSm:
        return ("ok", "GPU sm_%d is covered by the installed torch build." % gpu)
    return ("warn",
            "GPU sm_%d is NEWER than every architecture this torch build was compiled for "
            "(max sm_%d). CUDA kernels will fail with 'no kernel image is available'." % (gpu, maxSm))


def torchStatus(refresh=False):
    """Return a dict describing the installed torch build, the GPU, and CUDA usability.

    The facts come from a probe that imports torch in a separate process, so torch is never
    loaded into Slicer by this check. The probe takes several seconds, so its result is kept
    in the settings and reused until torch, torchvision, the GPU or the driver change.
    """
    status = {"installed": False, "version": None, "cudaBuild": None, "cudaAvailable": False,
              "device": None, "vramGb": None, "computeCapability": None, "archList": None,
              "mpsAvailable": False, "cudaSmokeTest": None, "error": None,
              "archStatus": "unknown", "archMessage": None}
    if _distVersion("torch") is None:
        return status

    signature = _probeSignature()
    probe = None if refresh else _probeCache.get(signature)
    if probe is None and not refresh:
        probe = _loadStoredProbe(signature)
    if probe is None:
        probe = _runProbe()
        if not probe.get("error"):
            _storeProbe(signature, probe)
    _probeCache.clear()
    _probeCache[signature] = probe

    status.update({key: value for key, value in probe.items() if key in status})
    capability = _versionTuple(status["computeCapability"])
    if capability and len(capability) > 1:
        status["archStatus"], status["archMessage"] = _archCompatibility(capability, status["archList"])
    smoke = status["cudaSmokeTest"]
    if smoke and smoke != "ok":
        status["archStatus"] = "warn"
        status["archMessage"] = "A test computation on the GPU failed: " + smoke
    return status


def cudaUsable(status):
    """True when the probe shows CUDA inference will work, not just that a GPU is listed."""
    return bool(status.get("installed") and status.get("cudaAvailable")
                and status.get("archStatus") != "warn" and status.get("cudaSmokeTest") in (None, "ok"))


def _settings():
    import qt
    return qt.QSettings()


def _loadStoredProbe(signature):
    try:
        stored = json.loads(_settings().value(PROBE_SETTINGS_KEY, "") or "{}")
        return stored.get("probe") if stored.get("signature") == signature else None
    except Exception:  # noqa: BLE001
        return None


def _storeProbe(signature, probe):
    try:
        _settings().setValue(PROBE_SETTINGS_KEY, json.dumps({"signature": signature, "probe": probe}))
    except Exception:  # noqa: BLE001
        pass


# --- Resume after a restart ------------------------------------------------------------
def resumeAttempts():
    """How many times in a row the install asked for a restart (0 = nothing pending)."""
    try:
        return int(_settings().value(RESUME_SETTINGS_KEY, 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def isResumePending():
    return resumeAttempts() > 0


def setResumeAttempts(count):
    try:
        if count:
            _settings().setValue(RESUME_SETTINGS_KEY, int(count))
        else:
            _settings().remove(RESUME_SETTINGS_KEY)
    except Exception:  # noqa: BLE001
        pass


# --- Reports ---------------------------------------------------------------------------
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


def summary():
    """One short paragraph for the panel: the hardware, the build it needs, what is installed.

    Cheap on purpose (one nvidia-smi call, no torch import), so it can run when the panel opens.
    """
    hardware = detectHardware()
    recommended = recommendedBuild(hardware)
    installed = installedTorch()
    action, why = torchAction(installed, hardware, recommended)
    build = recommended["backend"] or "standard"
    lines = [recommended["reason"]]
    if installed is None:
        lines.append("PyTorch is not installed yet. 'Install dependencies' installs the %s build." % build)
    elif action == "replace":
        lines.append(why + " 'Install dependencies' replaces it with the %s build." % build)
    else:
        lines.append("Installed: torch %s. Inference device: %s." % (installed["version"], recommended["device"].upper()))
    return "\n".join(lines)


def report(log):
    """Print a required-vs-installed dependency report and return True when ready to run.

    'Ready' means: Slicer new enough, NNUNet extension present, nnU-Net compatible, a PyTorch
    that suits this machine, and the extra packages present. Running on the CPU is a valid,
    slower path, so a missing GPU never makes the report fail.
    """
    ready = True
    repairTypingHooks()  # so the torch import nnU-Net does next cannot trip over a loaded pip
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
    log("PyTorch extension:    %s" % ("installed" if hasPT else "not yet (installed with the dependencies)"))

    nnVer, nnCompat = nnunetStatus()
    if nnVer:
        log("nnunetv2:             installed %s | compatible: %s" % (nnVer, "yes" if nnCompat else "NO (will be updated)"))
        ready = ready and nnCompat
    else:
        log("nnunetv2:             not installed (installed with the dependencies)")
        ready = False

    for dist, imp in EXTRA_PACKAGES.items():
        present = importlib.util.find_spec(imp) is not None
        log("%-20s  %s" % (dist + ":", ("installed " + str(_distVersion(dist))) if present else "MISSING (installed with the dependencies)"))
        ready = ready and present

    log("--- Hardware and PyTorch ---")
    hardware = detectHardware()
    for gpu in hardware.get("nvidia") or []:
        vram = (", %.1f GB" % (gpu["vramMb"] / 1024.0)) if gpu.get("vramMb") else ""
        log("NVIDIA GPU:           %s%s" % (_gpuLabel(gpu), vram))
    if not hardware.get("nvidia"):
        others = otherGpuNames()
        log("NVIDIA GPU:           none%s" % ((" | display adapters: " + ", ".join(others)) if others else ""))
    recommended = recommendedBuild(hardware)
    log("Build for this PC:    %s | %s" % (recommended["backend"] or "standard", recommended["reason"]))

    installed = installedTorch()
    action, why = torchAction(installed, hardware, recommended)
    if installed is None:
        log("torch:                not installed (installed with the dependencies)")
        ready = False
    else:
        log("torch:                %s | torchvision %s" % (installed["version"], installed.get("torchvision") or "not installed"))
    if action == "replace":
        log("PROBLEM: " + why)
        ready = False
    if torchImportBroken():
        log("PROBLEM: PyTorch failed to load earlier in this Slicer session. Restart Slicer.")
        ready = False

    if installed is not None and action == "none":
        ts = torchStatus()
        if ts["error"]:
            log("PROBLEM: PyTorch is installed but does not import: " + ts["error"])
            ready = False
        elif ts["archStatus"] == "warn":
            log("PROBLEM: " + (ts["archMessage"] or "the GPU cannot be used by this PyTorch build."))
            ready = False
        elif cudaUsable(ts):
            vram = (" (%.1f GB VRAM)" % ts["vramGb"]) if ts["vramGb"] else ""
            log("Inference device:     CUDA, %s%s. A test computation on the GPU succeeded." % (ts["device"], vram))
        elif ts["mpsAvailable"]:
            log("Inference device:     MPS (Apple GPU).")
        elif recommended["device"] == "cuda":
            log("Inference device:     CPU. The GPU is present but PyTorch cannot use it right now "
                "(driver not loaded?). Inference still runs, slowly.")
        else:
            log("Inference device:     CPU (slow, up to about 1 h per volume).")

    log("=== %s ===" % ("READY to run" if ready else "NOT ready: click 'Install dependencies'"))
    return ready


# --- Installation ----------------------------------------------------------------------
def _confirm(message):
    """Ask before installing. Without a main window (headless runs) there is nobody to ask."""
    try:
        if slicer.util.mainWindow() is None:
            return True
        return bool(slicer.util.confirmOkCancelDisplay(message, windowTitle="AUAE dependencies"))
    except Exception:  # noqa: BLE001
        return True


def _installPyTorchExtension():
    manager = slicer.app.extensionsManagerModel()
    if not manager.isExtensionInstalled("PyTorch") and not manager.installExtensionFromServer("PyTorch"):
        raise RuntimeError("Could not install the PyTorch extension from the Extensions server. "
                           "Install 'PyTorch' from the Extensions Manager, restart Slicer and try again.")


def isTorchLeftover(name):
    """True for the '~orch', '~orchvision', '~unctorch'... folders pip leaves in site-packages."""
    if not name.startswith("~"):
        return False
    stem = re.split(r"[-.]", name.lstrip("-~.=%0123456789"), maxsplit=1)[0]
    return len(stem) >= 3 and any(folder != stem and folder.endswith(stem) for folder in TORCH_FOLDERS)


def _torchLeftovers(includeInstalled=False):
    """Paths of stale PyTorch folders. includeInstalled also lists the live package folders."""
    paths = []
    for site in _sitePackagesDirs():
        for name in os.listdir(site):
            lowered = name.lower()
            stale = isTorchLeftover(name)
            if includeInstalled and not stale:
                stale = (lowered in TORCH_FOLDERS or lowered == "torchvision.libs"
                         or re.match(r"(torch|torchvision|torchaudio)-[\d.]+.*\.dist-info$", lowered) is not None)
            if stale:
                paths.append(os.path.join(site, name))
    return paths


def _deleteFolders(paths, log):
    """Delete folders; return the ones that could not be removed (files in use)."""
    failed = []
    for path in paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            log("Removed leftover " + os.path.basename(path))
        except Exception:  # noqa: BLE001
            failed.append(path)
    return failed


def _removeTorch(log):
    """Uninstall PyTorch and wipe what pip leaves behind. Raises when files are still in use."""
    import importlib.metadata as md
    packages = [name for name in ("torch", "torchvision", "torchaudio") if _distVersion(name)]
    # On Linux the CUDA runtime comes as separate nvidia-*-cuNN packages. Mixing the sets of
    # two CUDA versions is a classic source of load errors, so they go too; the new install
    # brings back exactly the ones it needs.
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "") if dist.metadata else ""
        if re.match(r"nvidia[-_].+[-_]cu\d+$", name.lower()) and name not in packages:
            packages.append(name)
    if packages:
        log("Removing the current PyTorch (%s)..." % ", ".join(packages))
        try:
            slicer.util.pip_uninstall(" ".join(packages))
        except Exception as exc:  # noqa: BLE001
            log("pip uninstall reported: %s. Cleaning up by hand." % exc)
    importlib.invalidate_caches()
    failed = _deleteFolders(_torchLeftovers(includeInstalled=True), log)
    if failed:
        raise RuntimeError("some PyTorch files are in use and could not be removed: "
                           + ", ".join(os.path.basename(path) for path in failed))
    importlib.invalidate_caches()


def runSlicerPython(arguments):
    """Run Slicer's own Python (PythonSlicer) as a separate process and wait for it.

    The output goes to the Slicer log while the application stays responsive, and a failing
    process raises subprocess.CalledProcessError. This is built from the public helpers of
    slicer.util on purpose: the private ``slicer.util._executePythonModule`` does the same,
    but it was dropped and later restored during the packaging rework of Slicer 5.12, so code
    that leans on it breaks on some builds.
    """
    executable = shutil.which("PythonSlicer") or sys.executable
    process = slicer.util.launchConsoleProcess([executable, *arguments], useStartupEnvironment=False)
    logOutput = getattr(slicer.util, "logProcessOutput", None)
    if logOutput is not None:
        logOutput(process)
        return
    for line in process.stdout:  # a Slicer without logProcessOutput: do what it does
        logging.info(line.rstrip())
        slicer.app.processEvents()
    process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, [executable, *arguments])


def _downloadNote(backend):
    """What to expect from the download: the CUDA builds bundle the CUDA runtime."""
    if backend and backend.startswith("cu"):
        return "The download is about 2-3 GB and can take several minutes..."
    return "The download is a few hundred MB..."


def _installTorch(backend, log):
    """Install torch + torchvision for one computation backend with light-the-torch.

    This is what ``PyTorchUtilsLogic.installTorch`` does (same tool, same arguments), run
    without its two in-process imports. That method imports ``light_the_torch._patch``, which
    loads pip and its private copy of typing_extensions into Slicer, and then imports torch.
    The second copy of typing_extensions takes over the generic-type checks of ``typing``, and
    the torch import that follows fails with "Type parameter ~_R without a default follows
    type parameter with a default"; the session then needs a restart although the install on
    disk is fine. Here light-the-torch only ever runs in its own process.
    """
    torchRequirement = ">=" + MIN_TORCH
    torchvisionRequirement = None
    if platform.system() == "Windows":
        torchRequirement += "," + WINDOWS_TORCH_CAP
        torchvisionRequirement = WINDOWS_TORCHVISION_CAP

    from PyTorchUtils import PyTorchUtilsLogic
    try:  # the extension's own argument builder carries its platform workarounds
        arguments = PyTorchUtilsLogic._getPipInstallArguments(backend, torchRequirement, torchvisionRequirement)
    except Exception:  # noqa: BLE001
        arguments = ["install", "torch" + torchRequirement, "torchvision" + (torchvisionRequirement or "")]
        if backend is not None:
            arguments.append("--pytorch-computation-backend=" + backend)

    if importlib.util.find_spec("light_the_torch") is None:
        log("Installing light-the-torch (the PyTorch installer)...")
        try:
            PyTorchUtilsLogic._installLightTheTorch()
        except Exception:  # noqa: BLE001
            slicer.util.pip_install("light-the-torch>=0.5")
        importlib.invalidate_caches()

    log("Installing PyTorch (%s build). %s" % (backend or "standard", _downloadNote(backend)))
    runSlicerPython(["-m", "light_the_torch", *arguments])
    importlib.invalidate_caches()
    if _distVersion("torch") is None:
        raise RuntimeError("the PyTorch installation did not complete.")
    log("Installed torch %s." % _distVersion("torch"))


def repairTypingHooks():
    """Undo what an in-process import of pip does to ``typing``, before torch is imported.

    typing_extensions installs its own versions of ``typing``'s generic-parameter checks. pip
    ships a private copy of typing_extensions; when something imports pip inside Slicer (the
    PyTorch extension does, to call light-the-torch) that copy replaces the checks, treats the
    type variables made by the real typing_extensions as if they had defaults, and ``import
    torch`` then fails. Pointing the checks back at the real module fixes it, as long as torch
    has not been imported yet. Returns True when a repair was made.
    """
    if "pip._vendor.typing_extensions" not in sys.modules:
        return False
    try:
        import typing
        import typing_extensions
    except Exception:  # noqa: BLE001
        return False
    repaired = False
    for name in ("_collect_parameters", "_collect_type_vars", "_check_generic"):
        real = getattr(typing_extensions, name, None)
        if real is not None and hasattr(typing, name) and getattr(typing, name) is not real:
            setattr(typing, name, real)
            repaired = True
    return repaired


def torchImportBroken():
    """True when an import of torch already failed halfway in this Slicer session.

    The compiled part of torch cannot be initialised twice in one process, so after a failed
    import only a restart of Slicer helps, whatever is installed on disk.
    """
    return "torch" not in sys.modules and "torch._C" in sys.modules


def _restartNeeded(log, why):
    """Record that the install must continue after a restart. Returns ensure()'s result."""
    attempts = resumeAttempts()
    if attempts >= 2:
        setResumeAttempts(0)
        log(why + " This already happened after a restart, so another module loads PyTorch as soon as "
                  "Slicer starts. Open Slicer, go straight to AUAE and click 'Install dependencies' "
                  "before using any other module.")
        return (False, False)
    setResumeAttempts(attempts + 1)
    log(why + " Restart Slicer: the installation continues by itself when AUAE opens again.")
    return (False, True)


def ensure(log, askConfirmation=True, refresh=True):
    """Install everything needed to run, in one call. Returns (ok, needsRestart).

    The steps are: PyTorch extension, the PyTorch build that suits this machine (replacing one
    that cannot work), nnU-Net through the NNUNet extension, the extra packages, then a check
    of the result. When Slicer has to be restarted in between (the PyTorch extension was just
    installed, or PyTorch is in use and must be replaced) needsRestart is True and the
    install picks up from there the next time AUAE opens.
    """
    userLog = log

    def log(message):  # every step also lands in the Slicer log file, for support
        logging.info("AUAE dependencies: %s", message)
        userLog(message)

    if not isNNUNetExtensionAvailable():
        log("The 'NNUNet' Slicer extension is required and was not found. Install it from the "
            "Extensions Manager (it is a declared dependency of this extension), restart Slicer "
            "and try again.")
        return (False, False)

    if not isPyTorchExtensionAvailable():
        if askConfirmation and not _confirm(
                "AUAE needs the PyTorch extension. It will be installed now and Slicer must then be "
                "restarted; the installation continues by itself after the restart."):
            log("Installation cancelled.")
            return (False, False)
        try:
            log("Installing the PyTorch extension...")
            _installPyTorchExtension()
        except Exception as exc:  # noqa: BLE001
            log(str(exc))
            return (False, False)
        setResumeAttempts(resumeAttempts() + 1)
        log("The PyTorch extension was installed. Restart Slicer to continue.")
        return (True, True)

    repairTypingHooks()
    hardware = detectHardware(refresh=refresh)
    recommended = recommendedBuild(hardware)
    installed = installedTorch()
    action, why = torchAction(installed, hardware, recommended)
    if action == "none":
        status = torchStatus()
        if status["error"]:
            action, why = "replace", "PyTorch is installed but does not import (%s)." % status["error"]
        elif status["archStatus"] == "warn" and recommended["device"] == "cuda":
            action, why = "replace", status["archMessage"] or "The GPU cannot be used by this PyTorch build."

    _nnVersion, nnCompatible = nnunetStatus()
    missingExtras = [dist for dist, imp in EXTRA_PACKAGES.items() if importlib.util.find_spec(imp) is None]

    steps = []
    build = recommended["backend"] or "standard"
    size = "about 2-3 GB" if build.startswith("cu") else "a few hundred MB"
    if action == "install":
        steps.append("install PyTorch, %s build (%s)" % (build, size))
    elif action == "replace":
        steps.append("replace PyTorch with the %s build (%s). %s" % (build, size, why))
    if not nnCompatible:
        steps.append("install nnU-Net (nnunetv2)")
    if missingExtras:
        steps.append("install " + ", ".join(missingExtras))

    if not steps:
        if torchImportBroken():
            return _restartNeeded(log, "Everything is installed, but PyTorch failed to load earlier in "
                                       "this Slicer session and cannot be loaded again without a restart.")
        _deleteFolders(_torchLeftovers(), log)  # tidy up what an earlier install left behind
        setResumeAttempts(0)
        log("All dependencies are installed. " + summary().splitlines()[-1])
        return (True, False)

    log(recommended["reason"])
    if askConfirmation and not _confirm(
            "AUAE will now:\n\n- " + "\n- ".join(steps) + "\n\nThis can take several minutes. Continue?"):
        setResumeAttempts(0)
        log("Installation cancelled.")
        return (False, False)

    ok = True
    try:
        if action in ("install", "replace"):
            if action == "replace" and installed and (installed["loaded"] or torchImportBroken()):
                return _restartNeeded(log, "PyTorch is in use by this Slicer session and cannot be replaced now.")
            try:
                # Also on a first install: a half-extracted earlier attempt may be lying around.
                _removeTorch(log)
            except RuntimeError as exc:
                return _restartNeeded(log, "Could not remove the old PyTorch (%s)." % exc)
            _installTorch(recommended["backend"], log)
            _checkInstalledTorch(log, recommended)

        if not nnCompatible:
            from SlicerNNUNetLib import InstallLogic
            installLogic = InstallLogic()
            installLogic.doAskConfirmation = False
            try:
                installLogic.progressInfo.connect(log)
            except Exception:  # noqa: BLE001
                pass
            log("Installing nnU-Net through the NNUNet extension...")
            ok = bool(installLogic.setupPythonRequirements()) and ok

        for dist in missingExtras:
            log("Installing " + dist + "...")
            try:
                slicer.util.pip_install(dist)
            except Exception as exc:  # noqa: BLE001
                log("Could not install " + dist + ": " + str(exc))
                ok = False

        # A dependency must not have swapped PyTorch for another build behind our back.
        importlib.invalidate_caches()
        lateAction, lateWhy = torchAction(installedTorch(), hardware, recommended)
        if lateAction != "none":
            log("PyTorch changed while the other packages were installed. " + lateWhy
                + " Click 'Install dependencies' again to repair it.")
            ok = False
    except Exception as exc:  # noqa: BLE001
        logging.error("AUAE dependencies: %s", traceback.format_exc())
        log("Installation failed: %s" % exc)
        ok = False

    repairTypingHooks()  # nnU-Net's installer may have loaded pip into this session
    if ok and torchImportBroken():
        return _restartNeeded(log, "The installation is complete, but PyTorch failed to load earlier in "
                                   "this Slicer session and cannot be loaded again without a restart.")
    setResumeAttempts(0)
    log("Dependency setup %s." % ("completed" if ok else "did not complete; see the messages above"))
    return (ok, False)


def _checkInstalledTorch(log, recommended):
    """Probe the fresh install in a separate process and say what inference will run on."""
    log("Checking the new PyTorch (this takes a few seconds)...")
    status = torchStatus(refresh=True)
    if status["error"]:
        raise RuntimeError("PyTorch was installed but does not import: " + status["error"])
    if cudaUsable(status):
        log("GPU inference works: %s, CUDA %s." % (status["device"], status["cudaBuild"]))
    elif status["mpsAvailable"]:
        log("GPU inference works through MPS (Apple GPU).")
    elif recommended["device"] == "cuda":
        log("WARNING: PyTorch was installed but cannot use the GPU%s. Inference runs on the CPU until "
            "this is fixed; updating the NVIDIA driver is the usual remedy."
            % ((": " + status["archMessage"]) if status["archMessage"] else ""))
    else:
        log("Inference will run on the CPU (slow, up to about 1 h per volume).")


# --- Device selection ------------------------------------------------------------------
def resolveDevice(deviceChoice="Auto"):
    """Turn the panel's device choice into the device nnU-Net gets: 'cuda', 'mps' or 'cpu'.

    'Auto' takes the best device the installed PyTorch can really use. An explicit 'CUDA' or
    'MPS' that cannot be used falls back to 'cpu', so a run never starts on a device that
    would crash halfway.
    """
    choice = (deviceChoice or "Auto").strip().lower()
    if choice == "cpu":
        return "cpu"
    status = torchStatus()
    if choice in ("auto", "cuda") and cudaUsable(status):
        return "cuda"
    if choice in ("auto", "mps") and status.get("mpsAvailable"):
        return "mps"
    return "cpu"


def parameterDeviceKwargs(deviceChoice):
    """Map a UI device choice to nnU-Net Parameter kwargs."""
    return {"device": resolveDevice(deviceChoice)}
