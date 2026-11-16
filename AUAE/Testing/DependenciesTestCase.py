"""Tests for the hardware-to-PyTorch-build decisions in AUAELib.Dependencies.

Only the pure functions are exercised (parsing nvidia-smi, choosing the build, deciding
whether an installed PyTorch must be replaced), so nothing is installed or imported.
"""

import os
import sys

_AUAE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AUAE_DIR not in sys.path:
    sys.path.insert(0, _AUAE_DIR)

from AUAELib import Dependencies as D  # noqa: E402


def _machine(name=None, driver="581.15", capability=None, system="Windows", apple=False):
    nvidia = [{"name": name, "driver": driver, "computeCapability": capability, "vramMb": 8192}] if name else []
    return {"system": system, "machine": "AMD64", "nvidia": nvidia, "appleSilicon": apple}


def _torch(version, vision=None):
    cuda = D._cudaFromVersion(version)
    return {"version": version, "cuda": None if cuda == "unknown" else cuda, "torchvision": vision, "loaded": False}


def test_parse_nvidia_smi():
    gpus = D.parseNvidiaSmi("NVIDIA GeForce RTX 3070, 621.24, 8.6, 8192\nNVIDIA GeForce RTX 5070 Ti, 581.15, 12.0, 16303\n")
    assert [g["computeCapability"] for g in gpus] == [(8, 6), (12, 0)]
    assert gpus[0]["driver"] == "621.24" and gpus[1]["vramMb"] == 16303
    # An old nvidia-smi without the compute_cap field: the capability comes from the name.
    old = D.parseNvidiaSmi("GeForce GTX 1060 6GB, 391.35, 6144\n", hasCapability=False)
    assert old[0]["computeCapability"] == (6, 1)
    assert D.parseNvidiaSmi("NVIDIA GeForce RTX 4090, 560.94, [N/A], 24564")[0]["computeCapability"] == (8, 9)
    assert D.parseNvidiaSmi("") == [] and D.parseNvidiaSmi(None) == []


def test_build_per_gpu_generation():
    cases = (
        ("NVIDIA GeForce RTX 5070", (12, 0), "581.15", "cu128", "cuda"),
        ("NVIDIA GeForce RTX 4090", (8, 9), "560.94", "cu126", "cuda"),
        ("NVIDIA GeForce RTX 3070", (8, 6), "621.24", "cu126", "cuda"),
        ("NVIDIA GeForce GTX 1660", (7, 5), "551.23", "cu126", "cuda"),
        ("NVIDIA GeForce GTX 1060", (6, 1), "546.33", "cu126", "cuda"),
        ("NVIDIA GeForce GTX 970", (5, 2), "546.33", "cu126", "cuda"),
        ("NVIDIA GeForce GTX 1060", (6, 1), "472.12", "cu118", "cuda"),   # driver too old for CUDA 12
        ("NVIDIA GeForce GTX 1060", (6, 1), "391.35", "cpu", "cpu"),      # driver too old for anything
        ("NVIDIA GeForce GTX 680", (3, 0), "474.11", "cpu", "cpu"),       # Kepler: no current build
        ("Some future card", None, "600.10", "cu126", "cuda"),            # unknown capability
    )
    for name, capability, driver, backend, device in cases:
        build = D.recommendedBuild(_machine(name, driver, capability))
        assert (build["backend"], build["device"]) == (backend, device), name
        assert name in build["reason"]


def test_build_without_nvidia():
    assert D.recommendedBuild(_machine())["backend"] == "cpu"                      # AMD / Intel graphics
    mac = D.recommendedBuild(_machine(system="Darwin", apple=True))
    assert (mac["backend"], mac["device"]) == (None, "mps")
    intelMac = D.recommendedBuild(_machine(system="Darwin"))
    assert (intelMac["backend"], intelMac["device"]) == (None, "cpu")
    linux = D.recommendedBuild(_machine("NVIDIA GeForce RTX 3070", "525.60.13", (8, 6), system="Linux"))
    assert linux["backend"] == "cu126"
    assert D.recommendedBuild(_machine("NVIDIA GeForce RTX 3070", "515.65", (8, 6), system="Linux"))["backend"] == "cu118"


def _action(installed, machine):
    return D.torchAction(installed, machine, D.recommendedBuild(machine))[0]


def test_working_installs_are_left_alone():
    rtx30 = _machine("NVIDIA GeForce RTX 3070", "621.24", (8, 6))
    assert _action(_torch("2.12.0+cu126", "0.27.0+cu126"), rtx30) == "none"
    assert _action(_torch("2.7.1+cu128", "0.22.1+cu128"), rtx30) == "none"   # not our pick, but it works
    assert _action(_torch("2.5.1+cu124", "0.20.1+cu124"), rtx30) == "none"
    rtx50 = _machine("NVIDIA GeForce RTX 5070", "581.15", (12, 0))
    assert _action(_torch("2.7.1+cu128", "0.22.1+cu128"), rtx50) == "none"
    assert _action(_torch("2.12.0+cu130", "0.27.0+cu130"), rtx50) == "none"
    # No NVIDIA GPU: any build runs on the CPU, so nothing is replaced.
    assert _action(_torch("2.12.0+cu126", "0.27.0+cu126"), _machine()) == "none"
    assert _action(_torch("2.12.0+cpu", "0.27.0+cpu"), _machine()) == "none"
    assert _action(_torch("2.12.0", "0.27.0"), _machine(system="Darwin", apple=True)) == "none"


def test_installs_that_cannot_work_are_replaced():
    rtx30 = _machine("NVIDIA GeForce RTX 3070", "621.24", (8, 6))
    rtx50 = _machine("NVIDIA GeForce RTX 5070", "581.15", (12, 0))
    gtx10 = _machine("NVIDIA GeForce GTX 1060", "546.33", (6, 1))
    assert _action(None, rtx30) == "install"
    assert _action(_torch("2.12.0+cpu", "0.27.0+cpu"), rtx30) == "replace"         # CPU build next to a GPU
    assert _action(_torch("2.12.0+cu126", "0.27.0+cu126"), rtx50) == "replace"     # no sm_120 kernels
    assert _action(_torch("2.7.1+cu128", "0.22.1+cu128"), gtx10) == "replace"      # Pascal dropped in 12.8
    assert _action(_torch("2.12.0+cu130", "0.27.0+cu130"), gtx10) == "replace"
    assert _action(_torch("2.12.0+cu126", "0.27.0+cpu"), rtx30) == "replace"       # mixed builds
    assert _action(_torch("2.12.0+cu126", "0.22.1+cu126"), rtx30) == "replace"     # mismatched releases
    oldDriver = _machine("NVIDIA GeForce RTX 3070", "516.94", (8, 6))
    assert _action(_torch("2.12.0+cu126", "0.27.0+cu126"), oldDriver) == "replace"
    assert _action(_torch("2.7.1+cu118", "0.22.1+cu118"), oldDriver) == "none"
    cuda13 = _machine("NVIDIA GeForce RTX 3070", "566.36", (8, 6))
    assert _action(_torch("2.12.0+cu130", "0.27.0+cu130"), cuda13) == "replace"    # CUDA 13 needs driver 580+


def test_version_helpers():
    assert D._cudaFromVersion("2.12.0+cu126") == (12, 6)
    assert D._cudaFromVersion("2.7.1+cu118") == (11, 8)
    assert D._cudaFromVersion("2.12.0+cu130") == (13, 0)
    assert D._cudaFromVersion("2.12.0+cpu") is None
    assert D._cudaFromVersion("2.12.0") == "unknown"
    assert not D.pairMismatch("2.12.0+cu126", "0.27.0+cu126")
    assert not D.pairMismatch("2.12.0+cu126", None)
    assert D.pairMismatch("2.12.0+cu126", "0.27.0+cu128")
    assert D._archCompatibility((12, 0), ["sm_50", "sm_86", "sm_90"])[0] == "warn"
    assert D._archCompatibility((12, 0), ["sm_75", "sm_100", "sm_120"])[0] == "ok"
    assert D._archCompatibility((8, 6), ["sm_50", "sm_86", "sm_90"])[0] == "ok"


def test_pip_leftover_names():
    for name in ("~orch", "~orchgen", "~orchvision", "~orchvision.libs", "~unctorch", "~-rch",
                 "~orch-2.12.0+cu126.dist-info"):
        assert D.isTorchLeftover(name), name
    for name in ("torch", "torchgen", "light_the_torch", "fft_conv_pytorch", "~umpy", "~cipy", "~orchmetrics"):
        assert not D.isTorchLeftover(name), name
