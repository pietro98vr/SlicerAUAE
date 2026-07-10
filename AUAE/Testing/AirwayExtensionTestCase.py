"""Headless tests for the AirwayExtension array logic.

These exercise the interface-free NumPy geometry (head-envelope decomposition, sinus recovery,
cephalostat-bubble removal, inferior extension and its padded merge, and the separable RAS
field). They need neither the nnU-Net model nor a GPU. Run them from the module self test
(SlicerPythonTestRunner) or directly in Slicer's Python console.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")  # the air envelope needs scipy.ndimage

# Make AUAELib importable when pytest collects this file on its own.
_AUAE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AUAE_DIR not in sys.path:
    sys.path.insert(0, _AUAE_DIR)

from AUAELib import AirwayExtension as AE  # noqa: E402

SPACING = 0.5
VOXEL_MM3 = SPACING ** 3
# A -> +J affine (anterior grows with the J axis), spacing 0.5 mm, identity axes.
IJK_TO_RAS = np.diag([SPACING, SPACING, SPACING, 1.0])


def _phantom():
    """A solid head with an enclosed airway tube, a large sinus, and a small bubble."""
    shape = (50, 70, 70)
    vol = np.full(shape, -1000.0, np.float32)
    vol[8:44, 12:60, 12:60] = 50.0                      # head (tissue)
    airway = np.zeros(shape, np.uint8)
    vol[10:42, 33:39, 33:39] = -1000.0                  # airway tube (enclosed)
    airway[10:42, 33:39, 33:39] = 1
    sinus = np.zeros(shape, bool)
    vol[12:22, 46:58, 28:50] = -1000.0                  # 10*12*22 = 2640 vox -> 330 mm^3 (> 250)
    sinus[12:22, 46:58, 28:50] = True
    bubble = np.zeros(shape, bool)
    vol[14:20, 46:52, 52:58] = -1000.0                  # 6*6*6 = 216 vox -> 27 mm^3 (< 250)
    bubble[14:20, 46:52, 52:58] = True
    return vol, airway, sinus, bubble


def test_ras_component_field_matches_affine():
    rng = np.random.default_rng(0)
    aff = np.eye(4)
    aff[:3, :3] = rng.normal(size=(3, 3))
    aff[:3, 3] = rng.normal(size=3)
    shape = (7, 9, 11)
    field = AE._rasComponentField(shape, aff, 1)
    worst = 0.0
    for _ in range(200):
        k, j, i = (int(rng.integers(n)) for n in shape)
        ras = aff @ np.array([i, j, k, 1.0])
        worst = max(worst, abs(ras[1] - field[k, j, i]))
    assert worst < 1e-9


def test_sinus_recovered_and_bubble_removed():
    vol, airway, sinus, bubble = _phantom()
    thr = AE._airThresholdFromAirway(vol, airway)
    sealed = AE._sealedHeadMask(vol, thr, (SPACING, SPACING, SPACING))
    air = vol <= thr
    internal = AE.computeInternalAirCavities(air, sealed, airway, VOXEL_MM3,
                                             minRegionMm3=AE.DEFAULT_MIN_INTERNAL_AIR_MM3).astype(bool)
    assert internal[sinus].all()                 # the sinus is fully recovered
    assert not internal[bubble].any()            # the cephalostat bubble is dropped
    assert not internal[airway > 0].any()        # the airway is not double-counted


def test_external_air_excludes_internal_and_airway():
    vol, airway, sinus, bubble = _phantom()
    thr = AE._airThresholdFromAirway(vol, airway)
    sealed = AE._sealedHeadMask(vol, thr, (SPACING, SPACING, SPACING))
    air = vol <= thr
    external = AE.computeExternalAirMask(air, airway, sealed, IJK_TO_RAS, VOXEL_MM3).astype(bool)
    assert external.sum() > 0
    assert not external[sinus].any()             # frontal sinus never leaks outside
    assert not external[bubble].any()
    assert not external[airway > 0].any()
    assert not external[sealed].any()            # everything kept is outside the head envelope


def test_extension_returns_pad_info_and_merges_airway_only():
    volumeNode = _makeVolumeNode((40, 40, 40), SPACING)
    try:
        airway = np.zeros((40, 40, 40), np.uint8)
        airway[5:35, 18:22, 18:22] = 1           # inferior end at K = 5 (min side)
        companion = np.zeros((40, 40, 40), np.uint8)
        companion[10:20, 5:9, 5:9] = 1           # a lower blob to stand in for external air
        extended, ijk, pad = AE.extendBinaryArray(airway, volumeNode, 10.0, "Inferior (neck)", None)
        padded = AE._padCompanionToExtendedGrid(companion, pad)
        side, nout = pad
        assert side == "min" and nout > 0
        assert extended.shape == padded.shape
        newSlices = slice(0, nout)
        assert extended[newSlices].sum() > 0     # the airway edge is replicated into the new slices
        assert padded[newSlices].sum() == 0      # the companion is only padded, never replicated
    finally:
        import slicer
        slicer.mrmlScene.RemoveNode(volumeNode)


def test_remove_small_islands_and_keep_largest():
    a = np.zeros((20, 20, 20), np.uint8)
    a[2:10, 2:10, 2:10] = 1                       # big component
    a[15, 15, 15] = 1                             # single-voxel speck
    largest = AE.keepLargestConnectedComponent(a)
    assert largest[15, 15, 15] == 0 and largest[2:10, 2:10, 2:10].all()
    cleaned = AE.removeSmallIslands(a, minSizeMm3=10.0, voxelVolumeMm3=1.0)
    assert cleaned[15, 15, 15] == 0 and cleaned[2:10, 2:10, 2:10].all()


def _makeVolumeNode(shape, spacing):
    import slicer
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "auae_test_ref")
    slicer.util.updateVolumeFromArray(node, np.zeros(shape, np.int16))
    node.SetSpacing(spacing, spacing, spacing)
    return node
