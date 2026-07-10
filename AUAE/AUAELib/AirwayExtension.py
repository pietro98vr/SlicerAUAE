"""Airway post-processing and extension logic.

This module is the distinctive part of this project, which is based on SlicerUpperAirwaySegmentator:
the geometric logic that extends the final binary airway mask beyond the acquired
field of view; the automatic connected-component cleanup applied during the segmentation
step (the embedded Segment Editor stays available for manual refinement); and the optional
external face-air segment used as an inlet volume for flow modelling.

The functions here are deliberately kept free of any Qt/UI dependency so they can be
unit-tested in a headless Slicer session. They operate on NumPy arrays in Slicer's
native KJI order (K = slice, J = row, I = column), the inverse of the IJK ordering
used by the VTK/MRML IJK->RAS matrix.

Ported from the Airway CBCT Pipeline (project v2) logic.
"""

import numpy as np
import vtk
import slicer


# --- Axis convention -----------------------------------------------------------------
# slicer.util.arrayFromVolume returns a NumPy array in KJI order; axis 0 is the slice
# stack we grow when extending. Keeping it in a constant avoids magic numbers and makes
# the whole geometry follow a single definition.
AXIS = 0
# Bridge between the two orderings: a NumPy (KJI) axis and its IJK counterpart. Needed
# when an array-space operation ("add N slices on axis 0") must be turned into an
# IJK->RAS origin shift. 0(K)->2, 1(J)->1, 2(I)->0 is exactly the KJI<->IJK reversal.
NUMPY_AXIS_TO_IJK_AXIS = {0: 2, 1: 1, 2: 0}

# UI-facing labels for the extension direction. Kept here so widget and logic agree.
EXTENSION_DIRECTIONS = (
    "Inferior (neck)",
    "Superior (cranial)",
)

# Single-label display, matching the upstream "Airway" segment appearance.
AIRWAY_SEGMENT_NAME = "Airway"
AIRWAY_SEGMENT_COLOR = (130 / 255.0, 177 / 255.0, 255 / 255.0)  # light blue
# Upstream default small-island threshold: 200 voxels at 0.3 mm isotropic.
DEFAULT_MIN_ISLAND_MM3 = (0.3 ** 3) * 200

# Second, optional segment: the ambient air in front of the face. For flow modelling this
# provides the external inlet volume the air enters through (nostrils / mouth), so a CFD mesh
# has a domain upstream of the airway rather than a bare opening.
EXTERNAL_AIR_SEGMENT_NAME = "External air (face)"
EXTERNAL_AIR_SEGMENT_COLOR = (255 / 255.0, 214 / 255.0, 102 / 255.0)  # warm yellow
# Ignore ambient-air blobs smaller than this (mm^3): keeps the meaningful front-of-face volume,
# drops trapped-air specks between skin folds, headrest gaps, etc.
DEFAULT_MIN_EXTERNAL_AIR_MM3 = 1000.0


def defaultOptions():
    """Return the default post-processing options shared by interactive and batch runs."""
    return {
        "removeSmallIslands": True,
        "minIslandMm3": DEFAULT_MIN_ISLAND_MM3,
        "keepLargestIsland": False,
        "smoothingFactor": 0.0,
        "segmentExternalAir": False,
        "mergeExternalAir": False,
        "extend": False,
        "direction": EXTENSION_DIRECTIONS[0],
        "lengthMm": 100.0,
    }


def postprocessSegmentation(segmentationNode, volumeNode, options, log=None):
    """Apply island cleanup and optional airway extension to a segmentation node.

    Single source of truth for both the interactive widget and the batch processor.
    Because removing the embedded Segment Editor also removes upstream's island effect,
    the cleanup here is done in NumPy on the exported binary labelmap. The optional
    extension may grow the geometry, so the result is rebuilt as a fresh segmentation
    node carrying a single "Airway" segment, then closed-surface converted for display
    and export. Returns the new segmentation node.
    """
    options = {**defaultOptions(), **(options or {})}

    # Export the airway mask to a labelmap using the reference (volume) geometry, so the
    # full slice stack is available for the inferior/superior extension computation.
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "airway_tmp_labelmap")
    try:
        if volumeNode is not None:
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        segLogic = slicer.modules.segmentations.logic()
        exported = segLogic.ExportAllSegmentsToLabelmapNode(
            segmentationNode, labelmapNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
        )
        if exported is False:
            raise RuntimeError("Could not export segmentation to labelmap for post-processing.")

        array = (slicer.util.arrayFromVolume(labelmapNode) > 0).astype(np.uint8)
        if int(array.sum()) == 0:
            raise RuntimeError("Segmentation is empty; nothing to post-process.")

        if options.get("keepLargestIsland", False):
            array = keepLargestConnectedComponent(array, log)
        if options.get("removeSmallIslands", True):
            voxelVolumeMm3 = float(np.prod(labelmapNode.GetSpacing()))
            array = removeSmallIslands(array, options.get("minIslandMm3", DEFAULT_MIN_ISLAND_MM3), voxelVolumeMm3, log)

        # Optional second segment: ambient air in front of the face (flow-modelling inlet).
        # Computed on the labelmap grid (same as the cleaned airway) so the two masks align.
        externalArray = None
        baseIJKToRAS = volumeIJKToRASArray(labelmapNode)
        if options.get("segmentExternalAir", False):
            voxelVolumeMm3 = float(np.prod(labelmapNode.GetSpacing()))
            externalArray = computeExternalAirMask(
                _volumeArrayLike(volumeNode, array), array, baseIJKToRAS, voxelVolumeMm3, log
            )
            if externalArray is not None and int(externalArray.sum()) == 0:
                externalArray = None
            # 'Merge into a single segmentation': union external air into the airway mask BEFORE
            # extension, so the whole fluid domain is one segment on one grid.
            if externalArray is not None and options.get("mergeExternalAir", False):
                array = ((array > 0) | (externalArray > 0)).astype(np.uint8)
                externalArray = None
                _log(log, "External air merged into the airway segment.")

        if options.get("extend", False) and float(options.get("lengthMm", 0.0)) > 0:
            array, ijkToRAS = extendBinaryArray(
                array, labelmapNode, float(options["lengthMm"]), options.get("direction", EXTENSION_DIRECTIONS[0]), log
            )
        else:
            ijkToRAS = baseIJKToRAS

        node = _buildAirwaySegmentation(array, ijkToRAS, segmentationNode.GetName(), log)
        if externalArray is not None:
            # Separate segment: import on the original (unextended) grid; Slicer resamples it
            # into the node's shared geometry.
            _addExternalAirSegment(node, externalArray, baseIJKToRAS, log)
        _applyDisplaySmoothing(node, options.get("smoothingFactor", 0.0))
        return node
    finally:
        slicer.mrmlScene.RemoveNode(labelmapNode)


def extendSegmentation(segmentationNode, volumeNode, options, log=None):
    """Apply only the airway extension to an existing segmentation, as a separate step.

    Used when segmentation and extension are decoupled: the user (or a batch) first obtains
    and refines a segmentation, then runs the extension on it. No island cleanup happens here,
    so any manual edits are preserved. Returns a new segmentation node carrying the extended
    "Airway" segment.
    """
    options = {**defaultOptions(), **(options or {})}
    lengthMm = float(options.get("lengthMm", 0.0))
    if lengthMm <= 0:
        raise RuntimeError("Set an extension length greater than 0 mm.")

    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "airway_tmp_labelmap")
    try:
        if volumeNode is not None:
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        segLogic = slicer.modules.segmentations.logic()
        exported = segLogic.ExportAllSegmentsToLabelmapNode(
            segmentationNode, labelmapNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
        )
        if exported is False:
            raise RuntimeError("Could not export segmentation to labelmap for extension.")

        array = (slicer.util.arrayFromVolume(labelmapNode) > 0).astype(np.uint8)
        if int(array.sum()) == 0:
            raise RuntimeError("Segmentation is empty; nothing to extend.")

        array, ijkToRAS = extendBinaryArray(
            array, labelmapNode, lengthMm, options.get("direction", EXTENSION_DIRECTIONS[0]), log
        )
        node = _buildAirwaySegmentation(array, ijkToRAS, segmentationNode.GetName(), log)
        _applyDisplaySmoothing(node, options.get("smoothingFactor", 0.0))
        return node
    finally:
        slicer.mrmlScene.RemoveNode(labelmapNode)


def _buildAirwaySegmentation(array, ijkToRAS, name, log=None):
    """Rebuild a single-segment 'Airway' segmentation node from a processed binary array."""
    processedLabelmap = slicer.util.addVolumeFromArray(
        array.astype(np.uint8), ijkToRAS=ijkToRAS, name="airway_tmp_processed",
        nodeClassName="vtkMRMLLabelMapVolumeNode",
    )
    try:
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
        segmentationNode.CreateDefaultDisplayNodes()
        segmentation = segmentationNode.GetSegmentation()
        segmentation.SetConversionParameter("Smoothing factor", "0.0")

        beforeIds = set(_segmentIds(segmentation))
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(processedLabelmap, segmentationNode)
        newIds = [sid for sid in _segmentIds(segmentation) if sid not in beforeIds]
        if not newIds:
            raise RuntimeError("Import produced no airway segment.")

        segment = segmentation.GetSegment(newIds[0])
        segment.SetName(AIRWAY_SEGMENT_NAME)
        segment.SetColor(*AIRWAY_SEGMENT_COLOR)
        displayNode = segmentationNode.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetSegmentOpacity3D(newIds[0], 0.8)

        segmentationNode.CreateClosedSurfaceRepresentation()  # replaces the removed Show3D button
        return segmentationNode
    finally:
        slicer.mrmlScene.RemoveNode(processedLabelmap)


def _segmentIds(segmentation):
    return [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]


def volumeIJKToRASArray(volumeNode):
    """Return the volume IJK->RAS matrix as a 4x4 NumPy array.

    This is the homogeneous affine that maps voxel indices (I,J,K) to physical RAS
    millimetres: it embeds spacing, origin and axis directions. We take it as a NumPy
    array so it composes with voxel vectors through the plain @ operator.
    """
    ijkToRASVtk = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(ijkToRASVtk)
    return slicer.util.arrayFromVTKMatrix(ijkToRASVtk)


def sliceSCoordinate(sliceIndex, arrayShape, ijkToRAS):
    """RAS-S (cranio-caudal height) of the centre of a given array slice.

    Used to decide which end of the slice stack is anatomically inferior: comparing the
    S of the first and last slice tells where the neck is, regardless of orientation.
    Note the KJI->IJK index reversal before the matrix product, and that S is row 2.
    """
    centerArray = [
        (arrayShape[0] - 1) / 2.0,
        (arrayShape[1] - 1) / 2.0,
        (arrayShape[2] - 1) / 2.0,
    ]
    centerArray[AXIS] = float(sliceIndex)
    ijk = np.array([centerArray[2], centerArray[1], centerArray[0], 1.0])
    ras = ijkToRAS @ ijk
    return float(ras[2])


# The three helpers below abstract per-slice access along AXIS: they build the correct
# NumPy index at runtime (slice(None) on every axis but AXIS) instead of hardcoding
# array[i] / array[:, i]. The extension logic stays independent of which axis is the stack.
def getAxisSlice(array, sliceIndex):
    """Extract the 2D slice of given index along the stack axis (AXIS)."""
    return np.take(array, sliceIndex, axis=AXIS)


def setAxisSlice(array, sliceIndex, slice2d):
    """Overwrite in place the slice of given index along AXIS with a 2D slice."""
    sliceTuple = [slice(None)] * array.ndim
    sliceTuple[AXIS] = sliceIndex
    array[tuple(sliceTuple)] = slice2d


def copyArrayIntoAxisRange(destination, startIndex, source):
    """Paste 'source' into 'destination' in a contiguous window along AXIS.

    Used to place the original mask inside the larger extended array: the window
    [startIndex, startIndex+length) leaves the newly added slices free to be filled.
    """
    sliceTuple = [slice(None)] * destination.ndim
    sliceTuple[AXIS] = slice(startIndex, startIndex + source.shape[AXIS])
    destination[tuple(sliceTuple)] = source


def resolveExtensionSideForDirection(direction, arrayShape, ijkToRAS):
    """Map an anatomical or array direction onto the min/max end of the KJI stack.

    'Array min'/'Array max' are deterministic and ignore geometry (handy for tests).
    'Inferior'/'Superior' are resolved from the RAS-S of the two ends, so the correct
    end is chosen whatever the patient orientation stored in the IJK->RAS matrix.
    """
    direction = (direction or "").lower()
    if "array min" in direction:
        return "min"
    if "array max" in direction:
        return "max"
    minS = sliceSCoordinate(0, arrayShape, ijkToRAS)
    maxS = sliceSCoordinate(arrayShape[AXIS] - 1, arrayShape, ijkToRAS)
    if "superior" in direction:
        return "min" if minS > maxS else "max"
    return "min" if minS < maxS else "max"


def keepLargestConnectedComponent(array, log=None):
    """Keep only the largest connected component, dropping spurious isolated voxels."""
    try:
        from scipy import ndimage
    except Exception as exc:  # noqa: BLE001
        _log(log, "Keep largest island skipped; scipy unavailable: " + str(exc))
        return array.astype(np.uint8)

    labeled, count = ndimage.label(array > 0)
    if count <= 1:
        return array.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # exclude background so argmax picks the main airway
    largestLabel = int(np.argmax(sizes))
    if log:
        _log(log, "Keep largest island: kept label %d of %d." % (largestLabel, count))
    return (labeled == largestLabel).astype(np.uint8)


def removeSmallIslands(array, minSizeMm3, voxelVolumeMm3, log=None):
    """Remove connected components smaller than a physical volume threshold.

    Replaces the upstream Segment Editor "Islands / remove small islands" effect, which
    is no longer available once the embedded editor is removed. The threshold matches the
    upstream default (200 voxels at 0.3 mm isotropic) but is expressed in mm^3 so it is
    correct for any spacing.
    """
    try:
        from scipy import ndimage
    except Exception as exc:  # noqa: BLE001
        _log(log, "Remove small islands skipped; scipy unavailable: " + str(exc))
        return array.astype(np.uint8)

    if voxelVolumeMm3 <= 0:
        return array.astype(np.uint8)
    minVoxels = int(np.ceil(float(minSizeMm3) / float(voxelVolumeMm3)))
    if minVoxels <= 1:
        return array.astype(np.uint8)

    labeled, count = ndimage.label(array > 0)
    if count == 0:
        return array.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    keptLabels = np.where(sizes >= minVoxels)[0]
    keptLabels = keptLabels[keptLabels != 0]  # never keep background
    cleaned = np.isin(labeled, keptLabels).astype(np.uint8)
    if log:
        removed = count - len(keptLabels)
        _log(log, "Remove small islands: dropped %d component(s) < %d voxels." % (removed, minVoxels))
    return cleaned


def _volumeArrayLike(volumeNode, referenceArray):
    """Grey-level volume array aligned to the airway labelmap grid, or None if unavailable."""
    if volumeNode is None:
        return None
    try:
        vol = slicer.util.arrayFromVolume(volumeNode)
    except Exception:  # noqa: BLE001
        return None
    return vol if vol is not None and vol.shape == referenceArray.shape else None


def _rasComponentField(shape, ijkToRAS, rasRow):
    """Per-voxel value of one RAS component (0=R, 1=A, 2=S) over a KJI array of given shape.

    The affine is separable, so the field is built from three 1-D ramps instead of a full
    matrix product per voxel. Array axes are KJI, hence the K/J/I coefficients are columns
    2/1/0 of the chosen RAS row.
    """
    nK, nJ, nI = shape
    row = ijkToRAS[rasRow]
    kRamp = np.arange(nK) * row[2]
    jRamp = np.arange(nJ) * row[1]
    iRamp = np.arange(nI) * row[0]
    return row[3] + kRamp[:, None, None] + jRamp[None, :, None] + iRamp[None, None, :]


def computeExternalAirMask(volumeArray, airwayArray, ijkToRAS, voxelVolumeMm3, log=None,
                           minRegionMm3=DEFAULT_MIN_EXTERNAL_AIR_MM3):
    """Segment the ambient air in front of the face as a binary mask on the airway grid.

    The air threshold is self-calibrated from the intensities under the known airway mask, so
    it holds whether the volume is in Hounsfield units (air ~ -1000) or intensity-shifted
    (air ~ 0). Ambient air is taken as the air components touching the volume border, clipped
    to the region anterior to the airway (the 'front of the face'), with the airway itself and
    negligible specks removed. Returns a uint8 array; an all-zero array means 'not found'.
    """
    if volumeArray is None:
        _log(log, "External air skipped: grey-level volume not available on the airway grid.")
        return np.zeros_like(airwayArray, dtype=np.uint8)
    try:
        from scipy import ndimage
    except Exception as exc:  # noqa: BLE001
        _log(log, "External air skipped; scipy unavailable: " + str(exc))
        return np.zeros_like(airwayArray, dtype=np.uint8)

    airwayVals = volumeArray[airwayArray > 0]
    if airwayVals.size == 0:
        return np.zeros_like(airwayArray, dtype=np.uint8)
    # Upper bound of the airway (air) intensity distribution -> everything this dark is air-like.
    threshold = float(airwayVals.mean() + 2.0 * (airwayVals.std() + 1e-6))
    airMask = volumeArray <= threshold

    labeled, count = ndimage.label(airMask)
    if count == 0:
        return np.zeros_like(airwayArray, dtype=np.uint8)
    borderLabels = set()
    for face in (labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1], labeled[:, :, 0], labeled[:, :, -1]):
        borderLabels.update(int(v) for v in np.unique(face))
    borderLabels.discard(0)
    if not borderLabels:
        return np.zeros_like(airwayArray, dtype=np.uint8)
    external = np.isin(labeled, list(borderLabels))
    external[airwayArray > 0] = False  # never double-count the airway lumen

    # Clip to the front of the face: keep voxels anterior to the airway's mid A-coordinate.
    try:
        aField = _rasComponentField(volumeArray.shape, ijkToRAS, 1)  # RAS 'A' (anterior +)
        airwayA = aField[airwayArray > 0]
        if airwayA.size:
            external &= aField >= float(np.median(airwayA))
    except Exception as exc:  # noqa: BLE001
        _log(log, "External air: anterior clip skipped (%s); keeping full ambient air." % exc)

    minVox = int(np.ceil(float(minRegionMm3) / max(float(voxelVolumeMm3), 1e-6)))
    if minVox > 1:
        lab2, c2 = ndimage.label(external)
        if c2 > 0:
            sizes = np.bincount(lab2.ravel())
            sizes[0] = 0
            keep = np.where(sizes >= minVox)[0]
            external = np.isin(lab2, keep)
    result = external.astype(np.uint8)
    _log(log, "External face air: %d voxels (air threshold <= %.1f)." % (int(result.sum()), threshold))
    return result


def _addExternalAirSegment(segmentationNode, externalArray, ijkToRAS, log=None):
    """Import the external-air mask as a second, distinctly coloured segment in the node."""
    tmp = slicer.util.addVolumeFromArray(
        externalArray.astype(np.uint8), ijkToRAS=ijkToRAS, name="external_air_tmp",
        nodeClassName="vtkMRMLLabelMapVolumeNode",
    )
    try:
        segmentation = segmentationNode.GetSegmentation()
        beforeIds = set(_segmentIds(segmentation))
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tmp, segmentationNode)
        newIds = [sid for sid in _segmentIds(segmentation) if sid not in beforeIds]
        if not newIds:
            _log(log, "External air import produced no segment.")
            return
        segment = segmentation.GetSegment(newIds[0])
        segment.SetName(EXTERNAL_AIR_SEGMENT_NAME)
        segment.SetColor(*EXTERNAL_AIR_SEGMENT_COLOR)
        displayNode = segmentationNode.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetSegmentOpacity3D(newIds[0], 0.4)
        segmentationNode.CreateClosedSurfaceRepresentation()
        _log(log, "External air added as a separate segment.")
    finally:
        slicer.mrmlScene.RemoveNode(tmp)


def _applyDisplaySmoothing(segmentationNode, factor):
    """Set the closed-surface smoothing factor (0-1) and rebuild the surface for display/export."""
    try:
        factor = float(factor)
    except Exception:  # noqa: BLE001
        return
    try:
        segmentationNode.GetSegmentation().SetConversionParameter("Smoothing factor", str(factor))
        segmentationNode.RemoveClosedSurfaceRepresentation()
        segmentationNode.CreateClosedSurfaceRepresentation()
    except Exception:  # noqa: BLE001
        pass


def extendBinaryArray(array, referenceNode, extensionMm, direction, log=None):
    """Extend the final binary mask by repeating its terminal non-empty slice.

    The requested length is applied in two steps (absorbing the old standalone extend
    logic): (1) fill the gap between the mask's terminal slice and the volume border,
    staying inside the acquired field of view; (2) if more is requested, grow the array
    beyond the border and replicate the terminal slice into the new slices. The IJK->RAS
    matrix is shifted when the array grows on the 'min' side so the extended mask stays
    connected and correctly positioned in space.

    Returns (extendedArray, ijkToRAS).
    """
    nonEmpty = np.where(np.any(array > 0, axis=tuple(i for i in range(3) if i != AXIS)))[0]
    if len(nonEmpty) == 0:
        raise RuntimeError("Cannot extend an empty final segmentation.")

    spacingIJK = referenceNode.GetSpacing()
    arraySpacing = [spacingIJK[2], spacingIJK[1], spacingIJK[0]]
    stepMm = float(arraySpacing[AXIS])
    slicesToAdd = int(round(float(extensionMm) / stepMm)) if stepMm > 0 else 0
    if slicesToAdd <= 0:
        return array.astype(np.uint8), volumeIJKToRASArray(referenceNode)

    ijkToRAS = volumeIJKToRASArray(referenceNode)
    extendSide = resolveExtensionSideForDirection(direction, array.shape, ijkToRAS)
    edgeIndex = int(nonEmpty.min()) if extendSide == "min" else int(nonEmpty.max())
    edgeSlice = getAxisSlice(array, edgeIndex).copy()
    axisLength = array.shape[AXIS]

    # Step 1: fill the in-FOV gap towards the volume border, working on a copy.
    filled = array.copy()
    gapSlices = edgeIndex if extendSide == "min" else (axisLength - 1 - edgeIndex)
    slicesToFillInside = min(slicesToAdd, gapSlices)
    for step in range(1, slicesToFillInside + 1):
        insideIndex = edgeIndex - step if extendSide == "min" else edgeIndex + step
        setAxisSlice(filled, insideIndex, edgeSlice)
    if slicesToFillInside > 0 and log:
        _log(log, "Extension: %d slice(s) filled inside the acquired FOV." % slicesToFillInside)

    # Step 2: if the request exceeds the in-FOV gap, grow the array beyond the border.
    slicesToAddOutside = slicesToAdd - slicesToFillInside
    if slicesToAddOutside <= 0:
        return filled.astype(np.uint8), volumeIJKToRASArray(referenceNode)

    oldShape = list(filled.shape)
    newShape = oldShape.copy()
    newShape[AXIS] += slicesToAddOutside
    extended = np.zeros(newShape, dtype=np.uint8)

    if extendSide == "min":
        copyArrayIntoAxisRange(extended, slicesToAddOutside, filled)
        for sliceIndex in range(slicesToAddOutside):
            setAxisSlice(extended, sliceIndex, edgeSlice)
        shiftVoxel = np.array([0.0, 0.0, 0.0, 0.0])
        shiftVoxel[NUMPY_AXIS_TO_IJK_AXIS[AXIS]] = -slicesToAddOutside
        shiftRAS = ijkToRAS @ shiftVoxel
        ijkToRAS[0:3, 3] += shiftRAS[0:3]
    else:
        copyArrayIntoAxisRange(extended, 0, filled)
        start = oldShape[AXIS]
        for offset in range(slicesToAddOutside):
            setAxisSlice(extended, start + offset, edgeSlice)

    if log:
        _log(log, "Extension: %d slice(s) added beyond the volume (side=%s)." % (slicesToAddOutside, extendSide))
    return extended.astype(np.uint8), ijkToRAS


def _log(log, message):
    """Call an optional log callback, ignoring failures so logging never breaks logic."""
    if log:
        try:
            log(message)
        except Exception:  # noqa: BLE001
            pass
