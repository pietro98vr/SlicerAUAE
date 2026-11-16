"""Airway post-processing and extension logic.

This module is the distinctive part of this project, which is based on SlicerUpperAirwaySegmentator:
the geometric logic that extends the final binary airway mask beyond the acquired
field of view; the automatic connected-component cleanup applied during the segmentation
step (the embedded Segment Editor stays available for manual refinement); the optional
recovery of the internal air cavities (paranasal sinuses) the model drops, folded back into
the airway; and the optional external face-air segment used as an inlet volume for flow
modelling. Internal cavities and external air are separated by one head-envelope decomposition,
so the sinuses never leak into the external-air segment.

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
# Internal air cavities the model often drops (paranasal sinuses, sinonasal complex). These are
# air ENCLOSED by the head envelope, so they belong to the airway, never to the external air.
# The threshold doubles as a small-island filter that drops the smaller cephalostat / head-holder
# air bubbles sitting near the frontal sinus, before those cavities are merged into the airway.
DEFAULT_MIN_INTERNAL_AIR_MM3 = 250.0


def defaultOptions():
    """Return the default post-processing options shared by interactive and batch runs."""
    return {
        "removeSmallIslands": True,
        "minIslandMm3": DEFAULT_MIN_ISLAND_MM3,
        "keepLargestIsland": False,
        "includeInternalAir": True,  # attempt frontal-sinus recovery by default
        "minInternalAirMm3": DEFAULT_MIN_INTERNAL_AIR_MM3,
        "smoothingFactor": 0.0,
        "segmentExternalAir": False,
        "mergeExternalAir": False,
        "extend": False,
        "lengthMm": 100.0,
    }


def postprocessSegmentation(segmentationNode, volumeNode, options, log=None):
    """Clean the raw model output and build the final airway segmentation node.

    Single source of truth for both the interactive widget and the batch processor. The steps
    run in NumPy on the mask exported to the reference volume grid: island cleanup, optional
    sinus recovery, optional external face air, optional inferior extension. The extension may
    grow the geometry, so the result is rebuilt as a fresh segmentation node carrying an
    "Airway" segment (plus "External air (face)" when requested and not merged). The closed
    surface is generated once, at the end. Returns the new segmentation node.
    """
    options = {**defaultOptions(), **(options or {})}

    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "airway_tmp_labelmap")
    try:
        airway = _exportBinaryMask(segmentationNode, volumeNode, labelmapNode)
        spacing = labelmapNode.GetSpacing()
        voxelVolumeMm3 = float(np.prod(spacing))
        baseIJKToRAS = volumeIJKToRASArray(labelmapNode)

        if options["keepLargestIsland"]:
            airway = keepLargestConnectedComponent(airway, log) > 0
        if options["removeSmallIslands"]:
            airway = removeSmallIslands(airway, options["minIslandMm3"], voxelVolumeMm3, log) > 0

        # Air handling on the labelmap grid (same as the cleaned airway, so the masks align).
        # Both features share one decomposition of the volume into a sealed head envelope:
        # air INSIDE the envelope = internal cavities (sinuses) that belong to the airway;
        # air OUTSIDE = ambient air. This is what keeps the frontal sinuses out of the
        # external-air segment and lets us fold them back into the airway.
        external = None
        wantExternal = bool(options["segmentExternalAir"])
        wantInternal = bool(options["includeInternalAir"])
        if wantExternal or wantInternal:
            volumeArray = _volumeArrayLike(volumeNode, airway)
            threshold = _airThresholdFromAirway(volumeArray, airway)
            sealedHead = None if threshold is None else _sealedHeadMask(volumeArray, threshold, spacing, log)
            if sealedHead is None:
                _log(log, "Air handling skipped: grey-level volume not available on the airway grid.")
            else:
                airMask = volumeArray <= threshold
                # Recover the internal air cavities (sinuses) the model drops, back INTO the airway.
                # The island cleanup above has already been applied to the main airway; the sinuses
                # are cleaned separately here, dropping the smaller cephalostat air bubbles by size
                # before they are merged in.
                if wantInternal:
                    airway |= computeInternalAirCavities(
                        airMask, sealedHead, airway, voxelVolumeMm3, log,
                        minRegionMm3=float(options["minInternalAirMm3"]),
                    ) > 0
                # External air = ambient air OUTSIDE the head envelope, in front of the face.
                if wantExternal:
                    external = computeExternalAirMask(
                        airMask, airway, sealedHead, baseIJKToRAS, voxelVolumeMm3, log
                    ) > 0
                    if not external.any():
                        external = None

        # Anatomical airway volume (airway plus any recovered sinuses), measured before the
        # extension pads it and before the external air is merged, so it stays clinically meaningful.
        airwayVolumeMm3 = float(np.count_nonzero(airway)) * voxelVolumeMm3
        _log(log, "Airway volume: %.2f mL (%.0f mm^3)." % (airwayVolumeMm3 / 1000.0, airwayVolumeMm3))

        # Extension is applied to the AIRWAY only, then (if requested) the external air is merged
        # in AFTER extension, padded onto the extended grid. This guarantees the caudal extension
        # replicates the airway's terminal slice, never the external air (which can sit lower).
        ijkToRAS = baseIJKToRAS
        merged = external is not None and bool(options["mergeExternalAir"])
        if options["extend"] and float(options["lengthMm"]) > 0:
            airway, ijkToRAS, padInfo = extendBinaryArray(airway, labelmapNode, float(options["lengthMm"]), log)
            if merged:
                external = _padCompanionToExtendedGrid(external, padInfo)
        if merged:
            airway = (airway > 0) | external
            external = None
            _log(log, "External air merged into the airway segment.")

        node = _buildAirwaySegmentation(airway, ijkToRAS, segmentationNode.GetName())
        node.SetAttribute("AUAE.airwayVolumeMm3", "%.1f" % airwayVolumeMm3)
        if external is not None:
            # Separate segment: import on the original (unextended) grid; Slicer resamples it
            # into the node's shared geometry.
            _addExternalAirSegment(node, external, baseIJKToRAS, log)
        applySurfaceSmoothing(node, options["smoothingFactor"])
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
    lengthMm = float(options["lengthMm"])
    if lengthMm <= 0:
        raise RuntimeError("Set an extension length greater than 0 mm.")

    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "airway_tmp_labelmap")
    try:
        airway = _exportBinaryMask(segmentationNode, volumeNode, labelmapNode)
        airway, ijkToRAS, _padInfo = extendBinaryArray(airway, labelmapNode, lengthMm, log)
        node = _buildAirwaySegmentation(airway, ijkToRAS, segmentationNode.GetName())
        applySurfaceSmoothing(node, options["smoothingFactor"])
        return node
    finally:
        slicer.mrmlScene.RemoveNode(labelmapNode)


def _exportBinaryMask(segmentationNode, volumeNode, labelmapNode):
    """Export every segment of a segmentation to labelmapNode and return it as a boolean array.

    The export uses the reference (volume) geometry, so the full slice stack is available for
    the extension and the mask lines up voxel for voxel with the grey-level volume.
    """
    if volumeNode is not None:
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    exported = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        segmentationNode, labelmapNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
    )
    if exported is False:
        raise RuntimeError("Could not export the segmentation to a labelmap.")
    mask = slicer.util.arrayFromVolume(labelmapNode) > 0
    if not mask.any():
        raise RuntimeError("The segmentation is empty; nothing to process.")
    return mask


def _importMaskAsSegment(segmentationNode, mask, ijkToRAS, name, color, opacity3D):
    """Import a binary mask as a new named segment; return its id, or None if nothing came in."""
    labelmapNode = slicer.util.addVolumeFromArray(
        np.asarray(mask, dtype=np.uint8), ijkToRAS=ijkToRAS, name="airway_tmp_import",
        nodeClassName="vtkMRMLLabelMapVolumeNode",
    )
    try:
        segmentation = segmentationNode.GetSegmentation()
        beforeIds = set(_segmentIds(segmentation))
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)
        newIds = [sid for sid in _segmentIds(segmentation) if sid not in beforeIds]
        if not newIds:
            return None
        segment = segmentation.GetSegment(newIds[0])
        segment.SetName(name)
        segment.SetColor(*color)
        displayNode = segmentationNode.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetSegmentOpacity3D(newIds[0], opacity3D)
        return newIds[0]
    finally:
        slicer.mrmlScene.RemoveNode(labelmapNode)


def _buildAirwaySegmentation(mask, ijkToRAS, name):
    """Build a segmentation node carrying a single 'Airway' segment from a binary mask.

    The closed surface is not generated here: the caller finishes with applySurfaceSmoothing,
    which builds it once with the right smoothing factor.
    """
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
    segmentationNode.CreateDefaultDisplayNodes()
    segmentId = _importMaskAsSegment(
        segmentationNode, mask, ijkToRAS, AIRWAY_SEGMENT_NAME, AIRWAY_SEGMENT_COLOR, 0.8
    )
    if segmentId is None:
        slicer.mrmlScene.RemoveNode(segmentationNode)
        raise RuntimeError("Import produced no airway segment.")
    return segmentationNode


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


def inferiorStackSide(arrayShape, ijkToRAS):
    """Return which end of the slice stack ('min' or 'max') is anatomically inferior.

    The extension always grows towards the neck. The RAS-S (cranio-caudal) coordinate of the
    centre of the first and last slice is compared, so the right end is chosen whatever the
    patient orientation stored in the IJK->RAS matrix. Note the KJI->IJK index reversal
    before the matrix product, and that S is row 2.
    """
    def sliceS(sliceIndex):
        center = [(n - 1) / 2.0 for n in arrayShape]
        center[AXIS] = float(sliceIndex)
        return float((ijkToRAS @ np.array([center[2], center[1], center[0], 1.0]))[2])

    return "min" if sliceS(0) < sliceS(arrayShape[AXIS] - 1) else "max"


def _axisWindow(ndim, start, stop):
    """Index tuple selecting the slices [start, stop) along AXIS and everything on the others."""
    window = [slice(None)] * ndim
    window[AXIS] = slice(start, stop)
    return tuple(window)


def _labelComponents(mask):
    """Label the connected components of a mask: (labels, voxels per label), background at 0.

    Returns (None, None) when scipy is missing, so callers can degrade to a no-op.
    """
    try:
        from scipy import ndimage
    except Exception:  # noqa: BLE001
        return None, None
    labeled, _count = ndimage.label(mask)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background never counts as a component
    return labeled, sizes


def _dropSmallComponents(mask, minVoxels):
    """Boolean mask without the components smaller than minVoxels, and how many were dropped.

    Selection goes through a per-label lookup table (keep[labeled]), which is a single pass
    over the volume and far cheaper than np.isin on a full-size label array.
    """
    mask = np.asarray(mask) > 0
    if minVoxels <= 1:
        return mask, 0
    labeled, sizes = _labelComponents(mask)
    if labeled is None:
        return mask, 0
    keep = sizes >= minVoxels
    dropped = int(np.count_nonzero(sizes)) - int(np.count_nonzero(keep))
    return (keep[labeled] if dropped else mask), dropped


def _minVoxels(minSizeMm3, voxelVolumeMm3):
    """Voxel count matching a physical volume threshold (0 when the spacing is unusable)."""
    if voxelVolumeMm3 <= 0:
        return 0
    return int(np.ceil(float(minSizeMm3) / float(voxelVolumeMm3)))


def keepLargestConnectedComponent(array, log=None):
    """Keep only the largest connected component, dropping every other island."""
    mask = np.asarray(array) > 0
    labeled, sizes = _labelComponents(mask)
    if labeled is None:
        _log(log, "Keep largest island skipped: scipy is unavailable.")
        return mask.astype(np.uint8)
    count = int(np.count_nonzero(sizes))
    if count <= 1:
        return mask.astype(np.uint8)
    _log(log, "Keep largest island: kept 1 of %d component(s)." % count)
    return (labeled == int(np.argmax(sizes))).astype(np.uint8)


def removeSmallIslands(array, minSizeMm3, voxelVolumeMm3, log=None):
    """Remove connected components smaller than a physical volume threshold.

    The automatic counterpart of the Segment Editor "Islands / remove small islands" effect.
    The default threshold matches upstream (200 voxels at 0.3 mm isotropic) but is expressed
    in mm^3, so it is correct for any spacing.
    """
    minVoxels = _minVoxels(minSizeMm3, voxelVolumeMm3)
    cleaned, dropped = _dropSmallComponents(array, minVoxels)
    if dropped:
        _log(log, "Remove small islands: dropped %d component(s) < %d voxels." % (dropped, minVoxels))
    return cleaned.astype(np.uint8)


def _volumeArrayLike(volumeNode, referenceArray):
    """Grey-level volume array aligned to the airway labelmap grid, or None if unavailable."""
    if volumeNode is None:
        return None
    try:
        vol = slicer.util.arrayFromVolume(volumeNode)
    except Exception:  # noqa: BLE001
        return None
    return vol if vol is not None and vol.shape == referenceArray.shape else None


def _rasRamps(shape, ijkToRAS, rasRow):
    """One RAS component (0=R, 1=A, 2=S) of a KJI grid, as an offset plus three 1-D ramps.

    The affine is separable: the component at voxel (k, j, i) is
    offset + kRamp[k] + jRamp[j] + iRamp[i]. Array axes are KJI, hence the K/J/I coefficients
    are columns 2/1/0 of the chosen RAS row. Working from the ramps avoids ever allocating a
    full-size float field, which on a large CBCT would cost about a gigabyte.
    """
    nK, nJ, nI = shape
    row = ijkToRAS[rasRow]
    return float(row[3]), np.arange(nK) * row[2], np.arange(nJ) * row[1], np.arange(nI) * row[0]


def _clipToAnteriorOf(mask, referenceMask, ijkToRAS):
    """Keep, in place, only the voxels of mask anterior to the median A of referenceMask."""
    offset, kRamp, jRamp, iRamp = _rasRamps(mask.shape, ijkToRAS, 1)  # RAS 'A' (anterior +)
    k, j, i = np.nonzero(referenceMask)
    if k.size == 0:
        return
    cutoff = float(np.median(offset + kRamp[k] + jRamp[j] + iRamp[i]))
    plane = offset + jRamp[:, None] + iRamp[None, :]
    for index in range(mask.shape[0]):  # slice by slice: the memory stays one 2-D plane
        mask[index] &= (plane + kRamp[index]) >= cutoff


def _airThresholdFromAirway(volumeArray, airwayArray):
    """Self-calibrated air/tissue threshold from the intensities under the known airway mask.

    The airway lumen is air, so its intensity distribution fixes the air threshold. This holds
    whether the volume is in Hounsfield units (air ~ -1000) or intensity-shifted (air ~ 0), so
    no fixed HU value is assumed. Returns None when the airway mask is empty.
    """
    if volumeArray is None:
        return None
    vals = volumeArray[airwayArray > 0]
    if vals.size == 0:
        return None
    return float(vals.mean() + 2.0 * (vals.std() + 1e-6))


def _sealedHeadMask(volumeArray, threshold, spacingIJK, log=None):
    """Solid head envelope: the tissue mask closed to seal small openings, then hole-filled.

    Everything brighter than the air threshold is tissue or bone; the largest such component is
    the patient. Morphological closing seals the narrow openings (nostrils, mouth, thin sinus
    ducts) so that hole-filling then treats the whole internal airspace as enclosed. The result
    is a solid mask whose interior holds the sinuses and airway and whose exterior is ambient
    air. Returns a boolean array, or None if scipy is unavailable.
    """
    try:
        from scipy import ndimage
    except Exception as exc:  # noqa: BLE001
        _log(log, "Air envelope skipped; scipy unavailable: " + str(exc))
        return None
    fullShape = volumeArray.shape

    # The head envelope is a coarse structure, so on large volumes the whole computation
    # (labelling included) runs on a downsampled copy and the result is resized back. This
    # keeps time and memory bounded without changing the envelope meaningfully; small volumes
    # are processed at full resolution.
    factor = 2 if volumeArray.size > 12_000_000 else 1
    tissue = (volumeArray[::factor, ::factor, ::factor] if factor > 1 else volumeArray) > threshold

    labeled, sizes = _labelComponents(tissue)
    if np.count_nonzero(sizes) > 1:
        tissue = labeled == int(np.argmax(sizes))  # the patient, not the table or props
    del labeled

    # Seal openings of about 3 mm; the iteration count scales with the working voxel size.
    minSpacing = min([float(s) for s in spacingIJK if float(s) > 0] or [1.0])
    iterations = max(1, min(8, int(round(3.0 / (minSpacing * factor)))))
    sealed = ndimage.binary_fill_holes(ndimage.binary_closing(tissue, iterations=iterations))
    if factor > 1:
        sealed = _upsampleNearest(sealed, fullShape, factor)
        _log(log, "Air envelope computed at 1/%d resolution for speed." % factor)
    return sealed


def _upsampleNearest(small, fullShape, factor):
    """Nearest-neighbour upsample a boolean array by an integer factor, cropped to fullShape."""
    up = small
    for axis in range(up.ndim):
        up = np.repeat(up, factor, axis=axis)
    return up[:fullShape[0], :fullShape[1], :fullShape[2]]


def computeInternalAirCavities(airMask, sealedHead, airwayArray, voxelVolumeMm3, log=None,
                               minRegionMm3=DEFAULT_MIN_INTERNAL_AIR_MM3):
    """Air enclosed by the head envelope and not already in the airway: the dropped sinuses.

    These paranasal / sinonasal cavities belong to the airway, so the caller unions them into
    it. Components below minRegionMm3 are dropped, which removes the smaller cephalostat /
    head-holder air bubbles near the frontal sinus before the merge. Returns a uint8 array; an
    all-zero array means 'nothing to add'.
    """
    if sealedHead is None:
        return np.zeros(np.shape(airwayArray), dtype=np.uint8)
    internal = airMask & sealedHead & ~(np.asarray(airwayArray) > 0)
    internal, _dropped = _dropSmallComponents(internal, _minVoxels(minRegionMm3, voxelVolumeMm3))
    _log(log, "Internal air cavities (sinuses) recovered for the airway: %d voxels."
         % int(np.count_nonzero(internal)))
    return internal.astype(np.uint8)


def computeExternalAirMask(airMask, airwayArray, sealedHead, ijkToRAS, voxelVolumeMm3, log=None,
                           minRegionMm3=DEFAULT_MIN_EXTERNAL_AIR_MM3):
    """Ambient air OUTSIDE the head envelope, in front of the face, as a binary mask.

    External air is the air outside the sealed head, so internal cavities such as the frontal
    sinuses are excluded by construction. It is restricted to the components touching the volume
    border and to the region anterior to the airway (the 'front of the face'), with the airway
    and negligible specks removed. Returns a uint8 array; an all-zero array means 'not found'.
    """
    empty = np.zeros(np.shape(airwayArray), dtype=np.uint8)
    if sealedHead is None:
        return empty
    airway = np.asarray(airwayArray) > 0
    external = airMask & ~sealedHead & ~airway  # air strictly outside the head envelope

    labeled, sizes = _labelComponents(external)
    if labeled is None:
        _log(log, "External air skipped: scipy is unavailable.")
        return empty
    faces = (labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1], labeled[:, :, 0], labeled[:, :, -1])
    touchesBorder = np.zeros(sizes.shape, dtype=bool)
    touchesBorder[np.unique(np.concatenate([face.ravel() for face in faces]))] = True
    touchesBorder[0] = False
    if not touchesBorder.any():
        return empty
    external = touchesBorder[labeled]  # the ambient air around the patient
    del labeled

    # Clip to the front of the face: keep voxels anterior to the airway's mid A-coordinate.
    try:
        _clipToAnteriorOf(external, airway, ijkToRAS)
    except Exception as exc:  # noqa: BLE001
        _log(log, "External air: anterior clip skipped (%s); keeping full ambient air." % exc)

    external, _dropped = _dropSmallComponents(external, _minVoxels(minRegionMm3, voxelVolumeMm3))
    _log(log, "External face air: %d voxels." % int(np.count_nonzero(external)))
    return external.astype(np.uint8)


def _addExternalAirSegment(segmentationNode, externalMask, ijkToRAS, log=None):
    """Import the external-air mask as a second, distinctly coloured segment in the node."""
    segmentId = _importMaskAsSegment(
        segmentationNode, externalMask, ijkToRAS, EXTERNAL_AIR_SEGMENT_NAME, EXTERNAL_AIR_SEGMENT_COLOR, 0.4
    )
    _log(log, "External air added as a separate segment." if segmentId
         else "External air import produced no segment.")


def applySurfaceSmoothing(segmentationNode, factor):
    """Set the closed-surface smoothing factor (0-1) and (re)build the surface when needed.

    The surface feeds the 3D view and the STL/OBJ export. Generating it is the slowest part of
    the display path, so nothing happens when the node already carries a surface built with the
    same factor.
    """
    try:
        factor = min(1.0, max(0.0, float(factor)))
        segmentation = segmentationNode.GetSegmentation()
        surfaceName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
        try:
            current = float(segmentation.GetConversionParameter("Smoothing factor"))
        except (TypeError, ValueError):
            current = None
        if current is not None and abs(current - factor) < 1e-6 and segmentation.ContainsRepresentation(surfaceName):
            return
        segmentation.SetConversionParameter("Smoothing factor", str(factor))
        segmentationNode.RemoveClosedSurfaceRepresentation()
        segmentationNode.CreateClosedSurfaceRepresentation()
    except Exception:  # noqa: BLE001
        pass  # display-only: a failed surface rebuild must never break the pipeline


def extendBinaryArray(array, referenceNode, extensionMm, log=None):
    """Extend a binary mask inferiorly by repeating its terminal non-empty slice.

    The requested length is consumed in two steps: (1) fill the gap between the mask's terminal
    slice and the volume border, staying inside the acquired field of view; (2) if more is
    requested, grow the array beyond the border and replicate the terminal slice into the new
    slices. The IJK->RAS matrix is shifted when the array grows on the 'min' side, so the
    extended mask stays connected and correctly positioned in space.

    Returns (extendedArray, ijkToRAS, padInfo) where padInfo = (side, slicesAddedOutside).
    padInfo lets a companion mask (e.g. the external air) be padded onto the same extended grid
    with _padCompanionToExtendedGrid, so it can be unioned in AFTER the extension.
    """
    mask = np.asarray(array) > 0
    occupied = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != AXIS)))
    if occupied.size == 0:
        raise RuntimeError("Cannot extend an empty segmentation.")

    ijkToRAS = volumeIJKToRASArray(referenceNode)
    stepMm = float(referenceNode.GetSpacing()[NUMPY_AXIS_TO_IJK_AXIS[AXIS]])
    slicesToAdd = int(round(float(extensionMm) / stepMm)) if stepMm > 0 else 0
    if slicesToAdd <= 0:
        return mask.astype(np.uint8), ijkToRAS, (None, 0)

    side = inferiorStackSide(mask.shape, ijkToRAS)
    edgeIndex = int(occupied[0] if side == "min" else occupied[-1])
    edgeSlice = mask[_axisWindow(mask.ndim, edgeIndex, edgeIndex + 1)]  # kept 3-D so it broadcasts

    # How much of the request fits inside the field of view, and how much has to be added outside.
    gapSlices = edgeIndex if side == "min" else mask.shape[AXIS] - 1 - edgeIndex
    inside = min(slicesToAdd, gapSlices)
    outside = slicesToAdd - inside

    padWidth = [(0, 0)] * mask.ndim
    padWidth[AXIS] = (outside, 0) if side == "min" else (0, outside)
    extended = np.pad(mask.astype(np.uint8), padWidth, mode="constant")  # also the working copy

    # One broadcast assignment fills both the in-FOV gap and the newly added slices.
    if side == "min":
        extended[_axisWindow(mask.ndim, edgeIndex - inside, edgeIndex + outside)] = edgeSlice
    else:
        extended[_axisWindow(mask.ndim, edgeIndex + 1, edgeIndex + 1 + slicesToAdd)] = edgeSlice

    if outside and side == "min":
        # The origin moves with the slices prepended in front of index 0.
        shiftVoxel = np.zeros(4)
        shiftVoxel[NUMPY_AXIS_TO_IJK_AXIS[AXIS]] = -outside
        ijkToRAS[0:3, 3] += (ijkToRAS @ shiftVoxel)[0:3]

    if inside:
        _log(log, "Extension: %d slice(s) filled inside the acquired FOV." % inside)
    if outside:
        _log(log, "Extension: %d slice(s) added beyond the volume (side=%s)." % (outside, side))
    return extended, ijkToRAS, (side, outside)


def _padCompanionToExtendedGrid(companion, padInfo):
    """Pad a companion mask with zeros so it matches an array extended by extendBinaryArray.

    The extension only ever adds 'slicesAddedOutside' slices on one end of AXIS; padding the
    companion (e.g. the external air) the same way puts it back on the extended grid without
    moving it, so it can be unioned after the airway has been extended.
    """
    side, nOutside = padInfo
    if not nOutside:
        return companion
    padWidth = [(0, 0)] * companion.ndim
    padWidth[AXIS] = (nOutside, 0) if side == "min" else (0, nOutside)
    return np.pad(companion, padWidth, mode="constant")


def _log(log, message):
    """Call an optional log callback, ignoring failures so logging never breaks logic."""
    if log:
        try:
            log(message)
        except Exception:  # noqa: BLE001
            pass
