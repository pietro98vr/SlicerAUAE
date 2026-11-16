"""Batch processing driver.

Added by this project. Runs the upstream nnU-Net segmentation over a set of inputs (a folder of
volumes and DICOM series, or the list in a JSON template), applies the same post-processing as
the module panel (AirwayExtension.postprocessSegmentation), and writes one output subfolder per
case with the requested mesh/labelmap files and the airway volume.

The default template lives in Resources/batch_template.json and documents every field.
"""

import json
import os
import re
from pathlib import Path

import slicer

from . import AirwayExtension
from . import Progress


# Volume file types accepted in a folder-in / folder-out batch (the classic nnU-Net style).
VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")


def defaultTemplatePath():
    """Path to the embedded batch template shipped in Resources."""
    return Path(__file__).parent.joinpath("..", "Resources", "batch_template.json").resolve()


def _looksLikeDicom(path):
    """True if a file is DICOM (by .dcm extension or the DICM magic at byte 128)."""
    if path.lower().endswith(".dcm"):
        return True
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except Exception:  # noqa: BLE001
        return False


def _isDicomFolder(folder):
    """True if a folder contains at least one DICOM file at any depth.

    The search recurses, so an exported layout like Patient/Study/Series/*.dcm is detected, and
    it stops at the first DICOM file found. One series per patient folder is assumed.
    """
    try:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if _looksLikeDicom(os.path.join(root, name)):
                    return True
    except Exception:  # noqa: BLE001
        pass
    return False


def listVolumesInFolder(folder):
    """List the inputs in a folder: single-file volumes plus DICOM series.

    Each entry is either a volume file path (.nii/.nrrd/...) or a folder holding a DICOM series
    (a subfolder, or the folder itself when it is a single series). The batch loads each entry
    accordingly, so a folder of per-patient DICOM subfolders works out of the box.
    """
    folder = str(folder)
    if not os.path.isdir(folder):
        return []
    inputs = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and any(name.lower().endswith(e) for e in VOLUME_EXTENSIONS):
            inputs.append(p)
    for name in sorted(os.listdir(folder)):
        sub = os.path.join(folder, name)
        if os.path.isdir(sub) and _isDicomFolder(sub):
            inputs.append(sub)
    if not inputs and _isDicomFolder(folder):
        inputs.append(folder)  # the folder itself is a single DICOM series
    return inputs


def folderConfig(inputFolder, outputRoot, exportFormats, postprocess=None, exportTargets=None):
    """Build a batch config from an input folder (classic nnU-Net-style folder-in/folder-out).

    Every volume in the folder is segmented with the given post-processing and written to a
    per-volume subfolder under outputRoot. If outputRoot is empty it defaults to an
    'AUAE_output' folder beside the inputs. postprocess uses the same option keys as
    AirwayExtension.defaultOptions (so the widget can pass its current settings straight in).
    exportTargets is any of 'airway', 'external', 'merged'.
    """
    inputFolder = str(inputFolder)
    outputRoot = str(outputRoot or "").strip() or os.path.join(inputFolder, "AUAE_output")
    return {
        "mode": "segment",
        "output_root": outputRoot,
        "subfolder_from": "filename",
        "export_formats": [str(f).strip().upper() for f in (exportFormats or ["STL", "NIFTI"])],
        "export_targets": [str(t).strip().lower() for t in (exportTargets or ["airway", "external"])],
        "postprocess": {**AirwayExtension.defaultOptions(), **(postprocess or {})},
        "volumes": listVolumesInFolder(inputFolder),
    }


def loadConfig(jsonPath):
    """Load and normalise a batch template file into a plain options dict.

    'volumes' may be given explicitly; alternatively 'input_folder' points at a directory and
    every volume inside it is used (classic folder-in/folder-out), and 'output_root' then
    defaults to an 'AUAE_output' folder beside the inputs.
    """
    with open(jsonPath, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    # A 'volumes' entry is a path, or {"path": ..., "name": ...} to choose the output subfolder
    # (the name may be nested, e.g. "P001/pre").
    volumes, caseNames = [], {}
    for item in raw.get("volumes", []):
        if isinstance(item, dict):
            path, name = str(item.get("path", "") or "").strip(), safeCaseName(item.get("name"))
        else:
            path, name = str(item).strip(), None
        if path:
            volumes.append(path)
            if name:
                caseNames[path] = name
    inputFolder = str(raw.get("input_folder", "") or "").strip()
    if inputFolder and not volumes:
        volumes = listVolumesInFolder(inputFolder)
    outputRoot = str(raw.get("output_root", "") or "").strip()
    if not outputRoot and inputFolder:
        outputRoot = os.path.join(inputFolder, "AUAE_output")
    extension = raw.get("extension", {}) or {}

    # Start from the shared defaults so a template that omits a key behaves like the module
    # panel, then apply only the keys the template actually sets.
    postprocess = AirwayExtension.defaultOptions()
    templateKeys = (
        ("remove_small_islands", "removeSmallIslands", bool),
        ("keep_largest_island", "keepLargestIsland", bool),
        ("min_island_mm3", "minIslandMm3", float),
        ("include_internal_air", "includeInternalAir", bool),
        ("min_internal_air_mm3", "minInternalAirMm3", float),
        ("segment_external_air", "segmentExternalAir", bool),
        ("merge_external_air", "mergeExternalAir", bool),
        ("smoothing_factor", "smoothingFactor", float),
    )
    for templateKey, optionKey, cast in templateKeys:
        if templateKey in raw:
            postprocess[optionKey] = cast(raw[templateKey])
    if "enabled" in extension:
        postprocess["extend"] = bool(extension["enabled"])
    if "length_mm" in extension:
        postprocess["lengthMm"] = float(extension["length_mm"])
    # Island options are mutually exclusive, mirroring the UI: keep-largest wins if both set.
    if postprocess["keepLargestIsland"]:
        postprocess["removeSmallIslands"] = False

    return {
        "mode": str(raw.get("mode", "segment") or "segment").strip().lower(),
        "output_root": outputRoot,
        "subfolder_from": str(raw.get("subfolder_from", "filename") or "filename"),
        "export_formats": [str(f).strip().upper() for f in raw.get("export_formats", ["STL", "NIFTI"])],
        "export_targets": [str(t).strip().lower() for t in raw.get("export_targets", ["airway", "external"])],
        "postprocess": postprocess,
        "volumes": volumes,
        "case_names": caseNames,
    }


def caseName(inputPath):
    """Name of a batch case: the input file name without its volume extension, or the folder name."""
    name = Path(inputPath).name
    for suffix in VOLUME_EXTENSIONS:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or "case"
    return name or "case"


def safeCaseName(name):
    """Clean a user-chosen output subfolder name; nested names keep their parts.

    Drive letters, absolute prefixes and '..' are dropped, so the result always stays inside
    the output folder. Returns None when nothing usable is left.
    """
    parts = [part.strip() for part in re.split(r"[\\/]+", str(name or ""))]
    parts = [re.sub(r'[<>:"|?*]', "_", part) for part in parts if part and part not in (".", "..")]
    return os.path.join(*parts) if parts else None


class BatchProcessor:
    """Sequentially process and export every input listed in a batch config."""

    def __init__(self, nnUnetFolder, exportFn, progressCallback=None):
        """
        :param nnUnetFolder: folder holding the downloaded nnU-Net weights.
        :param exportFn: callable(segmentationNode, folderPath, exportFormats, targets).
        :param progressCallback: callable(str) for progress lines.
        """
        self._nnUnetFolder = nnUnetFolder
        self._exportFn = exportFn
        self._userLog = progressCallback or (lambda msg: None)
        self._progress = self._logAndTrack  # every log line also feeds the progress tracker
        self._tracker = None
        self._onStage = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _logAndTrack(self, message):
        """Pass a log line on, and report the progress when the line moved it forward."""
        self._userLog(message)
        if self._tracker is not None and self._tracker.update(message):
            self._notifyStage()

    def _notifyStage(self):
        if self._onStage is not None and self._tracker is not None:
            try:
                self._onStage(self._tracker.snapshot())
            except Exception:  # noqa: BLE001
                pass  # a progress display must never break the run

    def run(self, config, exportFormatEnum, deviceKwargs=None, onProgress=None, onStage=None):
        """Process every input in config. Returns a per-case result summary list.

        'segment' mode runs the model on each volume and post-processes the result. 'extend'
        mode reruns only the airway extension on saved segmentation labelmaps (NIFTI/NRRD), so
        segmentation and extension can be two separate batch passes; it needs neither the
        model nor a GPU.

        :param exportFormatEnum: the ExportFormat flag value selected for all cases.
        :param deviceKwargs: optional Parameter device kwargs (e.g. {"device": "cuda"}).
        :param onProgress: optional callable(done, total), called after each case.
        :param onStage: optional callable(snapshot), called whenever the progress moves, within
            a case too. The snapshot is the dict of Progress.RunProgress.snapshot(): cases
            done, current case and stage, fractions of the case and of the run, time left.
        """
        outputRoot = config["output_root"]
        if not outputRoot:
            raise ValueError("Batch 'output_root' is not set.")
        inputs = config["volumes"]
        if not inputs:
            raise ValueError("The batch lists no inputs.")
        os.makedirs(outputRoot, exist_ok=True)

        options = config["postprocess"]
        targets = config.get("export_targets") or ["merged"]
        names = config.get("case_names") or {}
        self._caseNames = names
        self._tracker = Progress.RunProgress(len(inputs), config.get("mode", "segment"))
        self._onStage = onStage

        def export(segmentationNode, inputPath):
            subfolder = os.path.join(outputRoot, names.get(inputPath) or caseName(inputPath))
            os.makedirs(subfolder, exist_ok=True)
            self._progress("Exporting to " + subfolder)
            self._exportFn(segmentationNode, subfolder, exportFormatEnum, targets)
            return subfolder

        if config.get("mode", "segment") == "extend":
            def processOne(path, nodes):
                if not os.path.isfile(path):
                    raise FileNotFoundError("Segmentation not found: " + path)
                labelmapNode = slicer.util.loadLabelVolume(path)
                if labelmapNode is None:
                    raise RuntimeError("Could not load segmentation labelmap: " + path)
                nodes.append(labelmapNode)

                segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode", caseName(path) + "_Segmentation"
                )
                nodes.append(segmentationNode)
                segmentationNode.CreateDefaultDisplayNodes()
                segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(labelmapNode)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)

                extendedNode = AirwayExtension.extendSegmentation(
                    segmentationNode, labelmapNode, options, self._progress
                )
                nodes.append(extendedNode)
                return export(extendedNode, path)
        else:
            from SlicerNNUNetLib import Parameter, SegmentationLogic

            kwargs = dict(folds="0", modelPath=self._nnUnetFolder, **(deviceKwargs or {}))
            try:
                parameter = Parameter(**kwargs)
            except TypeError:  # older SlicerNNUNetLib without a device argument
                parameter = Parameter(folds="0", modelPath=self._nnUnetFolder)
            logic = SegmentationLogic()
            logic.progressInfo.connect(self._progress)

            def processOne(path, nodes):
                if not os.path.exists(path):
                    raise FileNotFoundError("Input not found: " + path)
                volumeNode = self._loadInput(path)
                nodes.append(volumeNode)

                logic.setParameter(parameter)
                logic.startSegmentation(volumeNode)
                logic.waitForSegmentationFinished()
                slicer.app.processEvents()

                rawSegmentation = logic.loadSegmentation()
                nodes.append(rawSegmentation)
                rawSegmentation.SetName(caseName(path) + "_Segmentation")
                segmentationNode = AirwayExtension.postprocessSegmentation(
                    rawSegmentation, volumeNode, options, self._progress
                )
                nodes.append(segmentationNode)

                subfolder = export(segmentationNode, path)
                self._writeAirwayVolume(segmentationNode, subfolder, path)
                return subfolder

        return self._runCases(inputs, processOne, onProgress)

    def _runCases(self, inputs, processOne, onProgress=None):
        """Run processOne(path, nodes) on each input, isolating failures and freeing the scene.

        processOne appends every scene node it creates to 'nodes'; they are removed after the
        case whatever happens, so a long batch (or a failing case) never accumulates nodes.
        """
        results = []
        total = len(inputs)
        for index, path in enumerate(inputs, start=1):
            if self._cancelled:
                self._progress("Batch cancelled by user.")
                break
            entry = {"volume": path, "status": "pending", "output": None, "error": None}
            if self._tracker is not None:
                name = getattr(self, "_caseNames", {}).get(path) or caseName(path)
                self._tracker.startCase(index, name.replace("\\", "/"))
                self._notifyStage()
            self._progress("[%d/%d] %s" % (index, total, path))
            nodes = []
            try:
                entry["output"] = processOne(path, nodes)
                entry["status"] = "ok"
                self._progress("[%d/%d] done -> %s" % (index, total, entry["output"]))
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "error"
                entry["error"] = str(exc)
                self._progress("[%d/%d] ERROR: %s" % (index, total, exc))
            finally:
                for node in reversed(nodes):
                    try:
                        slicer.mrmlScene.RemoveNode(node)
                    except Exception:  # noqa: BLE001
                        pass
                slicer.app.processEvents()
            results.append(entry)
            if self._tracker is not None:  # a failed case counts as done too: the run moves on
                self._tracker.finishCase()
                self._notifyStage()
            if onProgress:
                try:
                    onProgress(index, total)
                except Exception:  # noqa: BLE001
                    pass
        return results

    def _loadInput(self, path):
        """Load a batch input: a volume file, or a folder holding a DICOM series."""
        if os.path.isdir(path):
            return self._loadDicomSeries(path)
        node = slicer.util.loadVolume(path)
        if node is None:
            raise RuntimeError("Could not load volume: " + path)
        return node

    def _loadDicomSeries(self, folder):
        """Import and load a single DICOM series folder, returning its scalar volume node.

        One series per patient folder is required. If the folder yields more than one series the
        method raises, rather than guessing which one to segment; split the series into separate
        folders. Detection and import both recurse, so nested Patient/Study/Series layouts work.
        """
        from DICOMLib import DICOMUtils
        loadedIds = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(folder, db)
            for patientUID in db.patients():
                loadedIds.extend(DICOMUtils.loadPatientByUID(patientUID))
        volumes = [node for node in (slicer.mrmlScene.GetNodeByID(i) for i in loadedIds)
                   if node is not None and node.IsA("vtkMRMLScalarVolumeNode")]
        if not volumes:
            raise RuntimeError("No volume found in DICOM folder: " + folder)
        if len(volumes) > 1:
            for node in volumes:
                slicer.mrmlScene.RemoveNode(node)
            raise RuntimeError(
                "Found %d DICOM series in '%s'; one series per patient folder is required. "
                "Split the series into separate folders and re-run." % (len(volumes), folder))
        return volumes[0]

    def _writeAirwayVolume(self, segmentationNode, subfolder, sourcePath):
        """Write the airway volume (mL) computed during post-processing to a text file."""
        try:
            mm3 = segmentationNode.GetAttribute("AUAE.airwayVolumeMm3")
            if not mm3:
                return
            mm3 = float(mm3)
            with open(os.path.join(subfolder, "airway_volume.txt"), "w", encoding="utf-8") as handle:
                handle.write("case\t%s\n" % os.path.basename(sourcePath))
                handle.write("airway_volume_mL\t%.2f\n" % (mm3 / 1000.0))
                handle.write("airway_volume_mm3\t%.0f\n" % mm3)
        except Exception as exc:  # noqa: BLE001
            self._progress("Could not write airway volume: " + str(exc))
