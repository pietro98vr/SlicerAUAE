"""Batch processing driver.

Added by this project. Runs the upstream nnU-Net segmentation over a list of volumes defined in a
JSON template, applies the project post-processing (island cleanup + optional airway
extension), and writes one output subfolder per volume containing the segmentation and the
requested mesh/labelmap files, reusing the upstream export.

The default template lives in Resources/batch_template.json and documents every field.
"""

import json
import os
from pathlib import Path

import slicer

from . import AirwayExtension


# Volume file types accepted in a folder-in / folder-out batch (the classic nnU-Net style).
VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")


def defaultTemplatePath():
    """Path to the embedded batch template shipped in Resources."""
    return Path(__file__).parent.joinpath("..", "Resources", "batch_template.json").resolve()


def listVolumesInFolder(folder):
    """Return the sorted volume files directly inside a folder (folder-in/folder-out batch)."""
    folder = str(folder)
    if not os.path.isdir(folder):
        return []
    found = []
    for name in sorted(os.listdir(folder)):
        low = name.lower()
        if os.path.isfile(os.path.join(folder, name)) and any(low.endswith(e) for e in VOLUME_EXTENSIONS):
            found.append(os.path.join(folder, name))
    return found


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

    volumes = [str(v).strip() for v in raw.get("volumes", []) if str(v).strip()]
    inputFolder = str(raw.get("input_folder", "") or "").strip()
    if inputFolder and not volumes:
        volumes = listVolumesInFolder(inputFolder)
    outputRoot = str(raw.get("output_root", "") or "").strip()
    if not outputRoot and inputFolder:
        outputRoot = os.path.join(inputFolder, "AUAE_output")
    extension = raw.get("extension", {}) or {}
    # Island options are mutually exclusive, mirroring the UI: keep-largest wins if both set.
    keepLargest = bool(raw.get("keep_largest_island", False))
    removeSmall = bool(raw.get("remove_small_islands", True)) and not keepLargest
    return {
        "mode": str(raw.get("mode", "segment") or "segment").strip().lower(),
        "output_root": outputRoot,
        "subfolder_from": str(raw.get("subfolder_from", "filename") or "filename"),
        "export_formats": [str(f).strip().upper() for f in raw.get("export_formats", ["STL", "NIFTI"])],
        "export_targets": [str(t).strip().lower() for t in raw.get("export_targets", ["airway", "external"])],
        "postprocess": {
            "removeSmallIslands": removeSmall,
            "keepLargestIsland": keepLargest,
            "minIslandMm3": float(raw.get("min_island_mm3", AirwayExtension.DEFAULT_MIN_ISLAND_MM3)),
            "includeInternalAir": bool(raw.get("include_internal_air", False)),
            "minInternalAirMm3": float(raw.get("min_internal_air_mm3", AirwayExtension.DEFAULT_MIN_INTERNAL_AIR_MM3)),
            "segmentExternalAir": bool(raw.get("segment_external_air", False)),
            "mergeExternalAir": bool(raw.get("merge_external_air", False)),
            "smoothingFactor": float(raw.get("smoothing_factor", 0.0)),
            "extend": bool(extension.get("enabled", False)),
            "direction": str(extension.get("direction", AirwayExtension.EXTENSION_DIRECTIONS[0])),
            "lengthMm": float(extension.get("length_mm", 100.0)),
        },
        "volumes": volumes,
    }


class BatchProcessor:
    """Sequentially segment and export every volume listed in a batch template."""

    def __init__(self, nnUnetFolder, exportFn, progressCallback=None):
        """
        :param nnUnetFolder: folder holding the downloaded nnU-Net weights.
        :param exportFn: callable(segmentationNode, folderPath, ExportFormat) - the upstream export.
        :param progressCallback: callable(str) for progress lines.
        """
        self._nnUnetFolder = nnUnetFolder
        self._exportFn = exportFn
        self._progress = progressCallback or (lambda msg: None)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self, config, exportFormatEnum, deviceKwargs=None):
        """Process every volume in config. Returns a per-volume result summary list.

        :param exportFormatEnum: the ExportFormat flag value selected for all volumes.
        :param deviceKwargs: optional Parameter device kwargs (e.g. {"device": "cuda"}).
        """
        # 'extend' mode reruns only the airway extension on existing segmentation files, so
        # segmentation and extension can be two separate batch passes (no model / GPU needed).
        if config.get("mode", "segment") == "extend":
            return self._runExtend(config, exportFormatEnum)

        from SlicerNNUNetLib import Parameter, SegmentationLogic
        deviceKwargs = deviceKwargs or {}

        def makeParameter():
            kwargs = dict(folds="0", modelPath=self._nnUnetFolder)
            kwargs.update(deviceKwargs)
            try:
                return Parameter(**kwargs)
            except TypeError:
                return Parameter(folds="0", modelPath=self._nnUnetFolder)

        outputRoot = config["output_root"]
        if not outputRoot:
            raise ValueError("Batch 'output_root' is not set.")
        os.makedirs(outputRoot, exist_ok=True)

        volumes = config["volumes"]
        if not volumes:
            raise ValueError("Batch template lists no volumes.")

        results = []
        logic = SegmentationLogic()
        logic.progressInfo.connect(self._progress)

        for index, volumePath in enumerate(volumes, start=1):
            if self._cancelled:
                self._progress("Batch cancelled by user.")
                break
            entry = {"volume": volumePath, "status": "pending", "output": None, "error": None}
            self._progress("[%d/%d] %s" % (index, len(volumes), volumePath))
            volumeNode = None
            segmentationNode = None
            try:
                if not os.path.isfile(volumePath):
                    raise FileNotFoundError("Input volume not found: " + volumePath)
                volumeNode = slicer.util.loadVolume(volumePath)

                logic.setParameter(makeParameter())
                logic.startSegmentation(volumeNode)
                logic.waitForSegmentationFinished()
                slicer.app.processEvents()

                rawSegmentation = logic.loadSegmentation()
                rawSegmentation.SetName(Path(volumePath).stem + "_Segmentation")
                segmentationNode = AirwayExtension.postprocessSegmentation(
                    rawSegmentation, volumeNode, config["postprocess"], self._progress
                )
                slicer.mrmlScene.RemoveNode(rawSegmentation)

                subfolder = os.path.join(outputRoot, self._subfolderName(volumePath))
                os.makedirs(subfolder, exist_ok=True)
                self._exportFn(segmentationNode, subfolder, exportFormatEnum, config.get("export_targets") or ["merged"])

                entry["status"] = "ok"
                entry["output"] = subfolder
                self._progress("[%d/%d] done -> %s" % (index, len(volumes), subfolder))
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "error"
                entry["error"] = str(exc)
                self._progress("[%d/%d] ERROR: %s" % (index, len(volumes), exc))
            finally:
                # Free the scene between cases so long batches do not accumulate nodes.
                for node in (segmentationNode, volumeNode):
                    if node is not None:
                        try:
                            slicer.mrmlScene.RemoveNode(node)
                        except Exception:  # noqa: BLE001
                            pass
                slicer.app.processEvents()
            results.append(entry)

        return results

    def _runExtend(self, config, exportFormatEnum):
        """Extend a list of already-existing airway segmentations (batch 'extend' mode).

        Each item in 'volumes' is a saved airway segmentation labelmap (NIFTI/NRRD). It is
        loaded, extended with the template direction and length, and exported into its own
        subfolder. No nnU-Net inference runs, so this needs neither the model nor a GPU. This
        is the batch counterpart of the interactive two-step workflow: segment first, refine,
        then extend.
        """
        outputRoot = config["output_root"]
        if not outputRoot:
            raise ValueError("Batch 'output_root' is not set.")
        os.makedirs(outputRoot, exist_ok=True)

        segmentations = config["volumes"]
        if not segmentations:
            raise ValueError("Batch template lists no segmentations to extend.")

        results = []
        for index, path in enumerate(segmentations, start=1):
            if self._cancelled:
                self._progress("Batch cancelled by user.")
                break
            entry = {"volume": path, "status": "pending", "output": None, "error": None}
            self._progress("[%d/%d] extend %s" % (index, len(segmentations), path))
            labelmapNode = segmentationNode = extendedNode = None
            try:
                if not os.path.isfile(path):
                    raise FileNotFoundError("Segmentation not found: " + path)
                labelmapNode = slicer.util.loadLabelVolume(path)
                if labelmapNode is None:
                    raise RuntimeError("Could not load segmentation labelmap: " + path)

                segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode", Path(path).stem + "_Segmentation"
                )
                segmentationNode.CreateDefaultDisplayNodes()
                segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(labelmapNode)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)

                extendedNode = AirwayExtension.extendSegmentation(
                    segmentationNode, labelmapNode, config["postprocess"], self._progress
                )
                subfolder = os.path.join(outputRoot, self._subfolderName(path))
                os.makedirs(subfolder, exist_ok=True)
                self._exportFn(extendedNode, subfolder, exportFormatEnum, config.get("export_targets") or ["merged"])

                entry["status"] = "ok"
                entry["output"] = subfolder
                self._progress("[%d/%d] done -> %s" % (index, len(segmentations), subfolder))
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "error"
                entry["error"] = str(exc)
                self._progress("[%d/%d] ERROR: %s" % (index, len(segmentations), exc))
            finally:
                for node in (extendedNode, segmentationNode, labelmapNode):
                    if node is not None:
                        try:
                            slicer.mrmlScene.RemoveNode(node)
                        except Exception:  # noqa: BLE001
                            pass
                slicer.app.processEvents()
            results.append(entry)

        return results

    @staticmethod
    def _subfolderName(volumePath):
        stem = Path(volumePath).name
        for suffix in (".nii.gz", ".nrrd", ".nii", ".mha", ".mhd"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        return stem or "case"
