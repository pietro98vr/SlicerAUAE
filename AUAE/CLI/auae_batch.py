"""Headless command-line batch runner for AUAE.

Run it with Slicer's headless Python. Everything after the ``--`` is passed to this script:

    Slicer --no-main-window --python-script auae_batch.py -- \
        --input /data/cbct --output /data/out --formats STL NRRD \
        --targets airway external --device auto --external

The exit code is 0 when every case succeeded, 1 when at least one case failed, and 2 on a
fatal error (bad arguments, missing NNUNet extension, no inputs, or missing model weights).

The batch logic lives in AUAELib.BatchProcessor and needs no graphical interface, so this
script only builds a config from the arguments and drives it. The NNUNet Slicer extension must
be installed; ``--install-dependencies`` installs the Python packages, and the model weights
are downloaded on demand when they are missing (bundled model only).

``--log``, ``--progress`` and ``--summary`` write the log lines, the live progress and the
per-case results to files, which is how the MCP server (MCP/auae_mcp.py) follows a run without
reading the console.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# This file lives in AUAE/CLI/; make the AUAELib package importable.
_AUAE_DIR = str(Path(__file__).resolve().parent.parent)
if _AUAE_DIR not in sys.path:
    sys.path.insert(0, _AUAE_DIR)

_logPath = None


def _log(message):
    """Print a progress line and, with --log, append it to the log file right away.

    nnU-Net redraws its progress bar many times a second; those lines and the blank ones are
    printed but kept out of the file, which stays short enough to read.
    """
    message = str(message)
    print(message)
    text = message.strip()
    if _logPath and text and "it/s]" not in text and "s/it]" not in text:
        try:
            with open(_logPath, "a", encoding="utf-8") as handle:
                handle.write("%s %s\n" % (time.strftime("%H:%M:%S"), message))
        except Exception:  # noqa: BLE001
            pass


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="auae_batch", description="Headless AUAE batch runner.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="Input folder of volumes and/or DICOM series, or one volume file.")
    source.add_argument("--template", help="JSON batch template (advanced).")
    parser.add_argument("--output", default="", help="Output folder (default: AUAE_output beside the input).")
    parser.add_argument("--formats", nargs="+", default=["STL", "NIFTI"],
                        choices=["STL", "OBJ", "NIFTI", "NRRD"], help="Export formats.")
    parser.add_argument("--targets", nargs="+", default=["airway", "external"],
                        choices=["airway", "external", "merged"], help="What to export.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Inference device. 'auto' takes the GPU when PyTorch can use it, else the CPU.")
    parser.add_argument("--install-dependencies", action="store_true",
                        help="Install or repair PyTorch (the build that suits this computer), nnU-Net and "
                             "the extra packages. Can be used alone or before a run.")
    parser.add_argument("--check-dependencies", action="store_true",
                        help="Print the dependency report. Alone, the exit code is 0 when ready, 1 when not.")
    parser.add_argument("--mode", default=None, choices=["segment", "extend"],
                        help="'segment' runs the model; 'extend' extends existing segmentation labelmaps. "
                             "Default: 'segment', or the mode set in the template.")
    parser.add_argument("--sinus", action=argparse.BooleanOptionalAction, default=True,
                        help="Attempt frontal sinus recovery (default on; use --no-sinus to disable).")
    parser.add_argument("--external", action="store_true", help="Also segment the external face air.")
    parser.add_argument("--merge-external", action="store_true", help="Merge external air into the airway.")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest island.")
    parser.add_argument("--no-remove-islands", action="store_true", help="Do not remove small islands.")
    parser.add_argument("--smoothing", type=float, default=0.0, help="Surface smoothing, 0 to 1.")
    parser.add_argument("--min-island", type=float, default=None, help="Small-island threshold (mm^3).")
    parser.add_argument("--min-internal", type=float, default=None, help="Internal-cavity threshold (mm^3).")
    parser.add_argument("--extend", type=float, default=0.0, help="Inferior extension length in mm (0 = off).")
    parser.add_argument("--log", default="", help="Also append the progress lines to this file.")
    parser.add_argument("--summary", default="", help="Write the per-case results to this JSON file.")
    parser.add_argument("--progress", default="",
                        help="Keep this JSON file up to date with the progress of the run: cases done, "
                             "current case and stage, fractions, estimated time left.")
    args = parser.parse_args(argv)
    if not (args.input or args.template or args.install_dependencies or args.check_dependencies):
        parser.error("give --input or --template (or --check-dependencies / --install-dependencies alone)")
    return args


def _postprocess_from_args(args, AirwayExtension):
    options = AirwayExtension.defaultOptions()
    options["includeInternalAir"] = bool(args.sinus)
    options["segmentExternalAir"] = bool(args.external)
    options["mergeExternalAir"] = bool(args.merge_external)
    options["keepLargestIsland"] = bool(args.keep_largest)
    options["removeSmallIslands"] = (not args.no_remove_islands) and (not args.keep_largest)
    options["smoothingFactor"] = float(args.smoothing)
    if args.min_island is not None:
        options["minIslandMm3"] = float(args.min_island)
    if args.min_internal is not None:
        options["minInternalAirMm3"] = float(args.min_internal)
    if args.extend and args.extend > 0:
        options["extend"] = True
        options["lengthMm"] = float(args.extend)
    return options


def _airwayVolumeMl(outputFolder):
    """Read back the airway volume the batch wrote next to the exported files, if any."""
    try:
        with open(os.path.join(outputFolder, "airway_volume.txt"), "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition("\t")
                if key == "airway_volume_mL":
                    return float(value)
    except Exception:  # noqa: BLE001
        pass
    return None


def _writeSummary(path, state, **fields):
    """Write the machine-readable result of the run. 'state' is 'finished' or 'error'."""
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"state": state, **fields}, handle, indent=2)
    except Exception as exc:  # noqa: BLE001
        _log("Could not write the summary: %s" % exc)


def _fatal(args, message):
    _log("ERROR: " + message)
    _writeSummary(args.summary, "error", error=message)
    return 2


def main(argv):
    global _logPath
    args = _parse_args(argv)
    _logPath = args.log or None

    from AUAELib import BatchProcessor as BP
    from AUAELib import Dependencies
    from AUAELib import AirwayExtension
    from AUAELib.SegmentationWidget import SegmentationWidget
    from AUAELib.PythonDependencyChecker import PythonDependencyChecker

    if not Dependencies.isNNUNetExtensionAvailable():
        return _fatal(args, "the NNUNet Slicer extension is required. Install it and retry.")

    if args.install_dependencies:
        ok, needsRestart = Dependencies.ensure(_log, askConfirmation=False)
        if needsRestart:
            return _fatal(args, "the installation needs a restart. Run the same command again to continue.")
        if not ok:
            return _fatal(args, "the dependencies could not be installed.")
    ready = None
    if args.check_dependencies:
        ready = Dependencies.report(_log)
    if not (args.input or args.template):
        _writeSummary(args.summary, "finished", ready=ready, hardware=Dependencies.summary())
        return 0 if ready in (None, True) else 1

    postprocess = _postprocess_from_args(args, AirwayExtension)

    if args.template:
        if not os.path.isfile(args.template):
            return _fatal(args, "template not found: " + args.template)
        config = BP.loadConfig(args.template)
        formatFlag = SegmentationWidget._exportFormatFromNames(config.get("export_formats") or args.formats)
    else:
        if not os.path.exists(args.input):
            return _fatal(args, "input not found: " + args.input)
        config = BP.folderConfig(args.input, args.output, args.formats, postprocess, args.targets)
        if os.path.isfile(args.input):  # a single volume instead of a folder
            config["volumes"] = [args.input]
            config["output_root"] = args.output or os.path.join(os.path.dirname(args.input), "AUAE_output")
        formatFlag = SegmentationWidget._exportFormatFromNames(args.formats)
    if args.mode:  # an explicit --mode wins over the template; otherwise keep the config's mode
        config["mode"] = args.mode

    if not config.get("volumes"):
        return _fatal(args, "no inputs found to process.")

    # Model weights (bundled model, segment mode). Download only if missing; no dialogs headless.
    if config["mode"] == "segment":
        checker = PythonDependencyChecker()
        if checker.areWeightsMissing():
            _log("Model weights missing; downloading from the upstream release...")
            if not checker.downloadWeights(_log):
                return _fatal(args, "could not download the model weights.")

    deviceKwargs = Dependencies.parameterDeviceKwargs(args.device)
    _log("Inference device: %s" % deviceKwargs.get("device"))
    processor = BP.BatchProcessor(SegmentationWidget.nnUnetFolder(), SegmentationWidget.exportSegmentation, _log)

    def progress(done, total):
        _log("progress: %d/%d" % (done, total))

    lastWrite = {"time": 0.0, "key": None}

    def stage(snapshot):
        """Write the progress file, at most twice a second unless the case or stage changed."""
        if not args.progress:
            return
        key = (snapshot["done"], snapshot["case"], snapshot["stage"])
        now = time.time()
        if key == lastWrite["key"] and now - lastWrite["time"] < 0.5:
            return
        lastWrite["time"], lastWrite["key"] = now, key
        try:
            temporary = args.progress + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle)
            os.replace(temporary, args.progress)  # the reader never sees a half-written file
        except Exception:  # noqa: BLE001
            pass

    _log("progress: 0/%d" % len(config["volumes"]))
    results = processor.run(config, formatFlag, deviceKwargs=deviceKwargs, onProgress=progress, onStage=stage)
    ok = sum(1 for r in results if r["status"] == "ok")
    _log("Batch complete: %d/%d succeeded -> %s" % (ok, len(results), config["output_root"]))
    cases = []
    for entry in results:
        if entry["status"] != "ok":
            _log("  FAILED %s: %s" % (entry["volume"], entry["error"]))
        cases.append({
            "input": entry["volume"], "status": entry["status"], "output": entry["output"],
            "error": entry["error"],
            "airway_volume_mL": _airwayVolumeMl(entry["output"]) if entry["output"] else None,
        })
    _writeSummary(args.summary, "finished", succeeded=ok, total=len(results),
                  output_root=config["output_root"], device=deviceKwargs.get("device"), cases=cases)
    return 0 if (results and ok == len(results)) else 1


if __name__ == "__main__":
    exitCode = 2
    try:
        exitCode = main(sys.argv[1:])
    except SystemExit as exc:  # argparse failures land here
        exitCode = int(exc.code) if exc.code is not None else 2
    except Exception as exc:  # noqa: BLE001
        _log("FATAL: " + str(exc))
        exitCode = 2
    try:
        import slicer
        slicer.util.exit(exitCode)
    except Exception:  # noqa: BLE001
        sys.exit(exitCode)
