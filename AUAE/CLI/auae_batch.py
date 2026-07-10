"""Headless command-line batch runner for AUAE.

Run it with Slicer's headless Python. Everything after the ``--`` is passed to this script:

    Slicer --no-main-window --python-script auae_batch.py -- \
        --input /data/cbct --output /data/out --formats STL NRRD \
        --targets airway external --device auto --external

The exit code is 0 when every case succeeded, 1 when at least one case failed, and 2 on a
fatal error (bad arguments, missing NNUNet extension, no inputs, or missing model weights).

The batch logic lives in AUAELib.BatchProcessor and needs no graphical interface, so this
script only builds a config from the arguments and drives it. The NNUNet Slicer extension must
be installed and its Python dependencies present; the model weights are downloaded on demand
when they are missing (bundled model only).
"""

import argparse
import os
import sys
from pathlib import Path

# This file lives in AUAE/CLI/; make the AUAELib package importable.
_AUAE_DIR = str(Path(__file__).resolve().parent.parent)
if _AUAE_DIR not in sys.path:
    sys.path.insert(0, _AUAE_DIR)


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="auae_batch", description="Headless AUAE batch runner.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Input folder of volumes and/or DICOM series.")
    source.add_argument("--template", help="JSON batch template (advanced).")
    parser.add_argument("--output", default="", help="Output folder (default: AUAE_output beside the input).")
    parser.add_argument("--formats", nargs="+", default=["STL", "NIFTI"],
                        choices=["STL", "OBJ", "NIFTI", "NRRD"], help="Export formats.")
    parser.add_argument("--targets", nargs="+", default=["airway", "external"],
                        choices=["airway", "external", "merged"], help="What to export.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--mode", default="segment", choices=["segment", "extend"],
                        help="'segment' runs the model; 'extend' extends existing segmentation labelmaps.")
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
    return parser.parse_args(argv)


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


def main(argv):
    args = _parse_args(argv)

    from AUAELib import BatchProcessor as BP
    from AUAELib import Dependencies
    from AUAELib import AirwayExtension
    from AUAELib.SegmentationWidget import SegmentationWidget
    from AUAELib.PythonDependencyChecker import PythonDependencyChecker

    if not Dependencies.isNNUNetExtensionAvailable():
        print("ERROR: the NNUNet Slicer extension is required. Install it and retry.")
        return 2

    postprocess = _postprocess_from_args(args, AirwayExtension)

    if args.template:
        if not os.path.isfile(args.template):
            print("ERROR: template not found: " + args.template)
            return 2
        config = BP.loadConfig(args.template)
        formatFlag = SegmentationWidget._exportFormatFromNames(config.get("export_formats") or args.formats)
    else:
        if not os.path.isdir(args.input):
            print("ERROR: input folder not found: " + args.input)
            return 2
        config = BP.folderConfig(args.input, args.output, args.formats, postprocess, args.targets)
        formatFlag = SegmentationWidget._exportFormatFromNames(args.formats)
    config["mode"] = args.mode

    if not config.get("volumes"):
        print("ERROR: no inputs found to process.")
        return 2

    # Model weights (bundled model, segment mode). Download only if missing; no dialogs headless.
    if config["mode"] == "segment":
        checker = PythonDependencyChecker()
        if checker.areWeightsMissing():
            print("Model weights missing; downloading from the upstream release...")
            if not checker.downloadWeights(print):
                print("ERROR: could not download the model weights.")
                return 2

    deviceKwargs = Dependencies.parameterDeviceKwargs(args.device)
    processor = BP.BatchProcessor(
        SegmentationWidget.nnUnetFolder(), SegmentationWidget.exportSegmentation, print
    )

    def progress(done, total):
        print("progress: %d/%d" % (done, total))

    results = processor.run(config, formatFlag, deviceKwargs=deviceKwargs, onProgress=progress)
    ok = sum(1 for r in results if r["status"] == "ok")
    print("Batch complete: %d/%d succeeded -> %s" % (ok, len(results), config["output_root"]))
    for entry in results:
        if entry["status"] != "ok":
            print("  FAILED %s: %s" % (entry["volume"], entry["error"]))
    return 0 if (results and ok == len(results)) else 1


if __name__ == "__main__":
    exitCode = 2
    try:
        exitCode = main(sys.argv[1:])
    except SystemExit as exc:  # argparse failures land here
        exitCode = int(exc.code) if exc.code is not None else 2
    except Exception as exc:  # noqa: BLE001
        print("FATAL: " + str(exc))
        exitCode = 2
    try:
        import slicer
        slicer.util.exit(exitCode)
    except Exception:  # noqa: BLE001
        sys.exit(exitCode)
