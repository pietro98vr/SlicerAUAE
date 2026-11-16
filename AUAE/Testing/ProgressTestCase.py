"""Tests for AUAELib.Progress: the stage and fraction read from the lines a run logs."""

import importlib.util
import os

# Progress.py has no Slicer dependency; load it by path so the test also runs outside Slicer.
_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AUAELib", "Progress.py")
_SPEC = importlib.util.spec_from_file_location("auae_progress_under_test", _PATH)
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)

# The lines one segmentation logs, in order, as nnU-Net and AUAE print them.
RUN_LINES = (
    ("[1/2] D:/data/P001/pre", "Loading", 0.0),
    ("Transferring volume to nnUNet in C:/Temp/Slicer-abc", "Preparing the volume", 0.03),
    ("Starting nnUNet with the following parameters:", "Preprocessing", 0.11),
    ("nnUNet preprocessing...", "Preprocessing", 0.11),
    ("Predicting volume:", "Inference", 0.15),
    ("  0%|          | 0/20 [00:00<?, ?it/s]", "Inference", 0.15),
    (" 50%|#####     | 10/20 [00:12<00:12,  1.25s/it]", "Inference", 0.48),
    ("100%|##########| 20/20 [00:25<00:00,  1.25s/it]", "Inference", 0.81),
    ("sending off prediction to background worker for resampling and export", "Reading the result", 0.81),
    ("Remove small islands: dropped 38 component(s) < 200 voxels.", "Post-processing", 0.87),
    ("Airway volume: 141.78 mL (141776 mm^3).", "Post-processing", 0.93),
    ("Extension: 57 slice(s) added beyond the volume (side=min).", "Extension", 0.94),
    ("Exporting to D:/out/P001/pre", "Export", 0.95),
)


def test_stages_follow_the_log_and_never_go_back():
    case = P.CaseProgress()
    previous = 0.0
    for line, stage, fraction in RUN_LINES:
        case.update(line)
        assert case.stage == stage, line
        assert abs(case.fraction - fraction) < 1e-6, line
        assert case.fraction >= previous
        previous = case.fraction
    # A late progress bar or an early marker cannot pull the fraction back.
    assert not case.update(" 10%|#         | 2/20 [00:02<00:22,  1.25s/it]")
    assert not case.update("Transferring volume to nnUNet")
    assert (case.stage, case.fraction) == ("Export", 0.95)
    case.finish()
    assert (case.stage, case.fraction) == ("Done", 1.0)


def test_carriage_returns_chunks_and_noise():
    case = P.CaseProgress()
    case.update("Predicting volume:\r  5%|5   | 1/20\r 30%|###   | 6/20 [00:07<00:17]\n")
    assert case.stage == "Inference" and abs(case.fraction - (0.15 + 0.66 * 0.30)) < 1e-6
    assert not case.update("")                       # blank chunk
    assert not case.update(None)
    assert not case.update("There are 1 cases in the source folder")
    # A percent bar before the volume was handed to nnU-Net (a download, say) is not inference.
    fresh = P.CaseProgress()
    assert not fresh.update(" 40%|####      | 40/100 [00:01<00:02]")


def test_run_over_several_cases():
    run = P.RunProgress(4)
    assert run.snapshot()["overall_fraction"] == 0.0 and P.describe(run.snapshot()) == "Starting"
    run.startCase(1, "P001/pre")
    run.update("Predicting volume:")
    run.update(" 50%|#####     | 10/20")
    snapshot = run.snapshot()
    assert snapshot["case"] == 1 and snapshot["done"] == 0 and snapshot["stage"] == "Inference"
    assert abs(snapshot["overall_fraction"] - 0.48 / 4) < 1e-3
    run.finishCase()
    assert run.snapshot()["done"] == 1 and abs(run.snapshot()["overall_fraction"] - 0.25) < 1e-6
    run.startCase(2, "P001/post")
    assert abs(run.snapshot()["overall_fraction"] - 0.25) < 1e-6     # a new case starts from its own zero
    run.update("Airway volume: 100.00 mL")
    assert abs(run.snapshot()["overall_fraction"] - (1 + 0.93) / 4) < 1e-3
    for index in (2, 3, 4):
        run.startCase(index, "c%d" % index)
        run.finishCase()
    final = run.snapshot()
    assert final["done"] == 4 and final["overall_fraction"] == 1.0
    assert P.describe(final) == "Finished 4 of 4"


def test_extend_mode_has_its_own_stages():
    run = P.RunProgress(1, "extend")
    run.startCase(1, "seg")
    run.update(" 50%|#####     | 10/20")            # no model run in this mode: ignored
    assert run.snapshot()["case_fraction"] == 0.0
    run.update("Extension: 10 slice(s) filled inside the acquired FOV.")
    assert run.snapshot()["stage"] == "Extension" and run.snapshot()["case_fraction"] == 0.5
    run.update("Exporting to D:/out/seg")
    assert run.snapshot()["stage"] == "Export"


def test_text_for_people():
    snapshot = {"done": 1, "total": 5, "case": 2, "case_name": "P001/post", "stage": "Inference",
                "case_fraction": 0.62, "overall_fraction": 0.324, "elapsed_seconds": 200, "eta_seconds": 417}
    assert P.describe(snapshot) == "Case 2 of 5 (P001/post): Inference 62%, about 7 min left"
    bar = P.formatBar(snapshot, width=10)
    assert bar.startswith("[###-------] 32%  Case 2 of 5")
    single = dict(snapshot, total=1, done=0, case=1, eta_seconds=None)
    assert P.describe(single) == "Inference 62%"
    assert P.formatDuration(40) == "40 s" and P.formatDuration(4800) == "1 h 20 min"
    assert P.formatBar(None).startswith("[" + "-" * 24 + "] 0%")
