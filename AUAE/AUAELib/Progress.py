"""Progress of a run, worked out from the lines the run already logs.

Added by this project. nnU-Net, the post-processing and the batch runner all report what they
are doing as text lines. This module reads those lines and turns them into numbers: the stage
of the current case, how far that case is, how far the whole run is, and an estimate of the
time left. The panel shows them as a progress bar, the command-line runner writes them to a
file, and the MCP server hands them to the AI, so the three always agree.

Nothing here touches Slicer or Qt, so it can be tested on its own.
"""

import re
import time

# Share of a case reached when a marker line is seen. The shares come from timing a real run
# (a CBCT of 816 x 816 x 632 voxels on an RTX 3070): loading and handing the volume to nnU-Net
# take about a tenth, the model about two thirds, the rest is post-processing and export. They
# only shape how the bar moves; a slower computer stretches the stages roughly alike.
INFERENCE_START, INFERENCE_END = 0.15, 0.81
SEGMENT_STAGES = (
    (re.compile(r"Transferring volume to nnUNet"), "Preparing the volume", 0.03),
    (re.compile(r"Starting nnUNet|nnUNet preprocessing"), "Preprocessing", 0.11),
    (re.compile(r"Predicting volume|perform_everything_on_device"), "Inference", INFERENCE_START),
    (re.compile(r"sending off prediction|done with volume|Loading inference results"), "Reading the result", INFERENCE_END),
    (re.compile(r"Remove small islands|Keep largest island|Air envelope|Internal air cavities|"
                r"External face air|Air handling skipped"), "Post-processing", 0.87),
    (re.compile(r"Airway volume:"), "Post-processing", 0.93),
    (re.compile(r"Extension:"), "Extension", 0.94),
    (re.compile(r"^Exporting"), "Export", 0.95),
)
# 'extend' mode has no model run: loading, the extension, the export.
EXTEND_STAGES = (
    (re.compile(r"Extension:"), "Extension", 0.5),
    (re.compile(r"^Exporting"), "Export", 0.8),
)
# nnU-Net draws its sliding-window progress as ' 45%|####      | 9/20 [00:12<00:14, 1.3s/it]'.
_PERCENT_BAR = re.compile(r"(\d{1,3})%\|")


class CaseProgress:
    """Stage and fraction (0 to 1) of one case. The fraction never goes backwards."""

    def __init__(self, mode="segment"):
        self._stages = EXTEND_STAGES if mode == "extend" else SEGMENT_STAGES
        self.stage = "Loading"
        self.fraction = 0.0

    def update(self, text):
        """Read log text (one or more lines). Returns True when stage or fraction changed."""
        before = (self.stage, self.fraction)
        for line in re.split(r"[\r\n]+", str(text or "")):
            line = line.strip()
            if not line:
                continue
            bar = _PERCENT_BAR.search(line)
            if bar and self._stages is SEGMENT_STAGES and 0.03 <= self.fraction <= INFERENCE_END:
                percent = min(100, int(bar.group(1)))
                self._reach("Inference", INFERENCE_START + (INFERENCE_END - INFERENCE_START) * percent / 100.0)
                continue
            for pattern, stage, fraction in self._stages:
                if pattern.search(line):
                    self._reach(stage, fraction)
                    break
        return (self.stage, self.fraction) != before

    def _reach(self, stage, fraction):
        if fraction >= self.fraction:
            self.stage, self.fraction = stage, fraction

    def finish(self):
        self.stage, self.fraction = "Done", 1.0


class RunProgress:
    """Progress of a run of ``total`` cases: the current case plus the cases already done."""

    def __init__(self, total=1, mode="segment"):
        self.total = max(1, int(total))
        self.mode = mode
        self.done = 0
        self.index = 0          # 1-based number of the current case, 0 before the first
        self.caseName = ""
        self.case = CaseProgress(mode)
        self._started = time.time()

    def startCase(self, index, name=""):
        self.index = int(index)
        self.caseName = str(name or "")
        self.case = CaseProgress(self.mode)

    def update(self, text):
        """Feed log text to the current case. Returns True when something changed."""
        return self.index > 0 and self.case.update(text)

    def finishCase(self):
        self.case.finish()
        self.done = max(self.done, self.index)

    def overallFraction(self):
        running = self.case.fraction if self.index > self.done else 0.0
        return min(1.0, (self.done + running) / float(self.total))

    def snapshot(self):
        """The numbers to show: cases done, current case and stage, fractions, time left."""
        fraction = self.overallFraction()
        elapsed = time.time() - self._started
        # No estimate until enough of the run is behind us for the rate to mean something.
        measurable = fraction >= 0.05 and elapsed >= 10
        remaining = elapsed * (1.0 - fraction) / fraction if measurable else None
        return {
            "done": self.done,
            "total": self.total,
            "case": self.index,
            "case_name": self.caseName,
            "stage": self.case.stage,
            "case_fraction": round(self.case.fraction, 3),
            "overall_fraction": round(fraction, 3),
            "elapsed_seconds": int(elapsed),
            "eta_seconds": int(remaining) if remaining is not None else None,
        }


def formatDuration(seconds):
    """'40 s', '6 min', '1 h 20 min'."""
    seconds = int(seconds)
    if seconds < 60:
        return "%d s" % seconds
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return "%d min" % minutes
    return "%d h %02d min" % (minutes // 60, minutes % 60)


def describe(snapshot):
    """One line of text for a snapshot: 'Case 2 of 5 (P001/post): Inference 62%, about 6 min left'."""
    if not snapshot or not snapshot.get("case"):
        return "Starting"
    if snapshot["done"] >= snapshot["total"]:
        return "Finished %d of %d" % (snapshot["done"], snapshot["total"])
    text = "%s %d%%" % (snapshot["stage"], round(snapshot["case_fraction"] * 100))
    if snapshot["total"] > 1:
        name = " (%s)" % snapshot["case_name"] if snapshot.get("case_name") else ""
        text = "Case %d of %d%s: %s" % (snapshot["case"], snapshot["total"], name, text)
    if snapshot.get("eta_seconds") is not None:
        text += ", about %s left" % formatDuration(snapshot["eta_seconds"])
    return text


def formatBar(snapshot, width=24):
    """A text progress bar for places without widgets: '[#########-------] 38%  Case 2 of 5 ...'."""
    fraction = float((snapshot or {}).get("overall_fraction") or 0.0)
    filled = int(round(fraction * width))
    return "[%s%s] %d%%  %s" % ("#" * filled, "-" * (width - filled), round(fraction * 100), describe(snapshot))
