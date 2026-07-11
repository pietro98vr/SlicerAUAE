import slicer
from slicer.ScriptedLoadableModule import *

from AUAELib import SegmentationWidget


class AUAE(ScriptedLoadableModule):
    def __init__(self, parent):
        from slicer.i18n import tr, translate
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = tr("Automatic Upper Airways Extension (AUAE)")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        # Upstream authors are kept first; the maintainers are appended.
        self.parent.contributors = [
            "Alejandro Matos Camarillo (uAlberta)",
            "Silvia Capenakas (uAlberta)",
            "Manuel Lagravere (uAlberta)",
            "Pietro Montagna, DDS MSc PhD Student (University of Verona)",
            "Fabio Lonardi, MD OMFS PhD Student (University of Verona)",
        ]

        self.parent.helpText = tr(
            "Fully automatic AI segmentation of the upper airway (nasal cavity, nasopharynx, "
            "oropharynx) on CT and CBCT, based on the UpperAirwaySegmentator nnU-Net model.\n\n"
            "Based on capenaka/SlicerUpperAirwaySegmentator. AUAE keeps the upstream "
            "segmentation core and the multi-format export, and works in three steps: segment, "
            "refine in an embedded Segment Editor, then extend. It cleans the mask (remove small "
            "islands or keep the largest), recovers the sinuses the model drops, optionally "
            "segments the external face air for flow modelling, and grows the airway inferiorly. "
            "Export is STL, OBJ, NIFTI or NRRD, per segment or merged. Batch processing runs a "
            "folder of volumes and DICOM series, or a JSON template, and an autonomous, "
            "CUDA-aware preflight installs and checks the dependencies before a run."
        )
        self.parent.acknowledgementText = tr(
            "Developed by Dr. Pietro Montagna (DDS, MSc, PhD Student; pietro.montagna@univr.it) and "
            "Dr. Fabio Lonardi (MD, OMFS, PhD Student; fabio.lonardi@univr.it), Head and Neck "
            "Department, Department of Surgery, Dentistry, Pediatrics and Gynecology, University "
            "of Verona, Verona, Italy.\n\n"
            "Derived from "
            '<a href="https://github.com/capenaka/SlicerUpperAirwaySegmentator">'
            "capenaka/SlicerUpperAirwaySegmentator</a>, originally developed at the "
            '<a href="https://www.ualberta.ca/school-of-dentistry/">University of Alberta</a> '
            "(uAlberta) and licensed under Apache-2.0. Please cite the upstream paper "
            "(Gianoni-Capenakas et al., Journal of Dentistry 2026) and nnU-Net "
            "(Isensee et al., Nature Methods 2021)."
        )


class AUAEWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)
        widget = SegmentationWidget()
        self.logic = widget.logic
        self.layout.addWidget(widget)
        self.layout.addStretch()


class AUAETest(ScriptedLoadableModuleTest):
    def runTest(self):
        try:
            from SlicerPythonTestRunnerLib import RunnerLogic, RunnerWidget, RunSettings, isRunningInTestMode
            from pathlib import Path
        except ImportError:
            slicer.util.warningDisplay("Please install SlicerPythonTestRunner extension to run the self tests.")
            return

        currentDirTest = Path(__file__).parent.joinpath("Testing")
        results = RunnerLogic().runAndWaitFinished(
            currentDirTest,
            RunSettings(extraPytestArgs=RunSettings.pytestFileFilterArgs("*TestCase.py") + ["-m not slow"]),
            doRunInSubProcess=not isRunningInTestMode()
        )

        if results.failuresNumber:
            raise AssertionError(f"Test failed: \n{results.getFailingCasesString()}")

        slicer.util.delayDisplay(f"Tests OK. {results.getSummaryString()}")
