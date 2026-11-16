"""Main module widget.

This is based on capenaka/SlicerUpperAirwaySegmentator. The upstream automatic nnU-Net
segmentation core (input selection, weight download, inference through SlicerNNUNetLib) is
kept. The changes are:

  * the mask is post-processed in NumPy during the segmentation step (see AirwayExtension):
    island cleanup, sinus recovery, optional external face air; a non-destructive surface
    smoothing slider controls the exported mesh;
  * segmentation, refinement and airway extension are three separate steps: "Apply" segments
    and cleans the mask; an embedded Segment Editor (qMRMLSegmentEditorWidget), placed between
    segmentation and extension, lets the user refine the mask (paint, erase, islands, mask
    smoothing); then "Run airway extension" extends the selected segmentation inferiorly;
  * the export writes STL/OBJ meshes and NIFTI/NRRD labelmaps, per segment or merged;
  * a "Batch processing" section runs a folder of inputs, or a JSON template, into per-case
    output subfolders;
  * a "Dependencies & CUDA" section reports required-vs-installed versions and installs
    everything needed the Slicer-compliant way (via the NNUNet + PyTorch extensions).
"""

import os
from enum import Flag, auto
from pathlib import Path

import ctk
import qt
import slicer
import vtk

from . import AirwayExtension
from . import Dependencies
from . import LlmApi
from . import Progress
from . import BatchProcessor as BatchProcessorLib
from .IconPath import icon, iconPath
from .PythonDependencyChecker import PythonDependencyChecker
from .Utils import (
    createButton,
    addInCollapsibleLayout,
    set3DViewBackgroundColors,
    setConventionalWideScreenView,
    setBoxAndTextVisibilityOnThreeDViews,
)


class ExportFormat(Flag):
    STL = auto()
    OBJ = auto()
    NIFTI = auto()
    NRRD = auto()


def _stringArray(items):
    """Build a vtkStringArray from a list of strings (segment-id list for the export logic)."""
    arr = vtk.vtkStringArray()
    for value in items:
        arr.InsertNextValue(value)
    return arr


class SegmentationWidget(qt.QWidget):
    def __init__(self, logic=None, parent=None):
        super().__init__(parent)
        self.logic = logic or self._createSlicerSegmentationLogic()
        self._prevSegmentationNode = None
        self._lastExportFolder = ""
        self.segmentEditorNode = None
        self.segmentEditorWidget = None

        self.inputSelector = slicer.qMRMLNodeComboBox(self)
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.addEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)

        # Segmentation node selector. addEnabled=True restores the "Create new segmentation"
        # entry, so the user can pick or create the node to segment, refine, and extend.
        self.segmentationNodeSelector = slicer.qMRMLNodeComboBox(self)
        self.segmentationNodeSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationNodeSelector.selectNodeUponCreation = True
        self.segmentationNodeSelector.addEnabled = True
        self.segmentationNodeSelector.removeEnabled = True
        self.segmentationNodeSelector.renameEnabled = True
        self.segmentationNodeSelector.showHidden = False
        self.segmentationNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentationNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateSegmentationSelection)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.inputSelector)
        layout.addWidget(self.segmentationNodeSelector)

        self.applyButton = createButton(
            "Apply",
            callback=self.onApplyClicked,
            toolTip="Segment the selected volume and clean the mask (step 1).",
            icon=icon("start_icon.png"),
        )

        self.currentInfoTextEdit = qt.QTextEdit()
        self.currentInfoTextEdit.setReadOnly(True)
        self.currentInfoTextEdit.setLineWrapMode(qt.QTextEdit.NoWrap)
        self.currentInfoTextEdit.setMinimumHeight(160)  # taller: it holds progress and the airway volume
        self.fullInfoLogs = []

        self.stopButton = createButton("Stop", callback=self.onStopClicked, toolTip="Click to Stop the segmentation.")
        # Progress of the run in hand: one bar for a single segmentation and for a batch (where
        # it covers the whole batch), with a line saying which case and stage it is at.
        self._runTracker = None
        self.runProgressBar = qt.QProgressBar(self)
        self.runProgressBar.setRange(0, 1000)
        self.runProgressBar.setVisible(False)
        self.runStageLabel = qt.QLabel("", self)
        self.runStageLabel.setVisible(False)

        self.stopWidget = qt.QWidget(self)
        stopLayout = qt.QVBoxLayout(self.stopWidget)
        stopLayout.setContentsMargins(0, 0, 0, 0)
        stopLayout.addWidget(self.stopButton)
        stopLayout.addWidget(self.runProgressBar)
        stopLayout.addWidget(self.runStageLabel)
        stopLayout.addWidget(self.currentInfoTextEdit)
        self.stopWidget.setVisible(False)
        self.loading = qt.QMovie(iconPath("loading.gif"))
        self.loading.setScaledSize(qt.QSize(24, 24))
        self.loading.frameChanged.connect(self._updateStopIcon)
        self.loading.start()

        self.applyWidget = qt.QWidget(self)
        applyLayout = qt.QHBoxLayout(self.applyWidget)
        applyLayout.setContentsMargins(0, 0, 0, 0)
        applyLayout.addWidget(self.applyButton, 1)
        applyLayout.addWidget(
            createButton("", callback=self.showInfoLogs, icon=icon("info.png"), toolTip="Show logs.")
        )

        layout.addWidget(self.applyWidget)
        layout.addWidget(self.stopWidget)

        addInCollapsibleLayout(self._createPostprocessWidget(), layout, "Post-processing", isCollapsed=False)
        addInCollapsibleLayout(self._createEditorWidget(), layout, "Segment editor (refine)", isCollapsed=False)
        addInCollapsibleLayout(self._createExtensionWidget(), layout, "Airway extension", isCollapsed=False)

        exportWidget = self._createExportWidget()
        addInCollapsibleLayout(exportWidget, layout, "Export segmentation", isCollapsed=False)

        addInCollapsibleLayout(self._createBatchWidget(), layout, "Batch processing", isCollapsed=True)
        addInCollapsibleLayout(self._createDependenciesWidget(), layout, "Dependencies && CUDA", isCollapsed=True)
        addInCollapsibleLayout(self._createLlmApiWidget(), layout, "LLM API support (MCP)", isCollapsed=True)

        layout.addStretch()

        self.isStopping = False
        self._dependencyChecker = PythonDependencyChecker()
        self.processedVolumes = {}

        self.onInputChanged()
        self.updateSegmentationSelection()
        self.sceneCloseObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onSceneChanged)
        self.onSceneChanged(doStopInference=False)
        self._connectSegmentationLogic()

        # Fill the hardware summary once the panel is up, and pick up an installation that was
        # waiting for a restart.
        qt.QTimer.singleShot(0, self._updateDependencySummary)
        if Dependencies.isResumePending():
            qt.QTimer.singleShot(500, self._resumeDependencyInstall)

    # ------------------------------------------------------------------ UI builders ---
    def _createPostprocessWidget(self):
        # Island cleanup, applied during the segmentation step. The two options are mutually
        # exclusive (both may be off = no cleanup). Replaces the upstream "Islands" effect.
        widget = qt.QWidget()
        form = qt.QFormLayout(widget)
        self.removeIslandsCheckBox = qt.QCheckBox(widget)
        self.removeIslandsCheckBox.setChecked(True)
        self.removeIslandsCheckBox.setToolTip("Remove small disconnected components below a physical-volume threshold.")
        self.removeIslandsCheckBox.toggled.connect(self._onRemoveIslandsToggled)
        form.addRow("Remove small islands", self.removeIslandsCheckBox)

        self.keepLargestCheckBox = qt.QCheckBox(widget)
        self.keepLargestCheckBox.setChecked(False)
        self.keepLargestCheckBox.setToolTip(
            "Keep only the single largest connected component. NOTE: for the upper airway this "
            "drops the nasal cavity and sinuses (they are separate from the pharynx). Leave off "
            "unless you only want the pharyngeal airway."
        )
        self.keepLargestCheckBox.toggled.connect(self._onKeepLargestToggled)
        form.addRow("Keep largest island only", self.keepLargestCheckBox)

        # Recover the internal air cavities (paranasal sinuses) the model drops, back into the
        # airway. These are air enclosed by the head, so they never belong to the external air.
        self.internalAirCheckBox = qt.QCheckBox(widget)
        self.internalAirCheckBox.setChecked(True)  # on by default: the model often drops the sinuses
        self.internalAirCheckBox.setToolTip(
            "Attempt to recover the frontal and other paranasal sinuses the model misses, by "
            "thresholding the air enclosed inside the head and folding it into the airway. "
            "Smaller cephalostat air bubbles near the frontal sinus are dropped automatically."
        )
        form.addRow("Attempt frontal sinus segmentation", self.internalAirCheckBox)

        # Optional second segment: the ambient air in front of the face, for flow modelling.
        self.externalAirCheckBox = qt.QCheckBox(widget)
        self.externalAirCheckBox.setChecked(False)
        self.externalAirCheckBox.setToolTip(
            "Also segment the ambient air in front of the face as a second segment. For flow "
            "(CFD) modelling this is the inlet volume the air enters through (nostrils / mouth)."
        )
        self.externalAirCheckBox.toggled.connect(self._onExternalAirToggled)
        form.addRow("Segment external face air", self.externalAirCheckBox)

        self.mergeExternalAirCheckBox = qt.QCheckBox(widget)
        self.mergeExternalAirCheckBox.setChecked(False)
        self.mergeExternalAirCheckBox.setEnabled(False)
        self.mergeExternalAirCheckBox.setToolTip(
            "Merge the external air and the airway into a single segment (one continuous fluid "
            "domain) instead of keeping them as two separate segments."
        )
        form.addRow("Merge external air into airway", self.mergeExternalAirCheckBox)

        # Surface smoothing of the exported mesh (3D view + STL/OBJ), non-destructive.
        self.smoothingSlider = ctk.ctkSliderWidget(widget)
        self.smoothingSlider.minimum = 0.0
        self.smoothingSlider.maximum = 1.0
        self.smoothingSlider.singleStep = 0.1
        self.smoothingSlider.decimals = 2
        self.smoothingSlider.value = 0.3
        self.smoothingSlider.setToolTip(
            "Surface smoothing of the exported mesh (0 = raw voxels, 1 = very smooth). Applies to "
            "the 3D model and STL/OBJ, not to the labelmap. For mask-level smoothing use the "
            "editor's Smoothing effect (Median or Closing)."
        )
        self.smoothingSlider.connect("valueChanged(double)", self._onSmoothingChanged)
        form.addRow("Surface smoothing", self.smoothingSlider)
        return widget

    def _createEditorWidget(self):
        # Embedded Segment Editor between segmentation and extension, so the mask can be refined
        # (paint, erase, islands, smoothing) before extending it.
        self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        # The parameter node must exist before any segmentation or source volume is assigned,
        # otherwise the widget logs "need to set segment editor and segmentation nodes first"
        # and the segments model complains about an invalid display node. Create it up front.
        self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
        try:
            self.segmentEditorWidget.setMaximumNumberOfUndoStates(10)
            self.segmentEditorWidget.setSwitchToSegmentationsButtonVisible(False)
        except Exception:  # noqa: BLE001
            pass
        return self.segmentEditorWidget

    def _setEditorSourceVolume(self, volumeNode):
        # A source volume can only be set once the editor has BOTH a parameter node and a
        # segmentation node; setting it earlier triggers the VTK "need to set segment editor
        # and segmentation nodes first" warning. Guard on all three conditions.
        if (self.segmentEditorWidget is None or self.segmentEditorNode is None
                or volumeNode is None or self.getCurrentSegmentationNode() is None):
            return
        try:
            self.segmentEditorWidget.setSourceVolumeNode(volumeNode)
        except Exception:  # noqa: BLE001
            try:
                self.segmentEditorWidget.setMasterVolumeNode(volumeNode)
            except Exception:  # noqa: BLE001
                pass

    def _onSmoothingChanged(self, value):
        seg = self.getCurrentSegmentationNode()
        if seg and seg.GetSegmentation().GetNumberOfSegments() > 0:
            AirwayExtension.applySurfaceSmoothing(seg, value)

    def _createExtensionWidget(self):
        # Step 3, decoupled from segmentation: extends the currently selected segmentation
        # for flow modelling, so the user can refine the mask first, then extend it.
        widget = qt.QWidget()
        form = qt.QFormLayout(widget)

        # The extension always runs inferiorly (towards the neck); only the length is variable.
        self.extendLengthSpinBox = ctk.ctkDoubleSpinBox(widget)
        self.extendLengthSpinBox.minimum = 0.0
        self.extendLengthSpinBox.maximum = 300.0
        self.extendLengthSpinBox.singleStep = 5.0
        self.extendLengthSpinBox.value = 100.0
        self.extendLengthSpinBox.suffix = " mm"
        self.extendLengthSpinBox.setToolTip("How far to extend the airway inferiorly (towards the neck), beyond the scan.")
        form.addRow("Inferior length", self.extendLengthSpinBox)

        form.addRow(createButton(
            "Run airway extension", callback=self.onRunExtensionClicked, parent=widget,
            toolTip="Extend the selected segmentation inferiorly by repeating its terminal slice (step 3)."
        ))
        return widget

    def _createExportWidget(self):
        # Two clearly separated groups so it is obvious WHAT is written and in WHICH format:
        # "What to export" (which segments) and "File format" (mesh vs labelmap).
        exportWidget = qt.QWidget()
        layout = qt.QVBoxLayout(exportWidget)

        whatBox = qt.QGroupBox("What to export", exportWidget)
        whatLayout = qt.QVBoxLayout(whatBox)
        self.exportAirwayCheckBox = qt.QCheckBox("Airway", whatBox)
        self.exportAirwayCheckBox.setChecked(True)
        self.exportAirwayCheckBox.setToolTip("Write the airway segment.")
        self.exportExternalAirCheckBox = qt.QCheckBox("External air (if segmented)", whatBox)
        self.exportExternalAirCheckBox.setToolTip("Write the external face-air segment; skipped if it was not segmented.")
        self.exportMergedCheckBox = qt.QCheckBox("Merged (all segments in one file)", whatBox)
        self.exportMergedCheckBox.setToolTip("Write a single file that contains all segments together.")
        for cb in (self.exportAirwayCheckBox, self.exportExternalAirCheckBox, self.exportMergedCheckBox):
            whatLayout.addWidget(cb)
        layout.addWidget(whatBox)

        formatBox = qt.QGroupBox("File format", exportWidget)
        formatLayout = qt.QVBoxLayout(formatBox)
        self.stlCheckBox = qt.QCheckBox("STL (surface mesh)", formatBox)
        self.stlCheckBox.setChecked(True)
        self.objCheckBox = qt.QCheckBox("OBJ (surface mesh)", formatBox)
        self.niftiCheckBox = qt.QCheckBox("NIFTI (labelmap, .nii.gz)", formatBox)
        self.niftiCheckBox.setChecked(True)
        self.nrrdCheckBox = qt.QCheckBox("NRRD (labelmap, .nrrd)", formatBox)
        for cb in (self.stlCheckBox, self.objCheckBox, self.niftiCheckBox, self.nrrdCheckBox):
            formatLayout.addWidget(cb)
        layout.addWidget(formatBox)

        layout.addWidget(createButton(
            "Export", callback=self.onExportClicked, parent=exportWidget,
            toolTip="Write each selected target in each selected format into a folder you choose.",
        ))
        return exportWidget

    def _createBatchWidget(self):
        widget = qt.QWidget()
        layout = qt.QVBoxLayout(widget)

        # Simple path: folder in, folder out. No JSON needed; it reuses the current settings.
        folderBox = qt.QGroupBox("Batch by folder (recommended)", widget)
        folderLayout = qt.QVBoxLayout(folderBox)
        folderHint = qt.QLabel(
            "Segment every volume in a folder, using the current Post-processing and Export "
            "settings above. No JSON needed."
        )
        folderHint.setWordWrap(True)
        folderLayout.addWidget(folderHint)

        folderForm = qt.QFormLayout()
        self.batchInputFolderLineEdit = qt.QLineEdit(folderBox)
        self.batchInputFolderLineEdit.setToolTip(
            "Folder of inputs: single-file volumes (.nii/.nii.gz/.nrrd/.mha...) and/or DICOM "
            "series (one series per patient subfolder, files may be nested). Every input is "
            "processed; a folder with more than one series fails with an error."
        )
        inRow = qt.QHBoxLayout()
        inRow.addWidget(self.batchInputFolderLineEdit, 1)
        inRow.addWidget(createButton("Browse...", callback=self.onBrowseBatchInputFolder, parent=folderBox))
        inRowW = qt.QWidget(folderBox)
        inRowW.setLayout(inRow)
        folderForm.addRow("Input folder", inRowW)

        self.batchOutputFolderLineEdit = qt.QLineEdit(folderBox)
        self.batchOutputFolderLineEdit.setToolTip(
            "Output folder. Leave empty to use an 'AUAE_output' folder beside the inputs."
        )
        outRow = qt.QHBoxLayout()
        outRow.addWidget(self.batchOutputFolderLineEdit, 1)
        outRow.addWidget(createButton("Browse...", callback=self.onBrowseBatchOutputFolder, parent=folderBox))
        outRowW = qt.QWidget(folderBox)
        outRowW.setLayout(outRow)
        folderForm.addRow("Output folder", outRowW)
        folderLayout.addLayout(folderForm)

        folderLayout.addWidget(createButton(
            "Run batch (folder)", callback=self.onRunBatchFolderClicked, parent=folderBox,
            toolTip="Segment and export every volume in the input folder, using the current "
                    "post-processing and export settings, into per-volume subfolders."
        ))
        layout.addWidget(folderBox)

        # Advanced path: a JSON template for explicit per-run control (ordered list, extension).
        templateBox = qt.QGroupBox("Batch by template (advanced, optional)", widget)
        templateLayout = qt.QVBoxLayout(templateBox)
        templateHint = qt.QLabel(
            "Optional. For explicit control: an ordered list of volumes and every option in one "
            "JSON file. Not needed for the folder run above."
        )
        templateHint.setWordWrap(True)
        templateLayout.addWidget(templateHint)

        self.batchTemplateLineEdit = qt.QLineEdit(templateBox)
        self.batchTemplateLineEdit.text = str(BatchProcessorLib.defaultTemplatePath())
        self.batchTemplateLineEdit.setToolTip("JSON template listing the volumes and options.")
        browseRow = qt.QHBoxLayout()
        browseRow.addWidget(self.batchTemplateLineEdit, 1)
        browseRow.addWidget(createButton("Browse...", callback=self.onBrowseTemplate, parent=templateBox))
        browseRow.addWidget(createButton("Open folder", callback=self.onOpenTemplateFolder, parent=templateBox))
        rowWidget = qt.QWidget(templateBox)
        rowWidget.setLayout(browseRow)
        templateForm = qt.QFormLayout()
        templateForm.addRow("Template JSON", rowWidget)
        templateLayout.addLayout(templateForm)

        templateLayout.addWidget(createButton(
            "Run batch (template)", callback=self.onRunBatchClicked, parent=templateBox,
            toolTip="Segment and export every volume listed in the template into per-volume subfolders."
        ))
        layout.addWidget(templateBox)

        # Progress across the whole batch, shown only while a run is in flight. The same value
        # is on the bar next to the Stop button, with the case and stage spelled out.
        self.batchProgressBar = qt.QProgressBar(widget)
        self.batchProgressBar.setRange(0, 1000)
        self.batchProgressBar.setVisible(False)
        layout.addWidget(self.batchProgressBar)
        return widget

    def _showProgress(self, snapshot):
        """Show a progress snapshot (see Progress.RunProgress.snapshot) on the bars and the label."""
        value = int(round(snapshot["overall_fraction"] * 1000))
        for bar in (self.runProgressBar, self.batchProgressBar):
            bar.setRange(0, 1000)
            bar.setValue(value)
        self.runStageLabel.setText(Progress.describe(snapshot))
        slicer.app.processEvents()

    def _setProgressVisible(self, isVisible):
        self.runProgressBar.setVisible(isVisible)
        self.runStageLabel.setVisible(isVisible)
        if not isVisible:
            self._runTracker = None

    def _startRunProgress(self, caseName=""):
        """Begin tracking a single segmentation; its log lines move the bar from here on."""
        self._runTracker = Progress.RunProgress(1)
        self._runTracker.startCase(1, caseName)
        self._setProgressVisible(True)
        self._showProgress(self._runTracker.snapshot())

    def _createDependenciesWidget(self):
        widget = qt.QWidget()
        form = qt.QFormLayout(widget)
        self.dependencySummaryLabel = qt.QLabel("", widget)
        self.dependencySummaryLabel.setWordWrap(True)
        form.addRow(self.dependencySummaryLabel)
        self.deviceComboBox = qt.QComboBox(widget)
        for device in ("Auto", "CUDA", "MPS", "CPU"):
            self.deviceComboBox.addItem(device)
        self.deviceComboBox.setToolTip(
            "Inference device. 'Auto' uses the NVIDIA GPU (CUDA) when it works, the Apple GPU (MPS) "
            "on Apple Silicon, and the CPU otherwise."
        )
        form.addRow("Inference device", self.deviceComboBox)
        buttons = qt.QHBoxLayout()
        buttons.addWidget(createButton(
            "Install dependencies", callback=self.onInstallDependenciesClicked, parent=widget,
            toolTip="Detect the hardware and install everything AUAE needs: the PyTorch build that "
                    "suits this computer, nnU-Net and the extra packages. A PyTorch that cannot work "
                    "here is replaced; a working one is left alone.",
        ))
        buttons.addWidget(createButton(
            "Check", callback=self.onCheckDependenciesClicked, parent=widget,
            toolTip="Print what is installed, what this computer needs and the device inference will use.",
        ))
        rowWidget = qt.QWidget(widget)
        rowWidget.setLayout(buttons)
        form.addRow(rowWidget)
        return widget

    def _createLlmApiWidget(self):
        # Opt-in switch for the MCP server (MCP/auae_mcp.py). Off until the user ticks it; the
        # choice is kept between sessions. The server refuses every request while it is off.
        widget = qt.QWidget()
        layout = qt.QVBoxLayout(widget)
        self.llmApiCheckBox = qt.QCheckBox("Allow an agentic AI to run AUAE", widget)
        self.llmApiCheckBox.setToolTip(
            "When ticked, the MCP server shipped with AUAE accepts requests from an agentic AI "
            "such as Claude or Codex: describing input folders, starting segmentations, reading "
            "their results. Off by default; the choice is remembered. You are asked to accept a "
            "disclaimer on patient data before it is switched on."
        )
        self.llmApiCheckBox.setChecked(LlmApi.isEnabled())
        self.llmApiCheckBox.toggled.connect(self._onLlmApiToggled)
        layout.addWidget(self.llmApiCheckBox)

        LlmApi.syncSettingsFile()

        note = qt.QLabel(
            "The AI talks to AUAE through the MCP server, a small program that runs on Slicer's "
            "own Python. AUAE does not anonymise what the AI reads: protecting patient data is the "
            "responsibility of whoever uses the AI.", widget)
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addWidget(createButton(
            "Install MCP server", callback=self.onInstallMcpServerClicked, parent=widget,
            toolTip="Install the two packages the server needs into their own folder "
                    "(~/.auae_mcp/site, apart from Slicer's packages), then ask in which AI clients "
                    "to register the server: Claude Code, Codex, Claude Desktop, Gemini CLI, Cursor. "
                    "Only the ones you tick are changed.",
        ))
        self.llmApiStatusLabel = qt.QLabel("", widget)
        self.llmApiStatusLabel.setWordWrap(True)
        layout.addWidget(self.llmApiStatusLabel)

        manual = qt.QLabel("Server command, to register it by hand in any other MCP client:", widget)
        manual.setWordWrap(True)
        layout.addWidget(manual)
        self.llmApiCommandLineEdit = qt.QLineEdit(LlmApi.serverCommandLine(), widget)
        self.llmApiCommandLineEdit.setReadOnly(True)
        layout.addWidget(self.llmApiCommandLineEdit)
        self._updateLlmApiStatus()
        return widget

    def _updateLlmApiStatus(self):
        if LlmApi.isServerInstalled():
            self.llmApiStatusLabel.setText("The MCP server is installed.")
        else:
            self.llmApiStatusLabel.setText("The MCP server is not installed yet.")

    def _onLlmApiToggled(self, checked):
        # Switching the support on means accepting the disclaimer; declining leaves it off.
        if checked and not slicer.util.confirmYesNoDisplay(
                LlmApi.DISCLAIMER + "\n\nAllow an agentic AI to run AUAE?",
                windowTitle="LLM API support"):
            self.llmApiCheckBox.blockSignals(True)
            self.llmApiCheckBox.setChecked(False)
            self.llmApiCheckBox.blockSignals(False)
            return
        try:
            LlmApi.setEnabled(checked)
        except Exception as exc:  # noqa: BLE001
            slicer.util.errorDisplay("Could not save the LLM API setting:\n%s" % exc)

    def _buildMcpClientsDialog(self):
        """Dialog that lists the AI clients and lets the user tick where to register the server.

        Nothing is ticked in advance: a client's configuration is changed only if the user asks
        for it. Clients that are not installed are listed greyed out.
        """
        dialog = qt.QDialog(self)
        dialog.setWindowTitle("Install MCP server")
        layout = qt.QVBoxLayout(dialog)
        intro = qt.QLabel(
            "The server's packages are installed in any case. Tick the AI clients in which you "
            "want it registered. Only those are changed; the others are left alone and can be set "
            "up later, with this button or by hand.", dialog)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        checkBoxes = {}
        for label, found in LlmApi.detectClients().items():
            box = qt.QCheckBox(label if found else label + " (not found on this computer)", dialog)
            box.setEnabled(found)
            layout.addWidget(box)
            checkBoxes[label] = box
        buttons = qt.QDialogButtonBox(qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        return dialog, checkBoxes

    def onInstallMcpServerClicked(self, *_):
        dialog, checkBoxes = self._buildMcpClientsDialog()
        if not dialog.exec():
            return
        chosen = [label for label, box in checkBoxes.items() if box.checked]

        self.currentInfoTextEdit.clear()
        self._setApplyVisible(False)
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        result = None
        try:
            result = LlmApi.setupServer(self.onProgressInfo, chosen)
        except Exception as exc:  # noqa: BLE001
            self.onProgressInfo("MCP server installation failed: %s" % exc)
        finally:
            slicer.app.restoreOverrideCursor()
            self._setApplyVisible(True)
            self._updateLlmApiStatus()
        if not result or not result["installed"]:
            slicer.util.warningDisplay("The MCP server could not be installed. The log has the details.")
            return

        clients = result["clients"]
        registered = [label for label, state in clients.items() if state]
        failed = [label for label, state in clients.items() if state is False]
        message = "The MCP server is installed."
        if registered:
            message += "\n\nRegistered with: %s. Restart those that were open." % ", ".join(registered)
        elif not failed:
            message += ("\n\nIt was not registered with any AI client. Click the button again to "
                        "choose one, or register it by hand with the command shown in the panel.")
        if failed:
            message += "\n\nRegistration failed for: %s. The log has the details." % ", ".join(failed)
        if not self.llmApiCheckBox.checked:
            message += "\n\nThe support is still switched off: tick 'Allow an agentic AI to run AUAE' to let it in."
        slicer.util.infoDisplay(message)

    def _updateDependencySummary(self):
        try:
            self.dependencySummaryLabel.setText(Dependencies.summary())
        except Exception as exc:  # noqa: BLE001
            self.dependencySummaryLabel.setText("Hardware detection failed: %s" % exc)

    # ------------------------------------------------------------------ lifecycle -----
    def __del__(self):
        slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)
        if self.segmentEditorNode is not None:
            try:
                slicer.mrmlScene.RemoveNode(self.segmentEditorNode)
            except Exception:  # noqa: BLE001
                pass
        super().__del__()

    def onSceneChanged(self, *_, doStopInference=True):
        if doStopInference:
            self.onStopClicked()
        self.processedVolumes = {}
        self._prevSegmentationNode = None
        # (re)create the segment editor parameter node for the fresh scene
        if self.segmentEditorNode is not None:
            try:
                slicer.mrmlScene.RemoveNode(self.segmentEditorNode)
            except Exception:  # noqa: BLE001
                pass
        self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        if self.segmentEditorWidget is not None:
            self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
            # Re-point the editor at whatever is currently selected (both may be None right
            # after a scene close; the guards below then no-op).
            self.segmentEditorWidget.setSegmentationNode(self.getCurrentSegmentationNode())
            self._setEditorSourceVolume(self.getCurrentVolumeNode())
        self._initSlicerDisplay()

    @staticmethod
    def _initSlicerDisplay():
        """Initialize Slicer's display with a white background and no 3D box / labels."""
        set3DViewBackgroundColors([1, 1, 1], [1, 1, 1])
        setConventionalWideScreenView()
        setBoxAndTextVisibilityOnThreeDViews(False)

    def _updateStopIcon(self):
        # The loading QMovie keeps ticking after the widget is destroyed on a module reload;
        # guard against the stale signal touching an already-deleted button.
        try:
            self.stopButton.setIcon(qt.QIcon(self.loading.currentPixmap()))
        except (ValueError, RuntimeError):
            pass

    def onStopClicked(self):
        """Stop the running inference and restore the buttons once cleanup is done."""
        self.isStopping = True
        if self.logic is not None:
            self.logic.stopSegmentation()
            self.logic.waitForSegmentationFinished()
        slicer.app.processEvents()
        self.isStopping = False
        self._setApplyVisible(True)

    def onApplyClicked(self, *_):
        self.currentInfoTextEdit.clear()
        self._setApplyVisible(False)
        if not self._ensureReadyToRun():
            self._setApplyVisible(True)
            return
        if not self._dependencyChecker.downloadWeightsIfNeeded(self.onProgressInfo):
            self._setApplyVisible(True)
            return
        volumeNode = self.getCurrentVolumeNode()
        self._startRunProgress(volumeNode.GetName() if volumeNode else "")
        self._runSegmentation()

    def _setApplyVisible(self, isVisible):
        if isVisible:
            self._setProgressVisible(False)  # the run is over, whatever the outcome
        self.applyWidget.setVisible(isVisible)
        self.stopWidget.setVisible(not isVisible)
        self.inputSelector.setEnabled(isVisible)
        self.segmentationNodeSelector.setEnabled(isVisible)

    def _ensureReadyToRun(self):
        """Preflight before a run: print the report, then install or repair what is missing.

        When everything is in place this costs a fraction of a second. Otherwise the user is
        asked once, with the list of what will be installed.
        """
        if not Dependencies.report(self.onProgressInfo):
            ok, needsRestart = Dependencies.ensure(self.onProgressInfo, askConfirmation=True, refresh=False)
            self._updateDependencySummary()
            if needsRestart:
                self._offerRestart()
                return False
            if not ok:
                slicer.util.errorDisplay(
                    "The dependencies are not ready. Open 'Dependencies && CUDA', click "
                    "'Install dependencies' and check the log."
                )
                return False
        if self.logic is None and self.isNNUNetModuleInstalled():
            self.logic = self._createSlicerSegmentationLogic()
            self._connectSegmentationLogic()
        if self.logic is None:
            slicer.util.errorDisplay("The NNUNet extension is required. Install it and restart Slicer.")
            return False
        return True

    def _makeParameter(self):
        """Build an nnU-Net Parameter with the selected device, falling back if unsupported."""
        from SlicerNNUNetLib import Parameter
        kwargs = dict(folds="0", modelPath=self.nnUnetFolder())
        kwargs.update(Dependencies.parameterDeviceKwargs(self.deviceComboBox.currentText))
        try:
            return Parameter(**kwargs)
        except TypeError:
            return Parameter(folds="0", modelPath=self.nnUnetFolder())

    def _runSegmentation(self):
        deviceChoice = self.deviceComboBox.currentText.strip().lower()
        # resolveDevice falls back to 'cpu' when the requested GPU cannot be used.
        if deviceChoice != "cpu" and Dependencies.resolveDevice(deviceChoice) == "cpu":
            ret = qt.QMessageBox.question(
                self,
                "No usable GPU",
                "PyTorch cannot use a GPU on this computer right now.\n"
                "The segmentation will run on the CPU and may take up to 1 hour.\n"
                "'Dependencies && CUDA' shows why and what can be installed.\n"
                "Would you like to proceed on the CPU?",
            )
            if ret == qt.QMessageBox.No:
                self._setApplyVisible(True)
                return

        slicer.app.processEvents()
        self.logic.setParameter(self._makeParameter())
        self.logic.startSegmentation(self.getCurrentVolumeNode())

    def onInputChanged(self, *_):
        volumeNode = self.getCurrentVolumeNode()
        self.applyButton.setEnabled(volumeNode is not None)
        slicer.util.setSliceViewerLayers(background=volumeNode)
        slicer.util.resetSliceViews()
        self._setEditorSourceVolume(volumeNode)
        self._restoreProcessedSegmentation()

    def _restoreProcessedSegmentation(self):
        segmentationNode = self.processedVolumes.get(self.getCurrentVolumeNode())
        self.segmentationNodeSelector.setCurrentNode(segmentationNode)

    def _storeProcessedSegmentation(self):
        volumeNode = self.getCurrentVolumeNode()
        segmentationNode = self.getCurrentSegmentationNode()
        if volumeNode and segmentationNode:
            self.processedVolumes[volumeNode] = segmentationNode

    def updateSegmentationSelection(self, *_):
        """Toggle display of the selected segmentation and point the segment editor at it."""
        if self._prevSegmentationNode:
            self._prevSegmentationNode.SetDisplayVisibility(False)
        segmentationNode = self.getCurrentSegmentationNode()
        self._prevSegmentationNode = segmentationNode
        # Create the display node BEFORE handing the segmentation to the editor, otherwise the
        # segments model logs "Invalid segmentation display node".
        self._initializeSegmentationNodeDisplay(segmentationNode)
        # Wire the embedded editor (refine step) to the current segmentation + source volume.
        if self.segmentEditorWidget is not None and self.segmentEditorNode is not None:
            self.segmentEditorWidget.setSegmentationNode(segmentationNode)
            self._setEditorSourceVolume(self.getCurrentVolumeNode())

    def _initializeSegmentationNodeDisplay(self, segmentationNode):
        if not segmentationNode:
            return
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self.getCurrentVolumeNode())
        if not segmentationNode.GetDisplayNode():
            segmentationNode.CreateDefaultDisplayNodes()
            slicer.app.processEvents()
        segmentationNode.SetDisplayVisibility(True)
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDWidget.threeDView().rotateToViewAxis(3)
        slicer.util.resetThreeDViews()

    def getCurrentVolumeNode(self):
        return self.inputSelector.currentNode()

    def getCurrentSegmentationNode(self):
        return self.segmentationNodeSelector.currentNode()

    def onInferenceFinished(self, *_):
        if self.isStopping:
            self._setApplyVisible(True)
            return
        try:
            self.onProgressInfo("Loading inference results...")
            self._loadSegmentationResults()
            self.onProgressInfo("Segmentation ended successfully. Refine it if needed, then run the airway extension.")
        except RuntimeError as e:
            slicer.util.errorDisplay(e)
            self.onProgressInfo("Error loading results :\n" + str(e))
        finally:
            self._setApplyVisible(True)

    def _loadSegmentationResults(self):
        """Load the nnU-Net result and run island cleanup only (extension is a separate step)."""
        rawSegmentation = self.logic.loadSegmentation()
        rawSegmentation.SetName(self.getCurrentVolumeNode().GetName() + "_Segmentation")
        segmentationNode = AirwayExtension.postprocessSegmentation(
            rawSegmentation, self.getCurrentVolumeNode(), self._collectPostprocessOptions(), self.onProgressInfo
        )
        slicer.mrmlScene.RemoveNode(rawSegmentation)

        currentSegmentation = self.getCurrentSegmentationNode()
        if currentSegmentation is not None and currentSegmentation is not segmentationNode:
            self._copySegmentationResultsToExistingNode(currentSegmentation, segmentationNode)
        else:
            self.segmentationNodeSelector.setCurrentNode(segmentationNode)
        slicer.app.processEvents()
        self._updateSegmentationDisplay()
        self._storeProcessedSegmentation()

    def _collectPostprocessOptions(self):
        options = AirwayExtension.defaultOptions()
        options["removeSmallIslands"] = bool(self.removeIslandsCheckBox.checked)
        options["keepLargestIsland"] = bool(self.keepLargestCheckBox.checked)
        options["includeInternalAir"] = bool(self.internalAirCheckBox.checked)
        options["segmentExternalAir"] = bool(self.externalAirCheckBox.checked)
        options["mergeExternalAir"] = bool(self.mergeExternalAirCheckBox.checked)
        options["smoothingFactor"] = float(self.smoothingSlider.value)
        options["extend"] = False  # extension is an explicit, separate step
        return options

    def _collectExtensionOptions(self):
        return {
            "lengthMm": float(self.extendLengthSpinBox.value),
            "smoothingFactor": float(self.smoothingSlider.value),
        }

    @staticmethod
    def _copySegmentationResultsToExistingNode(currentSegmentation, segmentationNode):
        currentName = currentSegmentation.GetName()
        currentSegmentation.Copy(segmentationNode)
        currentSegmentation.SetName(currentName)
        slicer.mrmlScene.RemoveNode(segmentationNode)

    def _updateSegmentationDisplay(self):
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            return
        self._initializeSegmentationNodeDisplay(segmentationNode)
        # Apply the user-chosen surface smoothing factor (0 = raw voxel surface). A no-op when
        # the post-processing already built the surface with this factor.
        AirwayExtension.applySurfaceSmoothing(segmentationNode, self.smoothingSlider.value)
        slicer.util.resetThreeDViews()

    def onRunExtensionClicked(self):
        """Step 3: extend the currently selected segmentation, in place."""
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            slicer.util.warningDisplay("Select a segmentation to extend first.")
            return
        self.currentInfoTextEdit.clear()
        try:
            extended = AirwayExtension.extendSegmentation(
                segmentationNode, self.getCurrentVolumeNode(), self._collectExtensionOptions(), self.onProgressInfo
            )
            self._copySegmentationResultsToExistingNode(segmentationNode, extended)
            self._updateSegmentationDisplay()
            slicer.util.infoDisplay("Airway extension applied to '" + segmentationNode.GetName() + "'.")
        except Exception as exc:  # noqa: BLE001
            slicer.util.errorDisplay("Airway extension failed:\n" + str(exc))

    def onInferenceError(self, errorMsg):
        if self.isStopping:
            return
        self._setApplyVisible(True)
        slicer.util.errorDisplay("Encountered error during inference :\n" + errorMsg)

    def onProgressInfo(self, infoMsg):
        infoMsg = self.removeImageIOError(infoMsg)
        self.currentInfoTextEdit.insertPlainText(infoMsg + "\n")
        self.moveTextEditToEnd(self.currentInfoTextEdit)
        self.insertDatedInfoLogs(infoMsg)
        # A single run is tracked here; a batch reports through BatchProcessor's onStage.
        if self._runTracker is not None and self._runTracker.update(infoMsg):
            self._showProgress(self._runTracker.snapshot())
        slicer.app.processEvents()

    @staticmethod
    def removeImageIOError(infoMsg):
        return "\n".join([msg for msg in infoMsg.strip().splitlines() if "Error ImageIO factory" not in msg])

    def insertDatedInfoLogs(self, infoMsg):
        now = qt.QDateTime.currentDateTime().toString("yyyy/MM/dd hh:mm:ss.zzz")
        self.fullInfoLogs.extend([now + " :: " + msgLine for msgLine in infoMsg.splitlines()])

    def showInfoLogs(self):
        dialog = qt.QDialog()
        layout = qt.QVBoxLayout(dialog)
        textEdit = qt.QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.append("\n".join(self.fullInfoLogs))
        textEdit.setLineWrapMode(qt.QTextEdit.NoWrap)
        self.moveTextEditToEnd(textEdit)
        layout.addWidget(textEdit)
        dialog.setWindowFlags(qt.Qt.WindowCloseButtonHint)
        dialog.resize(slicer.util.mainWindow().size * .7)
        dialog.exec()

    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)

    # ------------------------------------------------------------------ export --------
    def getSelectedExportFormats(self):
        selectedFormats = ExportFormat(0)
        checkBoxes = {
            self.objCheckBox: ExportFormat.OBJ,
            self.stlCheckBox: ExportFormat.STL,
            self.niftiCheckBox: ExportFormat.NIFTI,
            self.nrrdCheckBox: ExportFormat.NRRD,
        }
        for checkBox, exportFormat in checkBoxes.items():
            if checkBox.isChecked():
                selectedFormats |= exportFormat
        return selectedFormats

    def getSelectedExportTargets(self):
        """Which segments to write: any of 'airway', 'external', 'merged'."""
        targets = []
        if self.exportAirwayCheckBox.isChecked():
            targets.append("airway")
        if self.exportExternalAirCheckBox.isChecked():
            targets.append("external")
        if self.exportMergedCheckBox.isChecked():
            targets.append("merged")
        return targets

    def onExportClicked(self):
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            slicer.util.warningDisplay("Please select a valid segmentation before exporting.")
            return

        selectedFormats = self.getSelectedExportFormats()
        if selectedFormats == ExportFormat(0):
            slicer.util.warningDisplay("Please select at least one export format (STL / OBJ / NIFTI / NRRD).")
            return
        targets = self.getSelectedExportTargets()
        if not targets:
            slicer.util.warningDisplay("Please select what to export (airway, external air, or merged).")
            return

        folderPath = qt.QFileDialog.getExistingDirectory(
            self, "Please select the export folder", self._lastExportFolder
        )
        if not folderPath:
            return
        self._lastExportFolder = folderPath  # start here next time

        with slicer.util.tryWithErrorDisplay("Export to " + folderPath + " failed.", waitCursor=True):
            self.exportSegmentation(segmentationNode, folderPath, selectedFormats, targets)
            slicer.util.infoDisplay("Export successful to " + folderPath + ".")

    @staticmethod
    def exportSegmentation(segmentationNode, folderPath, selectedFormats, targets=("merged",)):
        """Export the requested segments in the requested formats.

        targets is any of 'airway', 'external', 'merged'. 'airway'/'external' write only that
        segment; 'merged' writes all segments together in a single file. STL/OBJ are surface
        meshes; NIFTI/NRRD are labelmaps. Missing segments are skipped silently.
        """
        segmentation = segmentationNode.GetSegmentation()

        def segmentIdByName(name):
            for i in range(segmentation.GetNumberOfSegments()):
                sid = segmentation.GetNthSegmentID(i)
                if segmentation.GetSegment(sid).GetName() == name:
                    return sid
            return None

        # Build the export jobs: (vtkStringArray of segment ids or None, merge flag).
        jobs = []
        if "airway" in targets:
            sid = segmentIdByName(AirwayExtension.AIRWAY_SEGMENT_NAME)
            if sid:
                jobs.append((_stringArray([sid]), False))
        if "external" in targets:
            sid = segmentIdByName(AirwayExtension.EXTERNAL_AIR_SEGMENT_NAME)
            if sid:
                jobs.append((_stringArray([sid]), False))
        if "merged" in targets:
            jobs.append((None, True))

        logic = slicer.vtkSlicerSegmentationsModuleLogic
        for segmentIds, merge in jobs:
            for surfaceFormat in (ExportFormat.STL, ExportFormat.OBJ):
                if selectedFormats & surfaceFormat:
                    # signature: (folder, node, segmentIds, fileFormat, lps=True, sizeScale=1.0, merge)
                    # merge=True writes one combined mesh (STL only); keep LPS on as before.
                    logic.ExportSegmentsClosedSurfaceRepresentationToFiles(
                        folderPath, segmentationNode, segmentIds, surfaceFormat.name, True, 1.0, merge,
                    )
            for labelExt, labelFormat in (("nii.gz", ExportFormat.NIFTI), ("nrrd", ExportFormat.NRRD)):
                if selectedFormats & labelFormat:
                    # useCompression=True: a binary mask compresses from hundreds of MB to a few.
                    logic.ExportSegmentsBinaryLabelmapRepresentationToFiles(
                        folderPath, segmentationNode, segmentIds, labelExt, True,
                    )

    # ------------------------------------------------------------------ post-processing
    def _onRemoveIslandsToggled(self, checked):
        if checked and self.keepLargestCheckBox.checked:
            self.keepLargestCheckBox.setChecked(False)

    def _onKeepLargestToggled(self, checked):
        if checked and self.removeIslandsCheckBox.checked:
            self.removeIslandsCheckBox.setChecked(False)

    def _onExternalAirToggled(self, checked):
        # The merge flag only makes sense when the external air is being segmented.
        self.mergeExternalAirCheckBox.setEnabled(bool(checked))
        if not checked:
            self.mergeExternalAirCheckBox.setChecked(False)

    # ------------------------------------------------------------------ dependencies --
    def onCheckDependenciesClicked(self):
        self.currentInfoTextEdit.clear()
        try:
            ready = Dependencies.report(self.onProgressInfo)
        except Exception as exc:  # noqa: BLE001
            slicer.util.errorDisplay("Dependency check failed:\n" + str(exc))
            return
        self._updateDependencySummary()
        if ready:
            slicer.util.infoDisplay("All dependencies are present and the extension is ready to run.")
        else:
            slicer.util.warningDisplay(
                "Something is missing or does not suit this computer. Click 'Install dependencies'. "
                "The log has the details."
            )

    def onInstallDependenciesClicked(self, *_):
        self._installDependencies(askConfirmation=True)

    def _resumeDependencyInstall(self):
        """Continue an installation that had to wait for a restart. The user already agreed."""
        self.onProgressInfo("Continuing the dependency installation after the restart...")
        self._installDependencies(askConfirmation=False)

    def _installDependencies(self, askConfirmation):
        self.currentInfoTextEdit.clear()
        self._setApplyVisible(False)
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        ok, needsRestart = False, False
        try:
            ok, needsRestart = Dependencies.ensure(self.onProgressInfo, askConfirmation=askConfirmation)
        except Exception as exc:  # noqa: BLE001
            self.onProgressInfo("Dependency install failed: %s" % exc)
        finally:
            slicer.app.restoreOverrideCursor()
            self._setApplyVisible(True)
            self._updateDependencySummary()
        if needsRestart:
            self._offerRestart()
        elif ok:
            slicer.util.infoDisplay("The dependencies are installed and AUAE is ready to run.\n\n"
                                    + Dependencies.summary())
        else:
            slicer.util.warningDisplay("The dependency setup did not complete. The log has the details.")

    def _offerRestart(self):
        if slicer.util.confirmYesNoDisplay(
                "3D Slicer must be restarted to continue the installation. "
                "It picks up by itself when AUAE opens again.\n\nRestart now?"):
            slicer.util.restart()

    # ------------------------------------------------------------------ batch ---
    def onBrowseTemplate(self):
        path = qt.QFileDialog.getOpenFileName(self, "Select batch template JSON", "", "JSON (*.json)")
        if path:
            self.batchTemplateLineEdit.text = path

    def onOpenTemplateFolder(self):
        folder = os.path.dirname(self.batchTemplateLineEdit.text) or str(BatchProcessorLib.defaultTemplatePath().parent)
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(folder))

    def onBrowseBatchInputFolder(self):
        folder = qt.QFileDialog.getExistingDirectory(self, "Select the input folder of volumes")
        if folder:
            self.batchInputFolderLineEdit.text = folder

    def onBrowseBatchOutputFolder(self):
        folder = qt.QFileDialog.getExistingDirectory(self, "Select the batch output folder")
        if folder:
            self.batchOutputFolderLineEdit.text = folder

    def onRunBatchFolderClicked(self):
        """Classic folder-in / folder-out run, using the current UI post-processing settings."""
        inputFolder = self.batchInputFolderLineEdit.text.strip()
        if not os.path.isdir(inputFolder):
            slicer.util.warningDisplay("Select a valid input folder of volumes.")
            return
        exportFormats = self.getSelectedExportFormats()
        if not exportFormats:
            slicer.util.warningDisplay("Select at least one export format (STL / OBJ / NIFTI / NRRD).")
            return
        config = BatchProcessorLib.folderConfig(
            inputFolder, self.batchOutputFolderLineEdit.text.strip(),
            self._exportFormatNames(exportFormats), self._collectPostprocessOptions(),
            self.getSelectedExportTargets() or ["airway", "external"],
        )
        if not config["volumes"]:
            slicer.util.warningDisplay("No volumes found in the input folder.")
            return
        self._runBatch(config, exportFormats)

    def onRunBatchClicked(self):
        """Advanced run driven by a JSON template."""
        templatePath = self.batchTemplateLineEdit.text.strip()
        if not os.path.isfile(templatePath):
            slicer.util.warningDisplay("Batch template not found: " + templatePath)
            return
        try:
            config = BatchProcessorLib.loadConfig(templatePath)
        except Exception as exc:  # noqa: BLE001
            slicer.util.errorDisplay("Could not read batch template:\n" + str(exc))
            return
        if not config["volumes"]:
            slicer.util.warningDisplay("The batch template lists no volumes.")
            return
        if not config["output_root"]:
            folder = qt.QFileDialog.getExistingDirectory(self, "Select the batch output folder")
            if not folder:
                return
            config["output_root"] = folder
        self._runBatch(config, self._exportFormatFromNames(config["export_formats"]))

    def _runBatch(self, config, exportFormats):
        """Shared batch driver for both folder-in/folder-out and template runs."""
        self.currentInfoTextEdit.clear()
        self._setApplyVisible(False)
        self._setProgressVisible(True)
        self.batchProgressBar.setVisible(True)
        self._showProgress(Progress.RunProgress(len(config.get("volumes", []))).snapshot())
        try:
            if not self._ensureReadyToRun():
                return
            if not self._dependencyChecker.downloadWeightsIfNeeded(self.onProgressInfo):
                return
            deviceKwargs = Dependencies.parameterDeviceKwargs(self.deviceComboBox.currentText)
            processor = BatchProcessorLib.BatchProcessor(
                self.nnUnetFolder(), self.exportSegmentation, self.onProgressInfo
            )
            results = processor.run(config, exportFormats, deviceKwargs=deviceKwargs,
                                    onStage=self._showProgress)
        except Exception as exc:  # noqa: BLE001
            slicer.util.errorDisplay("Batch processing failed:\n" + str(exc))
            return
        finally:
            self._setApplyVisible(True)
            self.batchProgressBar.setVisible(False)

        ok = sum(1 for r in results if r["status"] == "ok")
        failed = [r for r in results if r["status"] != "ok"]
        message = "Batch complete: " + str(ok) + "/" + str(len(results)) + " volume(s) succeeded."
        if failed:
            lines = ["- " + os.path.basename(r["volume"]) + ": " + str(r["error"]) for r in failed]
            message += "\nFailed:\n" + "\n".join(lines)
        slicer.util.infoDisplay(message)

    @staticmethod
    def _exportFormatFromNames(names):
        mapping = {"STL": ExportFormat.STL, "OBJ": ExportFormat.OBJ,
                   "NIFTI": ExportFormat.NIFTI, "NRRD": ExportFormat.NRRD}
        selected = ExportFormat(0)
        for name in names:
            fmt = mapping.get(str(name).strip().upper())
            if fmt:
                selected |= fmt
        return selected if selected != ExportFormat(0) else (ExportFormat.STL | ExportFormat.NIFTI)

    @staticmethod
    def _exportFormatNames(exportFormats):
        """Flag -> list of format names (for the batch config's informational export list)."""
        names = []
        for name, fmt in (("STL", ExportFormat.STL), ("OBJ", ExportFormat.OBJ),
                          ("NIFTI", ExportFormat.NIFTI), ("NRRD", ExportFormat.NRRD)):
            if exportFormats & fmt:
                names.append(name)
        return names

    # ------------------------------------------------------------------ nnU-Net glue --
    @staticmethod
    def isNNUNetModuleInstalled():
        try:
            import SlicerNNUNetLib  # noqa: F401
            return True
        except ImportError:
            return False

    def _createSlicerSegmentationLogic(self):
        if not self.isNNUNetModuleInstalled():
            return None
        from SlicerNNUNetLib import SegmentationLogic
        return SegmentationLogic()

    def _connectSegmentationLogic(self):
        if self.logic is None:
            return
        self.logic.progressInfo.connect(self.onProgressInfo)
        self.logic.errorOccurred.connect(self.onInferenceError)
        self.logic.inferenceFinished.connect(self.onInferenceFinished)

    @classmethod
    def nnUnetFolder(cls):
        fileDir = Path(__file__).parent
        return fileDir.joinpath("..", "Resources", "ML").resolve()
