<div align="center">

<img src="AUAE/Resources/Icons/AUAE_full_icon.png" width="140" alt="AUAE logo"/>

# Automatic Upper Airways Extension (AUAE)

Automatic upper-airway segmentation on CT and CBCT in 3D Slicer, with mask extension for flow modelling, batch processing, and self-installing dependencies.

</div>

> [!IMPORTANT]
> AUAE is based on a previous project, made by other authors and available at: (https://github.com/capenaka/SlicerUpperAirwaySegmentator). 
> The segmentation model and the method were created by the
> upstream authors. If you use the present  tool, cite their paper (see [Citation](#citation)).
> AUAE adds tooling around that model; it does not replace or retrain it.

## What it does

AUAE takes a CT or CBCT volume, runs the upstream nnU-Net model to segment the upper airway
(nasal cavity, nasopharynx, oropharynx), cleans the mask, and lets you export it as STL, OBJ,
NIFTI, or NRRD. On top of the upstream extension, AUAE adds:

- **A three-step workflow:** *segment* (Apply), *refine*, *extend*. Segmentation and airway
  extension are separate steps, so the mask can be corrected in between.
- **Embedded Segment Editor:** a Slicer Segment Editor sits between segmentation and
  extension for manual refinement (paint, erase, islands, mask smoothing) before you extend.
- **Frontal sinus recovery:** on by default. The model often drops the paranasal sinuses;
  AUAE finds the air enclosed inside the head and folds it back into the airway, while removing
  the smaller cephalostat air bubbles by size.
- **Inferior airway extension for flow modelling:** grow the final mask towards the neck, past
  the scanned field of view, by a set number of millimetres. This closes the airway domain for
  downstream flow / CFD work. The direction is always inferior; only the length is variable.
- **External face-air segment:** optionally segment the ambient air in front of the face as a
  second segment (the inlet volume for CFD), with a flag to merge it into a single fluid-domain
  segment. When merged, the extension is applied to the airway before the merge.
- **Island cleanup and smoothing:** *remove small islands* or *keep the largest island only*
  (mutually exclusive), plus a non-destructive **surface-smoothing** slider for the exported mesh.
- **Flexible export:** STL, OBJ, NIFTI, NRRD, and independent targets (airway, external air, or a
  single merged file).
- **Batch processing:** a classic folder-in / folder-out run, or a JSON template, writing one
  output subfolder per case.
- **Dependency preflight:** it installs and checks everything before a run, reports the
  installed versions, and warns before a run when the GPU is newer than the installed torch
  build supports (for example an RTX 50-series), instead of crashing mid-inference.

## Requirements

- 3D Slicer 5.10 (the guide below is written for it).
- The **NNUNet** Slicer extension, installed from the Extensions Manager. It also brings in
  the **PyTorch** extension, which supplies a torch build that matches your CUDA setup.
- An NVIDIA GPU with a current driver for fast inference. CPU works but is slow.

The Python packages (`torch`, `nnunetv2`, `scipy`, `PyGithub`) install on demand from inside
the module; you do not install them by hand.

## Install and launch on Slicer 5.10

AUAE is not published to the Slicer Extensions Manager store. You install it directly from
this repository.

### 1. Install 3D Slicer 5.10

Download and install Slicer 5.10 from https://download.slicer.org, then open it.

### 2. Install the NNUNet extension (needed for every method)

In `View > Extensions Manager > Install Extensions`, search for **NNUNet**, install it, and
restart. It runs the nnU-Net inference and pulls in the PyTorch extension.

### 3. Add AUAE to Slicer (pick one method)

#### Method A: Extension Wizard (recommended, from a clone)

1. Clone or download the repository:
   ```bash
   git clone https://github.com/pietro98vr/SlicerAUAE.git
   ```
2. In Slicer, open `Developer Tools > Extension Wizard`.
3. Click `Select Extension` and choose the repository root (the folder that holds
   `CMakeLists.txt`).
4. Accept when it offers to load the module. AUAE appears under `Segmentation`.

#### Method B: Additional module path (ZIP or clone)

1. Download `SlicerAUAE-<version>.zip` from the Releases page (or use `Code > Download ZIP`),
   unzip it somewhere stable, or clone the
   repository.
2. Open `Edit > Application settings > Modules`, and next to *Additional module paths* click
   `Add`.
3. Select the module folder inside the repository:
   ```
   <clone>/AUAE
   ```
4. Click `OK` and restart Slicer.

### 4. First launch and dependencies

1. Open the module from `Modules > Segmentation > Automatic Upper Airways Extension (AUAE)`,
   or search `AUAE` in the module finder.
2. Expand **Dependencies & CUDA** and click **Install / update dependencies**. This installs
   torch (with the right CUDA build), nnU-Net, and the two helper packages.
3. If it installs the PyTorch extension, Slicer asks to restart. Restart, reopen the module,
   and click the button once more.
4. Click **Check dependencies**. The log should end with `READY to run` and, on a GPU
   machine, a line such as `GPU inference AVAILABLE`.

### 5. Run a segmentation

1. Load a CT or CBCT with the `DATA` or `DCM` button.
2. Select it in the input selector at the top.
3. Click **Apply**. The model weights download the first time, then inference starts.
4. When it finishes, export the airway from **Export segmentation**.

### Troubleshooting

- **"This module depends on the NNUNet module":** install NNUNet from the Extensions Manager
  (step 2) and restart.
- **Inference is slow / no GPU:** open **Dependencies & CUDA** and read the report. If it
  says the torch build is CPU-only, reinstall the dependencies so the PyTorch extension
  fetches a CUDA build, then restart. Confirm your NVIDIA driver is current.
- **Nothing happens after installing the PyTorch extension:** Slicer must restart once before
  torch is usable. Restart, then run again.
- **Weights fail to download:** check the network connection. The weights come from the
  upstream GitHub release.

#### Newer GPUs (RTX 50-series / sm_120) and the PyTorch build

**Symptom.** At startup, or when AUAE runs its preflight, you see a message like:

> NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible with the current
> PyTorch installation. The current PyTorch install supports CUDA capabilities sm_50 sm_60
> sm_70 sm_75 sm_80 sm_86 sm_90.

AUAE detects this and blocks the GPU run on purpose, so the inference does not crash halfway
with a "no kernel image is available" error. Until it is fixed, the GPU is unusable and you fall
back to CPU (slow).

**Cause.** The RTX 50-series (NVIDIA Blackwell) is compute capability **sm_120**. The PyTorch
build that Slicer's PyTorch extension installs by default is compiled for older architectures,
up to sm_90, so it has no kernels for your card. Only a PyTorch built for **CUDA 12.8** (torch
**2.7 or newer**) ships the sm_120 kernels.

**Fix on Windows.** Install a CUDA 12.8 torch build from inside Slicer:

1. Open the **PyTorch Utils** module (it comes with the PyTorch extension).
2. Uninstall the current torch there. Note where Slicer keeps its packages first, by running this
   once in the Python console:
   ```python
   import sysconfig; print(sysconfig.get_paths()["purelib"])
   ```
3. **Close Slicer, then delete the leftover torch folders by hand.** `pip uninstall` often leaves
   directories behind, and if the old `torchgen` folder (and `torch`, `torchvision`, `functorch`)
   is still there, the new install fails with a `torchgen` error. In the folder printed above,
   remove `torch`, `torchgen`, `torchvision`, `functorch`, their matching `*.dist-info` folders,
   and any half-removed leftover that pip renamed to start with a tilde (for example a folder
   named `~orch`).
4. Reopen Slicer. In PyTorch Utils, instead of letting it search automatically, choose the
   **CUDA 12.8 (cu128)** backend and install. It is a large download, around 4 GB. From the
   Python console the same thing is, for example:
   ```python
   import PyTorchUtils
   PyTorchUtils.PyTorchUtilsLogic().installTorch(askConfirmation=True, forceComputationBackend="cu128")
   ```
5. Restart Slicer, open AUAE, expand **Dependencies & CUDA**, and click **Check dependencies**.
   The report should now list `sm_120` under the torch architectures and say
   `GPU inference AVAILABLE`.

To confirm by hand, run `import torch; torch.cuda.get_arch_list()` in the Python console; the
list must include `sm_120`.

**Caveat on Linux.** As of mid-2025 the official Slicer Linux package is built on an old base
(CentOS 7, glibc 2.17). The torch 2.7+ wheels that carry sm_120 need a newer glibc (2.28), so
they do not run on the packaged Slicer. On Linux an RTX 50-series card therefore cannot be used
from the stock Slicer package until its base is updated; use Windows, build a compatible torch
yourself, or run on CPU.

**CPU fallback.** Set **Inference device** to **CPU** in Dependencies & CUDA. A run takes up to
about an hour on CPU, but it works and produces the same result.

## Use the extension

The module panel is laid out in the order you use it: pick the input, segment, refine, extend,
export. Two collapsed sections at the bottom hold batch processing and the dependency setup.

### 1. Input and segmentation

Choose the CT or CBCT in the first selector. The second selector picks the segmentation node to
write into; the "create new segmentation" entry is available, so you can keep several results in
one scene.

Press **Apply** to run the model. It runs the nnU-Net inference, then hands the raw mask to the
post-processing. The **Post-processing** section controls the cleanup:

- **Remove small islands** drops components below a physical-volume threshold. On by default.
- **Keep largest island only** keeps a single component. For the upper airway this drops the
  nasal cavity and the sinuses, so leave it off unless you want the pharyngeal airway alone. The
  two island options are mutually exclusive.
- **Attempt frontal sinus segmentation** is on by default. It recovers the paranasal sinuses the
  model misses and folds them into the airway, dropping the smaller cephalostat air bubbles by
  size first.
- **Segment external face air** adds a second segment with the ambient air in front of the face
  (the CFD inlet). When it is on, **Merge external air into airway** unions the two into one
  fluid-domain segment.
- **Surface smoothing** sets the mesh smoothing factor, from 0 (raw voxels) to 1 (very smooth). It
  changes the 3D model and the STL/OBJ output, not the labelmap.

### 2. Refinement

The **Segment editor (refine)** section is a standard Slicer Segment Editor between segmentation
and extension. Use it to paint, erase, or clean the mask. For airway masks, the Smoothing effect
with Median removes voxel noise while keeping the lumen, and Closing fills small holes. Edits here
are preserved, because the extension is a separate step.

### 3. Inferior extension

The **Airway extension** section grows the final mask towards the neck so the domain is closed for
flow work. The direction is always inferior, so only the length in millimetres is variable. Press
**Run airway extension** to apply it to the selected segmentation in place.

### 4. Export

The section has two groups. Tick what you want in each, then press **Export**:

- **What to export** (independent): **Airway**, **External air** (skipped if it was not
  segmented), and **Merged** (all segments in one file).
- **File format**: **STL** and **OBJ** are surface meshes; **NIFTI** and **NRRD** are labelmaps.

Each chosen target is written in each chosen format, into a folder you pick.

## How the post-processing works

All of this lives in `AirwayExtension` and runs on NumPy arrays in Slicer's KJI order. The mask is
exported to a labelmap on the reference volume geometry first, so the full slice stack is available.

### Island cleanup

`keepLargestConnectedComponent` keeps the single largest component. `removeSmallIslands` drops
components below a volume threshold in cubic millimetres, so the threshold is correct at any voxel
spacing. Cleanup runs on the airway before any sinus or external air is considered.

### The head-envelope decomposition

The sinus recovery and the external air share one idea. `_airThresholdFromAirway` reads the
intensities under the known airway mask and sets an air threshold from them. Because the airway
lumen is air, this holds whether the volume is in Hounsfield units (air near -1000) or
intensity-shifted (air near 0), with no fixed value assumed.

`_sealedHeadMask` then builds a solid head: everything brighter than the threshold is tissue or
bone, the largest such component is the patient, a morphological closing seals the narrow openings
(nostrils, mouth, thin sinus ducts), and hole-filling turns the internal airspace into a solid
mask. Inside that envelope sit the sinuses and the airway; outside it is ambient air. From this
single split:

- `computeInternalAirCavities` takes the air inside the envelope that is not already airway (the
  dropped sinuses), removes components below `min_internal_air_mm3` (250 mm^3 by default, which
  clears the cephalostat bubbles), and the result is unioned into the airway.
- `computeExternalAirMask` takes the air outside the envelope, keeps the border-touching
  components, and clips them to the region in front of the airway. Because it works from the
  outside of the sealed head, the frontal sinuses cannot leak into it.

### Extension and the merged case

The extension is applied to the airway alone. `extendBinaryArray` returns the extended array, the
updated geometry, and the padding it added. When the external air is merged, it is unioned after
the extension: `_padCompanionToExtendedGrid` pads it onto the extended grid so it keeps its
position. This guarantees the caudal extension repeats the airway's terminal slice, never the
external air, which can sit lower in the scan.

## Batch processing

The **Batch processing** section runs the same segmentation and post-processing over many volumes,
with two paths.

### Folder in, folder out (the classic way)

Point **Input folder** at a folder of inputs and press **Run batch (folder)**. Each input is
processed with the current Post-processing and Export settings, and each result goes to its own
subfolder. **Output folder** is optional; when empty, results land in an `AUAE_output` folder
beside the inputs.

An input is either a single-file volume (`.nii`, `.nii.gz`, `.nrrd`, `.nhdr`, `.mha`, `.mhd`) or a
DICOM series. For DICOM, put one series per patient in its own subfolder of the input folder; the
files may sit directly in that subfolder or nested deeper (`Patient/Study/Series/*.dcm`), since
detection and import both recurse. One series per patient folder is required: if a folder holds
more than one series, that case fails with a clear error rather than segmenting the wrong series,
so you split it and re-run. A folder that is itself a single series is treated as one case.

### Template (advanced)

A JSON template gives explicit control. The bundled template at `Resources/batch_template.json`
documents every field: `input_folder` or `volumes`, `output_root`, `export_formats` (STL, OBJ,
NIFTI, NRRD), `export_targets` (airway, external, merged), the island options, `include_internal_air`
with `min_internal_air_mm3`, `segment_external_air` and `merge_external_air`, `smoothing_factor`,
and the `extension` block. `mode` is `segment` to run the model, or `extend` to rerun only the
extension on existing segmentation files (no model or GPU needed).

## Headless (command line)

The batch logic needs no graphical interface, so AUAE can run from the command line through
Slicer's headless Python. The runner is `AUAE/CLI/auae_batch.py`, and everything after the `--`
is passed to it:

```
Slicer --no-main-window --python-script <path>/AUAE/CLI/auae_batch.py -- \
    --input /data/cbct --output /data/out --formats STL NRRD \
    --targets airway external --device auto --external
```

On Windows the Slicer launcher is `Slicer.exe`, for example
`"C:\Users\<you>\AppData\Local\slicer.org\3D Slicer 5.10.0\Slicer.exe"`.

The NNUNet extension must be installed and its Python dependencies present (run the module once
from the GUI to install them). The model weights are downloaded on demand when missing.

Arguments:

- `--input FOLDER` or `--template FILE.json`: a folder of volumes and/or DICOM series, or a JSON
  template. One of the two is required.
- `--output FOLDER`: output folder (default: an `AUAE_output` folder beside the input).
- `--formats`: any of `STL OBJ NIFTI NRRD` (default `STL NIFTI`).
- `--targets`: any of `airway external merged` (default `airway external`).
- `--device`: `auto`, `cuda`, or `cpu` (default `auto`).
- `--mode`: `segment` (default) or `extend`.
- `--sinus` / `--no-sinus`: frontal sinus recovery, on by default.
- `--external`, `--merge-external`: the external face-air segment and its merge.
- `--keep-largest`, `--no-remove-islands`: island cleanup.
- `--smoothing 0..1`, `--min-island MM3`, `--min-internal MM3`: post-processing thresholds.
- `--extend MM`: inferior extension length in millimetres (0 = off).

The exit code is `0` when every case succeeded, `1` when at least one failed, and `2` on a fatal
error (bad arguments, missing NNUNet extension, no inputs, or missing weights), so it slots into a
shell pipeline.

## Module and function reference

- **AirwayExtension** is the interface-free array logic. `postprocessSegmentation` is the single
  source of truth for the segmentation step (island cleanup, sinus recovery, external air,
  inferior extension, optional merge). `extendSegmentation` is the separate extension step and
  preserves manual edits. The head-envelope functions (`_airThresholdFromAirway`,
  `_sealedHeadMask`, `computeInternalAirCavities`, `computeExternalAirMask`) and the geometry
  helpers (`extendBinaryArray`, `_padCompanionToExtendedGrid`, `resolveExtensionSideForDirection`)
  are described above.
- **SegmentationWidget** is the module panel and the glue to the nnU-Net logic. It builds the
  three-step layout, collects the options, wires the embedded editor, and drives export and batch.
  `exportSegmentation(node, folder, formats, targets)` writes the requested segments in the
  requested formats. `_ensureReadyToRun` prints the dependency report and blocks a GPU run on the
  architecture conflict described in Troubleshooting.
- **BatchProcessor** holds `listVolumesInFolder`, `folderConfig`, `loadConfig`, and the
  `BatchProcessor.run` / `_runExtend` drivers.
- **Dependencies** is the autonomous preflight: `report` prints a required-versus-installed table
  and a CUDA verdict, `ensure` installs torch and nnU-Net through the NNUNet and PyTorch
  extensions, and `torchStatus` / `_archCompatibility` compare the GPU against the torch build.
- **PythonDependencyChecker** downloads the model weights from the upstream GitHub release.

## Notes and caveats

- **Intensity domain.** The model expects CT-like Hounsfield units and normalises with a fixed
  window. A CBCT with air near 0 rather than near -1000 falls outside that window and is
  under-segmented, which is the usual reason turbinates and sinuses go missing. The sinus recovery
  helps, but a volume already in Hounsfield units, or a model trained on your own CBCT domain, is
  the reliable path.
- **GPU.** See Troubleshooting for the RTX 50-series / sm_120 case. The preflight reports it and
  blocks the GPU run; set the device to CPU to run anyway.
- **Cephalostat bubbles.** The bubble threshold is a best-effort size filter. If a hypoplastic
  frontal sinus is smaller than expected, or a bubble is larger, adjust `min_internal_air_mm3` or
  correct the mask in the editor.

## Credits

**The segmentation model and the upstream extension** are the work of Alejandro Matos
Camarillo, Silvia Capenakas, and Manuel Lagravere at the University of Alberta, with the
co-authors of the paper below. Their repository is
[capenaka/SlicerUpperAirwaySegmentator](https://github.com/capenaka/SlicerUpperAirwaySegmentator).

**This project** (airway extension, batch processing, island options, dependency preflight) is
by Dr. Pietro Montagna (DDS, MSc, PhD Student, pietro.montagna@univr.it) and Dr. Fabio
Lonardi (MD, OMFS, PhD Student, fabio.lonardi@univr.it), Head and Neck Department, Department
of Surgery, Dentistry, Pediatrics and Gynecology, University of Verona, Verona, Italy.

## Citation

If this extension helps your work, cite the upstream paper and nnU-Net:

> Gianoni-Capenakas S, Matos A, Dot G, Schouman T, Chaurasia A, Pliska B, Lagravere M,
> Panithakumar K. Segmentation of the Upper Airway using Deep learning - nnUNet. Journal of
> Dentistry, 2026. https://doi.org/10.1016/j.jdent.2026.106507

> Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring
> method for deep learning-based biomedical image segmentation. Nature Methods,
> 2021;18(2):203-211. https://doi.org/10.1038/s41592-020-01008-z

## License

Apache-2.0, inherited from the upstream project. See [`LICENSE.md`](LICENSE.md) and
[`NOTICE.md`](NOTICE.md). Third-party components are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
