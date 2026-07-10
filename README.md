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

AUAE takes a CT or CBCT volume (HU calibrated), runs the upstream nnU-Net model to segment the upper airway
(nasal cavity, nasopharynx, oropharynx), cleans the mask, and lets you export it as STL, OBJ,
or NIFTI. On top of the upstream extension, AUAE adds:

- **A three-step workflow:** *segment* (Apply), *refine*, *extend*. Segmentation and airway
  extension are separate steps, so the mask can be corrected in between.
- **Embedded Segment Editor:** a Slicer Segment Editor sits between segmentation and
  extension for manual refinement (paint, erase, islands, mask smoothing) before you extend.
- **Airway extension for flow modelling:** extend the final mask past the scanned field of
  view by repeating its terminal slice, inferiorly or superiorly, by a set number of
  millimetres. This closes the airway domain for downstream flow / CFD work.
- **External face-air segment:** optionally segment the ambient air in front of the face as a
  second segment (the inlet volume for CFD), with a flag to merge it and the airway into a
  single fluid-domain segment.
- **Two island-cleanup options** that cannot both be on: *remove small islands* or *keep the
  largest island only*, plus a non-destructive **surface-smoothing** slider for the exported mesh.
- **Batch processing:** point a JSON template at a list of volumes and get one output
  subfolder per case, each with its segmentation and meshes.
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

## Use the extension

### One volume (three steps)

1. Load a CT or CBCT and pick it as the input.
2. In **Post-processing**, choose an island option (and, if you want a CFD inlet, *Segment
   external face air*, optionally *Merge external air into airway*), then click **Apply** to
   segment (step 1).
3. In **Segment editor (refine)**, correct the mask if needed with the paint, erase, islands,
   or smoothing tools (step 2). Use the **Surface smoothing** slider to smooth the exported mesh.
4. In **Airway extension**, set a direction (inferior or superior) and a length, then click
   **Run airway extension** to extend the selected segmentation (step 3).
5. Export the result from **Export segmentation**.

### A batch

1. Open **Batch processing**. It points to the bundled template at
   `Resources/batch_template.json`.
2. Copy the template, list your volumes under `volumes`, and set `output_root`, the export
   formats, the island / external-air / smoothing options, and the extension block.
3. Click **Run batch**. Each volume is written to its own subfolder under `output_root`.

Every field is described inside the template file.

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
