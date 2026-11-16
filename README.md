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
- **One-click dependencies:** one button reads the hardware (NVIDIA generation and driver,
  Apple Silicon, or no usable GPU), installs the PyTorch build that suits it together with
  nnU-Net, replaces a PyTorch that cannot work on the machine, and checks the result with a
  test computation on the GPU.
- **Agentic AI support:** an MCP server lets Claude, Codex, Gemini CLI, Cursor and similar
  clients run the extension from a request in plain language. It is off until you switch it
  on. AUAE does not anonymise what the AI reads: see the Disclaimer.

## Requirements

- 3D Slicer 5.10 or newer. AUAE is developed and tested on 5.10, and the guide below is
  written for it. The code uses only interfaces that are also present in 5.12, the current
  stable release, but it has not been run there yet.
- The **NNUNet** Slicer extension, installed from the Extensions Manager. AUAE adds the
  **PyTorch** extension by itself when it is missing.
- For fast inference, an NVIDIA GPU from the GTX 900 series onwards with a current driver, or
  an Apple Silicon Mac. Any other computer runs on the CPU, which works but is slow.

The Python packages (`torch`, `nnunetv2`, `scipy`, `PyGithub`) install on demand from inside
the module; you do not install them by hand. They go into the Python that ships inside Slicer,
which is separate from any Python installed on your computer: a `pip install` typed in a
terminal does not reach it. No Python has to be installed on the computer for AUAE.

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
2. Expand **Dependencies & CUDA**. The first lines name the GPU that was found and the PyTorch
   build it needs.
3. Click **Install dependencies** and confirm. AUAE installs the PyTorch extension if needed,
   the PyTorch build for this computer, nnU-Net, and the two helper packages. The PyTorch
   download is 2 to 3 GB.
4. If Slicer has to restart in between, accept. AUAE reopens after the restart and continues
   the installation without further clicks.
5. The log ends with the device inference will use. **Check** prints the full report at any
   time; it should end with `READY to run`.

Clicking **Apply** on a computer where something is missing starts the same installation, so
the button is not a required step.

Which PyTorch build is installed:

| Computer | Build | Inference runs on |
| --- | --- | --- |
| NVIDIA RTX 50 series (sm_120) | CUDA 12.8 (`cu128`) | GPU |
| NVIDIA GTX 900, GTX 10, GTX 16, RTX 20, RTX 30, RTX 40 | CUDA 12.6 (`cu126`) | GPU |
| The same cards with a driver older than 528 (Windows) or 525 (Linux) | CUDA 11.8 (`cu118`) | GPU |
| NVIDIA GTX 700 series and older | CPU build | CPU |
| Apple Silicon Mac | standard build | GPU (MPS) |
| AMD or Intel graphics, or no GPU | CPU build | CPU |

A PyTorch that is already installed and works is left as it is, even when it is not the build
from this table. It is replaced only when it cannot work on the computer: a CPU-only build next
to a usable NVIDIA GPU, a build without kernels for the GPU, `torch` and `torchvision` from
different builds, or an install that no longer imports. The replacement uninstalls the old
packages and deletes the folders pip leaves behind (`~orch`, `torchgen` and the like) before it
installs the new build.

### 5. Run a segmentation

1. Load a CT or CBCT with the `DATA` or `DCM` button.
2. Select it in the input selector at the top.
3. Click **Apply**. The model weights download the first time, then inference starts.
4. When it finishes, export the airway from **Export segmentation**.

### Troubleshooting

- **"This module depends on the NNUNet module":** install NNUNet from the Extensions Manager
  (step 2) and restart.
- **Inference is slow / no GPU:** open **Dependencies & CUDA** and click **Check**. A line that
  starts with `PROBLEM` says what is wrong with the installed PyTorch; **Install dependencies**
  repairs it. If the report says the GPU is present but PyTorch cannot use it, update the
  NVIDIA driver and click **Install dependencies** again.
- **"PyTorch is in use by this Slicer session":** PyTorch is loaded once a segmentation has run
  and its files cannot be replaced until Slicer restarts. Accept the restart; the installation
  continues when AUAE reopens.
- **The installation stops with `[WinError 206]` (file name too long):** this is PyTorch 2.13
  on Windows, whose files go past the path-length limit inside a default Slicer folder. AUAE
  installs a release below 2.13 on Windows for this reason. If another module installed 2.13,
  click **Install dependencies** to replace the broken install.
- **"Type parameter ~_R without a default follows type parameter with a default":** this
  appears when PyTorch is loaded in a Slicer session in which pip was loaded first, which the
  PyTorch Utils panel and the stock PyTorch installer do. pip carries its own copy of
  `typing_extensions`, and that copy takes over the type checks PyTorch relies on. The files
  on disk are fine. AUAE installs PyTorch in a separate process so that it never causes this,
  and repairs the checks before each run when another module caused it. If the message already
  appeared in the session, restart Slicer once.
- **Another extension changed PyTorch:** modules that install their own `torch` can leave a
  CPU-only build or a `torchvision` that does not match. **Check** reports it and **Install
  dependencies** puts back a matching pair.
- **Weights fail to download:** check the network connection. The weights come from the
  upstream GitHub release.

#### Newer GPUs (RTX 50-series / sm_120) and the PyTorch build

**Symptom.** At startup, or when AUAE runs its preflight, you see a message like:

> NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible with the current
> PyTorch installation. The current PyTorch install supports CUDA capabilities sm_50 sm_60
> sm_70 sm_75 sm_80 sm_86 sm_90.

**Cause.** The RTX 50-series (NVIDIA Blackwell) is compute capability **sm_120**. The PyTorch
build that Slicer's PyTorch extension picks by default is chosen from the driver version alone
and is compiled for older architectures, up to sm_90, so it has no kernels for the card. Only a
PyTorch built for **CUDA 12.8** (torch **2.7 or newer**) ships the sm_120 kernels.

**Fix.** Click **Install dependencies**. AUAE reads the compute capability of the card,
removes the build without sm_120 kernels, cleans the leftover folders, and installs the CUDA
12.8 build. If a segmentation was already attempted in the session, Slicer restarts once in
between. Until the fix is applied, `Auto` runs on the CPU instead of crashing halfway with a
"no kernel image is available" error.

**The same fix by hand**, should the automatic one fail:

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
5. Restart Slicer, open AUAE, expand **Dependencies & CUDA**, and click **Check**. The report
   should now say that a test computation on the GPU succeeded.

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
post-processing. While it runs, a progress bar under the **Stop** button shows how far the
segmentation is, with the stage in hand next to it (preparing the volume, preprocessing,
inference, reading the result, post-processing) and, once it can be estimated, the time left.
In a batch the same bar covers the whole batch and names the case being processed.

The **Post-processing** section controls the cleanup:

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
- `--device`: `auto`, `cuda`, `mps`, or `cpu` (default `auto`). A GPU that PyTorch cannot use
  falls back to the CPU.
- `--mode`: `segment` or `extend`. Without it the mode is `segment`, or the one set in the
  template.
- `--sinus` / `--no-sinus`: frontal sinus recovery, on by default.
- `--external`, `--merge-external`: the external face-air segment and its merge.
- `--keep-largest`, `--no-remove-islands`: island cleanup.
- `--smoothing 0..1`, `--min-island MM3`, `--min-internal MM3`: post-processing thresholds.
- `--extend MM`: inferior extension length in millimetres (0 = off).
- `--install-dependencies`: install or repair PyTorch, nnU-Net and the helper packages first,
  the same way the **Install dependencies** button does. If it reports that a restart is needed,
  run the command again. It can be used alone, without an input.
- `--check-dependencies`: print the dependency report. Alone, the exit code is `0` when ready.
- `--log FILE`, `--summary FILE.json`: also write the log lines and the per-case results
  (status, output folder, error, airway volume) to files.
- `--progress FILE.json`: keep a file up to date with the progress of the run: cases done, the
  case and stage in hand, the fraction of the case and of the whole run, and the estimated time
  left.

`--input` also accepts a single volume file. In a template, a `volumes` entry can be an object
such as `{"path": ".../scan.nrrd", "name": "P001/pre"}` to choose the output subfolder of that
case, which keeps a patient / time-point layout in the results.

The exit code is `0` when every case succeeded, `1` when at least one failed, and `2` on a fatal
error (bad arguments, missing NNUNet extension, no inputs, or missing weights), so it slots into a
shell pipeline.

## Use from an agentic AI (MCP server)

`MCP/auae_mcp.py` is a Model Context Protocol server. With it an agentic AI can run AUAE on
your behalf: you describe the job in words and the AI, called the assistant below, calls the
extension. MCP is an open protocol, so the server is not tied to one vendor. It works with
Claude Code, Claude Desktop (Cowork), OpenAI Codex, Gemini CLI, Cursor and any other client
that can start a local MCP server.

The server is thin on purpose. It exposes what the extension does, with every option as a
parameter, and it can describe what an input folder contains. Decisions about your data stay
with you and the assistant: which scans belong to which patient, which one is pre or post
operative, how the folders are named. The assistant sorts files with its own file tools, not
through AUAE.

### Which Python runs what

Slicer brings its own Python, separate from any Python installed on your computer. AUAE uses
only Slicer's, for the extension and for the server, so nothing has to be installed outside
Slicer.

| | The extension | The MCP server |
| --- | --- | --- |
| Started by | The Slicer application | The assistant, as `PythonSlicer MCP/auae_mcp.py`: Slicer's Python alone, without the application |
| What it does | Segmentation on the GPU, post-processing, export | Receives the assistant's requests and starts Slicer, without its window, for each job |
| Packages | `torch`, `nnunetv2`, `scipy`, `PyGithub` | `mcp`, plus `pydicom` to read DICOM headers |
| Installed by | **Install dependencies** | **Install MCP server** |
| Where they live | Slicer's `site-packages` | `~/.auae_mcp/site`, a folder only the server reads |

The server's packages stay out of Slicer's `site-packages` because `mcp` pins recent versions
of libraries that Slicer and its extensions also use, and installing it there could break
them.

The `python` and `pip` commands of a terminal belong to your computer's Python, if there is
one, and never to Slicer's. Do not use them to install `torch`, `nnunetv2` or `mcp` for AUAE.
Slicer would not see the first two, and the third would upgrade libraries your other Python
programs share. The two buttons put each package where it belongs.

The server needs no GPU and no PyTorch. It exchanges plain files with the Slicer it starts: a
job description goes in, a log and a results file come out.

### Setup

1. In Slicer, open AUAE and check that a segmentation runs (see "First launch and
   dependencies"). The server drives that same installation.
2. Expand **LLM API support (MCP)** and click **Install MCP server**. A dialog lists the
   clients it can set up (Claude Code, Codex, Claude Desktop, Gemini CLI, Cursor), with the ones
   missing from the computer greyed out. Tick those in which you want the server registered;
   nothing is ticked in advance and a client you leave unticked is not touched. The server's
   packages are installed in any case. Claude Code and Codex are registered through their own
   `mcp add` commands, the other three through their configuration files: the file is saved
   first with a `.auae-backup` suffix and only the `auae` entry is written into it. A
   registration made earlier under the name `auae` is replaced. Restart the clients that were
   open.
3. Tick **Allow an agentic AI to run AUAE**. The support is off by default and the choice is
   remembered between sessions. While it is off, the server answers only `environment` and
   refuses everything else. Ticking it shows a disclaimer on patient data that you have to
   accept (see "Patient data and responsibility").
4. Ask the assistant to check the AUAE environment. It should answer that the support is on
   and name the Slicer it found.

To register by hand instead, every client needs the same two things: the command, which is
Slicer's Python, and one argument, the server script. The panel shows both with your own paths
filled in. On Linux and macOS the interpreter is `bin/PythonSlicer`, without `.exe`.

Claude Code (`claude mcp list` should then show `auae` as connected):

```
claude mcp add --scope user auae -- "<Slicer folder>/bin/PythonSlicer.exe" "<repository>/MCP/auae_mcp.py"
```

OpenAI Codex, from the command line or as a table in `~/.codex/config.toml`, which the Codex
CLI and its IDE extension share:

```
codex mcp add auae -- "<Slicer folder>/bin/PythonSlicer.exe" "<repository>/MCP/auae_mcp.py"
```

```toml
[mcp_servers.auae]
command = "<Slicer folder>/bin/PythonSlicer.exe"
args = ["<repository>/MCP/auae_mcp.py"]
```

Claude Desktop (`claude_desktop_config.json`), Gemini CLI (`~/.gemini/settings.json`) and
Cursor (`~/.cursor/mcp.json`) take the same JSON entry:

```json
{
  "mcpServers": {
    "auae": {
      "command": "<Slicer folder>/bin/PythonSlicer.exe",
      "args": ["<repository>/MCP/auae_mcp.py"]
    }
  }
}
```

Any other client that starts local MCP servers over stdio is configured with the same command
and argument. The ChatGPT apps are the exception: they connect to remote servers over HTTPS and
do not start local programs, so they cannot reach this server. With OpenAI models, use Codex.

The server can also run on a Python of your own, 3.10 or newer, if you prefer to keep it apart
from Slicer. Give it a virtual environment, so that `mcp` does not upgrade the libraries of
your other programs, and register that environment's Python in place of `PythonSlicer`:

```
python -m venv "%USERPROFILE%\.auae_mcp\venv"
"%USERPROFILE%\.auae_mcp\venv\Scripts\python.exe" -m pip install -r "<repository>\MCP\requirements.txt"
```

The server looks for Slicer in its default install folders and takes the newest version. If
yours is elsewhere, set the `AUAE_SLICER` environment variable to the full path of the Slicer
executable (`"env": { "AUAE_SLICER": "..." }` in the Claude Desktop entry, or a system
environment variable for Claude Code).

### Working with it

Requests are written in plain language. A session on a folder of raw data can go like this:

- "Look at `D:\study\raw` and tell me which scans are there, by patient and date."
- "Copy them into `D:\study\sorted`, one folder per patient with `pre` and `post` subfolders.
  The earlier scan is pre. Show me the plan before you copy anything."
- "Segment every scan in `D:\study\sorted` with AUAE: STL and NIfTI, sinus recovery on,
  external air merged into the airway, 40 mm extension. Put the results in `D:\study\results`."
- "Make a table of the airway volumes, pre against post."

Behind the first request is `describe_inputs`. The copy in the second is done by the assistant
itself; ask for the plan first, so that it lists what it would do and waits for your answer.
The third becomes one `segment` call that lists the scans and names an output subfolder for
each, so the results mirror your layout:

```
results/
  P001/
    pre/    <scan>_Segmentation_Airway.stl   <scan>_Segmentation.nii.gz   airway_volume.txt
    post/   ...
  P002/
    pre/    ...
```

A run is a background job. `segment` returns a job id at once and the assistant then follows
the job with `wait_for_job`, which returns when the job ends or, after a timeout of some tens of
seconds, with the progress so far. The progress names the case and the stage in hand (loading,
preprocessing, inference, post-processing, export), gives the percentage of that case and of
the whole job and an estimate of the time left, and comes with a text bar the assistant can
show you, for example:

```
[########----------------] 32%  Case 2 of 5 (P001/post): Inference 62%, about 7 min left
```

In our tests one CBCT of 816 x 816 x 632 voxels took two to
four minutes on an RTX 3070; on the CPU allow up to an hour per scan. The job keeps running if
you close the assistant, and you can ask for the state of the AUAE jobs in a later session. A
scan that fails does not stop the others: the final report gives, for every scan, its status,
output folder, error message and airway volume in mL. Logs and results of the jobs are kept in
`~/.auae_mcp/jobs`.

### Tools

| Tool | What it does |
| --- | --- |
| `describe_inputs` | Lists the volume files and DICOM series under a folder, at any depth, with scan date, modality, descriptions, scanner, image matrix and whether a folder holds a single series. |
| `segment` | Segments a list of scans or every scan in a folder. Parameters below. |
| `extend` | Lengthens saved segmentation labelmaps downwards without running the model, for example after a manual correction. |
| `job_status` | State and progress of a job: cases done, case and stage in hand, percentages, time left, a text progress bar, and the results once it has ended. |
| `wait_for_job` | The same, but it waits: it returns when the job ends or after a timeout, so the assistant can follow a job without asking again and again. |
| `list_jobs`, `cancel_job` | List the recent jobs with their progress, and stop one. |
| `check_dependencies`, `install_dependencies` | The dependency report and the one-click installation, run in Slicer without its window. |
| `environment` | Whether the support is switched on, and where Slicer and the runner were found. |

Parameters of `segment`, which match the options of the panel:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `inputs` or `input_folder` | required, one of the two | A list of volume files and single-series DICOM folders, or a folder to process in full. |
| `output_folder` | `AUAE_output` beside the inputs | Where the per-scan subfolders are written. |
| `names` | derived from the input paths | Output subfolder of each input; may be nested, such as `P001/pre`. |
| `formats` | `STL`, `NIFTI` | Any of `STL`, `OBJ`, `NIFTI`, `NRRD`. |
| `targets` | `airway`, `external` | Any of `airway`, `external`, `merged`. |
| `device` | `auto` | `auto`, `cuda`, `mps` or `cpu`. |
| `recover_sinuses` | on | Add back the paranasal sinuses the model drops. |
| `min_internal_air_mm3` | 250 | Recovered cavities below this size are dropped (cephalostat bubbles). |
| `external_air` | off | Also segment the air in front of the face. |
| `merge_external_air` | off | Join that air to the airway as one fluid domain. |
| `remove_small_islands`, `min_island_mm3` | on, 5.4 | Drop disconnected specks below the threshold. |
| `keep_largest_island` | off | Keep only the largest connected component instead. |
| `smoothing` | 0 | Surface smoothing of the exported mesh, 0 to 1. |
| `extend_mm` | 0 | Inferior extension length in millimetres; 0 leaves it off. |

### Patient data and responsibility

AUAE does not anonymise or filter what the AI sees. The AI drives the extension, so the files
pass through its hands first: it reads the folders you point it at, including the patient
names, IDs and dates stored in DICOM headers and in the names of files and folders, and
`describe_inputs` reports the patient ID and name of each DICOM series so that scans can be
grouped by patient. What the AI reads is sent to the company that provides it.

The responsibility for this lies with the person who uses the AI, not with AUAE or its
authors. Before the support can be switched on, the panel shows this disclaimer and asks you
to accept it. In practice, you are the one who has to:

- de-identify the data before giving the AI access, where that is required. Anonymise the
  DICOM files at the source, or work on volumes converted to NRRD or NIfTI, and use folder and
  file names that carry no names;
- hold the consent and the approvals (ethics committee, institution) that cover this use of
  the data;
- check the terms of the AI provider: where the data is processed, whether it is kept, and
  whether it is used for training;
- comply with the data-protection rules that apply to you, for example the GDPR.

### When something does not work

| What you see | What to do |
| --- | --- |
| The assistant reports that LLM API support is switched off | Tick the box in the AUAE panel (step 3). No restart is needed. |
| The client lists `auae` as failed or disconnected (`claude mcp list`, `/mcp` in Codex) | Click **Install MCP server** again. If you registered a Python of your own, run `"<that python>" -c "import mcp"`; an error means the packages were installed with another Python. |
| `pip` answers "Fatal error in launcher" | That `pip` belongs to a Python that was removed from the computer. It is not needed: use the button, or `python -m pip` for a Python of your own. |
| "3D Slicer was not found" | Set `AUAE_SLICER` to the Slicer executable. |
| A job ends in `error` straight away | Ask the assistant to check the dependencies; the report names what is missing. The job status also carries the end of Slicer's console output. |
| "Found 2 DICOM series in ..." | AUAE takes one series per folder. Ask the assistant to split the series into separate folders. |

## Module and function reference

- **AirwayExtension** is the interface-free array logic. `postprocessSegmentation` is the single
  source of truth for the segmentation step (island cleanup, sinus recovery, external air,
  inferior extension, optional merge). `extendSegmentation` is the separate extension step and
  preserves manual edits. The head-envelope functions (`_airThresholdFromAirway`,
  `_sealedHeadMask`, `computeInternalAirCavities`, `computeExternalAirMask`) and the geometry
  helpers (`extendBinaryArray`, `_padCompanionToExtendedGrid`, `inferiorStackSide`) are described
  above. `applySurfaceSmoothing` sets the smoothing factor and rebuilds the closed surface only
  when the factor changes.
- **SegmentationWidget** is the module panel and the glue to the nnU-Net logic. It builds the
  three-step layout, collects the options, wires the embedded editor, and drives export and batch.
  `exportSegmentation(node, folder, formats, targets)` writes the requested segments in the
  requested formats. `_ensureReadyToRun` prints the dependency report and, when something is
  missing or unsuitable, starts the installation described in "First launch and dependencies".
- **BatchProcessor** holds `listVolumesInFolder`, `folderConfig`, `loadConfig`, `caseName`, and
  the `BatchProcessor.run` driver, which handles both the `segment` and the `extend` mode and
  removes the scene nodes of each case when it ends.
- **Dependencies** handles the installation. `detectHardware` reads the GPU through `nvidia-smi`
  before PyTorch exists, `recommendedBuild` maps it to a PyTorch build, and `torchAction` decides
  from the installed files (without importing torch) whether to install, replace, or leave it.
  `ensure` carries out the steps through the PyTorch and NNUNet extensions and resumes after a
  restart. `torchStatus` checks the result in a separate Python process, with a small convolution
  on the GPU, and keeps the answer until torch, the GPU or the driver change. `report` prints the
  table and `resolveDevice` turns `Auto` into `cuda`, `mps` or `cpu`.
- **PythonDependencyChecker** downloads the model weights from the upstream GitHub release.
- **Progress** reads the lines a run logs and turns them into a stage and a fraction.
  `CaseProgress` follows one case, `RunProgress` a run of several with the time left, and
  `describe` / `formatBar` put a snapshot into words. The panel, the command-line runner
  (`--progress`) and the MCP server all use it, so they report the same numbers.
- **LlmApi** holds the opt-in switch of the LLM API support (`isEnabled`, `setEnabled`), the
  disclaimer shown before it is switched on, and the server installation: `installServerPackages` puts `mcp` and `pydicom` into `~/.auae_mcp/site`
  with Slicer's Python, `detectClients` lists the AI clients found on the computer,
  `registerWith` adds the server to one of them, and `setupServer` installs the packages and
  registers the server with the clients the user ticked.
- **MCP/auae_mcp.py** is the server, outside the Slicer module. Each tool is a plain function
  (`describe_inputs`, `segment`, `extend`, `job_status` and the others); `_start_job` launches
  Slicer on `CLI/auae_batch.py` and `_job_state` reads the log and the results file back.

## Notes and caveats

- **Intensity domain.** The model expects CT-like Hounsfield units and normalises with a fixed
  window. A CBCT with air near 0 rather than near -1000 falls outside that window and is
  under-segmented, which is the usual reason turbinates and sinuses go missing. The sinus recovery
  helps, but a volume already in Hounsfield units, or a model trained on your own CBCT domain, is
  the reliable path.
- **GPU.** The build table reflects the PyTorch wheels available in November 2026: CUDA 12.6 is
  the last line with kernels for the GTX 900 and GTX 10 series, and CUDA 12.8 the first for the
  RTX 50 series. AMD and Intel graphics are not used; those computers run on the CPU. On Linux
  the replace step also removes the separate `nvidia-*` CUDA runtime packages of the old build.
- **Cephalostat bubbles.** The bubble threshold is a best-effort size filter. If a hypoplastic
  frontal sinus is smaller than expected, or a bubble is larger, adjust `min_internal_air_mm3` or
  correct the mask in the editor.

## Disclaimer

When an agentic AI (an LLM such as Claude, Codex or Gemini) runs AUAE
through the MCP server, it reads the files in the folders you point it at, patient names, IDs
and dates included, and that content is sent to the company that provides the AI. AUAE does
not anonymise or filter it. Whoever uses the AI is responsible for that use: for de-identifying
the data beforehand where required, for the consent and approvals that cover it, for the terms
of the AI provider, and for complying with the applicable data-protection rules, such as the
GDPR. The authors of AUAE take no responsibility for data disclosed to an AI or to its
provider. The support is off by default and is switched on only after this disclaimer is
accepted in the panel.

The software is provided "as is", without warranty of any kind, under the terms of the
Apache License 2.0 (see `LICENSE.md`).

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
