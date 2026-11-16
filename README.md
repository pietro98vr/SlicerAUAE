<div align="center">

<img src="AUAE/Resources/Icons/AUAE_full_icon.png" width="140" alt="AUAE logo"/>

# Automatic Upper Airways Extension (AUAE)

Automatic upper-airway segmentation on CT and CBCT in 3D Slicer, with mask extension for flow modelling, batch processing, and self-installing dependencies.

[Install](#install) · [Use](#use) · [Batch](#batch-processing) · [Command line](#command-line) · [AI agents](#use-from-an-ai-agent) · [Troubleshooting](#troubleshooting) · [How it works](#how-it-works) · [Citation](#citation)

</div>

> [!IMPORTANT]
> AUAE is based on a previous project by other authors,
> [capenaka/SlicerUpperAirwaySegmentator](https://github.com/capenaka/SlicerUpperAirwaySegmentator).
> The segmentation model and the method were created by the upstream authors. If you use this
> tool, cite their paper (see [Citation](#citation)). AUAE adds tooling around that model; it
> does not replace or retrain it.

## What it does

AUAE takes a CT or CBCT of the head, runs the upstream nnU-Net model to segment the upper airway
(nasal cavity, nasopharynx, oropharynx), cleans the result and exports it as a mesh or a
labelmap. Around the model it adds the following.

| Feature | What it adds |
| --- | --- |
| Three steps | Segment, refine by hand in an embedded Segment Editor, extend. The mask can be corrected before it is extended. |
| Sinus recovery | The model often drops the paranasal sinuses. AUAE finds the air enclosed in the head and adds it back, leaving out the small air bubbles of the cephalostat. |
| Inferior extension | Lengthens the airway towards the neck, past the scanned field of view, by a chosen number of millimetres. A flow simulation then has its outlet away from the region of interest. |
| External air | Segments the air in front of the face as the inlet volume, either as a separate segment or merged with the airway. |
| Cleanup and smoothing | Removes small islands or keeps the largest one. A slider smooths the exported surface. |
| Export | STL, OBJ, NIFTI or NRRD, for the airway, the external air, or both in one file. The airway volume in mL is reported with each result. |
| Batch | A folder of scans, or a JSON template, processed in one go from the panel or from the command line. |
| Dependencies | One button reads the GPU and installs the PyTorch build that suits it, together with nnU-Net. |
| AI agents | An MCP server lets Claude, Codex, Gemini CLI or Cursor run the extension from a request in plain language. Off by default. |

## Requirements

- 3D Slicer 5.10 or newer. AUAE is developed and tested on 5.10. It has not been run on 5.12
  yet, although it uses only interfaces that exist there too.
- The NNUNet extension, from Slicer's Extensions Manager. AUAE installs the PyTorch extension by
  itself.
- For fast inference, an NVIDIA GPU from the GTX 900 series onwards with a current driver, or an
  Apple Silicon Mac. Any other computer runs on the CPU, which takes up to an hour per scan.

No Python has to be installed on the computer. The packages AUAE needs go into the Python that
ships inside Slicer, which a `pip install` typed in a terminal does not reach.

## Install

AUAE is not in the Slicer Extensions Manager. It is installed from this repository.

1. Install Slicer 5.10 from https://download.slicer.org.
2. In `View > Extensions Manager`, install **NNUNet** and restart Slicer.
3. Get the code, with `git clone https://github.com/pietro98vr/SlicerAUAE.git` or with
   `Code > Download ZIP` on GitHub.
4. In `Edit > Application settings > Modules`, add the folder `<repository>/AUAE` to
   *Additional module paths* and restart. The Extension Wizard works as well:
   `Developer Tools > Extension Wizard > Select Extension`, then pick the repository root.
5. Open `Modules > Segmentation > Automatic Upper Airways Extension (AUAE)`, expand
   **Dependencies & CUDA** and click **Install dependencies**.

The last step names the GPU that was found and the PyTorch build it needs, then installs
PyTorch, nnU-Net and two helper packages. The PyTorch download is 2 to 3 GB. If Slicer has to
restart in between, accept: AUAE reopens and continues by itself. **Check** prints the full
report at any time and should end with `READY to run`. Pressing **Apply** on a computer where
something is missing starts the same installation.

| Computer | PyTorch build | Inference runs on |
| --- | --- | --- |
| NVIDIA RTX 50 series | CUDA 12.8 (`cu128`) | GPU |
| NVIDIA GTX 900 to RTX 40 | CUDA 12.6 (`cu126`) | GPU |
| The same cards with a driver older than 528 (Windows) or 525 (Linux) | CUDA 11.8 (`cu118`) | GPU |
| NVIDIA GTX 700 series and older | CPU | CPU |
| Apple Silicon Mac | standard | GPU (MPS) |
| AMD or Intel graphics, or no GPU | CPU | CPU |

A PyTorch that already works is left alone, even when it is not the build in the table. It is
replaced only when it cannot work on the computer: a CPU-only build next to a usable NVIDIA GPU,
a build without kernels for the GPU, `torch` and `torchvision` from different builds, or an
install that no longer imports. The old packages and the folders pip leaves behind (`~orch`,
`torchgen`) are removed before the new build goes in.

## Use

The panel follows the order of the work: input, segment, refine, extend, export.

### 1. Segment

Load a CT or CBCT with the `DATA` or `DCM` button, pick it in the first selector and press
**Apply**. The second selector chooses the segmentation node to write into, so several results
can live in one scene. The model weights are downloaded the first time.

While the model runs, a progress bar under **Stop** shows the stage in hand (preparing the
volume, preprocessing, inference, reading the result, post-processing) and, once it can be
estimated, the time left. The log reports the airway volume in mL.

The **Post-processing** section decides how the raw mask is cleaned.

| Option | Default | Effect |
| --- | --- | --- |
| Remove small islands | on | Drops the disconnected components below a volume threshold. |
| Keep largest island only | off | Keeps one component. On the upper airway this drops the nasal cavity and the sinuses, so use it only for the pharyngeal airway alone. It excludes the option above. |
| Attempt frontal sinus segmentation | on | Adds back the paranasal sinuses the model misses, after discarding the small cephalostat air bubbles. |
| Segment external face air | off | Adds a second segment with the air in front of the face. |
| Merge external air into airway | off | Joins the two segments into one fluid domain. |
| Surface smoothing | 0 | From 0 (raw voxels) to 1. It changes the 3D model and the STL and OBJ files, not the labelmap. |

### 2. Refine

**Segment editor (refine)** is a standard Slicer Segment Editor. Paint, erase or clean the mask
there. On airway masks the Smoothing effect with Median removes voxel noise without closing the
lumen, and Closing fills small holes. The edits survive the next step.

### 3. Extend

**Airway extension** lengthens the mask towards the neck by the number of millimetres you set.
The direction is always inferior. **Run airway extension** applies it to the selected
segmentation in place.

### 4. Export

Tick what to export (**Airway**, **External air**, **Merged**) and the formats (**STL** and
**OBJ** are surface meshes, **NIFTI** and **NRRD** are labelmaps), then press **Export** and
choose a folder. Each target is written in each format.

## Batch processing

The **Batch processing** section runs the same segmentation over many scans. A progress bar
covers the whole batch and names the scan in hand.

Point **Input folder** at a folder of scans and press **Run batch (folder)**. Every scan is
processed with the current Post-processing and Export settings and written to its own
subfolder, together with an `airway_volume.txt`. **Output folder** is optional: left empty, the
results go to an `AUAE_output` folder beside the inputs.

A scan is either a volume file (`.nii`, `.nii.gz`, `.nrrd`, `.nhdr`, `.mha`, `.mhd`) or a DICOM
series in its own subfolder. The DICOM files may sit directly in that subfolder or deeper
(`Patient/Study/Series/*.dcm`). Each subfolder must hold one series: with more than one, that
scan fails with an error instead of segmenting the wrong series.

For finer control, **Run batch (template)** takes a JSON file. The bundled
`AUAE/Resources/batch_template.json` documents every field. Its `mode` is `segment`, or `extend`
to lengthen saved segmentations without running the model. A `volumes` entry can be an object
such as `{"path": ".../scan.nrrd", "name": "P001/pre"}` to choose the output subfolder of that
scan.

## Command line

The batch needs no window, so it also runs from a terminal. Everything after `--` goes to the
runner.

```
Slicer --no-main-window --python-script <repository>/AUAE/CLI/auae_batch.py -- \
    --input /data/cbct --output /data/out --formats STL NRRD --targets airway --extend 40
```

On Windows the launcher is `Slicer.exe`, for example
`"C:\Users\<you>\AppData\Local\slicer.org\3D Slicer 5.10.0\Slicer.exe"`.

| Argument | Meaning |
| --- | --- |
| `--input PATH` | A folder of scans, or one volume file. |
| `--template FILE.json` | A batch template, instead of `--input`. |
| `--output FOLDER` | Where the results go. Default: `AUAE_output` beside the input. |
| `--formats` | Any of `STL OBJ NIFTI NRRD`. Default: `STL NIFTI`. |
| `--targets` | Any of `airway external merged`. Default: `airway external`. |
| `--device` | `auto`, `cuda`, `mps` or `cpu`. A GPU that PyTorch cannot use falls back to the CPU. |
| `--mode` | `segment` or `extend`. Default: `segment`, or the mode set in the template. |
| `--no-sinus` | Switch the sinus recovery off. |
| `--external`, `--merge-external` | Segment the external face air, and merge it into the airway. |
| `--keep-largest`, `--no-remove-islands` | Island cleanup. |
| `--smoothing`, `--min-island`, `--min-internal` | Surface smoothing (0 to 1) and the two size thresholds in mm³. |
| `--extend MM` | Inferior extension length. 0 leaves it off. |
| `--install-dependencies` | Install or repair the dependencies first, as the button does. Works alone, without an input. |
| `--check-dependencies` | Print the dependency report. |
| `--log`, `--summary`, `--progress` | Write the log, the per-scan results and the live progress to files. |

The exit code is `0` when every scan succeeded, `1` when at least one failed and `2` on a fatal
error such as bad arguments or a missing NNUNet extension.

## Use from an AI agent

`MCP/auae_mcp.py` is a Model Context Protocol server. With it an AI agent can run AUAE for you:
you describe the job in words and the agent calls the extension. MCP is an open protocol, so
the server works with Claude Code, Claude Desktop, OpenAI Codex, Gemini CLI, Cursor and any
other client that starts local MCP servers. The ChatGPT apps are the exception, because they
connect only to remote servers: with OpenAI models, use Codex.

The server does little by design. It exposes the processing, with every option as a parameter,
and it describes what a folder contains. Which scans belong to which patient, which one is pre
or post operative and how the folders are named are decisions for you and the agent, and the
agent moves files with its own tools.

### Setup

1. Check that a segmentation runs from the panel.
2. Expand **LLM API support (MCP)** and click **Install MCP server**. A dialog lists the clients
   it can set up. Tick the ones you want: nothing is ticked in advance, and a client you leave
   unticked is not touched. Restart the clients that were open.
3. Tick **Allow an agentic AI to run AUAE** and accept the disclaimer. The support is off by
   default and the choice is remembered. While it is off the server refuses every request.
4. Ask the agent to check the AUAE environment. It should name the Slicer it found.

The server runs on Slicer's own Python, started without the Slicer window. Its two packages
(`mcp` and `pydicom`) are installed in `~/.auae_mcp/site`, apart from Slicer's, because `mcp`
pins versions of libraries that Slicer also uses.

<details>
<summary>Registering the server by hand</summary>

Every client needs the same command, Slicer's Python, and one argument, the server script. The
panel shows both with your paths filled in. On Linux and macOS the interpreter is
`bin/PythonSlicer`, without `.exe`.

Claude Code:

```
claude mcp add --scope user auae -- "<Slicer folder>/bin/PythonSlicer.exe" "<repository>/MCP/auae_mcp.py"
```

OpenAI Codex, from the command line or in `~/.codex/config.toml`:

```
codex mcp add auae -- "<Slicer folder>/bin/PythonSlicer.exe" "<repository>/MCP/auae_mcp.py"
```

```toml
[mcp_servers.auae]
command = "<Slicer folder>/bin/PythonSlicer.exe"
args = ["<repository>/MCP/auae_mcp.py"]
```

Claude Desktop (`claude_desktop_config.json`), Gemini CLI (`~/.gemini/settings.json`) and Cursor
(`~/.cursor/mcp.json`):

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

The server looks for Slicer in its default install folders and takes the newest version. If
yours is elsewhere, set the `AUAE_SLICER` environment variable to the Slicer executable.

To run the server on a Python of your own (3.10 or newer) instead of Slicer's, give it a virtual
environment and register that Python:

```
python -m venv "%USERPROFILE%\.auae_mcp\venv"
"%USERPROFILE%\.auae_mcp\venv\Scripts\python.exe" -m pip install -r "<repository>\MCP\requirements.txt"
```

</details>

### Working with it

A session on a folder of raw data can go like this.

- "Look at `D:\study\raw` and tell me which scans are there, by patient and date."
- "Copy them into `D:\study\sorted`, one folder per patient with `pre` and `post` subfolders.
  The earlier scan is pre. Show me the plan before you copy anything."
- "Segment every scan in `D:\study\sorted` with AUAE: STL and NIfTI, external air merged into
  the airway, 40 mm extension. Put the results in `D:\study\results`."
- "Make a table of the airway volumes, pre against post."

Ask for the plan before any copy, so the agent lists what it would do and waits for your answer.
The results mirror the layout of the inputs:

```
results/
  P001/
    pre/    <scan>_Segmentation_Airway.stl   <scan>_Segmentation.nii.gz   airway_volume.txt
    post/   ...
  P002/
    pre/    ...
```

A run is a background job. The agent follows it and can show you where it is:

```
[########----------------] 32%  Case 2 of 5 (P001/post): Inference 62%, about 7 min left
```

One CBCT of 816 x 816 x 632 voxels took two to four minutes on an RTX 3070 in our tests. The job
keeps running if you close the agent, and you can ask about it in a later session. A scan that
fails does not stop the others. Logs and results of the jobs are kept in `~/.auae_mcp/jobs`.

### Tools

| Tool | What it does |
| --- | --- |
| `describe_inputs` | Lists the volume files and DICOM series under a folder: patient, scan date, modality, descriptions, scanner, image matrix. |
| `segment` | Segments a list of scans or every scan in a folder. |
| `extend` | Lengthens saved segmentations without running the model. |
| `job_status` | State and progress of a job, and its results once it has ended. |
| `wait_for_job` | The same, but it waits for the job to end or for a timeout. |
| `list_jobs`, `cancel_job` | List the recent jobs, and stop one. |
| `check_dependencies`, `install_dependencies` | The dependency report and installation. |
| `environment` | Whether the support is switched on, and where Slicer was found. |

<details>
<summary>Parameters of <code>segment</code></summary>

| Parameter | Default | Meaning |
| --- | --- | --- |
| `inputs` or `input_folder` | one is required | A list of volume files and single-series DICOM folders, or a folder to process in full. |
| `output_folder` | `AUAE_output` beside the inputs | Where the per-scan subfolders are written. |
| `names` | from the input paths | Output subfolder of each input. May be nested, such as `P001/pre`. |
| `formats` | `STL`, `NIFTI` | Any of `STL`, `OBJ`, `NIFTI`, `NRRD`. |
| `targets` | `airway`, `external` | Any of `airway`, `external`, `merged`. |
| `device` | `auto` | `auto`, `cuda`, `mps` or `cpu`. |
| `recover_sinuses` | on | Add back the paranasal sinuses the model drops. |
| `min_internal_air_mm3` | 250 | Recovered cavities below this size are dropped. |
| `external_air` | off | Also segment the air in front of the face. |
| `merge_external_air` | off | Join that air to the airway. |
| `remove_small_islands`, `min_island_mm3` | on, 5.4 | Drop disconnected components below the threshold. |
| `keep_largest_island` | off | Keep only the largest component instead. |
| `smoothing` | 0 | Surface smoothing of the exported mesh, 0 to 1. |
| `extend_mm` | 0 | Inferior extension length. 0 leaves it off. |

</details>

### Patient data

AUAE does not anonymise or filter what the agent sees. The agent reads the folders you point it
at, including the patient names, IDs and dates in DICOM headers and in file and folder names,
and `describe_inputs` reports the patient ID and name of each DICOM series so that scans can be
grouped. What the agent reads is sent to the company that provides it.

That use is the responsibility of the person who lets the agent in, as the
[Disclaimer](#disclaimer) says. If the agent must not learn who the patients are, give it a
de-identified copy: DICOM anonymised at the source, or volumes converted to NRRD or NIfTI, in
folders whose names carry no names.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| "This module depends on the NNUNet module" | Install NNUNet from the Extensions Manager and restart. |
| Inference is slow, or the GPU is not used | Click **Check** in **Dependencies & CUDA**. A line starting with `PROBLEM` says what is wrong with PyTorch, and **Install dependencies** repairs it. If the GPU is found but PyTorch cannot use it, update the NVIDIA driver first. |
| "PyTorch is in use by this Slicer session" | PyTorch cannot be replaced once a segmentation has run. Accept the restart: the installation continues when AUAE reopens. |
| "CUDA capability sm_120 is not compatible with the current PyTorch installation" | An RTX 50 card with a PyTorch build that has no kernels for it. **Install dependencies** replaces it with the CUDA 12.8 build. |
| `[WinError 206]`, file name too long | PyTorch 2.13 on Windows goes past the path-length limit inside the Slicer folder. AUAE installs an earlier release. If another module installed 2.13, **Install dependencies** replaces it. |
| "Type parameter ~_R without a default follows type parameter with a default" | Restart Slicer once. It happens when PyTorch is loaded after pip in the same session, which the PyTorch Utils panel causes. The files on disk are fine. |
| Another extension changed PyTorch | **Check** reports a CPU-only build or a `torchvision` that does not match, and **Install dependencies** puts back a matching pair. |
| The model weights do not download | Check the network. The weights come from the upstream GitHub release. |
| The agent says that LLM API support is switched off | Tick **Allow an agentic AI to run AUAE** in the panel. No restart is needed. |
| The client lists `auae` as failed or disconnected | Click **Install MCP server** again. |
| "3D Slicer was not found" | Set the `AUAE_SLICER` environment variable to the Slicer executable. |
| "Found 2 DICOM series in ..." | One series per folder. Split the series into separate folders. |

<details>
<summary>RTX 50 series: installing the CUDA 12.8 build by hand</summary>

The RTX 50 series is compute capability sm_120. The stock PyTorch installer of Slicer chooses
the build from the driver version alone and picks one compiled up to sm_90. Only a PyTorch built
for CUDA 12.8 (torch 2.7 or newer) has the sm_120 kernels. If **Install dependencies** fails:

1. Run `import sysconfig; print(sysconfig.get_paths()["purelib"])` in the Python console and
   note the folder.
2. Open the **PyTorch Utils** module and uninstall torch.
3. Close Slicer. In the folder noted above, delete `torch`, `torchgen`, `torchvision`,
   `functorch`, their `*.dist-info` folders and any folder that starts with a tilde, such as
   `~orch`. An old `torchgen` left behind makes the next install fail.
4. Reopen Slicer. In PyTorch Utils choose the **cu128** backend and install, or run this in the
   Python console:
   ```python
   import PyTorchUtils
   PyTorchUtils.PyTorchUtilsLogic().installTorch(askConfirmation=True, forceComputationBackend="cu128")
   ```
5. Restart Slicer and click **Check**. The report should say that a test computation on the
   GPU succeeded.

On Linux, as of mid-2025, the official Slicer package is built on CentOS 7 (glibc 2.17), and
the torch 2.7 wheels need glibc 2.28. An RTX 50 card therefore cannot be used from the stock
Linux package until its base is updated. Until then, set **Inference device** to **CPU**.

</details>

## How it works

The post-processing lives in `AirwayExtension` and runs on NumPy arrays.

Island cleanup runs on the airway first, with a threshold in cubic millimetres so that it holds
at any voxel size.

Sinus recovery and external air share one step. The intensities under the airway mask give
the air threshold, so no fixed value is assumed and a CBCT with shifted intensities works too.
Everything brighter is tissue, and its largest component is the patient. A morphological closing
seals the nostrils, the mouth and the thin sinus ducts, and hole filling turns the head into a
solid envelope. Air inside the envelope that is not yet airway is the sinuses: components
below 250 mm³, which is where the cephalostat bubbles fall, are discarded and the rest joins the
airway. Air outside the envelope, in front of the airway, is the external air. Because the two
come from opposite sides of the same envelope, a frontal sinus cannot end up in the external air.

The extension repeats the last slice of the airway towards the neck, first inside the
scanned volume and then beyond it. It is applied to the airway alone. When the external air is
merged, it is added afterwards on the enlarged grid, so the extension never repeats the external
air, which can sit lower in the scan.

<details>
<summary>Code map</summary>

| File | Role |
| --- | --- |
| `AUAE/AUAELib/AirwayExtension.py` | Array logic without any interface: `postprocessSegmentation` (cleanup, sinuses, external air, extension, merge), `extendSegmentation`, `applySurfaceSmoothing`. |
| `AUAE/AUAELib/SegmentationWidget.py` | The panel, the link to the nnU-Net logic, and `exportSegmentation`. |
| `AUAE/AUAELib/BatchProcessor.py` | `listVolumesInFolder`, `folderConfig`, `loadConfig` and the `BatchProcessor.run` driver for the `segment` and `extend` modes. |
| `AUAE/AUAELib/Dependencies.py` | `detectHardware`, `recommendedBuild`, `torchAction`, `ensure` (the installation, resumed after a restart), `torchStatus` (the check in a separate process), `report`, `resolveDevice`. |
| `AUAE/AUAELib/Progress.py` | Stage, fraction and time left of a run, read from its log lines. Shared by the panel, the command line and the MCP server. |
| `AUAE/AUAELib/LlmApi.py` | The opt-in switch and its disclaimer, the installation of the MCP server and its registration with the clients. |
| `AUAE/AUAELib/PythonDependencyChecker.py` | Download of the model weights from the upstream release. |
| `AUAE/CLI/auae_batch.py` | The command-line runner. |
| `MCP/auae_mcp.py` | The MCP server. Each tool is a plain function; `_start_job` launches Slicer on the runner. |
| `AUAE/Testing/` | Tests for the array logic, the dependency decisions and the progress, run by the module self test. |

</details>

## Limits

- The model expects CT-like Hounsfield units. A CBCT with air near 0 instead of -1000 is
  under-segmented, which is the usual reason turbinates and sinuses go missing. The sinus
  recovery helps, but a volume in Hounsfield units, or a model trained on your own CBCT, is the
  reliable path.
- The bubble filter works by size. A hypoplastic frontal sinus smaller than the threshold is
  discarded, and a bubble larger than it is kept. Adjust `min_internal_air_mm3` or correct the
  mask in the editor.
- The PyTorch build table reflects the wheels available in November 2026. AMD and Intel graphics
  are not used.

## Disclaimer

When an AI agent (an LLM such as Claude, Codex or Gemini) runs AUAE through the MCP server, it
reads the files in the folders you point it at, patient names, IDs and dates included, and that
content is sent to the company that provides the AI. AUAE does not anonymise or filter it.
Whoever uses the AI is responsible for that use: for de-identifying the data beforehand where
required, for the consent and approvals that cover it, for the terms of the AI provider, and
for complying with the applicable data-protection rules, such as the GDPR. The authors of AUAE
take no responsibility for data disclosed to an AI or to its provider. The support is off by
default and is switched on only after this disclaimer is accepted in the panel.

The software is provided "as is", without warranty of any kind, under the terms of the Apache
License 2.0 (see [`LICENSE.md`](LICENSE.md)).

## Credits

The segmentation model and the upstream extension are the work of Alejandro Matos Camarillo,
Silvia Capenakas and Manuel Lagravere at the University of Alberta, with the co-authors of the
paper below. Their repository is
[capenaka/SlicerUpperAirwaySegmentator](https://github.com/capenaka/SlicerUpperAirwaySegmentator).

AUAE is by Dr. Pietro Montagna (DDS, MSc, PhD Student, pietro.montagna@univr.it) and Dr. Fabio
Lonardi (MD, OMFS, PhD Student, fabio.lonardi@univr.it), Head and Neck Department, Department
of Surgery, Dentistry, Pediatrics and Gynecology, University of Verona, Verona, Italy.

## Citation

If this extension helps your work, cite the upstream paper and nnU-Net.

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
