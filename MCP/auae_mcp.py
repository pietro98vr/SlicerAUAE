"""MCP server for AUAE: lets an agentic AI run the extension.

MCP is an open protocol and this server uses its stdio transport, so it works with every
client that starts local MCP servers: Claude Code, Claude Desktop (Cowork), OpenAI Codex,
Gemini CLI, Cursor and others. The tool schemas stick to plain JSON types (no unions) so that
the stricter clients accept them.

The server is deliberately thin. It exposes what the extension does (segmentation,
post-processing, inferior extension, export, dependency setup) with every option as a tool
parameter, and it describes input folders so the assistant can decide what to run on what.
Sorting raw data, naming patients, choosing pre and post operative scans and similar decisions
stay with the assistant and its own file tools.

It is a separate program from the Slicer application. Normally it runs on Slicer's own Python
(PythonSlicer), with its two packages in ~/.auae_mcp/site: 'Install MCP server' in the AUAE
panel sets that up and registers the server with the assistant. It also runs on any other
Python 3.10+ that has ``mcp`` installed (``pydicom`` is optional and only used to describe
DICOM folders); use a virtual environment for that, see the README. Each run starts Slicer
without its window on AUAE/CLI/auae_batch.py and returns at once with a job id; the assistant
then polls ``job_status``. A job keeps running if the assistant session ends.

    PythonSlicer auae_mcp.py                   # stdio transport

Set AUAE_SLICER to the Slicer executable when it is not found by itself.

The support is opt-in. Until the user ticks 'Allow an agentic AI to run AUAE' in the panel
(off by default, remembered between sessions) every tool except ``environment`` refuses to act.

Nothing is anonymised here. The AI drives the extension and reads the files itself, so a filter
on what this server reports would protect nothing; describe_inputs returns the patient ID and
name as written in the DICOM headers. Protecting patient data is the responsibility of whoever
uses the AI: the user accepts a disclaimer to that effect before the support can be switched
on, and the README repeats it. Data that must stay unknown to the AI has to be de-identified
before the AI is given the folder.
"""

import functools
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

def _use_installed_packages():
    """Put the packages installed from the AUAE panel first on the import path.

    'Install MCP server' in the panel installs mcp and pydicom with Slicer's Python into
    ~/.auae_mcp/site, apart from Slicer's own packages. The folder is used only by the Python
    version it was installed for, so a server started from another Python (a virtual
    environment of your own) is not handed packages it cannot load.
    """
    folder = Path.home() / ".auae_mcp" / "site"
    try:
        version = (folder / "python_version.txt").read_text().strip()
    except OSError:
        return
    if version != "%d.%d" % sys.version_info[:2]:
        return
    import site
    before = list(sys.path)
    site.addsitedir(str(folder))  # also reads the .pth files some packages need (pywin32)
    added = [path for path in sys.path if path not in before]
    sys.path[:] = added + [path for path in sys.path if path not in added]


_use_installed_packages()

HERE = Path(__file__).resolve().parent
CLI_SCRIPT = HERE.parent / "AUAE" / "CLI" / "auae_batch.py"
JOBS_DIR = Path(os.environ.get("AUAE_MCP_HOME") or (Path.home() / ".auae_mcp")) / "jobs"
# Written by the AUAE panel (AUAELib/LlmApi.py) when the user ticks 'LLM API support'.
SWITCH_FILE = Path.home() / ".auae_mcp" / "settings.json"
DISABLED_MESSAGE = (
    "LLM API support is switched off in AUAE, so this server does nothing. Ask the user to open "
    "the AUAE module in 3D Slicer, expand 'LLM API support (MCP)' and tick 'Allow an agentic AI "
    "to run AUAE'. The choice is remembered; no restart is needed."
)

VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")
NOT_DICOM_EXTENSIONS = (".txt", ".json", ".xml", ".csv", ".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".zip",
                        ".stl", ".obj", ".ini", ".log", ".html", ".exe", ".dll", ".py")
EXPORT_FORMATS = ("STL", "OBJ", "NIFTI", "NRRD")
EXPORT_TARGETS = ("airway", "external", "merged")
DEVICES = ("auto", "cuda", "mps", "cpu")

_processes = {}  # job id -> Popen, for the jobs started by this server process
_progress = None  # the extension's Progress module, loaded on first use


# --- Slicer ----------------------------------------------------------------------------
def find_slicer():
    """Path of the Slicer executable: AUAE_SLICER, else the newest install in the usual places."""
    configured = os.environ.get("AUAE_SLICER")
    if configured:
        return configured if os.path.isfile(configured) else None
    candidates = []
    if sys.platform == "win32":
        for root in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
            if root:
                candidates += glob.glob(os.path.join(root, "slicer.org", "*", "Slicer.exe"))
                candidates += glob.glob(os.path.join(root, "NA-MIC", "*", "Slicer.exe"))
    elif sys.platform == "darwin":
        candidates += glob.glob("/Applications/Slicer*.app/Contents/MacOS/Slicer")
    else:
        candidates += glob.glob(os.path.expanduser("~/Slicer*/Slicer")) + glob.glob("/opt/Slicer*/Slicer")
    found = shutil.which("Slicer")
    if found:
        candidates.append(found)
    if not candidates:
        return None

    def version(path):  # ".../3D Slicer 5.10.0/Slicer.exe" -> (5, 10, 0)
        return tuple(int(n) for n in re.findall(r"\d+", os.path.basename(os.path.dirname(path))))

    return sorted(candidates, key=version)[-1]


def _slicer_command(cli_args):
    slicer = find_slicer()
    if slicer is None:
        raise RuntimeError("3D Slicer was not found. Set the AUAE_SLICER environment variable to the "
                           "Slicer executable (Slicer.exe on Windows).")
    if not CLI_SCRIPT.is_file():
        raise RuntimeError("AUAE command-line runner not found at %s" % CLI_SCRIPT)
    return [slicer, "--no-splash", "--no-main-window", "--python-script", str(CLI_SCRIPT), "--", *cli_args]


def _popen(command, stdout_path):
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True  # so cancel_job can stop Slicer and its children
    stdout = open(stdout_path, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs)


# --- Jobs ------------------------------------------------------------------------------
def _job_dir(job_id):
    if not re.fullmatch(r"[0-9a-f]{8}", str(job_id or "")):
        raise ValueError("Unknown job id: %r" % job_id)
    return JOBS_DIR / job_id


def _start_job(kind, cli_args, details):
    """Start Slicer on the runner and record the job on disk. Returns the job description."""
    job_id = uuid.uuid4().hex[:8]
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)
    log_path, summary_path = folder / "log.txt", folder / "summary.json"
    command = _slicer_command([*cli_args, "--log", str(log_path), "--summary", str(summary_path),
                               "--progress", str(folder / "progress.json")])
    process = _popen(command, folder / "console.txt")
    _processes[job_id] = process
    job = {"job_id": job_id, "kind": kind, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "pid": process.pid, **details}
    (folder / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    return {**job, "state": "running",
            "next": "Call wait_for_job with this job_id: it returns when the job ends, or after its "
                    "timeout with the progress so far, which you can pass on to the user before "
                    "calling it again. One volume takes about 2-4 minutes on a GPU and up to an "
                    "hour on the CPU."}


def _pid_alive(pid):
    if sys.platform == "win32":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _is_running(job):
    process = _processes.get(job["job_id"])
    if process is not None:
        return process.poll() is None
    return _pid_alive(job["pid"])


def _tail(path, lines):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return text[-int(lines):] if lines else []


def _progress_module():
    """The extension's Progress module, loaded from its file (the package itself needs Slicer)."""
    global _progress
    if _progress is None:
        import importlib.util
        path = HERE.parent / "AUAE" / "AUAELib" / "Progress.py"
        try:
            spec = importlib.util.spec_from_file_location("auae_progress", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _progress = module
        except Exception:  # noqa: BLE001
            _progress = False
    return _progress or None


def _read_progress(folder, log):
    """Progress of a job: the snapshot the runner keeps on disk, plus a text bar to show.

    {"done", "total", "case", "case_name", "stage", "case_fraction", "overall_fraction",
    "elapsed_seconds", "eta_seconds", "bar"}. Before the runner has written anything (Slicer is
    still starting) only the case counts found in the log are returned.
    """
    done = total = None
    for line in log:
        match = re.search(r"progress: (\d+)/(\d+)", line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
    progress = {"done": done, "total": total}
    try:
        snapshot = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        progress["bar"] = "Starting Slicer" if total is None else "%s of %s cases done" % (done, total)
        return progress
    progress.update(snapshot)
    module = _progress_module()
    progress["bar"] = module.formatBar(snapshot) if module else "%d%%" % round(snapshot.get("overall_fraction", 0) * 100)
    return progress


def _job_state(job_id, log_lines=15):
    folder = _job_dir(job_id)
    job_file = folder / "job.json"
    if not job_file.is_file():
        raise ValueError("Unknown job id: %r" % job_id)
    job = json.loads(job_file.read_text(encoding="utf-8"))
    log = _tail(folder / "log.txt", 100000)
    status = {**job, "progress": _read_progress(folder, log),
              "log_tail": log[-int(log_lines):] if log_lines else []}

    summary_file = folder / "summary.json"
    if summary_file.is_file():
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except ValueError:
            summary = None
        if summary is not None:
            status["result"] = summary
            failed = summary.get("total", 0) - summary.get("succeeded", 0) if "total" in summary else 0
            status["state"] = "error" if summary.get("state") == "error" else ("finished_with_failures" if failed else "finished")
            if "total" in summary and summary.get("state") != "error":
                status["progress"].update({"done": summary["total"], "total": summary["total"],
                                           "overall_fraction": 1.0, "eta_seconds": 0,
                                           "bar": "[%s] 100%%  Finished %d of %d" % (
                                               "#" * 24, summary["total"], summary["total"])})
            return status
    if (folder / "cancelled").is_file():
        status["state"] = "cancelled"
    elif _is_running(job):
        status["state"] = "running"
    else:
        status["state"] = "error"
        status["error"] = "Slicer stopped without writing a result. The end of its console output follows."
        status["console_tail"] = _tail(folder / "console.txt", 25)
    return status


# --- Inputs ----------------------------------------------------------------------------
def _is_volume_file(name):
    return name.lower().endswith(VOLUME_EXTENSIONS)


def _strip_volume_extension(name):
    for extension in VOLUME_EXTENSIONS:
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return name


def _looks_like_dicom(path):
    if path.lower().endswith(".dcm"):
        return True
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def _dicom_files(folder, names):
    """The DICOM files directly inside a folder (checked by extension, else by file signature)."""
    with_extension = [name for name in names if name.lower().endswith(".dcm")]
    if with_extension:
        return sorted(with_extension)
    # DICOM exports often have no extension (IM000001) or a UID as the file name.
    others = sorted(name for name in names
                    if not _is_volume_file(name) and not name.lower().endswith(NOT_DICOM_EXTENSIONS))
    if others and _looks_like_dicom(os.path.join(folder, others[0])):
        return others
    return []


def _read_dicom_header(path):
    try:
        import pydicom
    except ImportError:
        return None
    tags = ["PatientID", "PatientName", "StudyDate", "AcquisitionDate", "SeriesDate", "Modality",
            "StudyDescription", "SeriesDescription", "SeriesInstanceUID", "StudyInstanceUID",
            "Manufacturer", "ManufacturerModelName", "Rows", "Columns", "PixelSpacing", "SliceThickness"]
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags, force=True)
    except Exception:  # noqa: BLE001
        return None
    return {tag: (str(dataset.get(tag)) if dataset.get(tag) is not None else None) for tag in tags}


def _describe_dicom_folder(folder, files):
    entry = {"path": folder, "kind": "dicom_series", "files": len(files)}
    first = _read_dicom_header(os.path.join(folder, files[0]))
    if first is None:
        entry["note"] = "Install pydicom next to the MCP server to see the DICOM details."
        return entry
    last = _read_dicom_header(os.path.join(folder, files[-1])) or first
    entry.update({
        # As written in the header. Nothing is anonymised here: see the disclaimer in the README.
        "patient_id": first["PatientID"],
        "patient_name": first["PatientName"],
        "date": first["AcquisitionDate"] or first["SeriesDate"] or first["StudyDate"],
        "modality": first["Modality"],
        "study_description": first["StudyDescription"],
        "series_description": first["SeriesDescription"],
        "scanner": " ".join(part for part in (first["Manufacturer"], first["ManufacturerModelName"]) if part) or None,
        "matrix": "%s x %s" % (first["Rows"], first["Columns"]) if first["Rows"] else None,
        "pixel_spacing": first["PixelSpacing"],
        "slice_thickness": first["SliceThickness"],
        # AUAE needs one series per folder; a folder that mixes several must be split first.
        "one_series": first["SeriesInstanceUID"] == last["SeriesInstanceUID"],
    })
    return entry


def _nrrd_sizes(path):
    """Voxel grid of an NRRD file, read from its text header."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(4096).split(b"\n\n", 1)[0].decode("ascii", errors="replace")
    except OSError:
        return None
    match = re.search(r"^sizes:\s*(.+)$", header, re.MULTILINE)
    return match.group(1).strip() if match else None


def _describe_inputs(folder, max_entries=500):
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise ValueError("Not a folder: %s" % folder)
    entries, truncated = [], False
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        if len(entries) >= max_entries:
            truncated = True
            break
        for name in sorted(files):
            if _is_volume_file(name):
                path = os.path.join(root, name)
                entry = {"path": path, "kind": "volume", "size_mb": round(os.path.getsize(path) / 1e6, 1)}
                if name.lower().endswith(".nrrd"):
                    entry["sizes"] = _nrrd_sizes(path)
                entries.append(entry)
        dicom = _dicom_files(root, files)
        if dicom:
            entries.append(_describe_dicom_folder(root, dicom))
    return {"folder": folder, "count": len(entries), "inputs": entries, "truncated": truncated}


def _default_names(paths):
    """Output subfolder names that keep the folder structure the inputs share.

    ['R/P001/pre', 'R/P001/post', 'R/P002/pre.nrrd'] -> ['P001/pre', 'P001/post', 'P002/pre'].
    """
    absolute = [os.path.abspath(path) for path in paths]
    if len(absolute) == 1:
        return [_strip_volume_extension(os.path.basename(absolute[0]))]
    common = os.path.commonpath([os.path.dirname(path) for path in absolute])
    names = []
    for path in absolute:
        relative = os.path.relpath(path, common)
        head, tail = os.path.split(relative)
        names.append(os.path.join(head, _strip_volume_extension(tail)).replace("\\", "/"))
    return names


def _choice_list(values, allowed, label):
    chosen = [str(value).strip() for value in (values or [])]
    normalised = [value.upper() if allowed is EXPORT_FORMATS else value.lower() for value in chosen]
    bad = [value for value in normalised if value not in allowed]
    if bad or not normalised:
        raise ValueError("%s must be one or more of %s" % (label, ", ".join(allowed)))
    return normalised


def _start_batch(mode, inputs, input_folder, output_folder, names, formats, targets, device, options, extend_mm):
    inputs = [os.path.abspath(os.path.expanduser(path)) for path in (inputs or []) if str(path).strip()]
    input_folder = os.path.abspath(os.path.expanduser(input_folder)) if input_folder else ""
    if bool(inputs) == bool(input_folder):
        raise ValueError("Give either 'inputs' (explicit paths) or 'input_folder', not both and not neither.")
    if device not in DEVICES:
        raise ValueError("device must be one of " + ", ".join(DEVICES))
    template = {
        "mode": mode,
        "export_formats": _choice_list(formats, EXPORT_FORMATS, "formats"),
        "export_targets": _choice_list(targets, EXPORT_TARGETS, "targets"),
        "extension": {"enabled": float(extend_mm) > 0, "length_mm": float(extend_mm) if float(extend_mm) > 0 else 100.0},
        **options,
    }
    if inputs:
        missing = [path for path in inputs if not os.path.exists(path)]
        if missing:
            raise ValueError("Input not found: " + "; ".join(missing))
        names = [str(name) for name in names] if names else _default_names(inputs)
        if len(names) != len(inputs) or len(set(names)) != len(names):
            raise ValueError("'names' must hold one distinct output subfolder name per input.")
        template["volumes"] = [{"path": path, "name": name} for path, name in zip(inputs, names)]
        default_output = os.path.join(os.path.commonpath([os.path.dirname(path) for path in inputs]), "AUAE_output")
    else:
        if not os.path.isdir(input_folder):
            raise ValueError("Not a folder: %s" % input_folder)
        template["input_folder"] = input_folder
        default_output = os.path.join(input_folder, "AUAE_output")
    output_root = os.path.abspath(os.path.expanduser(output_folder)) if output_folder else default_output
    template["output_root"] = output_root

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    template_path = JOBS_DIR / ("template_%s.json" % uuid.uuid4().hex[:8])
    template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return _start_job(mode, ["--template", str(template_path), "--device", device],
                      {"output_root": output_root, "inputs": len(inputs) if inputs else "all in " + input_folder,
                       "template": str(template_path)})


# --- Tools -----------------------------------------------------------------------------
def environment() -> dict:
    """Say whether the server can run AUAE: whether the user switched the LLM API support on,
    where Slicer is, and where the jobs are kept.

    Call this first when something does not start. It does not launch Slicer and it is the
    only tool that answers while the support is switched off.
    """
    try:
        import pydicom  # noqa: F401
        has_pydicom = True
    except ImportError:
        has_pydicom = False
    enabled = llm_api_enabled()
    notes = []
    if not enabled:
        notes.append(DISABLED_MESSAGE)
    if not find_slicer():
        notes.append("Set AUAE_SLICER to the Slicer executable.")
    return {"llm_api_enabled": enabled, "slicer": find_slicer(), "runner": str(CLI_SCRIPT),
            "runner_found": CLI_SCRIPT.is_file(), "jobs_folder": str(JOBS_DIR), "pydicom": has_pydicom,
            "notes": notes}


def check_dependencies() -> dict:
    """Report what AUAE needs inside Slicer and whether it is ready to run.

    Starts Slicer without its window and prints the report: Slicer, NNUNet extension, nnU-Net,
    PyTorch, the GPU that was found, the PyTorch build this computer needs and the device
    inference will use. Takes about half a minute. If it is not ready, call
    install_dependencies.
    """
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    log_path, summary_path = JOBS_DIR / ("check_%s.txt" % stamp), JOBS_DIR / ("check_%s.json" % stamp)
    command = _slicer_command(["--check-dependencies", "--log", str(log_path), "--summary", str(summary_path)])
    process = _popen(command, JOBS_DIR / ("check_%s_console.txt" % stamp))
    try:
        process.wait(timeout=240)
    except subprocess.TimeoutExpired:
        process.kill()
        return {"ready": None, "error": "Slicer did not answer within 4 minutes.", "report": _tail(log_path, 40)}
    report = [re.sub(r"^\d\d:\d\d:\d\d ", "", line) for line in _tail(log_path, 60)]
    ready = None
    if summary_path.is_file():
        ready = json.loads(summary_path.read_text(encoding="utf-8")).get("ready")
    if not report:
        return {"ready": ready, "error": "Slicer gave no report.",
                "console_tail": _tail(JOBS_DIR / ("check_%s_console.txt" % stamp), 25)}
    return {"ready": ready, "report": report}


def install_dependencies() -> dict:
    """Install or repair everything AUAE needs inside Slicer, as a background job.

    Detects the GPU and installs the PyTorch build that suits it (CUDA 12.8 for the RTX 50
    series, CUDA 12.6 for GTX 900 up to RTX 40, the CPU build without an NVIDIA GPU), nnU-Net
    and the helper packages. A PyTorch that works is left alone; one that cannot work on this
    computer is replaced. The download can be 2-3 GB. Returns a job id for job_status. The
    NNUNet extension itself must already be installed from Slicer's Extensions Manager.
    """
    return _start_job("install_dependencies", ["--install-dependencies", "--check-dependencies"], {})


def describe_inputs(folder: str) -> dict:
    """List what AUAE can process under a folder, with the details needed to plan a run.

    Walks the folder at any depth and returns one entry per volume file (.nii, .nii.gz, .nrrd,
    .nhdr, .mha, .mhd) and one per folder of DICOM files, with the scan date, modality,
    descriptions, scanner, image matrix and whether the folder holds a single series (AUAE
    needs one series per folder). Use it to work out which scans belong to which patient and
    which are pre or post operative before organising files or starting a run.

    For DICOM folders the entry carries the patient ID and name as written in the header, so
    that scans can be grouped by patient. These are personal health data: use them only for
    the task the user asked for and do not repeat them where the task does not need them.
    """
    return _describe_inputs(folder)


def segment(
    inputs: list[str] = [],
    input_folder: str = "",
    output_folder: str = "",
    names: list[str] = [],
    formats: list[str] = ["STL", "NIFTI"],
    targets: list[str] = ["airway", "external"],
    device: str = "auto",
    recover_sinuses: bool = True,
    external_air: bool = False,
    merge_external_air: bool = False,
    remove_small_islands: bool = True,
    keep_largest_island: bool = False,
    min_island_mm3: float = 5.4,
    min_internal_air_mm3: float = 250.0,
    smoothing: float = 0.0,
    extend_mm: float = 0.0,
) -> dict:
    """Segment the upper airway on CT / CBCT scans with AUAE, as a background job.

    Give the scans in one of two ways. 'inputs' is an explicit list: each item is a volume file
    or a folder holding exactly one DICOM series. 'input_folder' is a folder whose volume files
    and DICOM subfolders are all processed. Each case is written to its own subfolder of
    output_folder (default: an 'AUAE_output' folder beside the inputs). With 'inputs' the
    subfolders keep the structure the inputs share, so .../P001/pre and .../P001/post become
    P001/pre and P001/post; 'names' sets them explicitly, one per input, and may be nested.

    Export: formats is any of STL, OBJ (surface meshes), NIFTI, NRRD (labelmaps). targets is
    any of 'airway', 'external' (the face-air segment, written only when external_air is true)
    and 'merged' (all segments in one file).

    Post-processing, the same options as the module panel:
    - recover_sinuses: add back the paranasal sinuses the model drops (on by default).
    - min_internal_air_mm3: recovered cavities smaller than this are dropped; removes the small
      air bubbles of the cephalostat near the frontal sinus.
    - external_air: also segment the air in front of the face, the inlet volume for airflow
      simulation. merge_external_air joins it to the airway as one fluid domain.
    - remove_small_islands / min_island_mm3: drop disconnected specks below the threshold.
    - keep_largest_island: keep only the largest connected component instead.
    - smoothing: surface smoothing of the exported mesh, 0 (raw voxels) to 1.
    - extend_mm: lengthen the airway downwards by this many millimetres by repeating its last
      slice (0 = off), for an outlet far from the region of interest in simulations.

    device is 'auto' (GPU when PyTorch can use it, else CPU), 'cuda', 'mps' or 'cpu'.
    Each case also gets an airway_volume.txt with the airway volume in mL.
    Returns a job id at once; call job_status to follow the run and read the results.
    """
    options = {
        "include_internal_air": bool(recover_sinuses),
        "min_internal_air_mm3": float(min_internal_air_mm3),
        "segment_external_air": bool(external_air),
        "merge_external_air": bool(merge_external_air),
        "remove_small_islands": bool(remove_small_islands) and not keep_largest_island,
        "keep_largest_island": bool(keep_largest_island),
        "min_island_mm3": float(min_island_mm3),
        "smoothing_factor": float(smoothing),
    }
    return _start_batch("segment", inputs, input_folder, output_folder, names, formats, targets, device,
                        options, extend_mm)


def extend(
    extend_mm: float,
    inputs: list[str] = [],
    input_folder: str = "",
    output_folder: str = "",
    names: list[str] = [],
    formats: list[str] = ["STL", "NIFTI"],
    targets: list[str] = ["airway"],
    smoothing: float = 0.0,
) -> dict:
    """Lengthen existing airway segmentations downwards, as a background job.

    The inputs are segmentation labelmaps saved earlier (.nii.gz or .nrrd), for example the
    files segment wrote or ones corrected by hand afterwards. Each is extended inferiorly by
    extend_mm millimetres by repeating its last slice, without running the model again, so no
    GPU is needed. inputs / input_folder / output_folder / names / formats / smoothing work as
    in segment. Returns a job id for job_status.
    """
    if float(extend_mm) <= 0:
        raise ValueError("extend_mm must be greater than 0.")
    return _start_batch("extend", inputs, input_folder, output_folder, names, formats, targets, "cpu",
                        {"smoothing_factor": float(smoothing)}, extend_mm)


def job_status(job_id: str, log_lines: int = 15) -> dict:
    """State of a job started by segment, extend or install_dependencies.

    state is 'running', 'finished', 'finished_with_failures', 'error' or 'cancelled'.

    progress says how far the job is: 'done' and 'total' count the cases, 'case' and
    'case_name' name the one in hand, 'stage' is what is being done to it (Loading, Preparing
    the volume, Preprocessing, Inference, Reading the result, Post-processing, Extension,
    Export), 'case_fraction' and 'overall_fraction' go from 0 to 1 for the case and for the
    whole job, 'eta_seconds' estimates the time left, and 'bar' is a one-line text progress bar
    ready to show to the user.

    When the job has ended, result lists every case with its status, output folder, error
    message and airway volume in mL. log_tail holds the last lines of the run's log. To follow
    a job, prefer wait_for_job, which waits for you instead of returning at once.
    """
    return _job_state(job_id, log_lines)


def wait_for_job(job_id: str, timeout_seconds: int = 45) -> dict:
    """Wait for a job, then return its status as job_status does.

    Returns as soon as the job ends, or after timeout_seconds (5 to 300) with the progress so
    far. Use it in a loop to follow a job: after each return, show the user progress.bar, and
    call it again while state is 'running'. This keeps the user informed without asking for the
    status many times in a row.
    """
    deadline = time.time() + max(5, min(300, int(timeout_seconds)))
    status = _job_state(job_id, 8)
    while status["state"] == "running" and time.time() < deadline:
        time.sleep(2)
        status = _job_state(job_id, 8)
    return status


def list_jobs(limit: int = 20) -> dict:
    """The most recent jobs, newest first, with their state, progress and output folder."""
    jobs = []
    for job_file in sorted(JOBS_DIR.glob("*/job.json"), key=lambda path: path.stat().st_mtime, reverse=True)[: int(limit)]:
        state = _job_state(job_file.parent.name, 0)
        jobs.append({key: state.get(key) for key in ("job_id", "kind", "started", "state", "progress", "output_root")})
    return {"jobs": jobs}


def cancel_job(job_id: str) -> dict:
    """Stop a running job. Cases already written stay on disk."""
    folder = _job_dir(job_id)
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    if not _is_running(job):
        return {"job_id": job_id, "state": _job_state(job_id, 0)["state"], "note": "The job was not running."}
    (folder / "cancelled").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    if sys.platform == "win32":  # Slicer.exe is a launcher: stop the whole process tree
        subprocess.run(["taskkill", "/PID", str(job["pid"]), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(int(job["pid"])), signal.SIGTERM)
        except OSError:
            pass
    return {"job_id": job_id, "state": "cancelled"}


TOOLS = (environment, check_dependencies, install_dependencies, describe_inputs, segment, extend,
         job_status, wait_for_job, list_jobs, cancel_job)

INSTRUCTIONS = (
    "AUAE segments the upper airway (nasal cavity, paranasal sinuses, pharynx) on head CT and CBCT "
    "scans inside 3D Slicer and exports meshes and labelmaps, for example for airflow simulation "
    "before and after orthognathic surgery. Typical use: describe_inputs to see what a folder "
    "holds, organise or select the scans with your own file tools, then segment and follow the "
    "job with wait_for_job, showing the user the progress bar it returns. Runs are background "
    "jobs. The scans are patient data: patient names and IDs "
    "appear in DICOM headers and often in file and folder names. Use them only as far as the "
    "user's task needs and do not repeat them otherwise."
)


def main():
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp 2.x
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x, same interface
        except ImportError:
            sys.exit("The 'mcp' package is missing. Install it with:  pip install mcp")
    server = Server("auae", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        server.tool()(_reporting_errors(tool))
    server.run()


def llm_api_enabled():
    """True when the user ticked 'LLM API support' in the AUAE panel (off by default)."""
    try:
        return json.loads(SWITCH_FILE.read_text(encoding="utf-8")).get("llm_api_enabled") is True
    except (OSError, ValueError, AttributeError):
        return False


def _reporting_errors(tool):
    """Refuse every tool while the user's switch is off, and return a bad request as
    {"error": message} so the assistant reads why it was refused."""
    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        if tool is not environment and not llm_api_enabled():
            return {"error": DISABLED_MESSAGE}
        try:
            return tool(*args, **kwargs)
        except (ValueError, RuntimeError, OSError) as exc:
            return {"error": str(exc)}
    return wrapper


if __name__ == "__main__":
    main()
