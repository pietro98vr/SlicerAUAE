"""LLM API support: the opt-in switch and the installation of the MCP server.

Added by this project. The MCP server (MCP/auae_mcp.py) lets an agentic AI run the extension.
MCP is an open protocol, so the same server works with Claude Code, Claude Desktop, OpenAI
Codex, Gemini CLI, Cursor and any other client that starts local MCP servers. Two things are
handled here.

The switch. Letting an agentic AI in is opt-in: the support is off until the user ticks it in
the panel and accepts the disclaimer, and the choice is kept between sessions. The server is a
separate program and cannot read Slicer's settings, so the choice is stored twice: in the
Slicer settings (what the checkbox shows) and in a small file in the user's home folder that
the server reads before every tool call. Without that file, or with the switch off, the server
refuses to do anything. There is no anonymisation layer: the AI drives the extension and reads
the files itself, so a filter on what AUAE reports would protect nothing. Protecting patient
data is the responsibility of whoever uses the AI, as DISCLAIMER says.

The installation. The server runs on Slicer's own Python, started directly without the Slicer
application, so nothing has to be installed on the computer besides Slicer. Its two packages
(``mcp`` and ``pydicom``) go into a folder of their own, ~/.auae_mcp/site, and never into
Slicer's site-packages: ``mcp`` pins recent versions of libraries Slicer also uses, and the
separate folder is seen only by the server. ``setupServer`` installs the packages and registers
the server with the AI clients the user picked; no client's configuration is touched unless the
user chose it.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SETTINGS_KEY = "AUAE/llmApiSupport"
SERVER_NAME = "auae"

# Shown before the support is switched on, and repeated in the README. AUAE does not filter
# what the AI sees: the AI drives the extension and reads the files itself, so keeping patient
# data protected is up to the person who lets the AI in.
DISCLAIMER = (
    "An agentic AI that runs AUAE works on your files. It can read everything in the folders "
    "you point it at, including the patient names, IDs and dates stored in DICOM headers and in "
    "file and folder names, and what it reads is sent to the company that provides the AI.\n\n"
    "AUAE does not anonymise or filter this data. You are responsible for how the AI is used: "
    "for de-identifying the data beforehand where that is required, for the consent and "
    "approvals that cover it, for the terms of the AI provider, and for complying with the "
    "data-protection rules that apply to you (for example the GDPR)."
)
SERVER_PACKAGES = ("mcp", "pydicom")


# --- The switch ------------------------------------------------------------------------
def serverHome():
    """Folder shared with the server: ~/.auae_mcp."""
    return Path(os.path.expanduser("~")) / ".auae_mcp"


def settingsFilePath():
    """The file the MCP server reads: ~/.auae_mcp/settings.json."""
    return serverHome() / "settings.json"


def _setting(key, default):
    import qt
    return str(qt.QSettings().value(key, default)).lower() in ("true", "1")


def isEnabled():
    """True when the user switched the LLM API support on. Off by default."""
    return _setting(SETTINGS_KEY, False)


def _writeSettingsFile():
    """Write the choice to the file the MCP server reads."""
    path = settingsFilePath()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"llm_api_enabled": isEnabled()}, handle)


def setEnabled(enabled):
    """Store the choice in the Slicer settings and in the file the MCP server reads."""
    import qt
    qt.QSettings().setValue(SETTINGS_KEY, bool(enabled))
    _writeSettingsFile()


def syncSettingsFile():
    """Make the file match the Slicer setting (it may be missing, or edited by hand)."""
    try:
        _writeSettingsFile()
    except Exception:  # noqa: BLE001
        pass


# --- The server ------------------------------------------------------------------------
def serverScriptPath():
    """Path of the MCP server script shipped with the extension (it may be absent in a package)."""
    return Path(__file__).resolve().parents[2] / "MCP" / "auae_mcp.py"


def serverSitePath():
    """Folder holding the server's packages, apart from Slicer's own: ~/.auae_mcp/site."""
    return serverHome() / "site"


def slicerPythonPath():
    """Slicer's Python interpreter (PythonSlicer), which runs the server without the application."""
    return Path(shutil.which("PythonSlicer") or sys.executable).resolve()


def serverCommand():
    """The command an assistant uses to start the server: [python, script]."""
    return [slicerPythonPath().as_posix(), serverScriptPath().as_posix()]


def isServerInstalled():
    """True when the server's packages were installed for this Slicer's Python version."""
    marker = serverSitePath() / "python_version.txt"
    try:
        return (serverSitePath() / "mcp").is_dir() and marker.read_text().strip() == "%d.%d" % sys.version_info[:2]
    except OSError:
        return False


def serverCommandLine():
    """The server command as one line, for the clients that are registered by hand."""
    return '"%s" "%s"' % tuple(serverCommand())


def installServerPackages(log):
    """Install mcp and pydicom with Slicer's Python into the server's own folder.

    ``pip --target`` puts the packages and all they depend on in that folder and leaves
    Slicer's site-packages alone. The folder is tied to the Python version, which is recorded
    so that the server never loads packages built for another one.
    """
    from . import Dependencies
    site = serverSitePath()
    site.mkdir(parents=True, exist_ok=True)
    log("Installing the MCP server packages (%s) into %s ..." % (", ".join(SERVER_PACKAGES), site))
    Dependencies.runSlicerPython(["-m", "pip", "install", "--upgrade", "--target", str(site), *SERVER_PACKAGES])
    (site / "python_version.txt").write_text("%d.%d" % sys.version_info[:2])
    log("MCP server packages installed.")


def _startupEnvironment():
    try:
        import slicer
        return slicer.util.startupEnvironment()
    except Exception:  # noqa: BLE001
        return None


def _run(command):
    """Run a shell command with the environment Slicer was started from. Returns (ok, output)."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, env=_startupEnvironment(), shell=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:  # noqa: BLE001
        return (False, str(exc))
    return (completed.returncode == 0, (completed.stdout + completed.stderr).strip())


# --- Registration with the AI clients ---------------------------------------------------
# The server speaks MCP over stdio, the transport every local client supports, so the same
# command works in all of them. Two kinds of client are registered automatically: those with a
# command-line tool that manages their MCP servers, and those that read a JSON file with an
# "mcpServers" table. Any other client is registered by hand with serverCommand().

def _which(program):
    environment = _startupEnvironment() or os.environ
    return shutil.which(program, path=environment.get("PATH"))


def _registerWithCli(label, program, removeArguments, addArguments, log):
    """Register through a client's own command-line tool. None when the tool is not installed.

    An entry of the same name is removed first, so the registration always points at this Slicer.
    """
    executable = _which(program)
    if not executable:
        return None
    _run(subprocess.list2cmdline([executable, *removeArguments]))  # fails when absent: fine
    ok, output = _run(subprocess.list2cmdline([executable, *addArguments, *serverCommand()]))
    log("%s: %s" % (label, "registered." if ok else "registration failed. " + output[-300:]))
    return ok


def registerWithClaudeCode(log, name=SERVER_NAME):
    """Register the server in Claude Code for the whole user account."""
    return _registerWithCli("Claude Code", "claude", ["mcp", "remove", name, "-s", "user"],
                            ["mcp", "add", "--scope", "user", name, "--"], log)


def registerWithCodex(log, name=SERVER_NAME):
    """Register the server in OpenAI Codex (CLI and IDE extension share ~/.codex/config.toml)."""
    return _registerWithCli("Codex", "codex", ["mcp", "remove", name], ["mcp", "add", name, "--"], log)


def claudeDesktopConfigPath():
    """Where Claude Desktop keeps its MCP configuration on this system."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")) / "Claude" / "claude_desktop_config.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def jsonConfigClients():
    """Clients that read a JSON file with an "mcpServers" table: {label: path of the file}."""
    return {
        "Claude Desktop": claudeDesktopConfigPath(),
        "Gemini CLI": Path.home() / ".gemini" / "settings.json",
        "Cursor": Path.home() / ".cursor" / "mcp.json",
    }


def registerInJsonConfig(label, configPath, log, name=SERVER_NAME):
    """Add the server to a client's JSON configuration file.

    Returns None when the client is not installed (the folder of the file is absent), else
    True or False. Only the entry of this server is written; everything else in the file is
    kept, and the file as it was is saved beside it with a .auae-backup suffix.
    """
    path = Path(configPath)
    if not path.parent.is_dir():
        return None
    try:
        config = {}
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            config = json.loads(text) if text.strip() else {}
            if not isinstance(config, dict):
                raise ValueError("unexpected content")
            shutil.copyfile(path, str(path) + ".auae-backup")
        command, script = serverCommand()
        servers = config.setdefault("mcpServers", {})
        servers[name] = {"command": command, "args": [script]}
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log("%s: could not update %s (%s). The file was left as it was." % (label, path, exc))
        return False
    log("%s: registered in %s. Restart it to load the server." % (label, path))
    return True


def detectClients():
    """Which AI clients are installed on this computer: {label: bool}, in display order.

    Nothing is changed here; the user picks from this list which ones to register.
    """
    found = {"Claude Code": bool(_which("claude")), "Codex": bool(_which("codex"))}
    for label, path in jsonConfigClients().items():
        found[label] = path.parent.is_dir()
    return found


def registerWith(label, log):
    """Register the server with one client. True, False, or None when it is not installed."""
    if label == "Claude Code":
        return registerWithClaudeCode(log)
    if label == "Codex":
        return registerWithCodex(log)
    path = jsonConfigClients().get(label)
    if path is None:
        log("%s: unknown client, skipped." % label)
        return None
    return registerInJsonConfig(label, path, log)


def setupServer(log, clients=()):
    """Install the server and register it with the clients the user chose.

    ``clients`` holds labels from detectClients(); with none, only the packages are installed
    and the server is left for the user to register by hand. No client is touched unless it is
    named here. Returns {"installed": bool, "clients": {label: True | False | None}}.
    """
    result = {"installed": False, "clients": {}}
    if not serverScriptPath().is_file():
        log("The MCP server script was not found at %s." % serverScriptPath())
        return result
    installServerPackages(log)
    result["installed"] = isServerInstalled()
    if not result["installed"]:
        log("The MCP server packages are missing after the installation; see the Slicer log.")
        return result
    for label in clients:
        result["clients"][label] = registerWith(label, log)
    return result
