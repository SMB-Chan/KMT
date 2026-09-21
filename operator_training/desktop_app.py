"""Register the human-operation cockpit as a desktop application.

The human-operation mode (``Start Manual`` in the browser cockpit) is
exposed as a freedesktop application: ``app install`` writes a
``.desktop`` entry, a launcher wrapper and the application icon into
the user's XDG directories, and ``app launch`` makes sure the
``operator_training serve`` HTTP server is running before opening
``http://<host>:<port>/cockpit/`` in the default browser.

Layout written by :func:`install` (all paths below ``$HOME``)::

    ~/.local/share/applications/kmt-operator-cockpit.desktop
    ~/.local/share/icons/hicolor/<size>x<size>/apps/kmt-operator-cockpit.png
    ~/.local/bin/kmt-operator-cockpit

Runtime state (server pid / log) lives in the git-ignored ``var/app``
directory of the repository so that ``app stop`` can reach a server
that ``app launch`` detached.

Everything is pure file handling plus ``subprocess``; the functions
take an explicit :class:`AppPaths` so tests can install into a
temporary ``$HOME`` without touching the real one.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

APP_ID = "kmt-operator-cockpit"
APP_NAME = "KMT Operator Cockpit"
APP_NAME_JA = "KMT 人間操作コックピット"
GENERIC_NAME = "Flying-Boat Drone Operator Cockpit"
GENERIC_NAME_JA = "飛行艇ドローン 人間操作コックピット"
COMMENT = ("Start the KMT operator server and open the manual-control "
           "cockpit in a browser")
COMMENT_JA = ("KMT の操縦サーバを起動し、人間操作モードのコックピットを"
              "ブラウザで開く")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
HOST_ENV = "KMT_OPERATOR_HOST"
PORT_ENV = "KMT_OPERATOR_PORT"

ICON_SIZES = (256, 128, 64, 48)
_ASSETS = Path(__file__).resolve().parent / "assets"
ICON_SOURCE = _ASSETS / f"{APP_ID}.png"

HEALTH_TIMEOUT_S = 20.0
STOP_TIMEOUT_S = 5.0

# Browser openers in preference order; the value is the argv prefix.
_OPENERS: tuple[tuple[str, list[str]], ...] = (
    ("xdg-open", ["xdg-open"]),
    ("gio", ["gio", "open"]),
    ("sensible-browser", ["sensible-browser"]),
    ("x-www-browser", ["x-www-browser"]),
    ("www-browser", ["www-browser"]),
    ("firefox", ["firefox"]),
    ("firefox-esr", ["firefox-esr"]),
    ("chromium", ["chromium"]),
    ("chromium-browser", ["chromium-browser"]),
    ("google-chrome", ["google-chrome"]),
)


@dataclass(frozen=True)
class AppPaths:
    """Every filesystem location the registration touches."""

    home: Path
    repo_root: Path
    applications_dir: Path
    desktop_file: Path
    icon_files: tuple[Path, ...]
    bin_dir: Path
    wrapper: Path
    desktop_dir: Path | None
    desktop_link: Path | None
    runtime_dir: Path
    pid_file: Path
    log_file: Path
    state_file: Path

    @property
    def icon_file(self) -> Path:
        """Largest installed icon; also the ``Icon=`` value."""
        return self.icon_files[0]


def resolve_paths(home: Path | str | None = None,
                  repo_root: Path | str | None = None) -> AppPaths:
    """Resolve XDG locations for ``home`` (default ``$HOME``).

    An explicit ``home`` ignores ``$XDG_DATA_HOME`` / ``$XDG_DESKTOP_DIR``
    so that tests and ``--home`` never touch the real user profile.
    """
    home_p = Path(home) if home is not None else Path(
        os.environ.get("HOME") or str(Path.home()))
    honor_env = home is None
    repo = Path(repo_root) if repo_root is not None else Path(
        __file__).resolve().parents[1]
    xdg_env = os.environ.get("XDG_DATA_HOME") if honor_env else None
    xdg_data = Path(xdg_env) if xdg_env else home_p / ".local" / "share"
    applications = xdg_data / "applications"
    icons_base = xdg_data / "icons" / "hicolor"
    icon_files = tuple(
        icons_base / f"{size}x{size}" / "apps" / f"{APP_ID}.png"
        for size in ICON_SIZES)
    bin_dir = home_p / ".local" / "bin"
    desktop_dir = _desktop_dir(home_p, honor_env)
    return AppPaths(
        home=home_p,
        repo_root=repo,
        applications_dir=applications,
        desktop_file=applications / f"{APP_ID}.desktop",
        icon_files=icon_files,
        bin_dir=bin_dir,
        wrapper=bin_dir / APP_ID,
        desktop_dir=desktop_dir,
        desktop_link=(desktop_dir / f"{APP_ID}.desktop"
                      if desktop_dir is not None else None),
        runtime_dir=repo / "var" / "app",
        pid_file=repo / "var" / "app" / "server.pid",
        log_file=repo / "var" / "app" / "server.log",
        state_file=repo / "var" / "app" / "install.json",
    )


def _desktop_dir(home: Path, honor_env: bool = True) -> Path | None:
    """Read ``XDG_DESKTOP_DIR`` from user-dirs.dirs, else guess."""
    env_dir = os.environ.get("XDG_DESKTOP_DIR") if honor_env else None
    if env_dir:
        return Path(env_dir)
    user_dirs = home / ".config" / "user-dirs.dirs"
    if user_dirs.is_file():
        for line in user_dirs.read_text(encoding="utf-8",
                                        errors="replace").splitlines():
            key, _, value = line.strip().partition("=")
            if key == "XDG_DESKTOP_DIR":
                value = value.strip().strip('"')
                value = value.replace("$HOME", str(home))
                if value:
                    return Path(value)
    for candidate in (home / "Desktop", home / "デスクトップ"):
        if candidate.is_dir():
            return candidate
    return None


def default_host() -> str:
    return os.environ.get(HOST_ENV) or DEFAULT_HOST


def default_port() -> int:
    raw = os.environ.get(PORT_ENV)
    if raw:
        return int(raw)
    return DEFAULT_PORT


def cockpit_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/cockpit/"


def health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


# ---------------------------------------------------------------- probing

def probe_health(host: str, port: int, timeout: float = 0.5) -> bool:
    """True when the operator server answers ``GET /health``."""
    try:
        with urllib.request.urlopen(health_url(host, port),
                                    timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_health(host: str, port: int,
                timeout_s: float = HEALTH_TIMEOUT_S,
                interval: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if probe_health(host, port):
            return True
        time.sleep(interval)
    return probe_health(host, port)


def port_busy(host: str, port: int, timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int | None) -> bool:
    """True while ``pid`` runs; zombies (terminated, unreaped) are dead."""
    if not pid:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            state = stat.read_text(encoding="ascii",
                                   errors="replace").rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            return False
        return state not in ("Z", "X", "x")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists but owned by someone else
        return True
    return True


# Popen handles of servers this process started, so stop can reap them.
_CHILDREN: dict[int, subprocess.Popen] = {}


def _reap(pid: int) -> None:
    proc = _CHILDREN.pop(pid, None)
    if proc is not None:
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass  # not our child; its own parent reaps it


def find_opener() -> tuple[str, list[str]] | None:
    for name, argv in _OPENERS:
        if shutil.which(name):
            return name, list(argv)
    return None


def open_url(url: str, opener: tuple[str, list[str]] | None = None) -> str:
    """Open ``url`` detached; returns the opener name used."""
    found = opener or find_opener()
    if found is None:
        raise RuntimeError(
            "no browser opener found (tried: "
            + ", ".join(name for name, _ in _OPENERS) + ")")
    name, argv = found
    subprocess.Popen(argv + [url], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return name


# ------------------------------------------------------- server lifecycle

def _serve_command(paths: AppPaths, host: str, port: int,
                   python_exe: str) -> list[str]:
    return [
        python_exe, "-m", "operator_training", "serve",
        "--host", host, "--port", str(port),
        "--recordings-root", str(paths.repo_root / "var" / "operator_sessions"),
        "--cockpit-dir", str(paths.repo_root / "web" / "operator"),
    ]


def start_server(paths: AppPaths, host: str, port: int,
                 python_exe: str = sys.executable) -> dict:
    """Detach ``operator_training serve`` and record its pid."""
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    log = open(paths.log_file, "ab", buffering=0)
    proc = subprocess.Popen(
        _serve_command(paths, host, port, python_exe),
        cwd=str(paths.repo_root),
        stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    log.close()
    _CHILDREN[proc.pid] = proc
    paths.pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    return {"pid": proc.pid, "log": str(paths.log_file)}


def stop_server(paths: AppPaths,
                timeout_s: float = STOP_TIMEOUT_S) -> dict:
    """SIGTERM (then SIGKILL) the server started by :func:`start_server`."""
    pid = read_pid(paths.pid_file)
    if pid is None:
        return {"stopped": False, "reason": "no pid file"}
    if not pid_alive(pid):
        _reap(pid)
        paths.pid_file.unlink(missing_ok=True)
        return {"stopped": False, "reason": "process already gone",
                "pid": pid}
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and pid_alive(pid):
        time.sleep(0.1)
    if pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
    _reap(pid)
    paths.pid_file.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid}


# ------------------------------------------------------------- rendering

def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def render_wrapper(paths: AppPaths, host: str, port: int,
                   python_exe: str) -> str:
    """Bash wrapper; keeps the (possibly non-ASCII) repo path quoted."""
    lines = [
        "#!/usr/bin/env bash",
        f"# {APP_NAME} launcher, generated by operator_training.desktop_app.",
        "# Re-generate with: python3 -m operator_training app install",
        "set -euo pipefail",
        f"REPO={_sh_quote(str(paths.repo_root))}",
        f"PYTHON={_sh_quote(python_exe)}",
        f"HOST={_sh_quote(host)}",
        f"PORT={_sh_quote(str(port))}",
        'cd "$REPO"',
        'exec "$PYTHON" -m operator_training app launch '
        '--host "$HOST" --port "$PORT" "$@"',
        "",
    ]
    return "\n".join(lines)


def render_desktop_entry(paths: AppPaths, port: int | None = None) -> str:
    """freedesktop ``.desktop`` entry text for the installed wrapper."""
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.1",
        f"Name={APP_NAME}",
        f"Name[ja]={APP_NAME_JA}",
        f"GenericName={GENERIC_NAME}",
        f"GenericName[ja]={GENERIC_NAME_JA}",
        f"Comment={COMMENT}",
        f"Comment[ja]={COMMENT_JA}",
        f"Exec={paths.wrapper}",
        f"Icon={paths.icon_file}",
        "Terminal=false",
        "Categories=Education;",
        "Keywords=kmt;drone;operator;cockpit;flight;simulator;uav;",
        "StartupNotify=false",
        f"X-KMT-Repo={paths.repo_root}",
        f"X-KMT-Port={port if port is not None else default_port()}",
        "",
    ]
    return "\n".join(lines)


def validate_desktop(desktop_file: Path) -> tuple[int, str]:
    """Run ``desktop-file-validate`` when available; (0, "skipped") else."""
    tool = shutil.which("desktop-file-validate")
    if tool is None:
        return 0, "skipped: desktop-file-validate not installed"
    proc = subprocess.run([tool, str(desktop_file)],
                          capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


# ------------------------------------------------------------- operations

def install(paths: AppPaths, host: str, port: int,
            python_exe: str = sys.executable,
            with_desktop_link: bool = False) -> dict:
    """Write icon, wrapper and ``.desktop`` entry; refresh the menu DB."""
    if not ICON_SOURCE.is_file():
        raise FileNotFoundError(
            f"icon source missing: {ICON_SOURCE} "
            "(run: python3 make_app_icon.py --all-sizes)")

    paths.applications_dir.mkdir(parents=True, exist_ok=True)
    paths.bin_dir.mkdir(parents=True, exist_ok=True)

    installed_icons = []
    for icon_path in paths.icon_files:
        icon_path.parent.mkdir(parents=True, exist_ok=True)
        size = int(icon_path.parent.parent.name.split("x")[0])
        source = _ASSETS / (f"{APP_ID}.png" if size == ICON_SIZES[0]
                            else f"{APP_ID}-{size}.png")
        if not source.is_file():
            source = ICON_SOURCE
        shutil.copyfile(source, icon_path)
        icon_path.chmod(0o644)
        installed_icons.append(str(icon_path))

    wrapper_text = render_wrapper(paths, host, port, python_exe)
    paths.wrapper.write_text(wrapper_text, encoding="utf-8")
    paths.wrapper.chmod(0o755)

    paths.desktop_file.write_text(render_desktop_entry(paths, port),
                                  encoding="utf-8")
    paths.desktop_file.chmod(0o644)

    link = None
    if with_desktop_link and paths.desktop_link is not None:
        paths.desktop_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths.desktop_file, paths.desktop_link)
        paths.desktop_link.chmod(0o755)
        link = str(paths.desktop_link)
        gio = shutil.which("gio")
        if gio:  # GNOME refuses to trust desktop files otherwise
            subprocess.run([gio, "set", link, "metadata::trusted", "true"],
                           capture_output=True)

    db_out = "skipped: update-desktop-database not installed"
    db_tool = shutil.which("update-desktop-database")
    if db_tool:
        proc = subprocess.run([db_tool, str(paths.applications_dir)],
                              capture_output=True, text=True)
        db_out = (proc.stdout + proc.stderr).strip() or f"ok ({proc.returncode})"

    code, message = validate_desktop(paths.desktop_file)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text(json.dumps({
        "app_id": APP_ID,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "port": port,
        "python": python_exe,
        "repo_root": str(paths.repo_root),
        "desktop_file": str(paths.desktop_file),
        "wrapper": str(paths.wrapper),
        "icons": installed_icons,
        "desktop_link": link,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "installed": True,
        "desktop_file": str(paths.desktop_file),
        "wrapper": str(paths.wrapper),
        "icons": installed_icons,
        "desktop_link": link,
        "desktop_database": db_out,
        "validate_returncode": code,
        "validate_output": message,
        "url": cockpit_url(host, port),
    }


def uninstall(paths: AppPaths, remove_runtime: bool = False) -> dict:
    """Remove every file :func:`install` wrote."""
    removed = []
    for target in (paths.desktop_file, paths.wrapper, paths.state_file,
                   *paths.icon_files):
        if target.exists():
            target.unlink()
            removed.append(str(target))
    if paths.desktop_link is not None and paths.desktop_link.exists():
        paths.desktop_link.unlink()
        removed.append(str(paths.desktop_link))
    db_tool = shutil.which("update-desktop-database")
    if db_tool and paths.applications_dir.is_dir():
        subprocess.run([db_tool, str(paths.applications_dir)],
                       capture_output=True, text=True)
    if remove_runtime and paths.runtime_dir.is_dir():
        shutil.rmtree(paths.runtime_dir)
        removed.append(str(paths.runtime_dir))
    return {"uninstalled": True, "removed": removed}


def status(paths: AppPaths, host: str, port: int) -> dict:
    pid = read_pid(paths.pid_file)
    healthy = probe_health(host, port)
    return {
        "app_id": APP_ID,
        "installed": paths.desktop_file.is_file(),
        "desktop_file": str(paths.desktop_file),
        "wrapper": str(paths.wrapper),
        "wrapper_exists": paths.wrapper.is_file(),
        "icon": str(paths.icon_file),
        "icon_exists": paths.icon_file.is_file(),
        "desktop_link": (str(paths.desktop_link)
                         if paths.desktop_link is not None else None),
        "server": {
            "healthy": healthy,
            "pid": pid,
            "pid_alive": pid_alive(pid),
            "port": port,
            "host": host,
        },
        "url": cockpit_url(host, port),
    }


def launch(paths: AppPaths, host: str, port: int,
           python_exe: str = sys.executable,
           open_browser: bool = True,
           opener: tuple[str, list[str]] | None = None,
           timeout_s: float = HEALTH_TIMEOUT_S) -> dict:
    """Ensure the server is up, then open the cockpit in a browser."""
    result: dict = {"host": host, "port": port,
                    "url": cockpit_url(host, port)}
    if probe_health(host, port):
        result.update(started=False, reused=True, pid=read_pid(paths.pid_file))
    else:
        if port_busy(host, port):
            raise RuntimeError(
                f"port {host}:{port} is busy but does not answer /health; "
                "stop the other process or choose another --port")
        info = start_server(paths, host, port, python_exe)
        if not wait_health(host, port, timeout_s=timeout_s):
            tail = ""
            if paths.log_file.is_file():
                tail = paths.log_file.read_text(
                    encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"server on {host}:{port} did not become healthy within "
                f"{timeout_s:.0f}s; log tail:\n{tail}")
        result.update(started=True, reused=False, pid=info["pid"],
                      log=info["log"])
    if open_browser:
        result["opener"] = open_url(result["url"], opener)
    else:
        result["opener"] = None
    return result


__all__ = [
    "APP_ID", "APP_NAME", "APP_NAME_JA", "DEFAULT_HOST", "DEFAULT_PORT",
    "HOST_ENV", "PORT_ENV", "ICON_SIZES", "ICON_SOURCE",
    "AppPaths", "resolve_paths", "default_host", "default_port",
    "cockpit_url", "health_url", "probe_health", "wait_health", "port_busy",
    "read_pid", "pid_alive", "find_opener", "open_url", "start_server",
    "stop_server", "render_wrapper", "render_desktop_entry",
    "validate_desktop", "install", "uninstall", "status", "launch",
]
