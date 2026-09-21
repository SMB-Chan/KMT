"""Tests for the desktop-application registration of the cockpit.

Covers ``operator_training/desktop_app.py`` and the ``app`` CLI
subcommand: rendering of the wrapper and ``.desktop`` entry, install /
uninstall into a throwaway ``$HOME``, the server lifecycle used by
``app launch`` / ``app stop``, and the icon asset.

Nothing here touches the real user profile: every test resolves paths
against a temporary home directory.
"""
import http.server
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

try:
    from PIL import Image  # noqa: F401  (only used to skip icon render test)
    _HAS_PIL = True
except ImportError:  # pragma: no cover - Pillow is in requirements
    _HAS_PIL = False

from operator_training import desktop_app as da

WORKSPACE = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    """Minimal stand-in for the operator server's /health endpoint."""

    def do_GET(self):  # noqa: N802 - stdlib naming
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # silence the test log
        pass


class DesktopAppTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name) / "home"
        self.home.mkdir(parents=True)
        self.paths = da.resolve_paths(home=self.home)
        self.port = _free_port()

    # ---------------------------------------------------------- rendering

    def test_render_wrapper_quotes_the_repo_path(self):
        text = da.render_wrapper(self.paths, "127.0.0.1", self.port,
                                 "/usr/bin/python3")
        self.assertIn("#!/usr/bin/env bash", text)
        self.assertIn("set -euo pipefail", text)
        self.assertIn(f"REPO='{WORKSPACE}'", text)
        self.assertIn(f"PORT='{self.port}'", text)
        self.assertIn('exec "$PYTHON" -m operator_training app launch', text)

    def test_render_wrapper_escapes_single_quotes(self):
        odd = da.resolve_paths(home=self.home,
                               repo_root=Path("/tmp/it's a repo"))
        text = da.render_wrapper(odd, "127.0.0.1", self.port, "python3")
        self.assertIn("REPO='/tmp/it'\\''s a repo'", text)

    def test_render_desktop_entry_has_required_keys(self):
        text = da.render_desktop_entry(self.paths, self.port)
        for key in ("[Desktop Entry]", "Type=Application",
                    f"Name={da.APP_NAME}", f"Name[ja]={da.APP_NAME_JA}",
                    f"Exec={self.paths.wrapper}",
                    f"Icon={self.paths.icon_file}",
                    "Terminal=false", "Categories=Education;"):
            self.assertIn(key, text)
        self.assertTrue(text.endswith("\n"))
        # Exactly one main category keeps desktop-file-validate quiet.
        self.assertEqual(text.count("Categories="), 1)

    def test_desktop_entry_passes_desktop_file_validate(self):
        tool = shutil.which("desktop-file-validate")
        if tool is None:
            self.skipTest("desktop-file-validate not installed")
        entry = self.home / "entry.desktop"
        entry.write_text(da.render_desktop_entry(self.paths, self.port),
                         encoding="utf-8")
        proc = subprocess.run([tool, str(entry)], capture_output=True,
                              text=True)
        self.assertEqual(proc.returncode, 0,
                         proc.stdout + proc.stderr)
        self.assertEqual((proc.stdout + proc.stderr).strip(), "")

    def test_wrapper_passes_bash_syntax_check(self):
        if shutil.which("bash") is None:
            self.skipTest("bash not installed")
        script = self.home / "wrapper.sh"
        script.write_text(da.render_wrapper(self.paths, "127.0.0.1",
                                            self.port, "/usr/bin/python3"),
                          encoding="utf-8")
        proc = subprocess.run(["bash", "-n", str(script)],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    # ------------------------------------------------------------ install

    def test_install_writes_entry_wrapper_and_icons(self):
        info = da.install(self.paths, "127.0.0.1", self.port)
        self.assertTrue(info["installed"])
        self.assertTrue(self.paths.desktop_file.is_file())
        self.assertTrue(self.paths.wrapper.is_file())
        self.assertEqual(self.paths.wrapper.stat().st_mode & 0o111, 0o111)
        self.assertEqual(self.paths.desktop_file.stat().st_mode & 0o777,
                         0o644)
        for icon in self.paths.icon_files:
            self.assertTrue(icon.is_file(), icon)
            self.assertTrue(icon.read_bytes().startswith(b"\x89PNG"))
        state = json.loads(self.paths.state_file.read_text(encoding="utf-8"))
        self.assertEqual(state["app_id"], da.APP_ID)
        self.assertEqual(state["port"], self.port)
        entry = self.paths.desktop_file.read_text(encoding="utf-8")
        self.assertIn(f"Exec={self.paths.wrapper}", entry)
        self.assertIn(f"X-KMT-Port={self.port}", entry)
        if shutil.which("desktop-file-validate"):
            self.assertEqual(info["validate_returncode"], 0,
                             info["validate_output"])

    def test_install_can_place_a_desktop_link(self):
        desktop_dir = self.home / "Desktop"
        desktop_dir.mkdir()
        # desktop_dir is discovered at resolve time, so resolve again
        paths = da.resolve_paths(home=self.home)
        info = da.install(paths, "127.0.0.1", self.port,
                          with_desktop_link=True)
        link = desktop_dir / f"{da.APP_ID}.desktop"
        self.assertEqual(info["desktop_link"], str(link))
        self.assertTrue(link.is_file())
        self.assertEqual(link.stat().st_mode & 0o111, 0o111)

    def test_install_without_a_desktop_dir_skips_the_link(self):
        info = da.install(self.paths, "127.0.0.1", self.port,
                          with_desktop_link=True)
        self.assertIsNone(info["desktop_link"])

    def test_uninstall_removes_every_installed_file(self):
        desktop_dir = self.home / "Desktop"
        desktop_dir.mkdir()
        da.install(self.paths, "127.0.0.1", self.port,
                   with_desktop_link=True)
        info = da.uninstall(self.paths)
        self.assertTrue(info["uninstalled"])
        for target in (self.paths.desktop_file, self.paths.wrapper,
                       self.paths.state_file, *self.paths.icon_files,
                       desktop_dir / f"{da.APP_ID}.desktop"):
            self.assertFalse(target.exists(), target)
        # runtime state survives unless explicitly removed
        self.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.assertTrue(self.paths.runtime_dir.is_dir())
        da.uninstall(self.paths, remove_runtime=True)
        self.assertFalse(self.paths.runtime_dir.exists())

    # ------------------------------------------------------------- status

    def test_status_on_a_clean_home(self):
        info = da.status(self.paths, "127.0.0.1", self.port)
        self.assertFalse(info["installed"])
        self.assertFalse(info["wrapper_exists"])
        self.assertFalse(info["icon_exists"])
        self.assertFalse(info["server"]["healthy"])
        self.assertIsNone(info["server"]["pid"])
        self.assertEqual(info["url"],
                         f"http://127.0.0.1:{self.port}/cockpit/")

    def test_status_after_install(self):
        da.install(self.paths, "127.0.0.1", self.port)
        info = da.status(self.paths, "127.0.0.1", self.port)
        self.assertTrue(info["installed"])
        self.assertTrue(info["wrapper_exists"])
        self.assertTrue(info["icon_exists"])

    # ------------------------------------------------------------- launch

    def test_launch_reuses_an_already_healthy_server(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port),
                                                 _HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        try:
            info = da.launch(self.paths, "127.0.0.1", self.port,
                             open_browser=False)
            self.assertTrue(info["reused"])
            self.assertFalse(info["started"])
            self.assertIsNone(info["opener"])
            # nothing to stop: the stub was not started by start_server
            stopped = da.stop_server(self.paths)
            self.assertFalse(stopped["stopped"])
            self.assertEqual(stopped["reason"], "no pid file")
        finally:
            server.shutdown()

    def test_launch_raises_when_the_port_is_busy_but_silent(self):
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", self.port))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        with self.assertRaises(RuntimeError):
            da.launch(self.paths, "127.0.0.1", self.port,
                      open_browser=False, timeout_s=1.0)

    def test_launch_starts_and_stop_terminates_the_real_server(self):
        info = da.launch(self.paths, "127.0.0.1", self.port,
                         python_exe=sys.executable, open_browser=False)
        self.addCleanup(da.stop_server, self.paths)
        self.assertTrue(info["started"])
        self.assertTrue(da.probe_health("127.0.0.1", self.port))
        import urllib.request
        with urllib.request.urlopen(info["url"], timeout=5) as resp:
            html = resp.read().decode("utf-8")
        self.assertIn("KMT Operator Cockpit", html)
        self.assertEqual(da.read_pid(self.paths.pid_file), info["pid"])

        stopped = da.stop_server(self.paths)
        self.assertTrue(stopped["stopped"])
        self.assertFalse(da.pid_alive(info["pid"]))
        self.assertFalse(da.probe_health("127.0.0.1", self.port))
        self.assertFalse(self.paths.pid_file.exists())

    # ---------------------------------------------------------------- env

    def test_host_and_port_come_from_the_environment(self):
        old_host = os.environ.get(da.HOST_ENV)
        old_port = os.environ.get(da.PORT_ENV)
        try:
            os.environ[da.HOST_ENV] = "localhost"
            os.environ[da.PORT_ENV] = "9123"
            self.assertEqual(da.default_host(), "localhost")
            self.assertEqual(da.default_port(), 9123)
        finally:
            for key, value in ((da.HOST_ENV, old_host),
                               (da.PORT_ENV, old_port)):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual(da.default_port(), da.DEFAULT_PORT)

    def test_explicit_home_ignores_xdg_data_home(self):
        old = os.environ.get("XDG_DATA_HOME")
        os.environ["XDG_DATA_HOME"] = str(self.home / "should-not-be-used")
        try:
            paths = da.resolve_paths(home=self.home)
        finally:
            if old is None:
                os.environ.pop("XDG_DATA_HOME", None)
            else:
                os.environ["XDG_DATA_HOME"] = old
        self.assertEqual(paths.applications_dir,
                         self.home / ".local" / "share" / "applications")

    # --------------------------------------------------------------- icon

    def test_icon_asset_is_a_256_px_png(self):
        self.assertTrue(da.ICON_SOURCE.is_file())
        blob = da.ICON_SOURCE.read_bytes()
        self.assertTrue(blob.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", blob[16:24])
        self.assertEqual((width, height), (256, 256))

    @unittest.skipUnless(_HAS_PIL, "Pillow not installed")
    def test_icon_rendering_is_deterministic(self):
        sys.path.insert(0, str(WORKSPACE))
        self.addCleanup(sys.path.pop, 0)
        import make_app_icon
        first = make_app_icon.render_icon(64).tobytes()
        second = make_app_icon.render_icon(64).tobytes()
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            make_app_icon.render_icon(8)

    # ---------------------------------------------------------------- CLI

    def _cli(self, *argv):
        return subprocess.run(
            [sys.executable, "-m", "operator_training", "app", *argv],
            cwd=str(WORKSPACE), capture_output=True, text=True, timeout=120)

    def test_cli_status_and_install_round_trip(self):
        proc = self._cli("status", "--home", str(self.home),
                         "--port", str(self.port))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(json.loads(proc.stdout)["installed"])

        proc = self._cli("install", "--home", str(self.home),
                         "--port", str(self.port))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["installed"])
        self.assertTrue(self.paths.desktop_file.is_file())

        proc = self._cli("uninstall", "--home", str(self.home))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(self.paths.desktop_file.exists())

    def test_cli_launch_and_stop_round_trip(self):
        proc = self._cli("launch", "--home", str(self.home),
                         "--port", str(self.port), "--no-browser")
        self.addCleanup(self._cli, "stop", "--home", str(self.home),
                        "--port", str(self.port))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["started"])
        self.assertTrue(da.probe_health("127.0.0.1", self.port))

        proc = self._cli("stop", "--home", str(self.home),
                         "--port", str(self.port))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["stopped"])
        self.assertFalse(da.probe_health("127.0.0.1", self.port))

    def test_cli_unknown_action_is_rejected(self):
        proc = self._cli("fly")
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
