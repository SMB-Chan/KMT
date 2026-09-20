"""CLI entry point for operator-training sessions.

Usage:

    python3 -m operator_training replay  <source_dir> <target_dir>
    python3 -m operator_training smoke  <out_dir>
    python3 -m operator_training headless-keys <out_dir>

The smoke and headless-keys drivers exist for P0 verification only;
the proper pad/UI integration is P1+. Both use ``FakePad`` to drive
the session deterministically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (
    ControlEnvelope,
    ControlInput,
    Curriculum,
    Session,
    SessionConfig,
)
from .calibration import GamepadProfile, to_control_input
from .envelope import ControlEnvelope as _ControlEnvelope
from .recording import FileSink, NullSink
from .replay import replay_session


def _smoke(out_dir: Path, *, seed: int = 42, ticks: int = 200) -> int:
    """Drive a hybrid session until handover, then exit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SessionConfig(
        curriculum=Curriculum(
            curriculum_id="hybrid_baseline",
            curriculum_version="1",
            scenario="hybrid",
            handover_altitude_m=4.0,
            handover_airspeed_m_s=8.0,
            handover_vertical_speed_m_s=4.0,
            handover_bank_deg=20.0,
            handover_duration_s=0.5,
        ),
        envelope=ControlEnvelope.beginner(),
        seed=seed,
        target_takeoff_alt_m=15.0,
    )
    sess = Session.create(config=cfg, sink=FileSink(root=out_dir))
    sess.ready()
    sess.start_takeoff()

    requested = False
    completed = False
    for i in range(ticks):
        sess.tick(dt=0.05, human_input=None, input_seq=i)
        if sess.state.lifecycle in ("FINISHED", "ABORTED"):
            break
        if not requested and sess.state.pending == "OFFER_MANUAL":
            sess.request_handover()
            requested = True
        if requested:
            auto = sess._capture_auto_setpoints()
            ci = ControlInput(auto["throttle"], auto["pitch_deg"],
                              auto["bank_deg"], auto["rudder"])
            sess.tick(dt=0.05, human_input=ci, input_seq=i + 1000)
            if sess.state.authority == "HUMAN":
                completed = True
                break

    summary = sess.finalize()
    print(json.dumps({
        "out_dir": str(out_dir),
        "handover_requested": requested,
        "handover_completed": completed,
        "lifecycle_end": summary["lifecycle_end"],
        "phase_end": summary["phase_end"],
        "authority_end": summary["authority_end"],
        "end_reason": summary["end_reason"],
        "tick_count": summary["tick_count"],
        "sim_t_end": summary["sim_t_end"],
    }, indent=2))
    return 0 if completed else 1


def _headless_keys(out_dir: Path, *, seed: int = 42, ticks: int = 400) -> int:
    """Drive a manual (HUMAN from start) session using a synthetic pad."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SessionConfig(
        curriculum=Curriculum(
            curriculum_id="manual_baseline",
            curriculum_version="1",
            scenario="manual_takeoff",
        ),
        envelope=ControlEnvelope.beginner(),
        seed=seed,
        target_takeoff_alt_m=20.0,
    )
    sess = Session.create(config=cfg, sink=FileSink(root=out_dir))
    sess.ready()
    # Promote directly into RUNNING under HUMAN authority so the
    # session simulates a manual mode without an auto takeoff phase.
    sess.adapter.arm()
    sess.state.lifecycle = "RUNNING"
    sess.state.authority = "HUMAN"
    sess.state.phase = "TAKEOFF"

    profile = GamepadProfile()
    for i in range(ticks):
        pitch = max(-8.0, min(12.0, 5.0 + 2.0 * _sin(i * 0.1)))
        raw = {"pit": pitch / 12.0, "ban": 0.1 * _sin(i * 0.05),
               "rud": 0.05 * _cos(i * 0.05), "thr": 0.9}
        axes = to_control_input(
            profile, raw,
            pitch_range_deg=(cfg.envelope.pitch_lo, cfg.envelope.pitch_hi),
            bank_abs_deg=cfg.envelope.bank_abs,
            rudder_abs=cfg.envelope.rudder_abs,
        )
        try:
            ci = cfg.envelope.validate(axes)
        except Exception:
            ci = None
        sess.tick(dt=0.05, human_input=ci, input_seq=i, raw_input=raw)
        if sess.state.lifecycle in ("FINISHED", "ABORTED"):
            break
    summary = sess.finalize()
    print(json.dumps({
        "out_dir": str(out_dir),
        "lifecycle_end": summary["lifecycle_end"],
        "authority_end": summary["authority_end"],
        "sim_t_end": summary["sim_t_end"],
        "max_altitude_m": max(
            (row["state"]["altitude_m"] for row in _read_ticks(out_dir)),
            default=0.0,
        ),
    }, indent=2))
    return 0


def _replay(source: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    _, report = replay_session(source_root=source, target_root=target)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 2


def _read_ticks(root: Path):
    import json
    p = root / "ticks.jsonl"
    if not p.exists():
        return []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sin(x: float) -> float:
    import math
    return math.sin(x)


def _cos(x: float) -> float:
    import math
    return math.cos(x)


def _split_hostport(spec: str, default_port: int) -> tuple[str, int]:
    if not spec:
        return "127.0.0.1", default_port
    if ":" in spec:
        host, _, port_s = spec.rpartition(":")
        return host or "127.0.0.1", int(port_s)
    return spec, default_port


async def _serve(host: str, port: int, recordings_root: Path,
                 allowed_origins: list[str], allowed_hosts: list[str],
                 cockpit_dir: Path | None,
                 tick_hz: float, heartbeat_s: float,
                 mavlink_port: int = 0, mavlink_gcs: str = "127.0.0.1:14550") -> int:
    """Run the OperatorServer until SIGINT / SIGTERM."""
    import asyncio
    import signal

    from .curriculum import Curriculum
    from .envelope import ControlEnvelope
    from .server import OperatorServer, ServerConfig

    if recordings_root is not None:
        recordings_root.mkdir(parents=True, exist_ok=True)

    cfg = ServerConfig(
        allowed_origins=tuple(allowed_origins) if allowed_origins else
                         ("http://localhost", "http://127.0.0.1"),
        allowed_hosts=tuple(allowed_hosts) if allowed_hosts else
                      ("localhost", "127.0.0.1"),
        recordings_root=recordings_root,
        require_csrf=False,
        tick_hz=tick_hz,
        heartbeat_s=heartbeat_s,
    )
    server = OperatorServer(
        config=cfg,
        envelope=ControlEnvelope.beginner(),
        curriculum=Curriculum(
            curriculum_id="hybrid_baseline",
            curriculum_version="1",
            scenario="hybrid",
        ),
    )
    if cockpit_dir is not None and cockpit_dir.exists():
        server.static_root = cockpit_dir

    # Serve static cockpit files when present.
    static_dirs = []
    if cockpit_dir is not None and cockpit_dir.exists():
        static_dirs.append(cockpit_dir)

    stop = asyncio.Event()

    def _signal_handler():
        stop.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows / restricted envs: signal handlers cannot be added.
            pass

    try:
        await server.start(host=host, port=port)
    except OSError as exc:
        import errno
        if exc.errno != errno.EADDRINUSE:
            raise
        print(f"HTTP port {host}:{port} is already in use. "
              "Choose another --port (or --port 0 for a free port), "
              "or stop the existing server in its terminal. "
              "--mavlink-port is a separate UDP port.", file=sys.stderr)
        return 2
    mav_task = None
    mav_bridge = None
    if mavlink_port:
        from .mavlink_udp import MavlinkUdpBridge

        def _vehicle():
            for rec in server.registry.values():
                return rec.session.adapter.vehicle
            return None

        def _on_manual(ci):
            for rec in server.registry.values():
                rec.session._pending_input = ci
                rec.session._pending_seq = rec.session.state.last_input_seq + 1
                return

        gcs_host, gcs_port = _split_hostport(mavlink_gcs, 14550)
        mav_bridge = MavlinkUdpBridge(
            get_vehicle=_vehicle,
            bind_host=host, bind_port=mavlink_port,
            gcs_host=gcs_host, gcs_port=gcs_port,
            on_manual=_on_manual,
        )
        await mav_bridge.start()

        async def _mav_loop():
            while not stop.is_set():
                mav_bridge.emit()
                await asyncio.sleep(0.05)

        mav_task = asyncio.create_task(_mav_loop())
    print(f"OperatorServer listening on http://{host}:{server.bound_port}",
          flush=True)
    if static_dirs:
        print(f"Static cockpit files served from {static_dirs[0]} "
              f"at http://{host}:{server.bound_port}/cockpit/",
              flush=True)
    print("Endpoints:", flush=True)
    print(f"  GET  /api/capabilities", flush=True)
    print(f"  POST /api/sessions", flush=True)
    print(f"  GET  /api/sessions/<id>/summary", flush=True)
    print(f"  WS   /ws/sessions/<id>", flush=True)
    print(f"  GET  /health", flush=True)
    if static_dirs:
        print(f"  GET  /cockpit/", flush=True)
    if mav_bridge is not None:
        print(f"  UDP  MAVLink {host}:{mav_bridge.bound_port} -> {mavlink_gcs}",
              flush=True)

    try:
        await stop.wait()
    finally:
        if mav_task is not None:
            mav_task.cancel()
        if mav_bridge is not None:
            await mav_bridge.stop()
        await server.stop()
        print("OperatorServer stopped", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="operator_training",
        description="Operator cockpit and MAVLink UDP SITL.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke", help="drive a hybrid smoke session")
    p_smoke.add_argument("out_dir", type=Path)
    p_smoke.add_argument("--seed", type=int, default=42)
    p_smoke.add_argument("--ticks", type=int, default=200)

    p_keys = sub.add_parser("headless-keys",
                            help="drive a manual session via fake pad")
    p_keys.add_argument("out_dir", type=Path)
    p_keys.add_argument("--seed", type=int, default=42)
    p_keys.add_argument("--ticks", type=int, default=400)

    p_replay = sub.add_parser("replay",
                              help="replay a recorded session")
    p_replay.add_argument("source", type=Path)
    p_replay.add_argument("target", type=Path)

    p_report = sub.add_parser("report",
                              help="post-flight analysis for a recorded session")
    p_report.add_argument("session_dir", nargs="?", type=Path, default=None,
                          help="session directory (default: --compare is used)")
    p_report.add_argument("--md", type=Path, default=None,
                          help="write markdown report to this path")
    p_report.add_argument("--csv", type=Path, default=None,
                          help="write long-format time series CSV to this path")
    p_report.add_argument("--compare", nargs="+", type=Path, default=None,
                          help="compare several session directories side-by-side")
    p_report.add_argument("--json", type=Path, default=None,
                          help="write the FlightMetrics record as JSON")

    p_serve = sub.add_parser("serve",
                             help="run the HTTP+WebSocket operator server")
    p_serve.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8765,
                        help="listen port (default: 8765)")
    p_serve.add_argument("--recordings-root", type=Path, default=Path("var/operator_sessions"),
                        help="where to write session artefacts")
    p_serve.add_argument("--cockpit-dir", type=Path, default=Path("web/operator"),
                        help="static cockpit files to serve at /cockpit/")
    p_serve.add_argument("--allowed-origin", action="append", default=[],
                        help="origin to allow (repeatable)")
    p_serve.add_argument("--allowed-host", action="append", default=[],
                        help="host to allow (repeatable)")
    p_serve.add_argument("--tick-hz", type=float, default=20.0,
                        help="physics tick rate (default: 20)")
    p_serve.add_argument("--heartbeat-s", type=float, default=30.0,
                         help="server-side heartbeat watchdog (default: 30)")
    p_serve.add_argument("--mavlink-port", type=int, default=0,
                         help="if set, also speak MAVLink UDP on this port (QGC/MAVProxy)")
    p_serve.add_argument("--mavlink-gcs", default="127.0.0.1:14550",
                         help="GCS listen address host:port (default 127.0.0.1:14550)")

    p_mav = sub.add_parser("mavlink",
                           help="MAVLink UDP SITL for QGroundControl / MAVProxy")
    p_mav.add_argument("--bind", default="127.0.0.1")
    p_mav.add_argument("--port", type=int, default=14551,
                       help="local UDP bind (default 14551)")
    p_mav.add_argument("--gcs", default="127.0.0.1:14550",
                       help="GCS address host:port (QGC listens on 14550)")
    p_mav.add_argument("--seed", type=int, default=42)
    p_mav.add_argument("--tick-hz", type=float, default=20.0)
    p_mav.add_argument("--duration", type=float, default=0.0,
                       help="exit after N seconds (0 = until SIGINT)")

    args = parser.parse_args(argv)
    if args.cmd == "smoke":
        return _smoke(args.out_dir, seed=args.seed, ticks=args.ticks)
    if args.cmd == "headless-keys":
        return _headless_keys(args.out_dir, seed=args.seed, ticks=args.ticks)
    if args.cmd == "replay":
        return _replay(args.source, args.target)
    if args.cmd == "serve":
        import asyncio
        try:
            return asyncio.run(_serve(
                host=args.host,
                port=args.port,
                recordings_root=args.recordings_root,
                allowed_origins=args.allowed_origin,
                allowed_hosts=args.allowed_host,
                cockpit_dir=args.cockpit_dir,
                tick_hz=args.tick_hz,
                heartbeat_s=args.heartbeat_s,
                mavlink_port=args.mavlink_port,
                mavlink_gcs=args.mavlink_gcs,
            ))
        except KeyboardInterrupt:
            return 0
    if args.cmd == "mavlink":
        import asyncio
        from .mavlink_udp import run_sitl
        gcs_host, gcs_port = _split_hostport(args.gcs, 14550)
        try:
            return asyncio.run(run_sitl(
                bind_host=args.bind, bind_port=args.port,
                gcs_host=gcs_host, gcs_port=gcs_port,
                tick_hz=args.tick_hz, seed=args.seed,
                duration_s=(args.duration or None),
            ))
        except KeyboardInterrupt:
            return 0
    if args.cmd == "report":
        return _report(args)
    parser.error(f"unknown command: {args.cmd}")
    return 1


def _report(args) -> int:
    """Post-flight analysis handler."""
    import json as _json
    from .flight_report import (
        FlightMetrics,
        compute_metrics,
        render_markdown,
        write_long_csv,
        compare_metrics,
    )

    if args.compare:
        if args.session_dir is not None:
            print("report: --compare ignores <session_dir>", file=sys.stderr)
        metrics_list = [compute_metrics(Path(d)) for d in args.compare]
        print(compare_metrics(metrics_list))
        return 0

    if args.session_dir is None:
        print("report: provide <session_dir> or use --compare <dirs>",
              file=sys.stderr)
        return 2

    sd = Path(args.session_dir)
    if not sd.exists():
        print(f"report: session directory not found: {sd}", file=sys.stderr)
        return 2

    m = compute_metrics(sd)
    md = render_markdown(m, sd)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md, encoding="utf-8")
    else:
        print(md)
    if args.csv:
        n = write_long_csv(sd, args.csv)
        print(f"\n(csv: {n} rows -> {args.csv})", file=sys.stderr)
    if args.json:
        from dataclasses import asdict
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            _json.dumps(asdict(m), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
