"""Post-flight analysis toolkit for recorded operator sessions.

A session directory produced by ``FileSink`` contains:

    config.json   - envelope, curriculum, seed, source SHA
    inputs.jsonl  - received input events (raw + request)
    ticks.jsonl   - applied control + normalised state + damage
    events.jsonl  - authority transitions, stop reasons, rejections
    summary.json  - lifecycle / phase / authority at session end

This module turns those records into a structured ``FlightMetrics``
record and a Markdown report, plus a long-format CSV for pandas /
matplotlib. It is purely read-only: it never modifies the session.

Typical use::

    from pathlib import Path
    from operator_training.flight_report import compute_metrics, render_markdown

    metrics = compute_metrics(Path("var/operator_sessions/<id>"))
    print(render_markdown(metrics, Path("var/operator_sessions/<id>")))

The CLI lives in ``__main__``::

    python3 -m operator_training report <session_dir> [--md out.md] [--csv out.csv]
    python3 -m operator_training report --compare <dir1> <dir2> [dir3 ...]
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# --- Metric record ------------------------------------------------------

@dataclass
class FlightMetrics:
    """Computed metrics for one recorded flight session."""

    # Identity
    session_id: str = ""
    curriculum_id: str = ""
    scenario: str = ""
    seed: int = 0
    spatial: bool = True

    # Time / counts
    duration_s: float = 0.0
    tick_count: int = 0
    input_count: int = 0

    # Outcome
    lifecycle_end: str = ""
    phase_end: str = ""
    authority_end: str = ""
    outcome: str = "incomplete"  # success | failure | aborted | incomplete
    failure_reason: Optional[str] = None
    end_reason: Optional[str] = None

    # Aircraft performance
    max_altitude_m: float = 0.0
    max_airspeed_m_s: float = 0.0
    max_forward_speed_m_s: float = 0.0
    max_bank_deg: float = 0.0
    max_pitch_deg: float = 0.0
    max_sink_rate_m_s: float = 0.0  # negative vertical speed
    min_keel_clearance_m: float = 0.0  # most negative = deepest penetration
    final_altitude_m: float = 0.0
    final_airspeed_m_s: float = 0.0

    # Authority / human factors
    time_auto_s: float = 0.0
    time_human_s: float = 0.0
    authority_changes: int = 0
    inputs_attempted: int = 0
    inputs_applied: int = 0
    inputs_rejected: int = 0

    # Pilot smoothness (RMS of d²control/dt²)
    pitch_jerk_rms: float = 0.0
    throttle_jerk_rms: float = 0.0
    bank_jerk_rms: float = 0.0

    # Damage
    damage_failed: bool = False
    final_water_kg: float = 0.0
    peak_water_kg: float = 0.0
    sling_count: int = 0

    # Energy proxy (∫ throttle dt)
    throttle_time_integral: float = 0.0

    # ASCII altitude profile (pre-rendered for the markdown report)
    altitude_profile: list[str] = field(default_factory=list)

    # Source
    session_dir: str = ""


# --- Loaders ------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --- Metric computation -------------------------------------------------

_SLING_DELTA_KG = 0.05  # heuristic: jump in water_kg implies a sling entry


def compute_metrics(session_dir: Path) -> FlightMetrics:
    """Read a session directory and compute its ``FlightMetrics``.

    Missing files are tolerated (returns zero metrics for that field),
    so partial recordings still produce a usable report.
    """
    session_dir = Path(session_dir)
    metrics = FlightMetrics(session_dir=str(session_dir))

    config = _read_json(session_dir / "config.json")
    metrics.session_id = str(config.get("session_id", session_dir.name))
    metrics.curriculum_id = str(config.get("curriculum_id", ""))
    metrics.scenario = str(config.get("scenario", ""))
    metrics.seed = int(config.get("seed", 0))
    metrics.spatial = bool(config.get("spatial", True))

    summary = _read_json(session_dir / "summary.json")
    metrics.lifecycle_end = str(summary.get("lifecycle_end", ""))
    metrics.phase_end = str(summary.get("phase_end", ""))
    metrics.authority_end = str(summary.get("authority_end", ""))
    metrics.failure_reason = summary.get("failure_reason")
    metrics.end_reason = summary.get("end_reason")
    metrics.duration_s = float(summary.get("sim_t_end", 0.0))
    metrics.tick_count = int(summary.get("tick_count", 0))

    ticks = _read_jsonl(session_dir / "ticks.jsonl")
    inputs = _read_jsonl(session_dir / "inputs.jsonl")
    events = _read_jsonl(session_dir / "events.jsonl")

    metrics.input_count = len(inputs)
    metrics.inputs_attempted = len(inputs)
    metrics.inputs_applied = sum(1 for r in inputs if r.get("applied"))
    metrics.inputs_rejected = metrics.inputs_attempted - metrics.inputs_applied

    if ticks:
        metrics = _populate_from_ticks(ticks, metrics)
        metrics.authority_changes = _count_authority_changes(events)
        if metrics.lifecycle_end:
            metrics.outcome = _classify_outcome(
                metrics.lifecycle_end,
                phase_end=metrics.phase_end,
                failure_reason=metrics.failure_reason,
                phase_reached_altitude=metrics.max_altitude_m,
                phase_reached_touchdown=metrics.phase_end == "TOUCHDOWN",
            )

    metrics.altitude_profile = _ascii_altitude_profile(
        [(t.get("sim_t", 0.0), (t.get("state") or {}).get("altitude_m", 0.0))
         for t in ticks],
        width=60, height=18,
    )
    return metrics


def _populate_from_ticks(ticks: list[dict], m: FlightMetrics) -> FlightMetrics:
    """Fill the state-derived metric fields from the tick stream."""
    if not ticks:
        return m

    times = [float(t.get("sim_t", 0.0)) for t in ticks]
    states = [t.get("state") or {} for t in ticks]
    controls = [t.get("control") or {} for t in ticks]
    damages = [t.get("damage") or {} for t in ticks]

    altitudes = [s.get("altitude_m", 0.0) for s in states]
    airspeeds = [s.get("airspeed_m_s", 0.0) for s in states]
    forward_speeds = [s.get("forward_speed_m_s", 0.0) for s in states]
    banks = [s.get("bank_deg", 0.0) for s in states]
    pitches = [float(controls[i].get("pitch_deg", 0.0))
               for i in range(len(controls))]
    vertical_speeds = [s.get("vertical_speed_m_s", 0.0) for s in states]
    keels = [s.get("keel_clearance_m", 0.0) for s in states]
    water = [d.get("water_kg", 0.0) for d in damages]

    m.max_altitude_m = max(altitudes)
    m.max_airspeed_m_s = max(airspeeds)
    m.max_forward_speed_m_s = max(forward_speeds)
    m.max_bank_deg = max(abs(b) for b in banks)
    m.max_pitch_deg = max(abs(p) for p in pitches)
    m.max_sink_rate_m_s = -min(vertical_speeds)  # positive magnitude
    m.min_keel_clearance_m = min(keels)
    m.final_altitude_m = altitudes[-1]
    m.final_airspeed_m_s = airspeeds[-1]

    # Authority time share
    auto_t = 0.0
    human_t = 0.0
    last_t = times[0]
    for i, t in enumerate(ticks):
        dt = max(0.0, float(t.get("sim_t", 0.0)) - last_t)
        last_t = float(t.get("sim_t", 0.0))
        if t.get("authority") == "AUTO":
            auto_t += dt
        elif t.get("authority") == "HUMAN":
            human_t += dt
    # Last interval to the final sample
    if times:
        tail = max(0.0, times[-1] - last_t)
        if ticks[-1].get("authority") == "AUTO":
            auto_t += tail
        elif ticks[-1].get("authority") == "HUMAN":
            human_t += tail
    m.time_auto_s = auto_t
    m.time_human_s = human_t

    # Jerk = RMS of d²control/dt² over piecewise constant samples.
    # Only HUMAN-authority ticks are scored here: AUTO guidance is
    # allowed to switch targets sharply between ticks.
    human_idx = [i for i, t in enumerate(ticks) if t.get("authority") == "HUMAN"]
    if len(human_idx) >= 3:
        h_times = [times[i] for i in human_idx]
        m.pitch_jerk_rms = _jerk_rms(
            h_times, [float(controls[i].get("pitch_deg", 0.0)) for i in human_idx],
        )
        m.throttle_jerk_rms = _jerk_rms(
            h_times, [float(controls[i].get("throttle", 0.0)) for i in human_idx],
        )
        m.bank_jerk_rms = _jerk_rms(
            h_times, [float(controls[i].get("bank_deg", 0.0)) for i in human_idx],
        )

    # Throttle-time integral (proxy for fuel/energy use)
    throttle_int = 0.0
    prev_t = times[0]
    prev_thr = float(controls[0].get("throttle", 0.0))
    for i in range(1, len(times)):
        dt = max(0.0, times[i] - prev_t)
        thr = float(controls[i].get("throttle", prev_thr))
        throttle_int += 0.5 * (thr + prev_thr) * dt
        prev_t = times[i]
        prev_thr = thr
    m.throttle_time_integral = throttle_int

    # Damage rollup
    if water:
        m.final_water_kg = water[-1]
        m.peak_water_kg = max(water)
    m.damage_failed = bool(damages[-1].get("failed", False))

    # Sling detection: rapid ingress jumps mark a hull re-entry
    sling = 0
    for i in range(1, len(water)):
        if water[i] - water[i - 1] > _SLING_DELTA_KG and water[i] > 0.0:
            sling += 1
    m.sling_count = sling
    return m


def _jerk_rms(times: list[float], values: list[float]) -> float:
    """RMS of second derivative via central differences (scaled)."""
    if len(values) < 3:
        return 0.0
    accels: list[float] = []
    for i in range(1, len(values) - 1):
        dt_p = max(1e-6, times[i + 1] - times[i])
        dt_m = max(1e-6, times[i] - times[i - 1])
        v_p = (values[i + 1] - values[i]) / dt_p
        v_m = (values[i] - values[i - 1]) / dt_m
        accels.append((v_p - v_m) / max(1e-6, 0.5 * (dt_p + dt_m)))
    if not accels:
        return 0.0
    return math.sqrt(sum(a * a for a in accels) / len(accels))


def _count_authority_changes(events: list[dict]) -> int:
    """Count AUTO <-> HUMAN transitions from the event stream."""
    last = None
    changes = 0
    for ev in events:
        auth = ev.get("authority")
        if auth in ("AUTO", "HUMAN") and last is not None and auth != last:
            changes += 1
        if auth in ("AUTO", "HUMAN"):
            last = auth
    return changes


def _classify_outcome(lifecycle_end: str, *, phase_end: str,
                      failure_reason: Optional[str],
                      phase_reached_altitude: float,
                      phase_reached_touchdown: bool) -> str:
    if lifecycle_end == "ABORTED":
        return "aborted"
    if failure_reason:
        return "failure"
    if lifecycle_end == "FINISHED":
        if phase_end == "TOUCHDOWN" or phase_reached_touchdown:
            return "success"
        return "success"
    return "incomplete"


# --- ASCII altitude profile ---------------------------------------------

def _ascii_altitude_profile(points: list[tuple[float, float]],
                            *, width: int = 60, height: int = 18) -> list[str]:
    """Render a compact ASCII profile of (time, altitude)."""
    if not points:
        return ["(no telemetry)"]
    alt_min = min(p[1] for p in points)
    alt_max = max(p[1] for p in points)
    span = max(1e-6, alt_max - alt_min)
    t_min = points[0][0]
    t_max = points[-1][0]
    t_span = max(1e-6, t_max - t_min)

    grid: list[list[str]] = [[" "] * width for _ in range(height)]
    # Axes
    for x in range(width):
        grid[height - 1][x] = "─"
    for y in range(height):
        grid[y][0] = "│"
    grid[height - 1][0] = "└"

    for t, alt in points:
        col = int(round((t - t_min) / t_span * (width - 2)))
        row = height - 2 - int(round((alt - alt_min) / span * (height - 2)))
        col = max(1, min(width - 1, col))
        row = max(1, min(height - 2, row))
        grid[row][col] = "●"

    alt_low = f"{alt_min:+.1f}m"
    alt_high = f"{alt_max:+.1f}m"
    lines = [f"  alt: {alt_high}".ljust(width + 4)]
    for r, row in enumerate(grid[:-1]):
        prefix = "       " if r == 0 else "       "
        lines.append(prefix + "".join(row))
    lines.append(f"       └{'─' * (width - 1)}  t: {t_min:.1f} → {t_max:.1f} s")
    return lines


# --- Markdown rendering -------------------------------------------------

_RISER = "▲"
_FALLER = "▽"
_CHECK = "✓"
_CROSS = "✗"


def render_markdown(m: FlightMetrics, session_dir: Optional[Path] = None) -> str:
    """Render a Markdown report for one ``FlightMetrics``."""
    out: list[str] = []
    out.append(f"# Flight Report — `{m.session_id or (session_dir.name if session_dir else 'unknown')}`")
    out.append("")
    out.append("## Outcome")
    out.append("")
    out.append(f"| Field | Value |")
    out.append(f"|---|---|")
    out.append(f"| Result | **{m.outcome.upper()}** {_outcome_glyph(m.outcome)} |")
    out.append(f"| Lifecycle end | `{m.lifecycle_end or 'n/a'}` |")
    out.append(f"| Phase end | `{m.phase_end or 'n/a'}` |")
    out.append(f"| Authority end | `{m.authority_end or 'n/a'}` |")
    if m.failure_reason:
        out.append(f"| Failure reason | `{m.failure_reason}` |")
    if m.end_reason:
        out.append(f"| End reason | `{m.end_reason}` |")
    out.append(f"| Duration | {m.duration_s:.2f} s ({m.tick_count} ticks) |")
    out.append(f"| Scenario | `{m.scenario or 'n/a'}` (curriculum `{m.curriculum_id or 'n/a'}`) |")
    out.append(f"| Seed | {m.seed} |")
    out.append(f"| Spatial | {m.spatial} |")
    out.append("")

    out.append("## Aircraft performance")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---:|")
    out.append(f"| Max altitude | {m.max_altitude_m:.2f} m |")
    out.append(f"| Max airspeed | {m.max_airspeed_m_s:.2f} m/s |")
    out.append(f"| Max forward speed | {m.max_forward_speed_m_s:.2f} m/s |")
    out.append(f"| Max bank | {m.max_bank_deg:.1f}° |")
    out.append(f"| Max pitch | {m.max_pitch_deg:.1f}° |")
    out.append(f"| Max sink rate | {m.max_sink_rate_m_s:.2f} m/s |")
    out.append(f"| Deepest keel penetration | {m.min_keel_clearance_m:.3f} m |")
    out.append(f"| Final altitude | {m.final_altitude_m:.2f} m |")
    out.append(f"| Final airspeed | {m.final_airspeed_m_s:.2f} m/s |")
    out.append("")

    out.append("## Authority & pilot inputs")
    out.append("")
    total = m.time_auto_s + m.time_human_s
    auto_pct = (100.0 * m.time_auto_s / total) if total > 0 else 0.0
    human_pct = 100.0 - auto_pct
    out.append(f"| Metric | Value |")
    out.append(f"|---|---:|")
    out.append(f"| Time on AUTO | {m.time_auto_s:.2f} s ({auto_pct:.0f}%) |")
    out.append(f"| Time on HUMAN | {m.time_human_s:.2f} s ({human_pct:.0f}%) |")
    out.append(f"| Authority changes | {m.authority_changes} |")
    out.append(f"| Inputs attempted | {m.inputs_attempted} |")
    out.append(f"| Inputs applied | {m.inputs_applied} |")
    out.append(f"| Inputs rejected | {m.inputs_rejected} |")
    out.append(f"| Pitch jerk RMS | {m.pitch_jerk_rms:.3f} °/s² |")
    out.append(f"| Throttle jerk RMS | {m.throttle_jerk_rms:.3f} /s² |")
    out.append(f"| Bank jerk RMS | {m.bank_jerk_rms:.3f} °/s² |")
    out.append(f"| Throttle-time integral | {m.throttle_time_integral:.2f} s |")
    out.append("")

    out.append("## Damage & integrity")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---:|")
    out.append(f"| Failed | `{m.damage_failed}` |")
    out.append(f"| Final water ingress | {m.final_water_kg:.4f} kg |")
    out.append(f"| Peak water ingress | {m.peak_water_kg:.4f} kg |")
    out.append(f"| Sling events (est.) | {m.sling_count} |")
    out.append("")

    if m.altitude_profile:
        out.append("## Altitude profile")
        out.append("")
        out.append("```")
        out.extend(m.altitude_profile)
        out.append("```")
        out.append("")
    return "\n".join(out)


def _outcome_glyph(outcome: str) -> str:
    return {
        "success": _CHECK,
        "failure": _CROSS,
        "aborted": "■",
        "incomplete": "?",
    }.get(outcome, "?")


# --- Long-format CSV export ---------------------------------------------

def write_long_csv(session_dir: Path, output: Path) -> int:
    """Write a long-format CSV of the time series (one row per tick).

    Returns the number of data rows written (excluding the header).
    """
    session_dir = Path(session_dir)
    ticks = _read_jsonl(session_dir / "ticks.jsonl")
    fields = [
        "sim_t", "tick", "authority",
        "altitude_m", "airspeed_m_s", "forward_speed_m_s",
        "vertical_speed_m_s", "bank_deg", "keel_clearance_m",
        "throttle", "pitch_deg", "bank_cmd_deg", "rudder_cmd",
        "water_kg", "damage_failed",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in ticks:
            state = row.get("state") or {}
            control = row.get("control") or {}
            damage = row.get("damage") or {}
            w.writerow({
                "sim_t": row.get("sim_t", ""),
                "tick": row.get("tick", ""),
                "authority": row.get("authority", ""),
                "altitude_m": state.get("altitude_m", ""),
                "airspeed_m_s": state.get("airspeed_m_s", ""),
                "forward_speed_m_s": state.get("forward_speed_m_s", ""),
                "vertical_speed_m_s": state.get("vertical_speed_m_s", ""),
                "bank_deg": state.get("bank_deg", ""),
                "keel_clearance_m": state.get("keel_clearance_m", ""),
                "throttle": control.get("throttle", ""),
                "pitch_deg": control.get("pitch_deg", ""),
                "bank_cmd_deg": control.get("bank_deg", ""),
                "rudder_cmd": control.get("rudder", ""),
                "water_kg": damage.get("water_kg", ""),
                "damage_failed": damage.get("failed", ""),
            })
    return len(ticks)


# --- Comparison ----------------------------------------------------------

def compare_metrics(metrics_list: list[FlightMetrics]) -> str:
    """Render a side-by-side comparison table for several flights."""
    if not metrics_list:
        return "(no sessions)"
    rows: list[tuple[str, str]] = [
        ("Session ID", "session_id"),
        ("Curriculum", "curriculum_id"),
        ("Scenario", "scenario"),
        ("Seed", "seed"),
        ("Outcome", "outcome"),
        ("Duration (s)", "duration_s"),
        ("Max altitude (m)", "max_altitude_m"),
        ("Max airspeed (m/s)", "max_airspeed_m_s"),
        ("Max bank (°)", "max_bank_deg"),
        ("Time AUTO (s)", "time_auto_s"),
        ("Time HUMAN (s)", "time_human_s"),
        ("Authority changes", "authority_changes"),
        ("Inputs applied / rejected",
         None),  # custom rendering
        ("Pitch jerk RMS", "pitch_jerk_rms"),
        ("Peak water (kg)", "peak_water_kg"),
        ("Sling events", "sling_count"),
        ("Throttle-time integral", "throttle_time_integral"),
    ]
    header = ["Metric"] + [m.session_id[:12] or f"#{i}" for i, m in enumerate(metrics_list)]
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join(["---"] + ["---:"] * len(metrics_list)) + "|")
    for label, key in rows:
        cells = [label]
        for m in metrics_list:
            if key is None:
                cells.append(f"{m.inputs_applied}/{m.inputs_rejected}")
            else:
                val = getattr(m, key)
                if isinstance(val, float):
                    cells.append(f"{val:.3f}" if abs(val) < 100 else f"{val:.1f}")
                else:
                    cells.append(str(val))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
