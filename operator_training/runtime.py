"""Fixed-step tick scheduler and watchdog.

Per design section 7.1, physics runs at a fixed 20 Hz. The runtime is
responsible for keeping the simulation on schedule and triggering a
SIM_OVERRUN stop when the loop falls more than two ticks behind.

The runtime is intentionally minimal so P0 can validate the contract
without dragging in an event loop. The browser/transport layer is
expected to call ``step()`` once per desired tick; ``FixedStepScheduler``
provides the deterministic driver for headless tests and CLI runs.

Watchdog is independent of the scheduler and tracks liveness signals
from the client: input freshness (last seen input seq) and heartbeat.
Both feed ``on_lapse`` which the session converts into an
``abort(reason)`` per design section 7.2.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Watchdog:
    """Track input freshness and heartbeat liveness.

    The thresholds are initial values; they are calibrated against the
    measured RTT in P4 (design section 11 P4 acceptance).
    """

    input_max_age_s: float = 0.25
    heartbeat_max_age_s: float = 0.5
    on_lapse: Optional[Callable[[str], None]] = None

    last_input_t: Optional[float] = None
    last_heartbeat_t: Optional[float] = None

    def note_input(self, t: Optional[float] = None) -> None:
        self.last_input_t = float(t if t is not None else time.monotonic())

    def note_heartbeat(self, t: Optional[float] = None) -> None:
        self.last_heartbeat_t = float(t if t is not None else time.monotonic())

    def check(self, now: Optional[float] = None) -> Optional[str]:
        now = float(now if now is not None else time.monotonic())
        if self.last_input_t is not None and (now - self.last_input_t) > self.input_max_age_s:
            return f"input_stale>={self.input_max_age_s}s"
        if (self.last_heartbeat_t is not None
                and (now - self.last_heartbeat_t) > self.heartbeat_max_age_s):
            return f"heartbeat_stale>={self.heartbeat_max_age_s}s"
        return None

    def trigger(self, reason: str) -> None:
        if self.on_lapse is not None:
            self.on_lapse(reason)


@dataclass
class FixedStepScheduler:
    """Drive ``session.tick`` at a fixed dt using a monotonic clock.

    Catches up at most two ticks per wake-up; deeper lag raises
    ``SimOverrun`` so the session can stop cleanly instead of silently
    dropping time on resume (design section 7.1).
    """

    dt: float = 0.05
    max_catchup_ticks: int = 2
    on_tick: Optional[Callable[[int], None]] = None
    on_overrun: Optional[Callable[[float], None]] = None

    _last_t: Optional[float] = field(default=None, init=False, repr=False)
    _tick_index: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._last_t = None
        self._tick_index = 0

    def pump(self, *, now: Optional[float] = None,
             feed: Callable[[float, int], None]) -> int:
        """Run as many catch-up ticks as allowed.

        ``feed(dt, index)`` is called for each tick. Returns the number
        of ticks executed this call.
        """
        now = float(now if now is not None else time.monotonic())
        if self._last_t is None:
            self._last_t = now
            return 0
        elapsed = now - self._last_t
        ticks_due = int(elapsed // self.dt)
        if ticks_due <= 0:
            return 0
        if ticks_due > self.max_catchup_ticks:
            overrun = elapsed - ticks_due * self.dt
            if self.on_overrun is not None:
                self.on_overrun(overrun)
            ticks_due = self.max_catchup_ticks
        executed = 0
        for _ in range(ticks_due):
            self._tick_index += 1
            feed(self.dt, self._tick_index)
            executed += 1
        self._last_t += ticks_due * self.dt
        return executed

    def step_once(self, *, feed: Callable[[float, int], None]) -> None:
        """Run a single tick; used by the headless driver."""
        self._tick_index += 1
        feed(self.dt, self._tick_index)


class SimOverrun(RuntimeError):
    """Raised when the scheduler falls more than the catch-up window behind."""
