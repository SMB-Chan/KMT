"""Run local phi3.5 as a simulated flying-boat pilot and save auditable logs."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from aircraft import Aircraft
from dynamics import hull_force
from mavlink_if import FlyingBoatVehicle
from ocean import Ocean
from ocean_real import load_buoy_default
from ollama_pilot import OllamaPilot, PilotError, observe, fallback_control


def positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=['takeoff', 'landing'], default='takeoff')
    p.add_argument('--model', default='phi3.5:latest')
    p.add_argument('--student', type=Path, help='offline distilled student checkpoint; no Ollama connection')
    p.add_argument('--host', default='http://127.0.0.1:11434')
    p.add_argument('--duration', type=positive, default=12)
    p.add_argument('--interval', type=positive, default=1, help='seconds of simulation between decisions')
    p.add_argument('--timeout', type=positive, default=60, help='HTTP socket timeout, wall seconds')
    p.add_argument('--dt', type=positive, default=.05)
    p.add_argument('--hs', type=float, default=.3)
    p.add_argument('--tp', type=positive, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ndbc', action='store_true', help='use bundled NDBC wave snapshot')
    p.add_argument('--target-alt', type=positive, default=8)
    p.add_argument('--target-speed', type=positive, default=8.5)
    p.add_argument('--output', type=Path, help='new output directory (must not exist)')
    p.add_argument('--check', action='store_true', help='check local model without flying')
    p.add_argument('--strict', action='store_true', help='abort on inference error instead of scripted fallback')
    return p


def run(args, pilot=None):
    if pilot is None:
        if args.student:
            from teacher_student import StudentPilot
            pilot = StudentPilot.load(args.student)
        else:
            pilot = OllamaPilot(args.model, args.host, args.timeout, args.seed)
    model_info = pilot.check_model()
    print(f"Local model: {model_info.get('name', args.model)}", flush=True)
    if args.check:
        return {'status': 'ready', 'model': model_info}
    if args.interval < args.dt:
        raise ValueError('interval must be at least dt')
    sea = load_buoy_default() if args.ndbc else Ocean(Hs=args.hs, Tp=args.tp, seed=args.seed)
    vehicle = FlyingBoatVehicle(Aircraft(), sea)
    vehicle.dt = args.dt
    if args.scenario == 'landing':
        vehicle.z = 25
        vehicle.Vx = 13 * math.cos(math.radians(8))
        vehicle.Vz = -13 * math.sin(math.radians(8))
    vehicle.arm()
    out = args.output or Path('results') / datetime.now(timezone.utc).strftime('ollama_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    mission = dict(scenario=args.scenario, target_altitude_m=args.target_alt if args.scenario == 'takeoff' else 0,
                   target_speed_m_s=args.target_speed, decision_interval_s=args.interval)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / 'config.json').write_text(json.dumps(dict(config=config, model=model_info), indent=2) + '\n')
    decisions, fallback_count, steps = 0, 0, 0
    next_decision, control, source = 0.0, None, 'fallback'
    status, error = 'time_limit', None
    started = time.monotonic()
    try:
        with (out / 'decisions.jsonl').open('w') as log, (out / 'trajectory.jsonl').open('w') as trajectory:
            while vehicle.t < args.duration - 1e-9:
                observation = observe(vehicle)
                if vehicle.t >= next_decision - 1e-9:
                    record = dict(observation=observation, mission=mission)
                    try:
                        control, metadata = pilot.decide(observation, mission, control)
                        source = 'student' if args.student else 'ollama'
                        record.update(metadata)
                    except PilotError as exc:
                        source = 'fallback'
                        fallback_count += 1
                        record['error'] = str(exc)
                        control = fallback_control(vehicle, args.scenario, args.target_alt, args.target_speed)
                        if args.strict:
                            record.update(source='aborted', control=None)
                            log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                            raise
                    decisions += 1
                    record.update(source=source, control=asdict(control))
                    log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                    print(f"t={vehicle.t:.2f} {source}: throttle={control.throttle:.3f} pitch={control.pitch_deg:.2f}", flush=True)
                    next_decision = vehicle.t + args.interval
                # Fallback feedback is recomputed every physics step, never a stale LLM command.
                if source == 'fallback':
                    control = fallback_control(vehicle, args.scenario, args.target_alt, args.target_speed)
                control.apply(vehicle)
                vehicle.step(dt=min(args.dt, args.duration - vehicle.t))
                steps += 1
                obs = observe(vehicle)
                snap = vehicle.read_telemetry()[2]
                trajectory.write(json.dumps(dict(observation=obs, control=asdict(control),
                                                 applied_throttle=vehicle.throttle,
                                                 source=source, thrust_N=snap['T']), allow_nan=False) + '\n')
                if vehicle.damage.failed:
                    status = 'damage_failure'; break
                if snap['N_water'] > 8 * vehicle.ac.W:
                    status = 'impact_failure'; break
                if args.scenario == 'takeoff':
                    if vehicle.z >= args.target_alt and vehicle.Vx >= args.target_speed:
                        status = 'success'; break
                elif obs['keel_clearance_m'] < 0:
                    contact = hull_force(vehicle.z, vehicle.Vx, vehicle.Vz, obs['wave_elevation_m'], vehicle.hull)
                    status = 'success' if abs(vehicle.Vz) < 1.5 and max(contact.N, snap['N_water']) < 3 * vehicle.ac.W else 'hard_landing'
                    break
    except (PilotError, KeyboardInterrupt) as exc:
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'pilot_error'
        error = str(exc)
    finally:
        vehicle.disarm()
    summary = dict(status=status, error=error, model=model_info.get('name', args.model), decisions=decisions,
                   fallback_decisions=fallback_count, physics_steps=steps,
                   simulation_seconds=vehicle.t, wall_seconds=time.monotonic() - started,
                   final_observation=observe(vehicle), output=str(out.resolve()),
                   timing='synchronous: simulation pauses during inference')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    args = parser().parse_args()
    try:
        summary = run(args)
    except (PilotError, ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    return 2 if summary['status'] in ('pilot_error', 'interrupted') else 0


if __name__ == '__main__':
    sys.exit(main())
