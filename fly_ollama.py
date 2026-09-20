"""Run local phi3.5 as a simulated flying-boat pilot and save auditable logs."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import sys
import time

from aircraft import Aircraft
from dynamics import hull_force
from mavlink_if import FlyingBoatVehicle
from atmosphere import AtmosphereConfig
from ocean_directional import DirectionalOcean
from ocean import Ocean
from ocean_real import load_buoy_default
from ollama_pilot import OllamaPilot, PilotError, observe, fallback_control, stabilize_lateral


def positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=['takeoff', 'landing'], default='takeoff')
    p.add_argument('--model', default='phi3.5:latest')
    models = p.add_mutually_exclusive_group()
    models.add_argument('--student', type=Path, help='offline distilled student checkpoint; no Ollama connection')
    models.add_argument('--policy', type=Path, help='saved RL policy, no Ollama connection')
    models.add_argument('--jev', action='store_true', help='local System One layer over native setpoints')
    p.add_argument('--jev-calibrator', type=Path, help='npz logistic weights for FlightJev')
    p.add_argument('--spatial', action='store_true')
    p.add_argument('--lateral-assist', action='store_true',
                   help='use per-step corridor feedback for bank/rudder, preserving pilot throttle/pitch')
    p.add_argument('--directional', action='store_true')
    p.add_argument('--theta-mean-deg', type=float, default=0.0)
    p.add_argument('--wind', type=float, nargs=3, default=(0., 0., 0.))
    p.add_argument('--gust-rms', type=float, default=0.0)
    p.add_argument('--render', action='store_true', help='save scene.png after flight')
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
    if args.lateral_assist and not args.spatial:
        raise ValueError("--lateral-assist requires --spatial")
    if args.spatial and args.student:
        raise ValueError('legacy student checkpoints do not support spatial control; use --policy')
    atmosphere = AtmosphereConfig(wind=tuple(args.wind), gust_rms=args.gust_rms)
    if not args.spatial and atmosphere != AtmosphereConfig():
        raise ValueError('wind configuration requires --spatial')
    if args.ndbc and args.directional:
        raise ValueError('--ndbc and --directional cannot be combined in this runner')
    if pilot is None:
        if args.jev:
            from flight_jev import FlightJev
            if args.jev_calibrator:
                pilot = FlightJev.load(args.jev_calibrator, spatial=args.spatial)
            else:
                pilot = FlightJev(spatial=args.spatial)
        elif getattr(args, 'jev_calibrator', None):
            raise ValueError('--jev-calibrator requires --jev')
        elif args.policy:
            from policy_pilot import PolicyPilot
            pilot = PolicyPilot(args.policy, spatial=args.spatial)
        elif args.student:
            from teacher_student import StudentPilot
            pilot = StudentPilot.load(args.student)
        else:
            pilot = OllamaPilot(args.model, args.host, args.timeout, args.seed, spatial=args.spatial)
    model_info = pilot.check_model()
    print(f"Local model: {model_info.get('name', args.model)}", flush=True)
    if args.check:
        return {'status': 'ready', 'model': model_info}
    if args.interval < args.dt:
        raise ValueError('interval must be at least dt')
    sea = load_buoy_default() if args.ndbc else Ocean(Hs=args.hs, Tp=args.tp, seed=args.seed)
    if args.directional:
        sea = DirectionalOcean(Hs=args.hs, Tp=args.tp, seed=args.seed,
                               theta_mean=math.radians(args.theta_mean_deg))
    vehicle = FlyingBoatVehicle(Aircraft(), sea, spatial=args.spatial,
                                atmosphere=atmosphere, seed=args.seed)
    vehicle.dt = args.dt
    if args.scenario == 'landing':
        vehicle.z = 25
        vehicle.Vx = 13 * math.cos(math.radians(8))
        vehicle.Vz = -13 * math.sin(math.radians(8))
    vehicle.arm()
    out = args.output or Path('results') / datetime.now(timezone.utc).strftime('ollama_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    mission = dict(spatial=args.spatial, scenario=args.scenario, target_altitude_m=args.target_alt if args.scenario == 'takeoff' else 0,
                   target_speed_m_s=args.target_speed, decision_interval_s=args.interval)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / 'config.json').write_text(json.dumps(dict(config=config, model=model_info, atmosphere=asdict(atmosphere),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).resolve().parent.glob('*.py')}), indent=2) + '\n')
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
                        source = 'jev' if args.jev else ('policy' if args.policy else ('student' if args.student else 'ollama'))
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
                elif args.jev:
                    control, _ = pilot.decide(observe(vehicle), mission, control)
                if args.spatial:
                    from ollama_pilot import SpatialControl
                    if not isinstance(control, SpatialControl):
                        raise PilotError('spatial pilot returned a longitudinal control')
                applied_control = stabilize_lateral(vehicle, control) if args.lateral_assist else control
                applied_control.apply(vehicle)
                vehicle.step(dt=min(args.dt, args.duration - vehicle.t))
                steps += 1
                obs = observe(vehicle)
                snap = vehicle.read_telemetry()[2]
                trajectory.write(json.dumps(dict(observation=obs, control=asdict(control),
                                                 applied_control=asdict(applied_control),
                                                 lateral_assist=args.lateral_assist,
                                                 applied_throttle=vehicle.throttle,
                                                 source=source, thrust_N=snap['T'], snapshot=snap,
                                                 telemetry=dict(hil=vehicle.read_telemetry()[0].fields,
                                                                global_position=vehicle.read_telemetry()[1].fields,
                                                                attitude=vehicle.read_attitude().fields)), allow_nan=False) + '\n')
                if vehicle.damage.failed:
                    status = 'damage_failure'; break
                if snap['N_water'] > 8 * vehicle.ac.W:
                    status = 'impact_failure'; break
                if args.spatial and abs(vehicle.y) > 50:
                    status = 'lateral_limit'; break
                if args.scenario == 'takeoff':
                    if vehicle.lateral_success() and vehicle.z >= args.target_alt and vehicle.Vx >= args.target_speed:
                        status = 'success'; break
                elif obs['keel_clearance_m'] < 0:
                    contact = hull_force(vehicle.z, vehicle.Vx, vehicle.Vz, obs['wave_elevation_m'], vehicle.hull)
                    status = 'success' if vehicle.lateral_success() and abs(vehicle.Vz) < 1.5 and max(contact.N, snap['N_water']) < 3 * vehicle.ac.W else 'hard_landing'
                    break
    except (PilotError, KeyboardInterrupt) as exc:
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'pilot_error'
        error = str(exc)
    finally:
        vehicle.disarm()
    summary = dict(status=status, error=error, lateral_assist=args.lateral_assist, model=model_info.get('name', args.model), decisions=decisions,
                   fallback_decisions=fallback_count, physics_steps=steps,
                   simulation_seconds=vehicle.t, wall_seconds=time.monotonic() - started,
                   final_observation=observe(vehicle), output=str(out.resolve()),
                   timing='synchronous: simulation pauses during inference')
    if args.scenario == 'takeoff':
        from flight_diagnostics import takeoff_progress
        summary['takeoff_diagnostics'] = takeoff_progress([entry[3] for entry in vehicle._msg_log])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if args.render and vehicle._msg_log:
        from visual3d import render_scene
        render_scene([entry[3] for entry in vehicle._msg_log], sea, output=str(out / 'scene.png'))
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
