"""First reproducible Phi teaching cycle: curriculum, validation, held-out flights."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from aircraft import Aircraft
from dynamics import hull_force
from mavlink_if import FlyingBoatVehicle
from ocean import Ocean
from ollama_pilot import OllamaPilot, PilotError, observe, fallback_control, SYSTEM_PROMPT
from teacher_student import StudentPilot, load_demonstrations

STAGES = [
    ('takeoff', 'taxi', None, .5, 0),
    ('takeoff', 'hump', None, 3, 0),
    ('takeoff', 'rotation', None, 8, .2),
    ('takeoff', 'climb', 3, 11, 1),
    ('takeoff', 'level', 8, 12, .2),
    ('landing', 'approach', 25, 13, -1.5),
    ('landing', 'descent', 5, 12, -1),
    ('landing', 'flare', .5, 10, -.6),
]


def mission(scenario):
    return dict(scenario=scenario, target_altitude_m=8 if scenario=='takeoff' else 0,
                target_speed_m_s=8.5, decision_interval_s=.05)


def curriculum():
    for seed, hs, split in [(10,0,'train'), (20,.3,'train'), (30,.8,'train'), (40,.5,'validation')]:
        for scenario, stage, altitude, speed, vz in STAGES:
            vehicle = FlyingBoatVehicle(Aircraft(), Ocean(Hs=hs, Tp=6, seed=seed))
            if altitude is not None:
                vehicle.z = altitude
                if stage=='flare':
                    vehicle.z += float(vehicle.sea.eta([0], 0)[0])
            vehicle.Vx, vehicle.Vz = speed, vz
            yield dict(id=f'{seed}_{scenario}_{stage}', split=split, seed=seed, Hs=hs,
                       stage=stage, observation=observe(vehicle), mission=mission(scenario))


def collect(out):
    pilot = OllamaPilot()
    model = pilot.check_model()
    config = dict(model=model, teacher_prompt=SYSTEM_PROMPT, curriculum=list(curriculum()))
    config_path = out/'curriculum.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('resume configuration/model mismatch')
    else:
        config_path.write_text(json.dumps(config, indent=2)+'\n')
    log_path=out/'teacher.jsonl'
    completed = set()
    if log_path.exists():
        completed={json.loads(line)['id'] for line in log_path.read_text().splitlines()}
    with log_path.open('a') as f:
        for i, record in enumerate(config['curriculum']):
            if record['id'] in completed: continue
            control, metadata = pilot.decide(record['observation'], record['mission'])
            record.update(source='ollama', control=asdict(control), **metadata)
            f.write(json.dumps(record, allow_nan=False)+'\n'); f.flush()
            print(f"Teacher {i+1}/32 {record['id']}: {control}", flush=True)
    return log_path


def flight(pilot, scenario, hs, seed, duration=40):
    v=FlyingBoatVehicle(Aircraft(), Ocean(Hs=hs,Tp=6,seed=seed))
    if scenario=='landing': v.z,v.Vx,v.Vz=25,13*math.cos(math.radians(8)),-13*math.sin(math.radians(8))
    v.arm(); status='time_limit'; peak=0
    for _ in range(round(duration/v.dt)):
        control = fallback_control(v,scenario,8,8.5) if pilot is None else pilot.decide(observe(v),mission(scenario))[0]
        control.apply(v); v.step()
        obs=observe(v); snap=v.read_telemetry()[2]; peak=max(peak,snap['N_water'])
        if v.damage.failed: status='damage_failure'; break
        if snap['N_water']>8*v.ac.W: status='impact_failure'; break
        if scenario=='takeoff' and v.z>=8 and v.Vx>=8.5: status='success'; break
        if scenario=='landing' and obs['keel_clearance_m']<0:
            contact=hull_force(v.z,v.Vx,v.Vz,obs['wave_elevation_m'],v.hull)
            status='success' if abs(v.Vz)<1.5 and max(contact.N,snap['N_water'])<3*v.ac.W else 'hard_landing'
            break
    return dict(scenario=scenario,Hs=hs,seed=seed,status=status,time=v.t,
                peak_load_weights=peak/v.ac.W,final=observe(v))


def _repo_root():
    return Path(__file__).resolve().parent


def train_evaluate(out, teacher_path, previous_path="results/phi_student_v1/student.npz", test_seeds=(101,102)):
    previous_path = Path(previous_path)
    if not previous_path.is_absolute():
        previous_path = _repo_root() / previous_path
    records=[json.loads(line) for line in teacher_path.read_text().splitlines()]
    for split in ('train','validation'):
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records if r['split']==split))
    x,y,provenance,_=load_demonstrations([out/'train.jsonl'])
    vx,vy,_,_=load_demonstrations([out/'validation.jsonl'])
    best_loss=float('inf'); best=None; learning=[]
    for seed in (0,1,2):
        student=StudentPilot(seed)
        for epoch in range(200,4001,200):
            student.fit(x,y,epochs=200)
            train_loss=float(np.mean((student.predict(x)-y)**2))
            val_loss=float(np.mean((student.predict(vx)-vy)**2))
            learning.append(dict(seed=seed,epochs=epoch,train_mse=train_loss,validation_mse=val_loss))
            if val_loss<best_loss:
                best_loss=val_loss
                best=StudentPilot(seed)
                best.body.set_flat(student.body.get_flat().copy())
                best.W,best.b=student.W.copy(),student.b.copy()
                selected=learning[-1].copy()
    path=out/'student.npz'; best.save(path)
    old=StudentPilot.load(previous_path)
    results={}
    # Test conditions are never used to select the checkpoint.
    for name,pilot in [('previous_student',old),('trained_student',best),('scripted',None)]:
        runs=[flight(pilot,scenario,hs,seed) for scenario in ('takeoff','landing')
              for hs in (0,.3,.8) for seed in test_seeds]
        results[name]=dict(runs=runs,successes=sum(r['status']=='success' for r in runs),total=len(runs))
        print(f"{name}: {results[name]['successes']}/{len(runs)} held-out flights",flush=True)
    report=dict(train_samples=len(x),validation_samples=len(vx),selected=selected,learning=learning,
                sources=provenance, previous_checkpoint=str(previous_path), test_seeds=list(test_seeds), teacher_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                evaluation=results,notes=['Teacher suggestions are not certified optimal controls.',
                'Curriculum consists of initialized phase states, not successful flight trajectories.',
                'Held-out test outcomes were not used for checkpoint selection.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/education_001'))
    p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'student.npz').exists(): raise ValueError('completed cycle exists; choose a new output directory')
    started=time.monotonic()
    path=args.output/'teacher.jsonl' if args.evaluate_only else collect(args.output)
    report=train_evaluate(args.output,path)
    print(json.dumps(dict(selected=report['selected'],wall_seconds=time.monotonic()-started),indent=2))


if __name__=='__main__': main()
