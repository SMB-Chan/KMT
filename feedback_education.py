"""Phi selects controls after observing counterfactual simulator outcomes."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
from aircraft import Aircraft
from dynamics import hull_force
from educate import train_evaluate, flight
from mavlink_if import FlyingBoatVehicle
from ocean import Ocean
from ollama_pilot import Control, OllamaPilot, PilotError, observe

FEEDBACK_PROMPT = '''You are a teacher for a simulated flying boat.
Choose exactly one candidate_id from the supplied simulator experiments.
Higher score is better for the stated mission. Failure is unacceptable when a
non-failed alternative exists. Use the measured outcomes to revise the previous
control; do not repeat a default action when the experiments show it is worse.
Return only JSON {"candidate_id": integer}. The learner will imitate your choice.
These are short-horizon model predictions, not evidence of real flight safety.'''


def restore(record):
    v=FlyingBoatVehicle(Aircraft(), Ocean(Hs=record['Hs'],Tp=6,seed=record['seed']))
    o=record['observation']
    v.t,v.x,v.z=o['t'],o['x_m'],o['altitude_m']
    v.Vx,v.Vz=o['forward_speed_m_s'],o['vertical_speed_m_s']
    v.damage.water_mass=o['water_mass_kg']
    v.arm()
    return v


def experiment(record, control, horizon=2.0):
    v=restore(record); initial=observe(v); peak=0; terminal=None
    for _ in range(round(horizon/v.dt)):
        control.apply(v); v.step()
        o=observe(v); snap=v.read_telemetry()[2]; peak=max(peak,snap['N_water']/v.ac.W)
        if not np.isfinite([v.z,v.Vx,v.Vz]).all(): terminal='nonfinite'; break
        if v.damage.failed or peak>8: terminal='failure'; break
        if record['mission']['scenario']=='landing' and o['keel_clearance_m']<0:
            contact=hull_force(v.z,v.Vx,v.Vz,o['wave_elevation_m'],v.hull)
            peak=max(peak,contact.N/v.ac.W)
            terminal='soft_touchdown' if abs(v.Vz)<1.5 and peak<3 else 'hard_touchdown'
            break
    end=observe(v)
    if terminal in ('nonfinite','failure','hard_touchdown'): score=-1000.0
    elif terminal=='soft_touchdown': score=100.0-10*abs(v.Vz)-peak
    elif record['mission']['scenario']=='takeoff':
        # Below liftoff speed emphasize acceleration. Once airborne track altitude.
        if initial['keel_clearance_m']<.1 and initial['forward_speed_m_s']<8.5:
            score=10*(v.Vx-initial['forward_speed_m_s']) - abs(v.Vz) - .1*peak
        else:
            score=-2*abs(v.z-8)-max(0,8.5-v.Vx)*3-abs(v.Vz)-.1*peak
    else:
        desired_sink=-min(1.2,max(.3,initial['keel_clearance_m']*.5))
        score=2*(initial['altitude_m']-v.z)-5*abs(v.Vz-desired_sink)-max(0,6.4-v.Vx)*5-.1*peak
    return dict(control=asdict(control),score=float(score),terminal=terminal,
                final=end,peak_load_weights=peak)


def candidates(record):
    base=Control.parse(json.dumps(record['control']))
    if record['mission']['scenario']=='takeoff':
        controls=[Control(t,p) for t in (.5,.8,1.) for p in (-4.,0.,4.,8.,12.)]
    else:
        controls=[Control(t,p) for t in (0.,.1,.3) for p in (-7.,-5.,-3.,0.,3.)]
    controls=list(dict.fromkeys([base,*controls]))
    results=[experiment(record,c) for c in controls]
    for i,r in enumerate(results): r['candidate_id']=i
    ranked=sorted(results,key=lambda r:r['score'],reverse=True)
    offered=ranked[:3]
    if all(r['candidate_id']!=0 for r in offered): offered.append(results[0])
    return results,offered


def choose(pilot, record, offered):
    ids=[r['candidate_id'] for r in offered]
    compact=[dict(candidate_id=r['candidate_id'],control=r['control'],score=round(r['score'],4),
                  terminal=r['terminal'],final_altitude=round(r['final']['altitude_m'],3),
                  final_speed=round(r['final']['forward_speed_m_s'],3),
                  final_vertical_speed=round(r['final']['vertical_speed_m_s'],3)) for r in offered]
    payload=dict(model=pilot.model,stream=False,
                 format={'type':'object','properties':{'candidate_id':{'type':'integer','enum':ids}},
                         'required':['candidate_id'],'additionalProperties':False},
                 messages=[{'role':'system','content':FEEDBACK_PROMPT},
                           {'role':'user','content':json.dumps(dict(mission=record['mission'],
                            observation=record['observation'],experiments=compact))}],
                 options={'temperature':0,'seed':pilot.seed,'num_predict':32,'num_ctx':2048},keep_alive='5m')
    start=time.monotonic(); response=pilot._request('/api/chat',payload)
    if response.get('done') is not True or response.get('done_reason')=='length':
        raise PilotError('incomplete teacher feedback')
    raw=response.get('message',{}).get('content','')
    try: value=json.loads(raw)
    except ValueError as exc: raise PilotError('invalid teacher feedback JSON') from exc
    if not isinstance(value,dict) or set(value)!= {'candidate_id'} or type(value['candidate_id']) is not int or value['candidate_id'] not in ids:
        raise PilotError('teacher selected an unavailable experiment')
    selected=next(r for r in offered if r['candidate_id']==value['candidate_id'])
    return selected,dict(raw_response=raw,latency_s=time.monotonic()-start)


def collect(out, source, pilot):
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    out.mkdir(parents=True,exist_ok=True)
    config=dict(model=pilot.check_model(),prompt=FEEDBACK_PROMPT,source=str(source.resolve()),
                source_sha256=__import__('hashlib').sha256(source.read_bytes()).hexdigest(),horizon_s=2)
    cp=out/'feedback_config.json'
    if cp.exists() and json.loads(cp.read_text())!=config: raise ValueError('resume configuration mismatch')
    cp.write_text(json.dumps(config,indent=2)+'\n')
    dest=out/'teacher.jsonl'; done=set()
    if dest.exists(): done={json.loads(l)['id'] for l in dest.read_text().splitlines()}
    with dest.open('a') as log:
        for i,record in enumerate(rows):
            if record['id'] in done: continue
            results,offered=candidates(record)
            selected,metadata=choose(pilot,record,offered)
            new={k:record[k] for k in ('id','split','seed','Hs','stage','observation','mission')}
            new.update(source='ollama',control=selected['control'],teacher_mode='simulator_feedback',
                       previous_control=record['control'],experiments=results,
                       selected_candidate_id=selected['candidate_id'],**metadata)
            log.write(json.dumps(new,allow_nan=False)+'\n');log.flush()
            print(f"Feedback {i+1}/{len(rows)} {record['id']}: {selected['control']} score={selected['score']:.2f}",flush=True)
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('results/education_001/teacher.jsonl'))
    p.add_argument('--output',type=Path,default=Path('results/education_002'))
    args=p.parse_args()
    if (args.output/'student.npz').exists(): raise ValueError('completed cycle exists')
    teacher=collect(args.output,args.source,OllamaPilot())
    # New test seeds; first cycle test seeds are no longer unseen.
    report=train_evaluate(args.output,teacher,previous_path='results/education_001/student.npz',test_seeds=(201,202))
    rows=[json.loads(l) for l in teacher.read_text().splitlines()]
    report['feedback']=dict(changed_controls=sum(r['control']!=r['previous_control'] for r in rows),
                          choices_at_best_score=sum(abs(r['experiments'][r['selected_candidate_id']]['score']-max(e['score'] for e in r['experiments']))<1e-8 for r in rows),
                          total=len(rows),horizon_s=2)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['feedback'],indent=2))


if __name__=='__main__':main()
