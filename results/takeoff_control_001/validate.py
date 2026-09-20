"""Paired held-out comparison of native and existing RL longitudinal control."""
import sys,json,math,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from ocean_directional import DirectionalOcean
from mavlink_if import FlyingBoatVehicle,MAV_CMD_NAV_TAKEOFF
from ollama_pilot import SpatialControl,observe
from policy_pilot import PolicyPilot
from flight_diagnostics import takeoff_progress
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
model=ROOT/'results/takeoff_preview_best_policy.npz'
rows=[]
for seed in range(9500,9505):
 for wind in (-3,3):
  for controller in ('native','rl_with_lateral_feedback'):
   pilot=PolicyPilot(model)
   v=FlyingBoatVehicle(Aircraft(),DirectionalOcean(Hs=.3,Tp=6,seed=seed),spatial=True,atmosphere=AtmosphereConfig(wind=(0,wind,0),gust_rms=.5),seed=seed)
   v.arm();v.send_command(MAV_CMD_NAV_TAKEOFF,dict(alt=8,speed=8.5));trajectory=[];status='time_limit'
   for _ in range(300):
    if controller=='rl_with_lateral_feedback':
     c,_=pilot.decide(observe(v),dict(scenario='takeoff'))
     bank,rudder=v.lateral_setpoint()
     SpatialControl(c.throttle,c.pitch_deg,math.degrees(bank),rudder).apply(v)
    v.step();snap=v.read_telemetry()[2];trajectory.append(snap)
    if v.damage.failed:status='damage_failure';break
    if snap['N_water']>8*v.ac.W:status='impact_failure';break
    if abs(v.y)>50:status='lateral_limit';break
    if v.z>=8 and v.Vx>=8.5 and v.lateral_success():status='success';break
   result=dict(seed=seed,wind_y=wind,controller=controller,status=status,final=trajectory[-1],diagnostics=takeoff_progress(trajectory))
   rows.append(result)
   if seed==9500:
    (OUT/f'{controller}_{wind:+d}_trajectory.json').write_text(json.dumps(trajectory,indent=2)+'\n')
   print(seed,wind,controller,status,round(v.Vx,2),flush=True)
report=dict(Hs=.3,Tp=6,gust_rms=.5,duration=15,seeds=list(range(9500,9505)),model=str(model),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),rows=rows,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
