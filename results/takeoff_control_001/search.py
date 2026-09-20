"""Search pitch schedules with unchanged aircraft, waves and damage."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from ocean_directional import DirectionalOcean
from mavlink_if import FlyingBoatVehicle,MAV_CMD_NAV_TAKEOFF
from ollama_pilot import SpatialControl
OUT=Path(__file__).resolve().parent

def trial(candidate,seed=9400,wind=3,save=False):
 v=FlyingBoatVehicle(Aircraft(),DirectionalOcean(Hs=.3,Tp=6,seed=seed),spatial=True,atmosphere=AtmosphereConfig(wind=(0,wind,0),gust_rms=.5),seed=seed)
 v.arm();v.send_command(MAV_CMD_NAV_TAKEOFF,dict(alt=8,speed=8.5))
 status='time_limit';rows=[]
 for i in range(300):
  if candidate['kind']=='baseline':v.step()
  else:
   if candidate['kind']=='speed':pitch=candidate['pitch'] if v.Vx<candidate['switch'] else 8
   else:pitch=candidate['pitch'] if v.t<candidate['switch'] else 8
   if v.z>2:
    alpha,_=v.ctl.takeoff_setpoint(dict(z=v.z,Vx=v.Vx,Vz=v.Vz),8,8.5)
    pitch=math.degrees(alpha)
   bank,rudder=v.lateral_setpoint()
   SpatialControl(1,pitch,math.degrees(bank),rudder).apply(v);v.step()
  snap=v.read_telemetry()[2];rows.append(snap)
  if v.damage.failed:status='damage_failure';break
  if snap['N_water']>8*v.ac.W:status='impact_failure';break
  if abs(v.y)>50:status='lateral_limit';break
  if v.z>=8 and v.Vx>=8.5 and v.lateral_success():status='success';break
 result=dict(candidate=candidate,seed=seed,wind_y=wind,status=status,steps=len(rows),x=v.x,z=v.z,Vx=v.Vx,max_Vx=max(r['Vx'] for r in rows),max_abs_y=max(abs(r['y']) for r in rows),water_mass=v.damage.water_mass)
 if save:result['trajectory']=rows
 return result

if __name__=='__main__':
 candidates=[dict(kind='baseline')]
 candidates += [dict(kind='speed',pitch=p,switch=s) for p in (-8,-4,0) for s in (3.1,3.5,4.0)]
 candidates += [dict(kind='time',pitch=-8,switch=s) for s in (1,3,5)]
 results=[]
 for c in candidates:
  r=trial(c);results.append(r);print(c,r['status'],round(r['max_Vx'],2),flush=True)
 (OUT/'search.json').write_text(json.dumps(results,indent=2)+'\n')
