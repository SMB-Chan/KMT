import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from ocean_directional import DirectionalOcean
from mavlink_if import FlyingBoatVehicle
from ollama_pilot import SpatialControl
rows=[]
for pitch in (0,4,8,12,15):
 for wind in (-3,3):
  v=FlyingBoatVehicle(Aircraft(),DirectionalOcean(Hs=.3,Tp=6,seed=9200),spatial=True,atmosphere=AtmosphereConfig(wind=(0,wind,0),gust_rms=.5),seed=9200)
  v.arm();status='time_limit'
  for _ in range(300):
   bank,rudder=v.lateral_setpoint()
   SpatialControl(1,pitch,math.degrees(bank),rudder).apply(v);v.step()
   if v.damage.failed: status='damage_failure';break
   if v.z>=8 and v.Vx>=8.5 and v.lateral_success():status='success';break
  rows.append(dict(pitch=pitch,wind_y=wind,status=status,x=v.x,z=v.z,Vx=v.Vx,y=v.y))
  print(rows[-1],flush=True)
Path(__file__).with_name('takeoff_sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
