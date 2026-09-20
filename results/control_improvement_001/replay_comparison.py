"""Paired replay: identical recorded Phi commands, different lateral assist."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from ocean_directional import DirectionalOcean
from mavlink_if import FlyingBoatVehicle
from ollama_pilot import SpatialControl,stabilize_lateral
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'results/spatial_connected_phi_001/decisions.jsonl'
commands=[SpatialControl(**json.loads(line)['control']) for line in source.read_text().splitlines()]
rows=[]
for seed in [42,*range(9300,9310)]:
 for wind in (-3,3):
  row=dict(seed=seed,wind_y=wind)
  for assisted in (False,True):
   v=FlyingBoatVehicle(Aircraft(),DirectionalOcean(Hs=.3,Tp=6,seed=seed),spatial=True,atmosphere=AtmosphereConfig(wind=(0,wind,0),gust_rms=.5),seed=seed)
   v.z=25;v.Vx=13*math.cos(math.radians(8));v.Vz=-13*math.sin(math.radians(8));v.arm()
   ys=[]
   for step in range(60):
    requested=commands[step//20]
    applied=stabilize_lateral(v,requested) if assisted else requested
    applied.apply(v);v.step();ys.append(abs(v.y))
   row['assisted' if assisted else 'baseline']=dict(final_y=v.y,max_abs_y=max(ys),Vy=v.Vy,z=v.z,damage_failed=v.damage.failed)
  rows.append(row)
report=dict(command_source=str(source.relative_to(ROOT)),duration=3,rows=rows,
            mean_max_abs_y={key:sum(r[key]['max_abs_y'] for r in rows)/len(rows) for key in ('baseline','assisted')},
            improved_pairs=int(sum(r['assisted']['max_abs_y']<r['baseline']['max_abs_y'] for r in rows)))
Path(__file__).with_name('replay_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['mean_max_abs_y'],report['improved_pairs'],'/',len(rows))
