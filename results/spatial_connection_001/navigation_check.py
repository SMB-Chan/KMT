"""Exercise native navigation commands with crosswind, gusts and damage."""
import json
import math
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from ocean_directional import DirectionalOcean
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_TAKEOFF, MAV_CMD_NAV_LAND
OUT = Path(__file__).resolve().parent
report=[]
for scenario in ('takeoff','landing'):
    for wy in (-3, 3):
        v=FlyingBoatVehicle(Aircraft(), DirectionalOcean(Hs=.3,Tp=6,seed=9200),
                           spatial=True, atmosphere=AtmosphereConfig(wind=(0,wy,0),gust_rms=.5),seed=9200)
        if scenario=='landing':
            v.z=25
            v.Vx=13*math.cos(math.radians(8))
            v.Vz=-13*math.sin(math.radians(8))
        v.arm()
        v.send_command(MAV_CMD_NAV_TAKEOFF if scenario=='takeoff' else MAV_CMD_NAV_LAND,
                       dict(alt=8 if scenario=='takeoff' else 0,speed=8.5,y=0))
        status='time_limit'
        rows=[]
        for _ in range(300 if scenario=='takeoff' else 600):
            v.step()
            snap=v.read_telemetry()[2]
            assert np.isfinite([v.x,v.y,v.z,v.Vx,v.Vy,v.Vz,v.bank,v.heading]).all()
            rows.append(snap)
            if v.damage.failed:
                status='damage_failure';break
            if snap['N_water']>8*v.ac.W:
                status='impact_failure';break
            if abs(v.y)>50:
                status='lateral_limit';break
            if scenario=='takeoff' and v.z>=8 and v.Vx>=8.5 and v.lateral_success():
                status='success';break
            if scenario=='landing' and v.z-v.hull.h_keel<v.wave_elevation():
                status='success' if abs(v.Vz)<1.5 and snap['N_water']<3*v.ac.W and v.lateral_success() else 'hard_landing'
                break
        v.disarm()
        case=dict(scenario=scenario,wind_y=wy,seed=9200,Hs=.3,Tp=6,gust_rms=.5,status=status,
                  steps=len(rows),max_abs_y=max(abs(r['y']) for r in rows),
                  max_abs_bank_deg=max(abs(math.degrees(r['bank'])) for r in rows),final=rows[-1])
        report.append(case)
        (OUT/f'navigation_{scenario}_{wy:+d}.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(scenario,wy,status,'max |y|',case['max_abs_y'],flush=True)
(OUT/'navigation.json').write_text(json.dumps(report,indent=2)+'\n')
