"""Fast real-MetaDrive API smoke test. Run BEFORE collecting any training data."""
from __future__ import annotations
import argparse,json
from traffic_v38 import spawn_lead,observe_front
from common_v38 import SCENARIOS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',default='S',choices=('S',))
    p.add_argument('--seed',type=int,default=128)
    p.add_argument('--gap',type=float,default=24.)
    p.add_argument('--steps',type=int,default=15)
    a=p.parse_args()
    from metadrive.envs.metadrive_env import MetaDriveEnv
    all_results=[]
    for scenario in SCENARIOS:
        env=MetaDriveEnv(dict(use_render=False,map=a.map,
                num_scenarios=a.seed+1,traffic_density=0.,horizon=600,log_level=40))
        try:
            env.reset(seed=a.seed)
            lead=spawn_lead(env,scenario,a.gap,a.seed)
            initial=observe_front(env)
            if not initial['present'] or abs(initial['gap_m']-a.gap)>6:
                raise RuntimeError(f'Lead was not observable: {initial}')
            snapshots=[{'step':0,**initial}]
            for t in range(a.steps):
                _,_,terminated,truncated,info=env.step([0.,0.])
                if terminated or truncated:
                    raise RuntimeError(f'Ego terminated during smoke at {t}: {info}')
                snapshots.append({'step':t+1,**observe_front(env)})
            lead_speed=float(lead.speed)
            if scenario=='stop_release' and lead_speed>0.5:
                raise RuntimeError(f'Stopped lead accelerated unexpectedly: {lead_speed:.3f} m/s')
            if scenario=='slow_release' and lead_speed>2.5:
                raise RuntimeError(f'Slow lead accelerated unexpectedly: {lead_speed:.3f} m/s')
            report=dict(scenario=scenario,lead_vehicle_id=str(lead.id),
                        lead_speed_after_mps=lead_speed,
                        initial_observation=initial,after_steps=snapshots[-1])
            all_results.append(report)
            print(json.dumps(report,indent=2,allow_nan=False),flush=True)
        finally:env.close()
    print('V3.8 MetaDrive smoke: PASS')
if __name__=='__main__':main()
