"""V3.11-A: closed-loop *disturbance recovery diagnostic* for frozen V3.10.

Applies an externally prescribed steering kick in MetaDrive, then fully releases
control back to the tested controller. Does not train or alter any model. Both
baseline and learned steering receive the same kick policy, route and seed.

WARNING: A requested lateral offset is not guaranteed. Report actual reached
offset and only count recovery when target was physically reached first.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import joblib

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'agent4_highspeed_training_v310'))
from common_v310 import check_pdd,check_bundle,parse_seed_spec,parse_values
import eval_v310 as v310

VERSION='V311A_DISTURBANCE_RECOVERY_DIAGNOSTIC'
VALID_MAPS=('S','SSSSS')
VALID_SPEEDS=(4.,6.,8.,10.)
PROFILES=v310.PROFILES_V310
SIGNALS=('positive','negative')
STEP_S=.1


def parse_targets(spec):
    values=[float(s.strip()) for s in spec.split(',') if s.strip()]
    if not values or len(set(values))!=len(values) or any(not math.isfinite(x) or not .05<=x<=.65 for x in values):
        raise ValueError('Targets must be unique finite offsets from 0.05 to 0.65 m')
    return values


@dataclass(frozen=True)
class KickSettings:
    start_step:int=65
    target_offset_m:float=.2
    sign:int=1
    steering_md:float=.10
    max_kick_steps:int=22
    maximum_offset_during_kick_m:float=.85
    maximum_heading_during_kick_rad:float=.20
    tolerance_m:float=.05
    tolerance_heading_rad:float=.02
    stable_steps:int=10
    window_steps:int=100
    sample_stride:int=10

    def validate(self):
        if self.start_step<5 or not .05<=self.target_offset_m<=.65 or self.sign not in (-1,1):
            raise ValueError('Invalid kick timing, target or sign')
        if not .01<=self.steering_md<=.25 or not 1<=self.max_kick_steps<=60:
            raise ValueError('Invalid kick amplitude or duration')
        if not .20<=self.maximum_offset_during_kick_m<=1.2 or not .05<=self.maximum_heading_during_kick_rad<=.4:
            raise ValueError('Invalid maximum kick safety threshold')
        if not 0<self.tolerance_m<self.target_offset_m or not 0<self.tolerance_heading_rad<=.1:
            raise ValueError('Recovery tolerance must be lower than offset target')
        if not 2<=self.stable_steps<=30 or not 10<=self.window_steps<=400 or not 1<=self.sample_stride<=100:
            raise ValueError('Invalid stable/window/sample values')
        return self


class DisturbanceMonitor:
    def __init__(self,settings):
        self.s=settings.validate()
        self.calls=0
        self.start_snapshot=None
        self.kick_steps=0
        self.released_at=None
        self.release_reason=None
        self.target_reached=False
        self.target_reached_step=None
        self.peak_abs_offset=0.
        self.peak_after_release=0.
        self.peak_abs_heading=0.
        self.recovery_streak=0
        self.recovery_step=None
        self.recovery_first_good_step=None
        self.observations=[]
        self.measurement_errors=[]
        self.last_offset=None
        self.last_heading=None
        self.last_speed=None

    def _measure(self,env):
        try:
            geom=v310.v30.planned_geometry(env.agent,2.)
            e=float(geom['lateral_right_m']);h=float(geom['heading_error_rad'])
            if not math.isfinite(e) or not math.isfinite(h):raise ValueError('Nonfinite geometry')
            return e,h
        except Exception as exc:
            self.measurement_errors.append(f'{self.calls}: {type(exc).__name__}: {exc}')
            return None,None

    def before_step(self,env,action):
        t=self.calls
        if len(action)!=2:raise ValueError('MetaDrive action must contain steering and longitudinal')
        base=float(action[0]);longitudinal=float(action[1]);
        if not all(math.isfinite(v) for v in (base,longitudinal)):
            raise ValueError('Nonfinite action')
        e,h=self._measure(env)
        # Keep units consistent with feature_for_model(): speed_km_h -> m/s.
        # MetaDrive agent.speed is already in m/s; do NOT divide it again.
        speed=float(env.agent.speed_km_h)/3.6
        if t==self.s.start_step and self.start_snapshot is None:
            self.start_snapshot=dict(step=t,offset_m=e,heading_rad=h,speed_mps=speed)
        extra=0.
        if self.s.start_step<=t and self.released_at is None:
            if e is None or h is None:
                self.released_at=t;self.release_reason='geometry_unavailable'
            elif self.kick_steps==0 and abs(e)>=self.s.target_offset_m:
                self.released_at=t;self.release_reason='offset_already_above_target_before_kick'
            elif self.kick_steps>0 and abs(e)>=self.s.target_offset_m:
                self.released_at=t;self.release_reason='target_reached'
                self.target_reached=True;self.target_reached_step=t-1
            elif self.kick_steps>=self.s.max_kick_steps:
                self.released_at=t;self.release_reason='max_kick_steps'
            elif abs(e)>=self.s.maximum_offset_during_kick_m or abs(h)>=self.s.maximum_heading_during_kick_rad:
                self.released_at=t;self.release_reason='kick_safety_limit'
            else:
                extra=self.s.sign*self.s.steering_md
                self.kick_steps+=1
        steer=max(-.35,min(.35,base+extra))
        return [steer,longitudinal],(e,h,speed),extra

    def after_step(self,env,before,extra,applied,info):
        t=self.calls
        e,h=self._measure(env)
        if e is not None:
            self.peak_abs_offset=max(self.peak_abs_offset,abs(e))
            self.peak_abs_heading=max(self.peak_abs_heading,abs(h))
            self.last_offset,self.last_heading=e,h
        if self.released_at is None and self.kick_steps>0:
            # Target can be crossed on the last action, even if the env ends then.
            if e is not None and abs(e)>=self.s.target_offset_m:
                self.released_at=t+1
                self.release_reason='target_reached'
                self.target_reached=True
                self.target_reached_step=t
            elif self.kick_steps>=self.s.max_kick_steps:
                self.released_at=t+1
                self.release_reason='max_kick_steps'
        if self.released_at is not None and t>=self.released_at and e is not None:
            self.peak_after_release=max(self.peak_after_release,abs(e))
            if t<self.released_at+self.s.window_steps and self.recovery_step is None:
                if abs(e)<=self.s.tolerance_m and abs(h)<=self.s.tolerance_heading_rad:
                    self.recovery_streak+=1
                    if self.recovery_streak>=self.s.stable_steps:
                        self.recovery_step=t
                        self.recovery_first_good_step=t-self.s.stable_steps+1
                else:self.recovery_streak=0
        if extra or (t>=self.s.start_step-1 and t<self.s.start_step+5) or (self.released_at is not None and t in (self.released_at,self.released_at+1)) or t%self.s.sample_stride==0:
            self.observations.append(dict(step=t,pre_offset_m=before[0],pre_heading_rad=before[1],
                pre_speed_mps=before[2],post_offset_m=e,post_heading_rad=h,
                injected_steer_md=extra,actual_steer_md=applied[0],
                release_step=self.released_at,route_completion=info.get('route_completion')))
        self.calls+=1

    def summary(self,episode_status):
        if self.start_snapshot is None:
            outcome='not_triggered'
        elif not self.target_reached:
            outcome='target_not_reached'
        elif self.recovery_step is not None:
            outcome='recovered'
        elif episode_status not in ('arrive_dest','step_limit'):
            outcome='failed_before_recovery'
        elif self.released_at is None or self.calls<self.released_at+self.s.window_steps:
            outcome='insufficient_observation'
        else:
            outcome='not_recovered'
        return dict(target_offset_m=self.s.target_offset_m,steering_kick_sign=self.s.sign,
            steering_kick_md=self.s.steering_md,kick_start_step=self.s.start_step,
            kick_steps=self.kick_steps,kick_release_step=self.released_at,
            kick_release_reason=self.release_reason,target_reached=self.target_reached,
            target_reached_step=self.target_reached_step,pre_kick_state=self.start_snapshot,
            peak_abs_offset_m=self.peak_abs_offset,
            peak_after_release_offset_m=self.peak_after_release,
            peak_abs_heading_rad=self.peak_abs_heading,
            recovered=self.target_reached and self.recovery_step is not None,
            recovery_first_good_step=self.recovery_first_good_step,
            recovery_confirmed_step=self.recovery_step,
            recovery_time_s=(round((self.recovery_first_good_step-self.released_at)*STEP_S,4)
                if self.recovery_first_good_step is not None and self.released_at is not None else None),
            recovery_outcome=outcome,measured_steps=self.calls,
            measurements_aligned=self.calls>=0 and not self.measurement_errors,
            measurement_errors=self.measurement_errors[:10])


@contextmanager
def injected_step(env,settings):
    """Alter ONLY external env action, after each tested controller predicted it."""
    monitor=DisturbanceMonitor(settings)
    original=env.step
    def wrapper(action):
        actual,before,extra=monitor.before_step(env,action)
        result=original(actual)
        _,_,_,_,info=result
        monitor.after_step(env,before,extra,actual,info)
        return result
    env.step=wrapper
    try:yield monitor
    finally:env.step=original


def run_one(env,map_name,seed,speed,profile,pdd,bundle,pdd_names,args,target,sign):
    conf=KickSettings(start_step=args.kick_step,target_offset_m=target,sign=sign,
          steering_md=args.kick_md,max_kick_steps=args.max_kick_steps,
          tolerance_m=args.recovery_tolerance,tolerance_heading_rad=args.heading_tolerance,
          stable_steps=args.stable_steps,window_steps=args.recovery_window,
          sample_stride=args.measure_stride)
    with injected_step(env,conf) as monitor:
        e=v310.run_one(env,map_name,seed,speed,profile,pdd,bundle,pdd_names,args.steps,args.trace_stride)
    e.pop('trace',None) # V3.11 stores a compact but precise disturbance trace instead.
    e['disturbance']=monitor.summary(e['summary']['status'])
    e['trace']=monitor.observations
    e['summary']['measured_steps_aligned']=monitor.calls==e['summary']['n_applied_steps']
    if not e['summary']['measured_steps_aligned']:
        e['errors'].append(f'Measured {monitor.calls} vs applied {e["summary"]["n_applied_steps"]}')
        e['summary']['error_count']=len(e['errors'])
    return e


def aggregate(episodes):
    groups=defaultdict(list)
    for e in episodes:
        groups[(e['map'],e['speed_mps'],e['profile'],e['disturbance']['target_offset_m'])].append(e)
    out=[]
    for (m,s,p,t),grp in sorted(groups.items()):
        valid=[e for e in grp if e['disturbance']['target_reached']]
        reached=len(valid)
        recovered=[e for e in valid if e['disturbance']['recovered']]
        out.append(dict(map=m,speed_mps=s,profile=p,target_offset_m=t,
          attempts=len(grp),targets_reached=reached,recovered=len(recovered),
          reached_target_rate=reached/len(grp),
          observed_recovery_rate=len(recovered)/reached if reached else None,
          arrived=sum(e['summary']['status']=='arrive_dest' for e in grp),
          mean_recovery_s=(sum(e['disturbance']['recovery_time_s'] for e in recovered)/len(recovered)
              if recovered else None)))
    return out


def run(args):
    maps=parse_values(args.maps,VALID_MAPS)
    speeds=parse_values(args.speeds,VALID_SPEEDS,float)
    seeds=parse_seed_spec(args.seeds)
    profiles=parse_values(args.profiles,PROFILES)
    signs=parse_values(args.directions,SIGNALS)
    targets=parse_targets(args.targets)
    if min(seeds)<150:raise ValueError('V3.11 diagnostic seeds >=150 (new holdout only)')
    if not 150<=args.steps<=5000:raise ValueError('Use 150..5000 steps')
    if args.kick_step>=args.steps-args.recovery_window:
        raise ValueError('kick-step must leave enough steps for recovery window')
    KickSettings(start_step=args.kick_step,target_offset_m=max(targets),
        steering_md=args.kick_md,max_kick_steps=args.max_kick_steps,
        tolerance_m=args.recovery_tolerance,tolerance_heading_rad=args.heading_tolerance,
        stable_steps=args.stable_steps,window_steps=args.recovery_window,
        sample_stride=args.measure_stride).validate()
    pdd=joblib.load(args.pdd);pdd_names=check_pdd(pdd)
    bundle=joblib.load(args.steering);check_bundle(bundle,pdd_names,seeds)
    from metadrive.envs.metadrive_env import MetaDriveEnv
    report=dict(version=VERSION,maps=maps,seeds=seeds,speeds_mps=speeds,profiles=profiles,
        target_offsets_m=targets,kick_directions=signs,
        settings=dict(kick_step=args.kick_step,kick_md=args.kick_md,
             max_kick_steps=args.max_kick_steps,recovery_tolerance_m=args.recovery_tolerance,
             heading_tolerance_rad=args.heading_tolerance,stable_steps=args.stable_steps,
             recovery_window_steps=args.recovery_window),
        warnings=['Disturbances are injected into MetaDrive actions, not set as exact initial offsets.',
            'Requested target may NOT be reached; only count recovered if target_reached first.',
            'A recovered offset is not necessarily a safe full route; inspect arrive_dest and road flags.',
            'Steering is learned only for V3.10. Speed governor is rule-based, no learned longitudinal.',
            'No teacher steering fallback in learned profile. Sim-only, privileged geometry, no curves or traffic.'],
        episodes=[])
    for m in maps:
        env=MetaDriveEnv(dict(use_render=False,map=m,num_scenarios=max(seeds)+1,
            traffic_density=0.,horizon=max(args.steps+10,1000),log_level=40,
            physics_world_step_size=.02,decision_repeat=5))
        try:
            for seed in seeds:
                for speed in speeds:
                    for target in targets:
                        for direction in signs:
                            sign=1 if direction=='positive' else -1
                            for profile in profiles:
                                e=run_one(env,m,seed,speed,profile,pdd,bundle,pdd_names,args,target,sign)
                                report['episodes'].append(e)
                                d=e['disturbance'];s=e['summary']
                                print(f'{m} seed={seed} v={speed:g} target={target:g} '
                                    f'{direction} {profile:13s} {s["status"]:26s} '
                                    f'kick={d["kick_steps"]} reached={d["target_reached"]} '
                                    f'offset={d["peak_abs_offset_m"]:.3f} '
                                    f'recovery={d["recovery_outcome"]} '
                                    f't={d["recovery_time_s"]}',flush=True)
        finally:env.close()
    report['aggregate']=aggregate(report['episodes'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False),encoding='utf-8')
    csv_path=args.out.with_suffix('.csv')
    cols=['map','seed','speed_mps','profile','target_offset_m','steering_kick_sign',
         'status','target_reached','recovery_outcome','recovered','recovery_time_s',
         'peak_abs_offset_m','peak_after_release_offset_m','kick_steps','kick_release_reason',
         'max_abs_planned_lateral_right_m','max_speed_mps','error_count','student_predictions_aligned',
         'measured_steps_aligned','teacher_fallback_calls']
    with csv_path.open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=cols);writer.writeheader()
        for e in report['episodes']:
            row=dict(map=e['map'],seed=e['seed'],speed_mps=e['speed_mps'],profile=e['profile'],
                 status=e['summary']['status'])
            for k in cols:
                if k not in row:row[k]=e['disturbance'].get(k,e['summary'].get(k))
            writer.writerow(row)
    print('JSON:',args.out.resolve(),'CSV:',csv_path.resolve())
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdd',required=True,type=Path)
    parser.add_argument('--steering',required=True,type=Path)
    parser.add_argument('--maps',default='SSSSS')
    parser.add_argument('--seeds',default='150,151')
    parser.add_argument('--speeds',default='4,6,8,10')
    parser.add_argument('--profiles',default=','.join(PROFILES))
    parser.add_argument('--targets',default='0.2')
    parser.add_argument('--directions',default=','.join(SIGNALS))
    parser.add_argument('--kick-md',type=float,default=.10)
    parser.add_argument('--kick-step',type=int,default=65)
    parser.add_argument('--max-kick-steps',type=int,default=22)
    parser.add_argument('--recovery-tolerance',type=float,default=.05)
    parser.add_argument('--heading-tolerance',type=float,default=.02)
    parser.add_argument('--stable-steps',type=int,default=10)
    parser.add_argument('--recovery-window',type=int,default=100)
    parser.add_argument('--measure-stride',type=int,default=10)
    parser.add_argument('--trace-stride',type=int,default=80)
    parser.add_argument('--steps',type=int,default=2500)
    parser.add_argument('--out',type=Path,default=Path('agent4_v311_recovery.json'))
    return run(parser.parse_args(argv))

if __name__=='__main__':main()
