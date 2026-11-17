"""V3.15 model-versus-V3.13 same-seed closed-loop evaluator. NO IDM fallback.

Only validation seeds can be used to select/promote the candidate. For final
independent evaluation, require new unseen seeds and a frozen model.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
from pathlib import Path
import json

import joblib
from common_v315 import (VERSION, STEP_SECONDS, allowed_evaluation_seeds,
                         atomic_json, check_baseline, digest, episode_status,
                         governor, legacy_modules, original_manifest, predict, read_step)


def run_episode(env,bundle,features,geometry,m,seed,speed,steps,profile):
    from metadrive.policy.idm_policy import IDMPolicy
    env.reset(seed=seed)
    pol=geometry.get_policy(env)
    if isinstance(pol,IDMPolicy):raise RuntimeError('IDM engine policy installed in MODEL-only eval')
    status='step_limit'; n=checked=near=road_marked=0; maxlat=maxhead=0.; minmargin=None; errs=[]; trace=[]
    for step in range(steps):
        try:
            pre=geometry.state_before_action(env.agent)
            lat=abs(float(pre['planned_lateral_right_m'])); head=abs(float(pre['heading_error_rad']))
            maxlat=max(maxlat,lat);maxhead=max(maxhead,head)
            margin=pre.get('edge_clearance_proxy_m')
            if margin is not None:
                margin=float(margin);minmargin=min(margin,minmargin) if minmargin is not None else margin
                near+=int(margin < .35)
            alpha=0.0 if profile=='baseline_v313' else None
            steer,base,correction,gate=predict(bundle,pre,features,alpha)
            longitude=governor(float(pre['speed_mps']),speed)
            terminated,truncated,info=read_step(env.step([steer,longitude]));n+=1
            road_marked+=int(bool(info.get('on_yellow_continuous_line')))
            applied=tuple(float(x) for x in env.agent.current_action)
            if len(applied)!=2 or not all(math.isfinite(x) for x in applied) or max(abs(applied[0]-steer),abs(applied[1]-longitude))>5e-4:
                raise RuntimeError('Model steering or throttle command not applied')
            if geometry.get_policy(env) is not pol:raise RuntimeError('Engine policy changed')
            checked+=1
            if step%25==0 or terminated or truncated:
                trace.append(dict(step=step,lateral_m=lat,heading_error_rad=head,
                                  edge_clearance_proxy_m=margin,
                                  base_steer=base,correction=correction,gate=gate,
                                  predicted_steer=steer,applied_steer=applied[0]))
            if terminated or truncated:
                status=episode_status(terminated,truncated,info);break
        except Exception as exc:
            errs.append(f'{type(exc).__name__}: {exc}');status='runtime_or_prediction_error';break
    arrived=(status=='arrive_dest' and n==checked and not errs)
    quality=(arrived and maxlat<=.6 and maxhead<=.12 and
             minmargin is not None and minmargin>=.3 and near==0 and road_marked==0)
    return dict(profile=profile,map=m,seed=seed,speed_mps=speed,
                status=status,confirmed_success=arrived,quality_pass=quality,
                applied_steps=n,verified_prediction_steps=checked,
                teacher_fallback_calls=0,idm_policy_used=False,learned_longitudinal=False,
                max_abs_lateral_m=maxlat,max_abs_heading_error_rad=maxhead,
                min_edge_clearance_proxy_m=minmargin,near_edge_warning_steps=near,
                road_marked_steps=road_marked,
                errors=errs,trace=trace)


def summarize(episodes):
    result={}
    for profile in ('baseline_v313','candidate_v315'):
        rows=[e for e in episodes if e['profile']==profile]
        result[profile]=dict(total=len(rows),arrived=sum(e['confirmed_success'] for e in rows),
                             quality=sum(e['quality_pass'] for e in rows),
                             failures=[dict(map=e['map'],seed=e['seed'],speed=e['speed_mps'],
                                            status=e['status'],quality=e['quality_pass'])
                                       for e in rows if not e['quality_pass']])
    return result


def acceptance(summary,mode):
    b=summary['baseline_v313'];c=summary['candidate_v315']
    if b['total']!=c['total'] or b['total']==0: return False
    # Require at least one new quality success without regressing any baseline-quality episode.
    # A strict promotion rule prevents gaining aggregate rate while losing a previously safe case.
    # Individual-pair regressions are checked in compare() below.
    return c['arrived']>=b['arrived'] and c['quality']>b['quality']


def compare(episodes,mode):
    by={(e['profile'],e['map'],e['seed'],e['speed_mps']):e for e in episodes}
    b=[e for e in episodes if e['profile']=='baseline_v313']
    c=[e for e in episodes if e['profile']=='candidate_v315']
    if len(b)!=len(c):return dict(accepted=False,regressions=['missing pairs'])
    regressions=[]; gains=[]
    for e in b:
        p=by.get(('candidate_v315',e['map'],e['seed'],e['speed_mps']))
        if p is None:
            regressions.append(f'candidate_missing:{e["map"]}/{e["seed"]}/{e["speed_mps"]}')
            continue
        if e['confirmed_success'] and not p['confirmed_success']:
            regressions.append(f'arrival_regression:{e["map"]}/{e["seed"]}/{e["speed_mps"]}')
        if e['quality_pass'] and not p['quality_pass']:
            regressions.append(f'quality_regression:{e["map"]}/{e["seed"]}/{e["speed_mps"]}')
        if not e['quality_pass'] and p['quality_pass']:
            gains.append(f'{e["map"]}/{e["seed"]}/{e["speed_mps"]}')
    summary=summarize(episodes)
    accepted=(mode=='validation' and acceptance(summary,mode) and not regressions)
    return dict(accepted=accepted,regressions=regressions,quality_gains=gains,
                criterion='validation: strictly more quality passes, zero baseline-quality/arrival regressions; final never tunes')


def evaluate(args):
    features,geometry=legacy_modules(args.legacy_root)
    manifest,splits,manifest_hash=original_manifest(args.dataset)
    maps=geometry.parse_maps(args.maps)
    seeds=allowed_evaluation_seeds(geometry.parse_seed_spec(args.seeds),splits,args.mode,args.confirm_frozen)
    speeds=geometry.parse_speeds(args.speeds)
    if args.steps<100:raise ValueError('--steps must be >=100')
    bundle=joblib.load(args.model) # Trusted local artifact only
    if bundle.get('version')!=VERSION or bundle.get('teacher_fallback') is not False:
        raise ValueError('Invalid V3.15 no-fallback model')
    check_baseline(bundle['baseline'],manifest_hash,features)
    if bundle.get('trained_on_original_manifest_sha256') != manifest_hash:
        raise ValueError('V3.15 model trained using a different V3.12B dataset')
    config=dict(maps=maps,seeds=seeds,speeds_mps=speeds,steps=args.steps,
                mode=args.mode,model_sha256=digest(args.model),traffic_density=0.0,
                no_idm_fallback=True,learned_longitudinal=False)
    path=Path(args.out)
    if path.exists():
        if not args.resume:raise FileExistsError('Output exists; use --resume or different --out')
        report=json.loads(path.read_text(encoding='utf8'))
        if report.get('config')!=config or report.get('version')!=VERSION:raise ValueError('Resume config mismatch')
    else:
        report=dict(version=VERSION,config=config,episodes=[],summary={},promotion={},
                    limitations=['Privileged MetaDrive geometry; no traffic; throttle is RULE BASED',
                                 'No IDM teacher instantiated during evaluation',
                                 'Validation is for model selection; final seeds must be untouched'])
    existing={(e['profile'],e['map'],e['seed'],e['speed_mps']) for e in report['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    for m in maps:
        needed=[(profile,seed,speed) for seed in seeds for speed in speeds
                for profile in ('baseline_v313','candidate_v315') if (profile,m,seed,speed) not in existing]
        if not needed:continue
        env=MetaDriveEnv(dict(map=m,use_render=False,num_scenarios=max(seeds)+1,
                            traffic_density=0.,horizon=max(1000,args.steps+10),log_level=40,
                            physics_world_step_size=.02,decision_repeat=5))
        try:
            for profile,seed,speed in needed:
                row=run_episode(env,bundle,features,geometry,m,seed,speed,args.steps,profile)
                report['episodes'].append(row)
                report['summary']=summarize(report['episodes'])
                report['promotion']=compare(report['episodes'],args.mode)
                atomic_json(path,report)
                print(f'{profile} {m} seed={seed} v={speed:g}: {row["status"]} '
                      f'quality={int(row["quality_pass"])} lateral={row["max_abs_lateral_m"]:.3f}',flush=True)
        finally:env.close()
    report['summary']=summarize(report['episodes'])
    report['promotion']=compare(report['episodes'],args.mode)
    atomic_json(path,report)
    print('Result:',path.resolve(),'PROMOTE:',report['promotion']['accepted'])
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('..'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--mode',choices=['validation','final'],default='validation')
    p.add_argument('--maps',default='C,X')
    p.add_argument('--seeds',default='186-189')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--out',type=Path,default=Path('../agent4_v315_validation.json'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--confirm-frozen',action='store_true')
    return evaluate(p.parse_args(argv))

if __name__=='__main__':main()
