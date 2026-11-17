"""A/B closed-loop validation: same frozen V3.13 steering, ONLY gas/brake differs.

Baseline: V3.15 rule speed governor. Candidate: V3.16 model's direct output.
No IDM is instantiated or used at evaluation. No rule-governor fallback in
candidate. Final holdout seeds must be previously unseen >=202.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import joblib
from common_v316 import (VERSION,allowable_seeds,dependencies,make_features,
                         predict_long,sha,speed_schedule,validate_model,verified_action)
PROFILES=('rule_governor_v313','learned_longitudinal_v316')


def episode(env, baseline, bundle, c, feat, geom, m, seed, base_speed, steps, profile):
    from metadrive.policy.idm_policy import IDMPolicy
    env.reset(seed=seed);owner=geom.get_policy(env)
    if isinstance(owner,IDMPolicy):raise RuntimeError('IDM is not allowed in evaluation')
    status='step_limit';n=checked=near=road=brakes=overspeed=0
    speed_error_sum=0.;maxlat=maxhead=0.;minmargin=None;errors=[];trace=[]
    for step in range(steps):
        try:
            pre=geom.state_before_action(env.agent)
            target=speed_schedule(base_speed,step)
            row={**pre,'target_speed_mps':target}
            # Check identical pre-action observation schema for both profiles.
            make_features(row,feat)
            steer=feat.steering_prediction(baseline,pre)
            longitude=(c.governor(float(pre['speed_mps']),target) if profile==PROFILES[0]
                       else predict_long(bundle,row,feat))
            if not -1<=longitude<=1:raise ValueError('Invalid signed model action')
            speed_error_sum+=abs(target-float(pre['speed_mps']))
            overspeed+=int(float(pre['speed_mps'])>target+.45)
            lat=abs(float(pre['planned_lateral_right_m']));head=abs(float(pre['heading_error_rad']))
            maxlat=max(maxlat,lat);maxhead=max(maxhead,head)
            margin=pre.get('edge_clearance_proxy_m')
            if margin is not None:
                margin=float(margin)
                minmargin=margin if minmargin is None else min(margin,minmargin)
                near+=int(margin<.35)
            term,trunc,info=c.read_step(env.step([steer,longitude]));n+=1
            verified_action(env.agent,steer,longitude);checked+=1
            if geom.get_policy(env) is not owner:raise RuntimeError('Ego policy switched')
            road+=int(bool(info.get('on_yellow_continuous_line')))
            brakes+=int(longitude<-.02)
            if step%30==0 or term or trunc:
                trace.append(dict(step=step,speed_mps=float(pre['speed_mps']),
                    desired_mps=target,steering=steer,signed_longitudinal=longitude,
                    lateral_m=lat,edge_clearance_proxy_m=margin))
            if term or trunc:
                status=c.episode_status(term,trunc,info);break
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}');status='episode_error';break
    arrived=(status=='arrive_dest' and n==checked and not errors)
    quality=(arrived and maxlat<=.60 and maxhead<=.12 and
             minmargin is not None and minmargin>=.30 and near==0 and road==0)
    return dict(profile=profile,map=m,seed=seed,speed_mps=base_speed,status=status,
        confirmed_success=arrived,quality_pass=quality,steps=n,verified_steps=checked,
        speed_mae_mps=(speed_error_sum/n if n else None),overspeed_steps=overspeed,
        model_brake_steps=brakes,max_abs_lateral_m=maxlat,
        max_abs_heading_error_rad=maxhead,min_edge_clearance_proxy_m=minmargin,
        near_edge_warning_steps=near,road_marked_steps=road,
        teacher_calls=0,idm_policy_used=False,learned_longitudinal=profile==PROFILES[1],
        errors=errors,trace=trace)


def summarize(episodes):
    result={}
    for prof in PROFILES:
        rows=[e for e in episodes if e['profile']==prof]
        s=sum(e['speed_mae_mps']*e['steps'] for e in rows if e['speed_mae_mps'] is not None)
        n=sum(e['steps'] for e in rows if e['speed_mae_mps'] is not None)
        result[prof]=dict(total=len(rows),arrived=sum(e['confirmed_success'] for e in rows),
            quality=sum(e['quality_pass'] for e in rows),speed_mae_mps=(s/n if n else None),
            brake_steps=sum(e['model_brake_steps'] for e in rows),
            overspeed_steps=sum(e['overspeed_steps'] for e in rows),
            failed=[dict(map=e['map'],seed=e['seed'],speed=e['speed_mps'],status=e['status'])
                    for e in rows if not e['quality_pass']])
    return result


def comparison(episodes,mode):
    by={(e['profile'],e['map'],e['seed'],e['speed_mps']):e for e in episodes}
    base=[e for e in episodes if e['profile']==PROFILES[0]]
    candidate=[e for e in episodes if e['profile']==PROFILES[1]]
    regress=[]
    for e in base:
        q=by.get((PROFILES[1],e['map'],e['seed'],e['speed_mps']))
        if not q:
            regress.append(f'missing:{e["map"]}/{e["seed"]}/{e["speed_mps"]}');continue
        if e['confirmed_success'] and not q['confirmed_success']:
            regress.append(f'arrival:{e["map"]}/{e["seed"]}/{e["speed_mps"]}')
        if e['quality_pass'] and not q['quality_pass']:
            regress.append(f'quality:{e["map"]}/{e["seed"]}/{e["speed_mps"]}')
    summary=summarize(episodes);b=summary[PROFILES[0]];q=summary[PROFILES[1]]
    complete=(b['total']>0 and b['total']==q['total'] and len(base)==len(candidate)
              and set((e['map'],e['seed'],e['speed_mps']) for e in base)
              ==set((e['map'],e['seed'],e['speed_mps']) for e in candidate))
    tolerance=max(.05,.05*b['speed_mae_mps']) if b['speed_mae_mps'] is not None else None
    tracking_ok=(complete and b['speed_mae_mps'] is not None and q['speed_mae_mps'] is not None
                 and q['speed_mae_mps']<=b['speed_mae_mps']+tolerance)
    accepted=(mode=='validation' and complete and not regress and tracking_ok
              and q['arrived']>=b['arrived'] and q['quality']>=b['quality']
              and q['brake_steps']>=10)
    return dict(accepted=accepted,complete=complete,regressions=regress,
        speed_tracking_within_tolerance=tracking_ok,
        validation_criterion='no arrival/quality regression; speed MAE <= baseline + max(0.05 m/s, 5%); >=10 learned braking steps; final never promotes')


def evaluate(args):
    c,feat,geom=dependencies(args.legacy_root)
    manifest,splits,manifest_hash=c.original_manifest(args.dataset)
    maps=geom.parse_maps(args.maps);speeds=geom.parse_speeds(args.speeds)
    if not set(maps)<=set(manifest['config']['maps']) or not set(speeds)<=set(manifest['config']['speeds_mps']):
        raise ValueError('Unknown maps/speeds')
    if args.mode=='final' and not args.confirm_frozen:raise ValueError('Final holdout requires --confirm-frozen')
    selected=allowable_seeds(geom.parse_seed_spec(args.seeds),splits,'val' if args.mode=='validation' else 'final')
    if args.steps<180:raise ValueError('Need >=180 steps for speed-change schedule')
    baseline=joblib.load(args.baseline)
    c.check_baseline(baseline,manifest_hash,feat)
    model=joblib.load(args.model)
    validate_model(model,manifest_hash,sha(args.baseline),feat)
    config=dict(maps=maps,seeds=selected,speeds_mps=speeds,steps=args.steps,
        mode=args.mode,model_sha256=sha(args.model),baseline_sha256=sha(args.baseline),
        speed_schedule='cruise 65, slow 55, cruise 60 ticks',traffic_density=0.0)
    path=Path(args.out)
    if path.exists():
        if not args.resume:raise FileExistsError('Output exists; use --resume or new file')
        doc=json.loads(path.read_text(encoding='utf8'))
        if doc.get('config')!=config or doc.get('version')!=VERSION:raise ValueError('Resume mismatch')
    else:
        doc=dict(version=VERSION,config=config,episodes=[],summary={},decision={},
            caveats=['No traffic; privileged MetaDrive geometry; no physical safety guarantee',
                     'V3.13 frozen steering in both profiles; longitudinal model only differs',
                     'No IDM teacher or rule fallback for learned profile',
                     'Validation cannot prove generalization; final data never tunes model'])
    done={(e['profile'],e['map'],e['seed'],e['speed_mps']) for e in doc['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    for m in maps:
        todo=[(prof,s,v) for s in selected for v in speeds for prof in PROFILES
              if (prof,m,s,v) not in done]
        if not todo:continue
        env=MetaDriveEnv(dict(map=m,use_render=False,num_scenarios=max(selected)+1,
            traffic_density=0.,horizon=max(1000,args.steps+10),log_level=40,
            physics_world_step_size=.02,decision_repeat=5))
        try:
            for prof,s,v in todo:
                row=episode(env,baseline,model,c,feat,geom,m,s,v,args.steps,prof)
                doc['episodes'].append(row)
                doc['summary']=summarize(doc['episodes']);doc['decision']=comparison(doc['episodes'],args.mode)
                c.atomic_json(path,doc)
                print(f'{prof} {m} seed={s} v={v:g}: {row["status"]} '
                      f'q={int(row["quality_pass"])} speedMAE={row["speed_mae_mps"]} brakes={row["model_brake_steps"]}',flush=True)
        finally:env.close()
    doc['summary']=summarize(doc['episodes']);doc['decision']=comparison(doc['episodes'],args.mode)
    c.atomic_json(path,doc)
    print('Saved:',path.resolve(),'accepted:',doc['decision']['accepted'])
    return doc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--mode',choices=['validation','final'],default='validation')
    p.add_argument('--confirm-frozen',action='store_true')
    p.add_argument('--maps',default='S,C,X')
    p.add_argument('--seeds',default='186-189')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--out',type=Path,default=Path('agent4_v316_validation.json'))
    p.add_argument('--resume',action='store_true')
    return evaluate(p.parse_args(argv))

if __name__=='__main__':main()
