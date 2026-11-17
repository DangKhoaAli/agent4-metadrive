"""Evaluate V3.16.1 stabilization vs the frozen V3.16 and rule speed governor.

When --reference is supplied, previously recorded matching V3.16 validation
rollouts are REUSED without claiming a new run, so only V3.16.1 is simulated.
Candidate inference uses learned V3.16 estimator + causal signal filter ONLY.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# v316 module is an untouched sibling; do not copy or overwrite it.
_root=Path(__file__).resolve().parent.parent
_old=_root/'agent4_longitudinal_v316'
if not (_old/'common_v316.py').is_file():
    raise FileNotFoundError('Place agent4_longitudinal_v3161 beside agent4_longitudinal_v316')
if str(_old) not in sys.path:
    sys.path.insert(0,str(_old))

import joblib
from common_v316 import (allowable_seeds, dependencies, make_features,
                         predict_long, sha, speed_schedule, validate_model, verified_action)
from common_v3161 import VERSION, FilterConfig, StabilizedLongitudinal, file_sha256

RULE='rule_governor_v313'
RAW='learned_longitudinal_v316'
STABLE='stabilized_longitudinal_v3161'
PROFILES=(RULE, RAW, STABLE)
SCHEDULE='cruise 65, slow 55, cruise 60 ticks'


def run_episode(env, baseline, bundle, c, feat, geom, map_name, seed, base_speed, steps, profile, filter_config):
    from metadrive.policy.idm_policy import IDMPolicy
    env.reset(seed=seed)
    policy=geom.get_policy(env)
    if isinstance(policy,IDMPolicy):
        raise RuntimeError('IDM-controlled ego is forbidden at evaluation')
    stabilizer=StabilizedLongitudinal(filter_config) if profile==STABLE else None
    status='step_limit'; n=checked=near=road=brakes=overspeed=0
    speed_error_sum=0.0;maxlat=maxhead=0.0;minmargin=None
    action_variation=0.0;reversals=0;previous_action=None;previous_sign=0
    errors=[];trace=[]
    for step in range(steps):
        try:
            pre=geom.state_before_action(env.agent)
            target=speed_schedule(base_speed,step)
            row={**pre,'target_speed_mps':target}
            make_features(row,feat)
            steer=feat.steering_prediction(baseline,pre)
            if profile==RULE:
                action=c.governor(float(pre['speed_mps']),target)
            elif profile==RAW:
                action=predict_long(bundle,row,feat)
            elif profile==STABLE:
                action=stabilizer.apply(predict_long(bundle,row,feat))
            else:
                raise ValueError('Unknown profile')
            if not -1<=action<=1:
                raise ValueError('Invalid longitudinal action')
            speed_error_sum+=abs(target-float(pre['speed_mps']))
            overspeed+=int(float(pre['speed_mps'])>target+0.45)
            lat=abs(float(pre['planned_lateral_right_m']))
            head=abs(float(pre['heading_error_rad']))
            maxlat=max(maxlat,lat); maxhead=max(maxhead,head)
            margin=pre.get('edge_clearance_proxy_m')
            if margin is not None:
                margin=float(margin)
                minmargin=margin if minmargin is None else min(minmargin,margin)
                near+=int(margin<.35)
            term,trunc,info=c.read_step(env.step([steer,action]))
            n+=1
            verified_action(env.agent,steer,action); checked+=1
            if geom.get_policy(env) is not policy:
                raise RuntimeError('Ego policy changed during rollout')
            road+=int(bool(info.get('on_yellow_continuous_line')))
            brakes+=int(action<-.02)
            if previous_action is not None:
                action_variation+=abs(action-previous_action)
            s=1 if action>.02 else (-1 if action<-.02 else 0)
            if s!=0:
                if previous_sign!=0 and s!=previous_sign:reversals+=1
                previous_sign=s
            previous_action=action
            if step%30==0 or term or trunc:
                trace.append(dict(step=step,speed_mps=float(pre['speed_mps']),
                    desired_mps=target,steering=steer,signed_longitudinal=action,
                    lateral_m=lat,edge_clearance_proxy_m=margin))
            if term or trunc:
                status=c.episode_status(term,trunc,info);break
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}');status='episode_error';break
    arrived=status=='arrive_dest' and n==checked and not errors
    quality=(arrived and maxlat<=.60 and maxhead<=.12 and minmargin is not None
             and minmargin>=.30 and near==0 and road==0)
    return dict(profile=profile,map=map_name,seed=seed,speed_mps=base_speed,
        status=status,confirmed_success=arrived,quality_pass=quality,steps=n,
        verified_steps=checked,speed_mae_mps=(speed_error_sum/n if n else None),
        overspeed_steps=overspeed,model_brake_steps=brakes,
        max_abs_lateral_m=maxlat,max_abs_heading_error_rad=maxhead,
        min_edge_clearance_proxy_m=minmargin,near_edge_warning_steps=near,
        road_marked_steps=road,teacher_calls=0,idm_policy_used=False,
        learned_longitudinal=profile!=RULE,errors=errors,
        action_total_variation=action_variation,action_sign_reversals=reversals,
        trace=trace)


def summary_for(episodes):
    result={}
    for profile in PROFILES:
        rows=[e for e in episodes if e['profile']==profile]
        n=sum(e['steps'] for e in rows if e.get('speed_mae_mps') is not None)
        mae=sum(e['speed_mae_mps']*e['steps'] for e in rows if e.get('speed_mae_mps') is not None)
        result[profile]=dict(total=len(rows),arrived=sum(e['confirmed_success'] for e in rows),
            quality=sum(e['quality_pass'] for e in rows),speed_mae_mps=(mae/n if n else None),
            brake_steps=sum(e['model_brake_steps'] for e in rows),
            overspeed_steps=sum(e['overspeed_steps'] for e in rows),
            action_total_variation=(sum(e['action_total_variation'] for e in rows)
                if rows and all('action_total_variation' in e for e in rows) else None),
            action_sign_reversals=(sum(e['action_sign_reversals'] for e in rows)
                if rows and all('action_sign_reversals' in e for e in rows) else None),
            failed=[dict(map=e['map'],seed=e['seed'],speed=e['speed_mps'],
                         status=e['status']) for e in rows if not e['quality_pass']])
    return result


def compare(episodes,mode,require_full_suite=False):
    indexed={}
    for e in episodes:
        key=(e['profile'],e['map'],int(e['seed']),float(e['speed_mps']))
        if key in indexed:raise ValueError(f'Duplicate evaluation row {key}')
        indexed[key]=e
    groups={p:{(k[1],k[2],k[3]) for k in indexed if k[0]==p} for p in PROFILES}
    complete=bool(groups[RULE] and all(groups[p]==groups[RULE] for p in PROFILES))
    required={(m,seed,float(speed)) for m in ('S','C','X')
              for seed in range(186,190) for speed in (2,3,4)}
    full_validation_suite=(mode=='validation' and complete and groups[RULE]==required)
    regressions=[]
    for key in sorted(groups[RULE]):
        out=indexed.get((STABLE,*key))
        if out is None:
            regressions.append(f'missing:{key}');continue
        for baseline in (RULE, RAW):
            b=indexed.get((baseline,*key))
            if b is None:
                regressions.append(f'missing:{baseline}:{key}');continue
            if b['confirmed_success'] and not out['confirmed_success']:
                regressions.append(f'arrival_vs_{baseline}:{key}')
            if b['quality_pass'] and not out['quality_pass']:
                regressions.append(f'quality_vs_{baseline}:{key}')
    s=summary_for(episodes)
    b,r,q=(s[p] for p in PROFILES)
    tolerance=(max(.05,.05*b['speed_mae_mps']) if b['speed_mae_mps'] is not None else None)
    tracking=(complete and q['speed_mae_mps'] is not None and tolerance is not None
              and q['speed_mae_mps']<=b['speed_mae_mps']+tolerance)
    quality=(complete and q['quality']>=max(b['quality'],r['quality']))
    arrived=(complete and q['arrived']>=max(b['arrived'],r['arrived']))
    braking=(complete and q['brake_steps']>=10)
    # More than 4,292 brake steps in raw V3.16 were suspicious. An observed
    # reduction is tracked separately; it is NOT a substitute for safety.
    brake_reduction=(complete and q['brake_steps']<r['brake_steps'])
    accepted=bool(mode=='validation' and complete and (not require_full_suite or full_validation_suite) and not regressions
        and tracking and quality and arrived and braking and brake_reduction)
    return dict(accepted=accepted,complete=complete,full_validation_suite=full_validation_suite,regressions=regressions,
        speed_tracking_within_tolerance=tracking,
        braking_observed=braking,brake_step_reduction_vs_v316=brake_reduction,
        criterion='No arrival/quality regression vs rule OR raw V3.16; no mean speed MAE regression beyond +max(0.05m/s,5%); >=10 AI brakes; fewer AI-brake steps than raw; final never automatically promotes')


def load_reference(file,model_sha,baseline_sha,maps,seeds,speeds,steps):
    doc=json.loads(Path(file).read_text(encoding='utf8'))
    cfg=doc.get('config',{})
    if (doc.get('version')!='AGENT4_V316_SHADOW_IDM_LONGITUDINAL'
        or cfg.get('mode')!='validation' or cfg.get('model_sha256')!=model_sha
        or cfg.get('baseline_sha256')!=baseline_sha or cfg.get('steps')!=steps
        or cfg.get('traffic_density')!=0.0 or cfg.get('speed_schedule')!=SCHEDULE):
        raise ValueError('V3.16 validation reference model/baseline/settings mismatch')
    wanted={(m,s,v) for m in maps for s in seeds for v in speeds}
    found={}
    for ep in doc.get('episodes',[]):
        key=(ep['map'],int(ep['seed']),float(ep['speed_mps']))
        if key not in wanted or ep['profile'] not in (RULE,RAW):continue
        idx=(ep['profile'],*key)
        if idx in found:raise ValueError('Duplicate V3.16 reference episode')
        if ep['errors'] or ep['steps']!=ep['verified_steps'] or ep['teacher_calls']!=0 or ep['idm_policy_used']:
            raise ValueError('Invalid V3.16 reference episode')
        found[idx]=ep
    expected={(profile,*k) for k in wanted for profile in (RULE,RAW)}
    if set(found)!=expected:
        raise ValueError(f'Reference does not contain complete matched scenarios: {len(found)}/{len(expected)}')
    return list(found.values())


def evaluate(args):
    from common_v316 import VERSION as ORIGINAL_VERSION
    c,feat,geom=dependencies(args.legacy_root)
    orig,splits,hash_orig=c.original_manifest(args.dataset)
    maps=geom.parse_maps(args.maps)
    speeds=geom.parse_speeds(args.speeds)
    if not set(maps)<=set(orig['config']['maps']) or not set(speeds)<=set(orig['config']['speeds_mps']):
        raise ValueError('Unexpected map/speed')
    if args.mode=='final' and not args.confirm_frozen:
        raise ValueError('Final evaluation requires --confirm-frozen')
    if args.mode=='final' and args.reference:
        raise ValueError('Never reuse a validation reference in a final holdout')
    seeds=allowable_seeds(geom.parse_seed_spec(args.seeds),splits,
                          'val' if args.mode=='validation' else 'final')
    if args.steps<180:raise ValueError('Need >=180 steps')
    baseline=joblib.load(args.baseline)
    c.check_baseline(baseline,hash_orig,feat)
    model=joblib.load(args.model)
    validate_model(model,hash_orig,sha(args.baseline),feat)
    if model.get('version')!=ORIGINAL_VERSION:
        raise ValueError('Expected frozen original V3.16 model')
    config_filter=FilterConfig(up_rate=args.up_rate,down_rate=args.down_rate,
        max_step_up=args.max_step_up,max_step_down=args.max_step_down,
        deadband=args.deadband,max_throttle=args.max_throttle,max_brake=args.max_brake)
    model_sha=sha(args.model);baseline_sha=sha(args.baseline)
    config=dict(maps=maps,seeds=seeds,speeds_mps=speeds,steps=args.steps,
        mode=args.mode,model_sha256=model_sha,baseline_sha256=baseline_sha,
        filter=config_filter.to_dict(),speed_schedule=SCHEDULE,traffic_density=0.0,
        reference_sha256=(file_sha256(args.reference) if args.reference else None))
    path=Path(args.out)
    if path.exists():
        if not args.resume:raise FileExistsError('Output exists: --resume or new path')
        doc=json.loads(path.read_text(encoding='utf8'))
        if doc.get('version')!=VERSION or doc.get('config')!=config:
            raise ValueError('Resume config mismatch (model/settings/reference may differ)')
    else:
        if path.parent.exists() and path.is_dir():raise ValueError('Expected output JSON file')
        doc=dict(version=VERSION,config=config,episodes=[],summary={},decision={},
          caveats=['V3.16 estimator frozen; V3.16.1 is signal filtering, NOT retraining',
                   'No traffic, privileged MetaDrive geometry; no real-world safety guarantee',
                   'Rule governor used only for baseline; no IDM or rule fallback in either AI profile',
                   'Reference A/B episodes are reused from prior validation only when verified',
                   'Validation scenarios 186..189 already inspected; final seeds >=202 must stay untouched'])
        if args.reference:
            doc['episodes'].extend(load_reference(args.reference,model_sha,baseline_sha,maps,seeds,speeds,args.steps))
        c.atomic_json(path,doc)
    done={(e['profile'],e['map'],int(e['seed']),float(e['speed_mps'])) for e in doc['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    for m in maps:
        todo=[(prof,s,v) for s in seeds for v in speeds for prof in PROFILES
              if (prof,m,s,v) not in done]
        if not todo:continue
        env=MetaDriveEnv(dict(map=m,use_render=False,num_scenarios=max(seeds)+1,
             traffic_density=0.0,horizon=max(1000,args.steps+10),log_level=40,
             physics_world_step_size=.02,decision_repeat=5))
        try:
            for prof,s,v in todo:
                ep=run_episode(env,baseline,model,c,feat,geom,m,s,v,args.steps,prof,config_filter)
                doc['episodes'].append(ep)
                doc['summary']=summary_for(doc['episodes'])
                doc['decision']=compare(doc['episodes'],args.mode,require_full_suite=True)
                c.atomic_json(path,doc)
                print(f'{prof} {m} seed={s} v={v:g}: {ep["status"]} Q={int(ep["quality_pass"])} '
                    f'mae={ep["speed_mae_mps"]} brakes={ep["model_brake_steps"]}',flush=True)
        finally:
            env.close()
    doc['summary']=summary_for(doc['episodes']);doc['decision']=compare(doc['episodes'],args.mode,require_full_suite=True)
    c.atomic_json(path,doc)
    print('Saved:',path.resolve(),'accepted:',doc['decision']['accepted'])
    return doc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True,help='FROZEN agent4_v316_longitudinal.joblib')
    p.add_argument('--reference',type=Path,help='Existing V3.16 validation JSON; reuse verified rule/raw runs')
    p.add_argument('--mode',choices=('validation','final'),default='validation')
    p.add_argument('--confirm-frozen',action='store_true')
    p.add_argument('--maps',default='S,C,X');p.add_argument('--seeds',default='186-189')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--out',type=Path,default=Path('agent4_v3161_validation.json'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--up-rate',type=float,default=.30)
    p.add_argument('--down-rate',type=float,default=.65)
    p.add_argument('--max-step-up',type=float,default=.16)
    p.add_argument('--max-step-down',type=float,default=.35)
    p.add_argument('--deadband',type=float,default=.065)
    p.add_argument('--max-throttle',type=float,default=.70)
    p.add_argument('--max-brake',type=float,default=.80)
    return evaluate(p.parse_args(argv))

if __name__=='__main__':main()
