"""Paired closed-loop A/B: frozen V3.13 steering + RULE, original V3.16, new V3.16.2.

When --reference is supplied, verified RULE / V3.16 validation episodes are reused;
only the new V3.16.2 candidate is simulated. Final forbids reference reuse.
"""
from __future__ import annotations
import argparse,json,sys,math
from pathlib import Path
import joblib
from common_v3162 import (VERSION,dependencies,allowable_seeds,make_features,
    predict_long,predict_v3162,sha,speed_schedule,validate_model,validate_v3162,verified_action)
RULE='rule_governor_v313';RAW='learned_longitudinal_v316';NEW='learned_longitudinal_v3162'
PROFILES=(RULE,RAW,NEW)
SCHEDULE='cruise 65, slow 55, cruise 60 ticks'


def run_episode(env,baseline,actor,candidate,c,feat,geom,m,seed,speed,steps,profile):
    from metadrive.policy.idm_policy import IDMPolicy
    env.reset(seed=seed);owner=geom.get_policy(env)
    if isinstance(owner,IDMPolicy):raise RuntimeError('IDM is forbidden in evaluation')
    status='step_limit';n=checked=near=road=brake=overspeed=0
    speed_error=0.0;maxlat=maxhead=0.0;minmargin=None;errors=[];trace=[]
    action_variation=0.0;reversals=0;previous=None;previous_sign=0
    for step in range(steps):
        try:
            pre=geom.state_before_action(env.agent)
            target=speed_schedule(speed,step)
            row={**pre,'target_speed_mps':target}
            make_features(row,feat)
            steer=feat.steering_prediction(baseline,pre)
            if profile==RULE:long=c.governor(float(pre['speed_mps']),target)
            elif profile==RAW:long=predict_long(actor,row,feat)
            elif profile==NEW:long=predict_v3162(candidate,row,feat)
            else:raise ValueError('Unknown evaluation profile')
            if not math.isfinite(long) or not -1.0<=long<=1.0:raise ValueError('Invalid action')
            sp=float(pre['speed_mps'])
            speed_error+=abs(target-sp);overspeed+=int(sp>target+.45)
            lat=abs(float(pre['planned_lateral_right_m']));head=abs(float(pre['heading_error_rad']))
            maxlat=max(maxlat,lat);maxhead=max(maxhead,head)
            margin=pre.get('edge_clearance_proxy_m')
            if margin is not None:
                margin=float(margin)
                minmargin=margin if minmargin is None else min(minmargin,margin)
                near+=int(margin<.35)
            term,trunc,info=c.read_step(env.step([steer,long]));n+=1
            verified_action(env.agent,steer,long);checked+=1
            if geom.get_policy(env) is not owner:raise RuntimeError('Ego policy was replaced')
            road+=int(bool(info.get('on_yellow_continuous_line')));brake+=int(long<-.02)
            if previous is not None:action_variation+=abs(long-previous)
            sign=1 if long>.02 else (-1 if long<-.02 else 0)
            if sign and previous_sign and sign!=previous_sign:reversals+=1
            if sign:previous_sign=sign
            previous=long
            if step%30==0 or term or trunc:
                trace.append(dict(step=step,speed_mps=sp,desired_mps=target,
                    steering=steer,signed_longitudinal=long,lateral_m=lat,edge_clearance_proxy_m=margin))
            if term or trunc:
                status=c.episode_status(term,trunc,info);break
        except Exception as e:
            errors.append(f'{type(e).__name__}: {e}');status='episode_error';break
    arrived=status=='arrive_dest' and checked==n and not errors
    quality=(arrived and maxlat<=.60 and maxhead<=.12 and
             minmargin is not None and minmargin>=.30 and near==0 and road==0)
    return dict(profile=profile,map=m,seed=seed,speed_mps=speed,status=status,
        confirmed_success=arrived,quality_pass=quality,steps=n,verified_steps=checked,
        speed_mae_mps=speed_error/n if n else None,overspeed_steps=overspeed,
        model_brake_steps=brake,max_abs_lateral_m=maxlat,max_abs_heading_error_rad=maxhead,
        min_edge_clearance_proxy_m=minmargin,near_edge_warning_steps=near,
        road_marked_steps=road,teacher_calls=0,idm_policy_used=False,
        learned_longitudinal=profile!=RULE,errors=errors,
        action_total_variation=action_variation,action_sign_reversals=reversals,trace=trace)


def summary_for(episodes):
    result={}
    for profile in PROFILES:
        rows=[e for e in episodes if e['profile']==profile]
        n=sum(e['steps'] for e in rows if e.get('speed_mae_mps') is not None)
        mae=sum(e['speed_mae_mps']*e['steps'] for e in rows if e.get('speed_mae_mps') is not None)
        result[profile]=dict(total=len(rows),arrived=sum(e['confirmed_success'] for e in rows),
            quality=sum(e['quality_pass'] for e in rows),speed_mae_mps=mae/n if n else None,
            brake_steps=sum(e['model_brake_steps'] for e in rows),
            overspeed_steps=sum(e['overspeed_steps'] for e in rows),
            action_total_variation=(sum(e['action_total_variation'] for e in rows)
                if rows and all('action_total_variation' in e for e in rows) else None),
            action_sign_reversals=(sum(e['action_sign_reversals'] for e in rows)
                if rows and all('action_sign_reversals' in e for e in rows) else None),
            failed=[dict(map=e['map'],seed=e['seed'],speed=e['speed_mps'],status=e['status'])
                    for e in rows if not e['quality_pass']])
    return result


def compare(episodes,mode,require_full_suite=True):
    indexed={}
    for ep in episodes:
        key=(ep['profile'],ep['map'],int(ep['seed']),float(ep['speed_mps']))
        if key in indexed:raise ValueError('Duplicate profile/scenario')
        indexed[key]=ep
    groups={p:{(k[1],k[2],k[3]) for k in indexed if k[0]==p} for p in PROFILES}
    complete=bool(groups[RULE] and all(groups[p]==groups[RULE] for p in PROFILES))
    full={('S',s,v) for s in range(186,190) for v in (2.,3.,4.)}|\
         {('C',s,v) for s in range(186,190) for v in (2.,3.,4.)}|\
         {('X',s,v) for s in range(186,190) for v in (2.,3.,4.)}
    full_suite=(mode=='validation' and complete and groups[RULE]==full)
    regress=[]
    for k in sorted(groups[RULE]):
        out=indexed.get((NEW,*k))
        if out is None:regress.append(f'missing:{k}');continue
        for ref in (RULE,RAW):
            b=indexed.get((ref,*k))
            if b is None:regress.append(f'missing:{ref}:{k}');continue
            if b['confirmed_success'] and not out['confirmed_success']:regress.append(f'arrival_vs_{ref}:{k}')
            if b['quality_pass'] and not out['quality_pass']:regress.append(f'quality_vs_{ref}:{k}')
    summary=summary_for(episodes);a,b,q=(summary[p] for p in PROFILES)
    if complete:
        tracking=all(q['speed_mae_mps']<=t['speed_mae_mps']+max(.05,.05*t['speed_mae_mps']) for t in (a,b))
    else:tracking=False
    no_failure=complete and not regress and q['quality']>=max(a['quality'],b['quality']) and q['arrived']>=max(a['arrived'],b['arrived'])
    braking=complete and q['brake_steps']>=10
    accepted=bool(mode=='validation' and complete and (not require_full_suite or full_suite) and
                  no_failure and tracking and braking)
    return dict(accepted=accepted,complete=complete,full_validation_suite=full_suite,
         regressions=regress,speed_tracking_within_tolerance=tracking,braking_observed=braking,
         brake_steps_below_v316=complete and q['brake_steps']<b['brake_steps'],
         criterion='No per-case arrival/quality regression vs RULE or V3.16; MAE <= both refs + max(.05m/s,5%); learned braking >=10; full suite required; final cannot promote')


def load_reference(path,actor_sha,baseline_sha,maps,seeds,speeds,steps):
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    cfg=doc.get('config',{})
    if (doc.get('version')!='AGENT4_V316_SHADOW_IDM_LONGITUDINAL' or
        cfg.get('mode')!='validation' or cfg.get('model_sha256')!=actor_sha or
        cfg.get('baseline_sha256')!=baseline_sha or cfg.get('steps')!=steps or
        cfg.get('traffic_density')!=0.0 or cfg.get('speed_schedule')!=SCHEDULE):
        raise ValueError('Incompatible V3.16 A/B reference')
    wanted={(m,int(s),float(v)) for m in maps for s in seeds for v in speeds}
    found={}
    for ep in doc['episodes']:
        k=(ep['map'],int(ep['seed']),float(ep['speed_mps']))
        if k not in wanted or ep['profile'] not in (RULE,RAW):continue
        key=(ep['profile'],*k)
        if key in found:raise ValueError('Duplicate original reference')
        if (ep['errors'] or ep['steps']!=ep['verified_steps'] or
            ep['teacher_calls'] or ep['idm_policy_used']):raise ValueError('Corrupt original reference')
        found[key]=ep
    expected={(p,*k) for k in wanted for p in (RULE,RAW)}
    if set(found)!=expected:raise ValueError(f'Incomplete reference {len(found)} / {len(expected)}')
    return list(found.values())


def evaluate(args):
    c,feat,geom=dependencies(args.legacy_root)
    orig,splits,digest=c.original_manifest(args.dataset)
    maps=geom.parse_maps(args.maps);speeds=geom.parse_speeds(args.speeds)
    if not set(maps)<=set(orig['config']['maps']) or not set(speeds)<=set(orig['config']['speeds_mps']):
        raise ValueError('Invalid maps/speeds')
    if args.mode=='final' and not args.confirm_frozen:raise ValueError('Final must confirm frozen')
    if args.mode=='final' and args.reference:raise ValueError('Final cannot reuse validation results')
    seeds=allowable_seeds(geom.parse_seed_spec(args.seeds),splits,
                          'val' if args.mode=='validation' else 'final')
    if args.steps<180:raise ValueError('Need >=180 steps')
    baseline=joblib.load(args.baseline);c.check_baseline(baseline,digest,feat)
    actor=joblib.load(args.actor);validate_model(actor,digest,sha(args.baseline),feat)
    candidate=joblib.load(args.candidate)
    validate_v3162(candidate,digest,sha(args.baseline),sha(args.actor),feat)
    cfg=dict(maps=maps,seeds=seeds,speeds_mps=speeds,steps=args.steps,mode=args.mode,
         actor_sha256=sha(args.actor),candidate_sha256=sha(args.candidate),
         baseline_sha256=sha(args.baseline),speed_schedule=SCHEDULE,traffic_density=0.0,
         reference_sha256=sha(args.reference) if args.reference else None)
    path=Path(args.out)
    if path.exists():
        if not args.resume:raise FileExistsError('Output exists. Use --resume')
        doc=json.loads(path.read_text(encoding='utf-8'))
        if doc.get('version')!=VERSION or doc.get('config')!=cfg:raise ValueError('Resume mismatch')
    else:
        doc=dict(version=VERSION,config=cfg,episodes=[],summary={},decision={},
          caveats=['V3.16.2 is trained only from original and DAgger TRAIN data',
            'V3.13 steering frozen; no rule governor/IDM fallback for AI profiles',
            'No traffic, privileged MetaDrive geometry; no real-world safety claim',
            'Previously examined val seeds 186..189 cannot be final holdout',
            'Final seeds >=202 must remain untouched until candidate is frozen'])
        if args.reference:
            doc['episodes'].extend(load_reference(args.reference,sha(args.actor),
                            sha(args.baseline),maps,seeds,speeds,args.steps))
        c.atomic_json(path,doc)
    done={(e['profile'],e['map'],int(e['seed']),float(e['speed_mps'])) for e in doc['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    for m in maps:
        todo=[(p,s,v) for s in seeds for v in speeds for p in PROFILES if (p,m,s,v) not in done]
        if not todo:continue
        env=MetaDriveEnv(dict(map=m,use_render=False,num_scenarios=max(seeds)+1,
            traffic_density=0.0,horizon=max(1000,args.steps+10),log_level=40,
            physics_world_step_size=.02,decision_repeat=5))
        try:
            for profile,s,v in todo:
                ep=run_episode(env,baseline,actor,candidate,c,feat,geom,m,s,v,args.steps,profile)
                doc['episodes'].append(ep)
                doc['summary']=summary_for(doc['episodes'])
                doc['decision']=compare(doc['episodes'],args.mode)
                c.atomic_json(path,doc)
                print(f'{profile} {m} seed={s} speed={v}: {ep["status"]} quality={ep["quality_pass"]} MAE={ep["speed_mae_mps"]:.4f}',flush=True)
        finally:env.close()
    doc['summary']=summary_for(doc['episodes']);doc['decision']=compare(doc['episodes'],args.mode)
    c.atomic_json(path,doc)
    print('Saved',path.resolve(),'accepted',doc['decision']['accepted'])
    return doc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True,help='Frozen original V3.16 model')
    p.add_argument('--candidate',type=Path,required=True,help='Trained V3.16.2 model')
    p.add_argument('--reference',type=Path,help='Verified full V3.16 validation file (validation mode only)')
    p.add_argument('--mode',choices=('validation','final'),default='validation')
    p.add_argument('--confirm-frozen',action='store_true')
    p.add_argument('--maps',default='S,C,X');p.add_argument('--seeds',default='186-189')
    p.add_argument('--speeds',default='2,3,4');p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--out',type=Path,default=Path('agent4_v3162_validation.json'))
    p.add_argument('--resume',action='store_true')
    return evaluate(p.parse_args(argv))

if __name__=='__main__':main()
