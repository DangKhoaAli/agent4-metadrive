"""Collect V3.16 longitudinal training/validation labels from shadow MetaDrive IDM.

The steering V3.13 and the RULE speed governor drive env.step. IDM is a
separate label-only policy (NOT installed in the environment). Labels are
counterfactual teacher outputs: do not claim they were applied to the vehicle.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import joblib
from common_v316 import (VERSION, allowable_seeds, dependencies, make_features,
                         pair_action, sha, speed_schedule, verified_action)

META=('split','map','seed','base_speed_mps','step','target_speed_mps',
      'teacher_longitudinal','raw_teacher_longitudinal','teacher_label_valid',
      'label_reason','applied_steering','applied_longitudinal','teacher_label_only')


def label_quality(pre, raw):
    reason=[]
    if not math.isfinite(raw) or abs(raw)>1.+5e-4:reason.append('teacher_action_unbounded')
    if abs(float(pre['planned_lateral_right_m']))>1.15:reason.append('lateral_far')
    if abs(float(pre['heading_error_rad']))>.22:reason.append('heading_far')
    if int(float(pre['lookahead_5m_available']))!=1 or int(float(pre['lookahead_10m_available']))!=1:
        reason.append('no_required_lookahead')
    margin=pre.get('edge_clearance_proxy_m')
    if margin is None or not math.isfinite(float(margin)) or float(margin)<.15:
        reason.append('low_clearance')
    return not reason,';'.join(reason)


def set_shadow_target(policy, desired_mps):
    # Verified against V3.6 native IDM speed-schedule collection pattern.
    kmh=float(desired_mps)*3.6
    if not math.isfinite(kmh) or not .36<=kmh<=43.2:
        raise ValueError('Invalid desired speed for shadow IDM')
    policy.NORMAL_SPEED=kmh
    policy.CREEP_SPEED=min(kmh,5.0)
    policy.target_speed=kmh


def collect_episode(env, m, split, seed, speed, steps, baseline, feat, geom, c, writer):
    from metadrive.policy.idm_policy import IDMPolicy
    env.reset(seed=seed)
    owner=geom.get_policy(env)
    if isinstance(owner,IDMPolicy):raise RuntimeError('Refusing IDM-controlled student rollout')
    Shadow=type('ShadowIDM_V316',(IDMPolicy,),{'NORMAL_SPEED':float(speed)*3.6})
    shadow=Shadow(control_object=env.agent,random_seed=seed+1017)
    errors=[];status='step_limit';n=valid=brakes=transitions=verified=queries=0
    last_target=None
    try:
        for step in range(steps):
            pre=geom.state_before_action(env.agent)
            target=speed_schedule(speed,step)
            record={k:pre.get(k) for k in feat.FEATURES}
            record['target_speed_mps']=target
            make_features(record,feat) # fail closed on input schema before simulator advances
            set_shadow_target(shadow,target)
            try:
                raw_pair=pair_action(shadow.act());queries+=1
                raw_long=raw_pair[1]
                label=raw_long if abs(raw_long)<=1.+5e-4 else 0.
                ok,reason=label_quality(pre,raw_long)
            except Exception as e:
                raw_long=0.;label=0.;ok=False;reason=f'shadow_error:{type(e).__name__}:{e}'
            # The learner-only vehicle action is based on frozen V3.13 steering;
            # the rule speed governor is used ONLY as the collector's behavior policy.
            steer=feat.steering_prediction(baseline,pre)
            throttle=c.governor(float(pre['speed_mps']),target)
            term,trunc,info=c.read_step(env.step([steer,throttle]))
            verified_action(env.agent,steer,throttle)
            if geom.get_policy(env) is not owner:raise RuntimeError('Student ego policy was replaced')
            record.update(split=split,map=m,seed=seed,base_speed_mps=speed,step=step,
                          teacher_longitudinal=label,raw_teacher_longitudinal=raw_long,
                          teacher_label_valid=int(ok),label_reason=reason,
                          applied_steering=steer,applied_longitudinal=throttle,
                          teacher_label_only=1)
            writer.writerow(record)
            n+=1;verified+=1;valid+=int(ok)
            brakes+=int(ok and label<-.02)
            transitions+=int(last_target is not None and target!=last_target)
            last_target=target
            if term or trunc:
                status=c.episode_status(term,trunc,info);break
    except Exception as e:
        errors.append(f'{type(e).__name__}: {e}');status='episode_error'
    finally:del shadow
    return dict(split=split,map=m,seed=seed,speed_mps=speed,status=status,
                steps=n,verified_learner_steps=verified,valid_labels=valid,
                teacher_brake_labels=brakes,speed_transitions=transitions,
                shadow_queries=queries,teacher_driving_calls=0,errors=errors)


def collect(args):
    c,feat,geom=dependencies(args.legacy_root)
    manifest,splits,digest=c.original_manifest(args.dataset)
    chosen={'train':allowable_seeds(geom.parse_seed_spec(args.train_seeds),splits,'train'),
            'val':allowable_seeds(geom.parse_seed_spec(args.val_seeds),splits,'val')}
    if set(chosen['train'])&set(chosen['val']):raise ValueError('Train/val seeds overlap')
    maps=geom.parse_maps(args.maps);speeds=geom.parse_speeds(args.speeds)
    if not set(maps)<=set(manifest['config']['maps']) or not set(speeds)<=set(manifest['config']['speeds_mps']):
        raise ValueError('Maps/speeds must match the original V3.12B coverage')
    if args.steps<180:raise ValueError('Need at least 180 steps for a full speed-change cycle')
    baseline=joblib.load(args.baseline) # trusted local file only
    c.check_baseline(baseline,digest,feat)
    config=dict(maps=maps,speeds_mps=speeds,train_seeds=chosen['train'],val_seeds=chosen['val'],
                steps=args.steps,original_manifest_sha256=digest,baseline_sha256=sha(args.baseline),
                schedule='cruise 65, slow 55, cruise 60 ticks',traffic_density=0.0)
    root=Path(args.out_dir);index=root/'collection_manifest.json'
    if index.exists():
        if not args.resume:raise FileExistsError('Existing collection. Use --resume or new directory')
        report=json.loads(index.read_text(encoding='utf-8'))
        if report.get('version')!=VERSION or report.get('config')!=config:
            raise ValueError('Resume config mismatch')
    else:
        if root.exists() and any(root.iterdir()):raise FileExistsError('Output directory not empty')
        report=dict(version=VERSION,config=config,episodes=[],
            note='Shadow IDM supplies labels only. Student V3.13 + RULE governor drive the data-collection vehicle.')
        c.atomic_json(index,report)
    completed={(r['split'],r['map'],r['seed'],r['speed_mps']) for r in report['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    for m in maps:
        todo=[(sp,s,v) for sp in ('train','val') for s in chosen[sp] for v in speeds
              if (sp,m,s,v) not in completed]
        if not todo:continue
        env=MetaDriveEnv(dict(map=m,use_render=False,num_scenarios=max(chosen['train']+chosen['val'])+1,
                              traffic_density=0.,horizon=max(1000,args.steps+10),
                              log_level=40,physics_world_step_size=.02,decision_repeat=5))
        try:
            for sp,s,v in todo:
                rel=Path('episodes')/sp/f'{m}_seed{s}_speed{str(v).replace(".","p")}.csv'
                dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_suffix('.partial')
                try:
                    with temp.open('w',encoding='utf-8',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(feat.FEATURES)+list(META));writer.writeheader()
                        row=collect_episode(env,m,sp,s,v,args.steps,baseline,feat,geom,c,writer)
                    temp.replace(dest)
                finally:
                    if temp.exists():temp.unlink()
                row['csv']=rel.as_posix()
                report['episodes'].append(row)
                c.atomic_json(index,report)
                print(f'{sp} {m} seed={s} v={v:g}: {row["status"]} '
                      f'labels={row["valid_labels"]}/{row["steps"]} brakes={row["teacher_brake_labels"]}',flush=True)
        finally:env.close()
    print('Saved:',index.resolve())
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--maps',default='S,C,X')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--train-seeds',default='170-177')
    p.add_argument('--val-seeds',default='186-189')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v316_data'))
    p.add_argument('--resume',action='store_true')
    return collect(p.parse_args(argv))

if __name__=='__main__': main()
