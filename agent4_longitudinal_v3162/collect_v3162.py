"""On-policy DAgger collection for V3.16.2: frozen V3.13 steering + frozen V3.16 gas/brake drive.

IDM exists only as shadow teacher for supervision, NEVER as actuating policy.
Collect ONLY TRAIN episodes; existing V3.16 validation remains held out.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import joblib
from common_v3162 import (VERSION,allowable_seeds,dependencies,make_features,
                           pair_action,predict_long,sha,speed_schedule,validate_model,verified_action)
from collect_v316 import META,label_quality,set_shadow_target


def collect_episode(env,m,seed,speed,steps,baseline,actor,feat,geom,c,writer):
    from metadrive.policy.idm_policy import IDMPolicy
    env.reset(seed=seed)
    owner=geom.get_policy(env)
    if isinstance(owner,IDMPolicy):raise RuntimeError('Ego cannot be controlled by IDM')
    Shadow=type('ShadowIDM_V3162',(IDMPolicy,),{'NORMAL_SPEED':float(speed)*3.6})
    shadow=Shadow(control_object=env.agent,random_seed=seed+1017)
    errors=[];status='step_limit';n=checked=valid=brakes=queries=transitions=0;last_target=None
    try:
        for step in range(steps):
            pre=geom.state_before_action(env.agent)
            target=speed_schedule(speed,step)
            record={k:pre.get(k) for k in feat.FEATURES}
            record['target_speed_mps']=target
            make_features(record,feat)
            set_shadow_target(shadow,target)
            try:
                raw=pair_action(shadow.act())[1];queries+=1
                ok,reason=label_quality(pre,raw)
                label=raw if math.isfinite(raw) and abs(raw)<=1.0005 else 0.0
            except Exception as e:
                raw=0.0;label=0.0;ok=False;reason=f'shadow_error:{type(e).__name__}:{e}'
            steer=feat.steering_prediction(baseline,pre)
            longitude=predict_long(actor,record,feat)  # STRICTLY learned V3.16 actor
            term,trunc,info=c.read_step(env.step([steer,longitude]))
            verified_action(env.agent,steer,longitude)
            if geom.get_policy(env) is not owner:raise RuntimeError('Ego policy was replaced')
            record.update(split='train',map=m,seed=seed,base_speed_mps=speed,step=step,
                          teacher_longitudinal=label,raw_teacher_longitudinal=raw,
                          teacher_label_valid=int(ok),label_reason=reason,
                          applied_steering=steer,applied_longitudinal=longitude,teacher_label_only=1)
            writer.writerow(record)
            n+=1;checked+=1;valid+=int(ok)
            brakes+=int(ok and label<-.02)
            transitions+=int(last_target is not None and target!=last_target)
            last_target=target
            if term or trunc:
                status=c.episode_status(term,trunc,info);break
    except Exception as e:
        errors.append(f'{type(e).__name__}: {e}');status='episode_error'
    finally:del shadow
    return dict(split='train',map=m,seed=seed,speed_mps=speed,status=status,
                steps=n,verified_learner_steps=checked,valid_labels=valid,
                teacher_brake_labels=brakes,speed_transitions=transitions,
                shadow_queries=queries,teacher_driving_calls=0,errors=errors)


def collect(args):
    c,feat,geom=dependencies(args.legacy_root)
    orig,splits,digest=c.original_manifest(args.dataset)
    seeds=allowable_seeds(geom.parse_seed_spec(args.train_seeds),splits,'train')
    maps=geom.parse_maps(args.maps);speeds=geom.parse_speeds(args.speeds)
    if not set(maps)<=set(orig['config']['maps']) or not set(speeds)<=set(orig['config']['speeds_mps']):
        raise ValueError('Unexpected maps or speeds')
    if args.steps<180:raise ValueError('Need >=180 steps')
    baseline=joblib.load(args.baseline)
    c.check_baseline(baseline,digest,feat)
    actor=joblib.load(args.actor)
    validate_model(actor,digest,sha(args.baseline),feat)
    cfg=dict(maps=maps,speeds_mps=speeds,train_seeds=seeds,steps=args.steps,
             original_manifest_sha256=digest,baseline_sha256=sha(args.baseline),
             actor_sha256=sha(args.actor),schedule='cruise 65, slow 55, cruise 60 ticks',
             traffic_density=0.0,behavior_policy='frozen V3.16 gas/brake + V3.13 steering')
    root=Path(args.out_dir);index=root/'collection_manifest.json'
    if index.exists():
        if not args.resume:raise FileExistsError('Collection exists; use --resume')
        report=json.loads(index.read_text(encoding='utf-8'))
        if report.get('version')!=VERSION or report.get('config')!=cfg:raise ValueError('Resume config mismatch')
    else:
        if root.exists() and any(root.iterdir()):raise FileExistsError('Directory not empty')
        report=dict(version=VERSION,config=cfg,episodes=[],
                    note='TRAIN ONLY: shadow IDM labels on V3.16-visited states; IDM never drives')
        c.atomic_json(index,report)
    done={(e['map'],int(e['seed']),float(e['speed_mps'])) for e in report['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    for m in maps:
        todo=[(s,v) for s in seeds for v in speeds if (m,s,v) not in done]
        if not todo:continue
        env=MetaDriveEnv(dict(map=m,use_render=False,num_scenarios=max(seeds)+1,
            traffic_density=0.0,horizon=max(1000,args.steps+10),log_level=40,
            physics_world_step_size=.02,decision_repeat=5))
        try:
            for s,v in todo:
                rel=Path('episodes')/'train'/f'{m}_seed{s}_speed{str(v).replace(".","p")}.csv'
                dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_suffix('.partial')
                try:
                    with temp.open('w',encoding='utf-8',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(feat.FEATURES)+list(META));writer.writeheader()
                        record=collect_episode(env,m,s,v,args.steps,baseline,actor,feat,geom,c,writer)
                    temp.replace(dest)
                finally:
                    if temp.exists():temp.unlink()
                record['csv']=rel.as_posix();report['episodes'].append(record)
                c.atomic_json(index,report)
                print(f'C/X/S train {m} seed={s} v={v:g}: {record["status"]} valid={record["valid_labels"]} brake={record["teacher_brake_labels"]}',flush=True)
        finally:env.close()
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True,help='Frozen original V3.16 model')
    p.add_argument('--maps',default='S,C,X');p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--train-seeds',default='178-185');p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--out-dir',type=Path,default=Path('agent4_v3162_dagger_data'))
    p.add_argument('--resume',action='store_true')
    return collect(p.parse_args(argv))

if __name__=='__main__':main()
