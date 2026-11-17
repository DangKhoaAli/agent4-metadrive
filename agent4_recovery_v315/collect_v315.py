"""V3.15 DAgger-style on-policy state collection with *shadow* IDM steering labels.

The only action sent to env.step is V3.13 + deterministic bounded perturbations.
The shadow teacher is NOT installed as env.agent_policy and never actuates.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import random

import joblib
from common_v315 import (VERSION, STEP_SECONDS, allowed_train_seeds, atomic_json,
                         bounded, check_baseline, digest, episode_status,
                         governor, legacy_modules, original_manifest, read_step)

FIELDS = ('map','seed','target_speed_mps','step','teacher_steer_raw','teacher_steer',
          'base_steer','sent_steer','applied_steer','applied_longitudinal','injected_noise',
          'label_valid','label_reason','margin_proxy_m','rollout_status','learner_policy_name')


def noise_at_step(seed, step, pre, amplitude):
    """Deterministic two-sided micro-pulses; no random test/validation contamination."""
    if amplitude <= 0 or step < 95 or step % 95 >= 6:
        return 0.0
    if abs(float(pre['planned_lateral_right_m'])) > 0.55:
        return 0.0
    margin = pre.get('edge_clearance_proxy_m')
    if margin is not None and float(margin) < 0.40:
        return 0.0
    sign = 1.0 if (seed + step // 95) % 2 else -1.0
    return float(sign * amplitude)


def trainable(pre, expert, raw, steer, reasons):
    if not math.isfinite(raw) or abs(raw) > 1.0 + 5e-4:
        reasons.append('expert_raw_out_of_range')
    if not (math.isfinite(expert) and math.isfinite(steer)):
        reasons.append('nonfinite_action')
    if abs(float(pre['planned_lateral_right_m'])) > 1.15:
        reasons.append('outside_recovery_envelope')
    if abs(float(pre['heading_error_rad'])) > .22:
        reasons.append('heading_unrecoverable')
    if not pre['lookahead_5m_available'] or not pre['lookahead_10m_available']:
        reasons.append('required_lookahead_missing')
    margin = pre.get('edge_clearance_proxy_m')
    if margin is None or float(margin) < .15:
        reasons.append('not_enough_edge_margin')
    return not reasons


def episode(env, m, seed, target, steps, amplitude, baseline, features, geometry, writer):
    from metadrive.policy.idm_policy import IDMPolicy
    env.reset(seed=int(seed))
    actual_policy = geometry.get_policy(env)
    if isinstance(actual_policy, IDMPolicy):
        raise RuntimeError('Refusing to collect: the learner would be controlled by IDM')
    policy_name = type(actual_policy).__name__
    # Instantiate an independent expert with the SAME vehicle state, but do not register it
    # with the engine. This is a label oracle, not the vehicle-driving policy.
    speed_kmh = float(target) * 3.6
    ShadowIDM = type('ShadowIDM_V315', (IDMPolicy,), {'NORMAL_SPEED': speed_kmh})
    shadow = ShadowIDM(control_object=env.agent, random_seed=int(seed) + 1017)
    count = valid = invalid = impulses = checked = 0
    status = 'step_limit'; errors = []; maxlat = 0.; minmargin = None; near = 0; teacher_calls=0
    try:
        for step in range(steps):
            pre = geometry.state_before_action(env.agent)
            base = features.steering_prediction(baseline, pre)
            # Shadow teacher only reads current state and updates its private PID state.
            try:
                action = shadow.act()
                teacher_calls += 1
                raw = float(action[0])
                if not math.isfinite(raw): raise ValueError('nonfinite teacher action')
                teacher = bounded(raw)
                teacher_issue = ''
            except Exception as exc:
                raw = 0.; teacher = 0.; teacher_issue = f'expert_query:{type(exc).__name__}'
            reasons = [teacher_issue] if teacher_issue else []
            sample_ok = trainable(pre, teacher, raw, base, reasons)
            noise = noise_at_step(int(seed), step, pre, amplitude)
            sent = bounded(base + noise)
            throttle = governor(float(pre['speed_mps']), float(target))
            terminated, truncated, info = read_step(env.step([sent, throttle]))
            applied = tuple(float(x) for x in env.agent.current_action)
            if (len(applied) != 2 or any(not math.isfinite(x) for x in applied) or
                abs(applied[0] - sent) > 5e-4 or abs(applied[1] - throttle) > 5e-4):
                raise RuntimeError(f'Learner action not applied: model={sent, throttle}, vehicle={applied}')
            checked += 1
            if geometry.get_policy(env) is not actual_policy:
                raise RuntimeError('Environment replaced the learner policy')
            margin = pre.get('edge_clearance_proxy_m')
            if margin is not None:
                margin = float(margin)
                minmargin = min(margin, minmargin) if minmargin is not None else margin
                near += int(margin < 0.35)
            maxlat = max(maxlat, abs(float(pre['planned_lateral_right_m'])))
            impulses += int(abs(noise) > 0)
            row = {k: pre.get(k) for k in features.FEATURES}
            row.update(map=m, seed=int(seed), target_speed_mps=float(target), step=step,
                       teacher_steer_raw=raw, teacher_steer=teacher, base_steer=base,
                       sent_steer=sent, applied_steer=applied[0], applied_longitudinal=applied[1],
                       injected_noise=noise, label_valid=int(sample_ok),
                       label_reason=';'.join(reasons), margin_proxy_m=margin,
                       rollout_status='pretermination', learner_policy_name=policy_name)
            writer.writerow(row)
            count += 1; valid += int(sample_ok); invalid += int(not sample_ok)
            if terminated or truncated:
                status = episode_status(terminated, truncated, info)
                break
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        status = 'episode_error'
    finally:
        del shadow
    return dict(map=m,seed=int(seed),speed_mps=float(target),status=status,steps=count,
                valid_teacher_rows=valid,invalid_teacher_rows=invalid,perturbed_steps=impulses,
                learner_steps_verified=checked,shadow_teacher_queries=teacher_calls,
                teacher_driving_calls=0,learner_policy=policy_name,
                max_abs_lateral_m=maxlat,min_edge_clearance_proxy_m=minmargin,
                near_edge_steps=near,errors=errors)


def collect(args):
    features, geometry = legacy_modules(args.legacy_root)
    ds, splits, manifest_hash = original_manifest(args.dataset)
    seeds = allowed_train_seeds(geometry.parse_seed_spec(args.seeds), splits)
    maps = geometry.parse_maps(args.maps)
    speeds = geometry.parse_speeds(args.speeds)
    if any(m not in ds['config']['maps'] for m in maps) or any(v not in ds['config']['speeds_mps'] for v in speeds):
        raise ValueError('Only maps and speeds from original V3.12B dataset are supported')
    if not 0 <= args.noise <= .15 or args.steps < 100:
        raise ValueError('Require noise in [0,.15] and at least 100 steps')
    baseline = joblib.load(args.baseline)  # ONLY load user-trusted model files.
    check_baseline(baseline, manifest_hash, features)
    config = dict(maps=maps,seeds=seeds,speeds_mps=speeds,steps=args.steps,
                  noise=args.noise,baseline_sha256=digest(args.baseline),
                  original_manifest_sha256=manifest_hash,traffic_density=0.0)
    out = Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    manifest_path = out/'collection_manifest.json'
    if manifest_path.exists():
        report = json.loads(manifest_path.read_text(encoding='utf-8'))
        if report.get('config') != config or not args.resume:
            raise ValueError('Existing collection: supply --resume with identical config or a new --out-dir')
    else:
        report = dict(version=VERSION,config=config,episodes=[],
                      warning='Dataset for TRAIN seeds only. Shadow IDM labels are queried but NOT actuated.')
    seen = {(e['map'], e['seed'], e['speed_mps']) for e in report['episodes']}
    from metadrive.envs.metadrive_env import MetaDriveEnv
    for m in maps:
        todo = [(s,v) for s in seeds for v in speeds if (m,s,v) not in seen]
        if not todo:continue
        env = MetaDriveEnv(dict(map=m,use_render=False,num_scenarios=max(seeds)+1,
                                traffic_density=0.0,horizon=max(args.steps+10,1000),
                                log_level=40,physics_world_step_size=.02,decision_repeat=5))
        try:
            for seed,target in todo:
                name = f'{m}_seed{seed}_speed{str(target).replace(".","p")}.csv'
                path = out/'episodes'/name; path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('w',encoding='utf-8',newline='') as f:
                    writer = csv.DictWriter(f,fieldnames=list(features.FEATURES)+list(FIELDS))
                    writer.writeheader()
                    record = episode(env,m,seed,target,args.steps,args.noise,baseline,features,geometry,writer)
                record['csv'] = str(Path('episodes')/name).replace('\\','/')
                report['episodes'].append(record)
                atomic_json(manifest_path,report)
                print(f'{m} seed={seed} speed={target:g} status={record["status"]} '
                      f'valid={record["valid_teacher_rows"]}/{record["steps"]} '
                      f'perturbed={record["perturbed_steps"]}',flush=True)
        finally:
            env.close()
    print('Collection complete:',manifest_path.resolve())
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,default=Path('..'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--maps',default='C,X')
    p.add_argument('--seeds',default='170-185')
    p.add_argument('--speeds',default='2,3,4')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--noise',type=float,default=.08)
    p.add_argument('--out-dir',type=Path,default=Path('../agent4_v315_data'))
    p.add_argument('--resume',action='store_true')
    return collect(p.parse_args(argv))

if __name__ == '__main__': main()
