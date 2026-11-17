"""Agent 4 V3.14: read-only audit of V3.13 steering closed-loop JSON.

No MetaDrive dependency, no model tuning, no training. Never assumes a road
clearance proxy is a safety certificate; conflicting proxies are flagged.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

VERSION='AGENT4_V314_READ_ONLY_DIAGNOSTIC'


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        doc=json.load(f)
    if not isinstance(doc, dict) or doc.get('version')!='AGENT4_V313_IDM_IMITATION_STEERING':
        raise ValueError('Expected V3.13 JSON')
    return doc


def finite_number(value, label):
    try:
        n=float(value)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError(f'Invalid {label}: {value!r}') from exc
    if not math.isfinite(n):
        raise ValueError(f'Non-finite {label}')
    return n


def trace_findings(trace):
    """Only flags possible measurement discontinuities, never diagnoses cause."""
    flags=[]
    last=None
    for current in trace:
        step=int(current['step'])
        lat=finite_number(current['lateral_m'], 'trace lateral')
        if last is not None:
            last_step,last_lat,last_circ=last
            if step<=last_step:raise ValueError('Trace steps not increasing')
            delta=lat-last_lat
            # Sampling can hide motion: report observations, not a definitive jump.
            if abs(delta)>=1.0 and lat*last_lat<0:
                flags.append(dict(kind='lateral_sign_reversal',from_step=last_step,to_step=step,
                                  from_lateral_m=last_lat,to_lateral_m=lat,delta_m=delta,
                                  note='May reflect lane-reference switch or real large lateral motion; inspect full-step geometry'))
        last=(step,lat,bool(current.get('is_circular_lane')))
    return flags


def review_episode(ep):
    mapname=str(ep.get('map'))
    seed=int(ep['seed'])
    speed=finite_number(ep['speed_mps'],'speed_mps')
    status=ep.get('status')
    success=bool(ep.get('confirmed_success'))
    quality=bool(ep.get('quality_pass'))
    lat=finite_number(ep['max_abs_planned_lateral_m'],'max_abs_planned_lateral_m')
    heading=finite_number(ep['max_abs_heading_error_rad'],'max_abs_heading_error_rad')
    clearance=finite_number(ep['min_edge_clearance_proxy_m'],'min_edge_clearance_proxy_m')
    warning=int(ep.get('near_edge_warning_steps',0))
    n=int(ep['n_applied_steps']);verified=int(ep['n_verified_prediction_steps'])
    notes=[]
    if status=='out_of_road':notes.append('road_departure')
    if status!='arrive_dest' and status!='out_of_road':notes.append('non_arrival')
    if not ep.get('model_predictions_valid') or n!=verified or ep.get('teacher_fallback_calls') or ep.get('idm_policy_used'):
        notes.append('model_provenance_issue')
    if quality and not success:notes.append('inconsistent_success_quality')
    if success and not quality:notes.append('arrived_but_quality_failed')
    if lat>.6:notes.append('lateral_threshold_exceeded')
    if heading>.12:notes.append('heading_threshold_exceeded')
    if clearance<.3 or warning:notes.append('near_edge_or_low_clearance')
    if lat>.6 and clearance>=.3 and warning==0:
        notes.append('lateral_vs_clearance_proxy_disagreement')
    flags=trace_findings(ep.get('trace') or [])
    if flags:notes.append('potential_lateral_reference_discontinuity')
    return dict(map=mapname,seed=seed,speed_mps=speed,status=status,
        confirmed_success=success,quality_pass=quality,model_predictions_valid=bool(ep.get('model_predictions_valid')),
        n_applied_steps=n,n_verified_prediction_steps=verified,
        max_abs_planned_lateral_m=lat,max_abs_heading_error_rad=heading,
        min_edge_clearance_proxy_m=clearance,near_edge_warning_steps=warning,
        has_circular_lane=bool(ep.get('has_circular_lane')),
        suspected_findings=notes,trace_flags=flags)


def summarize(episodes):
    agg={}
    for field in ('map','speed_mps'):
        groups=defaultdict(list)
        for ep in episodes:groups[str(ep[field])].append(ep)
        agg['by_'+('map' if field=='map' else 'speed')]=[
            dict(key=k,total=len(v),arrived=sum(int(e['confirmed_success']) for e in v),
                 quality_accepted=sum(int(e['quality_pass']) for e in v),
                 out_of_road=sum(e['status']=='out_of_road' for e in v))
            for k,v in sorted(groups.items())]
    agg['total']=len(episodes)
    agg['arrived']=sum(e['confirmed_success'] for e in episodes)
    agg['quality_accepted']=sum(e['quality_pass'] for e in episodes)
    agg['out_of_road']=sum(e['status']=='out_of_road' for e in episodes)
    agg['verified_prediction_steps']=sum(e['n_verified_prediction_steps'] for e in episodes)
    agg['action_steps']=sum(e['n_applied_steps'] for e in episodes)
    agg['near_edge_warning_steps']=sum(e['near_edge_warning_steps'] for e in episodes)
    agg['finding_counts']=dict(Counter(note for ep in episodes for note in ep['suspected_findings']))
    return agg


def analyze(test_path, metrics_path=None, val_path=None):
    test=read_json(test_path)
    eps=test.get('episodes',[])
    if not eps:raise ValueError('No V3.13 episodes')
    identities=[(ep['map'],int(ep['seed']),float(ep['speed_mps'])) for ep in eps]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate closed-loop episodes')
    checked=[review_episode(ep) for ep in eps]
    model_sha=test.get('config',{}).get('model_sha256')
    report=dict(version=VERSION,source=str(test_path),model_sha256=model_sha,
        summary=summarize(checked),episodes=checked,attention=[e for e in checked if not e['quality_pass']],
        notes=['Only source-derived observations; diagnostic flags are hypotheses, not verified root causes.',
               'A passing arrival is not equal to satisfying quality thresholds.',
               'Clearance is a proxy; lateral and clearance may use different reference lanes.',
               'Do not tune models against previously inspected test seeds; keep a new blind holdout.'])
    if metrics_path is not None:
        metrics=read_json(metrics_path)
        report['offline']=dict(selected_model=metrics.get('selected_model'),
            n_samples=metrics.get('n_samples'),
            validation_balanced_mae=metrics.get('selection_score'),
            test_mae=metrics.get('selected_model_test',{}).get('overall',{}).get('mae'),
            test_rmse=metrics.get('selected_model_test',{}).get('overall',{}).get('rmse'))
    if val_path is not None:
        val=read_json(val_path)
        if val.get('config',{}).get('model_sha256')!=model_sha:
            raise ValueError('Validation/test model hashes differ')
        v=[review_episode(ep) for ep in val.get('episodes',[])]
        report['validation_smoke']=summarize(v)
        report['validation_attention']=[e for e in v if not e['quality_pass']]
    return report


def emit(report, output):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    outcsv=output.with_suffix('.csv')
    cols=['map','seed','speed_mps','status','confirmed_success','quality_pass','n_applied_steps',
          'max_abs_planned_lateral_m','max_abs_heading_error_rad','min_edge_clearance_proxy_m',
          'near_edge_warning_steps','suspected_findings']
    with outcsv.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for ep in report['episodes']:
            row={k:ep.get(k) for k in cols}
            row['suspected_findings']=';'.join(ep['suspected_findings'])
            w.writerow(row)
    return outcsv


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test',type=Path,required=True)
    parser.add_argument('--metrics',type=Path)
    parser.add_argument('--val',type=Path)
    parser.add_argument('--out',type=Path,default=Path('agent4_v314_diagnostic.json'))
    args=parser.parse_args(argv)
    report=analyze(args.test,args.metrics,args.val)
    csvpath=emit(report,args.out)
    print('Arrived:',report['summary']['arrived'],'/',report['summary']['total'])
    print('Quality:',report['summary']['quality_accepted'],'/',report['summary']['total'])
    print('Action verification:',report['summary']['verified_prediction_steps'],'/',report['summary']['action_steps'])
    print('Attention:',[(e['map'],e['seed'],e['speed_mps'],e['suspected_findings']) for e in report['attention']])
    print('Saved:',args.out,csvpath)

if __name__=='__main__':main()
