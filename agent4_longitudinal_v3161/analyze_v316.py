"""Summarize V3.16 validation regressions from saved JSON. No MetaDrive required."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import defaultdict

BASE='rule_governor_v313'
RAW='learned_longitudinal_v316'


def analyze(doc):
    eps=doc['episodes']
    pair=defaultdict(dict)
    for e in eps:
        key=(e['map'],int(e['seed']),float(e['speed_mps']))
        if e['profile'] in pair[key]:
            raise ValueError(f'Duplicate scenario/profile: {key}, {e["profile"]}')
        pair[key][e['profile']]=e
    if any(set(v)!={BASE,RAW} for v in pair.values()):
        raise ValueError('Incomplete V3.16 pair comparison')
    rows=[]
    for (m,s,v),p in sorted(pair.items()):
        b,n=p[BASE],p[RAW]
        rows.append(dict(map=m,seed=s,speed_mps=v,
            baseline_quality=b['quality_pass'],learned_quality=n['quality_pass'],
            mae_rule_mps=b['speed_mae_mps'],mae_learned_mps=n['speed_mae_mps'],
            brake_rule=b['model_brake_steps'],brake_learned=n['model_brake_steps'],
            near_rule=b['near_edge_warning_steps'],near_learned=n['near_edge_warning_steps'],
            maxlat_rule_m=b['max_abs_lateral_m'],maxlat_learned_m=n['max_abs_lateral_m']))
    regress=[r for r in rows if r['baseline_quality'] and not r['learned_quality']]
    improvements=[r for r in rows if not r['baseline_quality'] and r['learned_quality']]
    return dict(n_pairs=len(rows),regressions=regress,improvements=improvements,
                total_brakes_rule=sum(r['brake_rule'] for r in rows),
                total_brakes_learned=sum(r['brake_learned'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('agent4_v316_validation.json'))
    args=p.parse_args()
    d=json.loads(args.input.read_text(encoding='utf8'))
    if d.get('version')!='AGENT4_V316_SHADOW_IDM_LONGITUDINAL':
        raise ValueError('Expected V3.16 validation JSON')
    print(json.dumps(analyze(d),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
