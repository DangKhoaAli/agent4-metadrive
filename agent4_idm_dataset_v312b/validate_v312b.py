"""Validate an exported V3.12B dataset offline without MetaDrive.

Checks all per-episode audit/clean CSVs, split leakage, step indexing, label
provenance, sample-level filtering, summary counts and manifest aggregates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math

from collect_v312b import VERSION, FIELDS, clip, stable_hash, refresh, TOL


def check_dataset(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['version']!=VERSION:raise ValueError('Wrong manifest version')
    cfg=manifest['config']
    if stable_hash(cfg)!=manifest['config_sha256']:raise ValueError('Config fingerprint mismatch')
    split_seeds=cfg['split_seeds']
    a=[(seed,split) for split,seeds in split_seeds.items() for seed in seeds]
    if len({seed for seed,_ in a})!=len(a):raise ValueError('Split leakage: seed shared')
    seen=set();summary=[]
    for episode in manifest['episodes']:
        ident=(episode['split'],episode['map'],episode['seed'],float(episode['speed_mps']))
        if ident in seen:raise ValueError(f'Duplicate episode {ident}')
        seen.add(ident)
        split,m,seed,speed=ident
        if seed not in split_seeds[split] or m not in cfg['maps'] or speed not in cfg['speeds_mps']:
            raise ValueError(f'Unrecognized episode config {ident}')
        audit=root/episode['audit_file']
        if not audit.is_file():raise FileNotFoundError(audit)
        audit_count=0;clean_count=0;spikes=0;expected_clean=[]
        with audit.open('r',encoding='utf-8',newline='') as fd:
            reader=csv.DictReader(fd)
            if reader.fieldnames!=FIELDS:raise ValueError(f'Audit header mismatch in {audit}')
            for row in reader:
                idx=audit_count;audit_count+=1
                if row['step']!=str(idx):raise ValueError(f'Misaligned step at {audit}#{idx}')
                if row['split']!=split or row['map']!=m or int(row['seed'])!=seed:
                    raise ValueError(f'Episode metadata mismatch at {audit}#{idx}')
                raw=[float(row[k]) for k in ('raw_steering','raw_throttle_brake')]
                applied=[float(row[k]) for k in ('applied_steering','applied_throttle_brake')]
                if not all(math.isfinite(x) for x in raw+applied):raise ValueError('Nonfinite label')
                if max(abs(clip(raw[i])-applied[i]) for i in (0,1))>TOL:
                    raise ValueError(f'Mislabeled applied action at {audit}#{idx}')
                if row['current_action_verified']!='1' or row['label_valid']!='1':
                    raise ValueError('Missing applied action provenance')
                spike=any(abs(x)>1+TOL for x in raw)
                spikes+=spike
                if int(row['raw_out_of_range'])!=int(spike):raise ValueError('Wrong spike label')
                if row['trainable_row']=='1':
                    if spike or row['excluded_reason']:raise ValueError('Invalid trainable row')
                    expected_clean.append(row)
        if audit_count!=episode['steps'] or spikes!=episode['raw_out_of_range_steps']:
            raise ValueError(f'Episode counts disagree: {audit}')
        if episode['quality_pass']:
            cpath=root/episode['clean_file']
            if not cpath.is_file():raise FileNotFoundError(cpath)
            with cpath.open('r',encoding='utf-8',newline='') as fd:
                r=csv.DictReader(fd)
                if r.fieldnames!=FIELDS:raise ValueError('Clean header mismatch')
                for clean_count,row in enumerate(r,1):
                    if row!=expected_clean[clean_count-1]:
                        raise ValueError(f'Clean CSV differs from audit filtering: {cpath}')
            if clean_count!=len(expected_clean):raise ValueError('Missing clean data rows')
        else:
            if episode.get('clean_file'):raise ValueError('Rejected episode has clean file')
        if clean_count!=episode['actual_trainable_steps']:
            raise ValueError('Manifest clean count mismatch')
        summary.append({'map':m,'split':split,'seed':seed,'speed':speed,'quality':episode['quality_pass'],
                        'audit':audit_count,'clean':clean_count,'spikes':spikes})
    old=manifest['totals'].copy()
    if refresh(manifest)['totals']!=old:raise ValueError('Manifest aggregate totals mismatch')
    return {'episodes':len(summary),'quality_accepted':sum(s['quality'] for s in summary),
            'audit_steps':sum(s['audit'] for s in summary),
            'clean_steps':sum(s['clean'] for s in summary),
            'raw_overrange_steps':sum(s['spikes'] for s in summary),
            'by_split':{split:sum(s['clean'] for s in summary if s['split']==split)
                        for split in ('train','val','test')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    args=parser.parse_args()
    report=check_dataset(args.dataset)
    print('V3.12B OFFLINE DATASET VALIDATION: PASS')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
