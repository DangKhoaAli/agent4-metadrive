"""Re-score an EXISTING V3.11B JSON with corrected V3.10 attribution checks.

No MetaDrive, PDD model, or simulator rollouts; never modifies original by default.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from curve_eval_v311b import VERSION, keys, enrich_episode, refresh, atomic_json, export_csv


def rescore(src: Path, dst: Path, overwrite: bool = False):
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        raise FileExistsError(f'{dst} already exists; choose another path or --overwrite')
    report=json.loads(src.read_text(encoding='utf-8'))
    if report.get('version') != VERSION:
        raise ValueError(f'Unexpected report version: {report.get("version")!r}; expected {VERSION}')
    eps=report.get('episodes')
    if not isinstance(eps, list) or not eps:
        raise ValueError('Expected non-empty episodes list')
    episode_keys=[keys(e) for e in eps]
    if len(episode_keys)!=len(set(episode_keys)):
        raise ValueError('Cannot rescore report with duplicate map/seed/speed/profile keys')
    changed=[]
    for ep in eps:
        old=(ep['summary'].get('model_predictions_valid'),ep['summary'].get('confirmed_success'))
        enrich_episode(ep)
        new=(ep['summary']['model_predictions_valid'],ep['summary']['confirmed_success'])
        if old != new:
            changed.append({'map':ep['map'],'seed':ep['seed'], 'speed_mps':ep['speed_mps'],
                            'profile':ep['profile'],'old':old,'new':new})
    refresh(report)
    report.setdefault('warnings', []).append(
        'V3.11B scores recalculated offline: the older scorer required missing '
        'measured_steps_aligned from V3.10. No episodes rerun and no model changed.')
    atomic_json(dst,report)
    export_csv(report,dst.with_suffix('.csv'))
    print(f'Rescored {len(eps)} saved episodes; changed flags on {len(changed)} episodes:')
    for c in changed:
        print(f"  {c['map']} seed={c['seed']} v={c['speed_mps']:g} "
              f"{c['profile']}: {c['old']} -> {c['new']}")
    print(f'JSON: {dst.resolve()}')
    print(f'CSV:  {dst.with_suffix(".csv").resolve()}')
    return report,changed


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args=parser.parse_args(argv)
    return rescore(args.input,args.out,args.overwrite)


if __name__=='__main__':main()
