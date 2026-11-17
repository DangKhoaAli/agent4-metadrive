"""Export a compact, public-safe summary. Never publishes model weights or step traces."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from common_v315 import atomic_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reports/v315_validation_summary.json'))
    a=p.parse_args(argv)
    report=json.loads(a.evaluation.read_text(encoding='utf-8'))
    if report['config']['mode']!='validation':
        raise ValueError('Publishing final holdout summaries requires a separate explicit review')
    compact=dict(version=report['version'],mode='validation',
                 maps=report['config']['maps'],speeds_mps=report['config']['speeds_mps'],
                 seed_count=len(report['config']['seeds']),
                 run_count=len(report['episodes']),summary=report['summary'],promotion=report['promotion'],
                 warnings=['No raw traces, model weights or private local file paths included',
                           'Simulator only; throttle is rule-based'])
    atomic_json(a.output,compact)
    print('Publishable summary:',a.output.resolve())

if __name__=='__main__':main()
