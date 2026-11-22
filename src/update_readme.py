"""Refresh the README comparison from completed measurements, never placeholders."""
import json
from common import ROOT
from train import REVISION

def main():
    metrics=json.loads((ROOT/'results'/'metrics.json').read_text())
    baseline=json.loads((ROOT/'results'/'baselines.json').read_text())
    assert len(metrics)==4 and all(r.get('revision')==REVISION for r in metrics.values())
    p=baseline['persistence']['test']
    lines=['## Measured results','',
        '| Model | Validation MAE | Test MAE | Test RMSE | Training seconds | Epochs / selected |',
        '|---|---:|---:|---:|---:|---:|']
    labels={'temporal':'Temporal-only','fixed':'Fixed graph','adaptive':'Adaptive graph','hybrid':'Hybrid graph'}
    for mode,label in labels.items():
        r=metrics[mode]
        lines.append(f'| {label} | {r["validation"]["mae"]:.2f} | {r["test"]["mae"]:.2f} | {r["test"]["rmse"]:.2f} | {r["training_seconds"]:.1f} | {r["epochs_run"]} / {r["best_epoch"]} |')
    lines.append(f'| Persistence | {baseline["persistence"]["validation"]["mae"]:.2f} | {p["mae"]:.2f} | {p["rmse"]:.2f} | 0 (untrained) | -- |')
    best=min(metrics,key=lambda m:metrics[m]['test']['mae'])
    lines.extend(['',f'Errors are vehicles per 15-minute interval. Every method uses the same 26,389 test windows and {p["evaluated_values"]:,} valid sensor-horizon targets. The lowest test MAE among the trained models is {labels[best]} ({metrics[best]["test"]["mae"]:.2f}). Checkpoints are selected by validation only. This is a single-seed comparison, and graph variants have additional parameters.',
        '', 'Training times are synchronized XPU wall times including epoch validation, independently measured and rounded. Exact totals and per-epoch durations are retained in `results/metrics.json` and history files. Full-test NPU deployment metrics are in `results/npu_metrics.json`.',
        '', 'Following independent human review, this revision adds a local path, per-sensor scaling, stronger adaptive initialization, a longer budget, and persistence checks. Private review notes are excluded from the public repository and submission. These changes were made together; the rerun is not an ablation isolating a single cause.',''])
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    start=text.index('## Measured results');end=text.index('## Environment',start)
    (ROOT/'README.md').write_text(text[:start]+'\n'.join(lines)+'\n'+text[end:],encoding='utf-8')
    print('Updated README from completed measurements.')

if __name__=='__main__':main()
