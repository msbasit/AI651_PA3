"""Verify executed notebooks, model artifacts, metrics, report links and identities."""
import json
import hashlib
import numpy as np
import nbformat
import fitz
import torch
from common import ROOT,save_json
from train import REVISION

def main():
    notebooks={}
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4)
        cells=[c for c in nb.cells if c.cell_type=='code']
        assert cells and all(c.execution_count is not None for c in cells),path
        assert all(o.output_type!='error' for c in cells for o in c.outputs),path
        notebooks[path.name]=dict(code_cells=len(cells),all_executed=True,no_errors=True)
    assert len(notebooks)==4
    metrics=json.loads((ROOT/'results'/'metrics.json').read_text())
    npu=json.loads((ROOT/'results'/'npu_metrics.json').read_text())
    assert set(metrics)==set(npu)=={'temporal','fixed','adaptive','hybrid'}
    counts=[metrics[m]['test']['evaluated_values'] for m in metrics]
    assert len(set(counts))==1
    config=json.loads((ROOT/'results'/'config.json').read_text())
    baselines=json.loads((ROOT/'results'/'baselines.json').read_text())
    assert config['revision']==REVISION and config['max_epochs']==40 and config['patience']==5
    assert config['embedding']==10 and config['embedding_init_std']==1
    assert config['embedding_weight_decay']==0.
    graphs=json.loads((ROOT/'results'/'graph_analysis.json').read_text())
    for mode in ['adaptive','hybrid']:
        assert min(graphs[mode]['embedding_std'])>.05,'Graph embeddings collapsed.'
        assert graphs[mode]['max_abs_departure_from_uniform']>.001,'Learned graph is effectively uniform.'
    assert len(config['mean'])==len(config['std'])==716
    prep=json.loads((ROOT/'results'/'preparation.json').read_text())
    assert np.allclose(config['mean'],prep['mean'],rtol=1e-6,atol=1e-4)
    assert np.allclose(config['std'],prep['std'],rtol=1e-6,atol=1e-4)
    assert all(v['test']['evaluated_values']==counts[0] and v['revision']==REVISION for v in baselines.values())
    graph_below_persistence={m:metrics[m]['test']['mae']<baselines['persistence']['test']['mae'] for m in ['fixed','adaptive','hybrid']}
    assert all(graph_below_persistence.values()),'Inspect a graph model that fails the persistence sanity check.'
    for mode in metrics:
        assert (ROOT/'checkpoints'/f'{mode}.pt').is_file()
        checkpoint=torch.load(ROOT/'checkpoints'/f'{mode}.pt',map_location='cpu',weights_only=True)
        assert checkpoint['config']==config
        assert metrics[mode]['revision']==npu[mode]['revision']==REVISION
        assert checkpoint['best_epoch']==metrics[mode]['best_epoch']
        assert all(torch.isfinite(value).all() for value in checkpoint['state_dict'].values())
        assert npu[mode]['windows']==26389
        assert npu[mode]['evaluated_values']==counts[0]
        assert 'NPU' in npu[mode]['execution_devices']
        assert abs(npu[mode]['mae']-metrics[mode]['test']['mae'])<.05,'NPU metrics disagree with the selected GPU checkpoint.'
        for split in ['validation','test']:
            assert metrics[mode][split]['rmse']>=metrics[mode][split]['mae']>0
    pdf=ROOT/'AI651_PA3_Muhammad_Sami_Ul_Basit_25280025.pdf'
    doc=fitz.open(pdf)
    first=doc[0].get_text()
    assert '25280025' in first and 'Muhammad Sami Ul Basit' in first
    assert any(x.get('uri')=='https://github.com/msbasit/AI651_PA3' for x in doc[0].get_links())
    text='\n'.join(p.get_text() for p in doc)
    for fragment in ['What does a sum remember','Which way is the slowdown moving',
                     'Dataset preparation','Spatio-temporal forecasting','Fixed, adaptive',
                     'Experimental analysis','Reproducibility','AI Acknowledgement']:
        assert fragment in text,fragment
    tex=(ROOT/'report'/'report.tex').read_text(encoding='utf-8')
    assert all('sec:q'+str(i) in tex for i in range(1,8))
    assert r'\begin{equation}' in tex and r'\begin{align}' in tex
    assert 'No LMS upload was performed' not in text
    assert 'independently review the derivations' not in text
    assert 'Persistence' in text and 'Human contribution and independent review' in text
    assert 'issues.md' not in text and 'ISSUE_RESOLUTION.md' not in text
    ignored=(ROOT/'.gitignore').read_text().splitlines()
    assert 'issues.md' in ignored and 'ISSUE_RESOLUTION.md' in ignored
    save_json('submission_checks.json',dict(notebooks=notebooks,common_test_metric_count=counts[0],
        all_models_have_checkpoints=True,consistent_training_configs=True,finite_checkpoint_weights=True,
        full_test_npu=True,report_pages=len(doc),native_latex_source=True,first_page_identity_and_public_link=True,
        review_revision=REVISION,persistence_same_test_masks=True,
        graph_models_below_persistence=graph_below_persistence,
        private_review_notes_ignored=True,
        sha256_report=hashlib.file_digest(pdf.open('rb'),'sha256').hexdigest()))
    print('PASS: all four notebooks executed without errors, checkpoints and common full-test metrics present, NPU execution confirmed, report identity/link verified.')

if __name__=='__main__':
    main()
