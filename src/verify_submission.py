"""Verify executed notebooks, model artifacts, metrics, report links and identities."""
import json
import hashlib
import numpy as np
import nbformat
import fitz
import torch
from common import ROOT,save_json

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
    for mode in metrics:
        assert (ROOT/'checkpoints'/f'{mode}.pt').is_file()
        checkpoint=torch.load(ROOT/'checkpoints'/f'{mode}.pt',map_location='cpu',weights_only=True)
        assert checkpoint['config']==config
        assert checkpoint['best_epoch']==metrics[mode]['best_epoch']
        assert all(torch.isfinite(value).all() for value in checkpoint['state_dict'].values())
        assert npu[mode]['windows']==26389
        assert npu[mode]['evaluated_values']==counts[0]
        assert 'NPU' in npu[mode]['execution_devices']
        for split in ['validation','test']:
            assert metrics[mode][split]['rmse']>=metrics[mode][split]['mae']>0
    pdf=ROOT/'AI651_PA3_Muhammad_Sami_Ul_Basit_25280025.pdf'
    doc=fitz.open(pdf)
    first=doc[0].get_text()
    assert '25280025' in first and 'Muhammad Sami Ul Basit' in first
    assert any(x.get('uri')=='https://github.com/msbasit/AI651_PA3' for x in doc[0].get_links())
    text='\n'.join(p.get_text() for p in doc)
    for fragment in ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','AI-use disclosure']:
        assert fragment in text,fragment
    save_json('submission_checks.json',dict(notebooks=notebooks,common_test_metric_count=counts[0],
        all_models_have_checkpoints=True,consistent_training_configs=True,finite_checkpoint_weights=True,
        full_test_npu=True,report_pages=len(doc),first_page_identity_and_public_link=True,
        sha256_report=hashlib.file_digest(pdf.open('rb'),'sha256').hexdigest()))
    print('PASS: all four notebooks executed without errors, checkpoints and common full-test metrics present, NPU execution confirmed, report identity/link verified.')

if __name__=='__main__':
    main()
