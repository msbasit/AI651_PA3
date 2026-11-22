"""Meaningful checks for ordering, dates, masked windows, and graph gradients."""
import json
import numpy as np
import pandas as pd
import torch
from common import ROOT,load_data,save_json
from models import Forecaster,normalize_fixed
from train import Windows,SEED,seed_all

def main():
    torch.set_num_threads(8); seed_all()
    d=load_data(); meta=pd.read_csv(ROOT/'data'/'sd_meta.csv')
    assert np.array_equal(d['ids'],meta.ID.to_numpy())
    limits={'train':('2020-01-01','2021-01-01',35136),'val':('2021-01-01','2021-04-01',8640),
            'test':('2021-04-01','2022-01-01',26400)}
    for name,(start,end,length) in limits.items():
        dates=pd.to_datetime(d[name+'_dates'])
        assert len(dates)==length and d[name].shape==(length,716)
        assert dates[0]==pd.Timestamp(start) and dates[-1]<pd.Timestamp(end)
        assert np.all(np.diff(dates.asi8)==15*60*10**9)
        assert np.isfinite(d[name]).all()
        w=Windows(d[name],d[name+'_mask'],d['mean'],d['std'],'cpu')
        ids=torch.tensor([0,w.size-1])
        x,y,mask=w.batch(ids)
        assert x.shape==(2,8,716,1) and y.shape==(2,4,716,1)
        assert np.allclose((y[-1,:,:,0].numpy()*d['std']+d['mean']),d[name][-4:],atol=.001)
        assert np.array_equal(mask[-1,:,:,0],d[name+'_mask'][-4:])
    assert d['mean'].shape==d['std'].shape==(716,)
    assert np.allclose(d['train'].astype(np.float64).mean(axis=0),d['mean'],atol=.001)
    expected=d['train'].astype(np.float64).std(axis=0)
    expected=np.where(expected<1e-6,1.,expected)
    assert np.allclose(expected,d['std'],atol=.001)
    a=normalize_fixed(torch.tensor(np.load(ROOT/'data'/'sd_adj.npy')))
    assert torch.allclose(a.sum(1),torch.ones(716),atol=1e-6)
    raw=np.load(ROOT/'data'/'sd_adj.npy').copy();np.fill_diagonal(raw,0)
    assert np.allclose(a.diag().numpy(),1/(raw.sum(1)+1),atol=1e-6)
    shared={};initial_entropies={}
    for mode in ['temporal','fixed','adaptive','hybrid']:
        seed_all(); m=Forecaster(mode,a)
        shared[mode]={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('temporal.','pool.','local.','head.'))}
        x=torch.randn(2,8,716,1)
        y=m(x)
        assert y.shape==(2,4,716,1) and torch.isfinite(y).all()
        if mode!='temporal':
            assert torch.allclose(m.adjacency().sum(1),torch.ones(716),atol=1e-6)
        if mode in ['adaptive','hybrid']:
            a0=torch.softmax(torch.relu(m.E1@m.E2.T),dim=1)
            initial_entropies[mode]=float(-(a0*a0.clamp_min(1e-30).log()).sum(1).mean().detach())
            assert initial_entropies[mode]<np.log(716)-.25
            y.square().mean().backward()
            for e in [m.E1,m.E2]:
                assert torch.isfinite(e.grad).all() and e.grad.abs().sum()>0
        if mode!='temporal':
            # With graph weights zeroed, an own-node route must still agree
            # exactly with the shared temporal baseline, independently of A.
            with torch.no_grad(): m.spatial.weight.zero_()
            seed_all(); baseline=Forecaster('temporal',a)
            assert torch.allclose(m(x),baseline(x),atol=1e-6)
    for mode in shared:
        assert all(torch.equal(v,shared['temporal'][k]) for k,v in shared[mode].items())
    save_json('checks.json',dict(sensor_order=True,exact_split_lengths=True,regular_15_min_grid=True,
        split_local_first_last_windows=True,target_masks=True,train_only_stats=True,
        row_stochastic_graphs=True,exactly_one_self_loop=True,local_path_survives_graph_removal=True,
        per_sensor_inverse_scaling=True,adaptive_gradients_nonzero=True,identical_shared_initial_weights=True,
        initial_adaptive_entropy=initial_entropies,
        input_shape=[2,8,716,1],output_shape=[2,4,716,1],seed=SEED))
    print('PASS: ordering, exact dates, split-local windows, target masks, training statistics, normalized graphs, gradients, shared initialization.')

if __name__=='__main__':
    main()
