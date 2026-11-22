"""Saved-checkpoint predictions, adjacency diagnostics, and synchronized XPU latency."""
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common import ROOT,load_data,save_json,figure
from models import Forecaster,normalize_fixed

def haversine(meta):
    lat=np.radians(meta.Lat.to_numpy()); lon=np.radians(meta.Lng.to_numpy())
    dlat=lat[:,None]-lat[None,:]; dlon=lon[:,None]-lon[None,:]
    h=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 6371*2*np.arcsin(np.sqrt(np.clip(h,0,1)))

def graph_stats(a,road,dist):
    a=a.copy(); np.fill_diagonal(a,-np.inf)
    top=np.argsort(a,axis=1)[:,-10:]
    i=np.arange(len(a))[:,None]
    return dict(top10_road_overlap=float((road[i,top]>0).mean()),
                top10_mean_distance_km=float(dist[i,top].mean()),
                top10_median_distance_km=float(np.median(dist[i,top])),
                top10_under_5km_fraction=float((dist[i,top]<5).mean()),
                top10_edges=int(top.size))

@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    data=load_data(); mu=data['mean']; std=data['std']
    meta=pd.read_csv(ROOT/'data'/'sd_meta.csv')
    prep=json.loads((ROOT/'results'/'preparation.json').read_text())
    metrics=json.loads((ROOT/'results'/'metrics.json').read_text())
    A=np.load(ROOT/'data'/'sd_adj.npy'); road=A.copy(); np.fill_diagonal(road,0)
    fixed=normalize_fixed(torch.tensor(A,device='xpu'))
    dates=pd.to_datetime(data['test_dates'])
    # Predetermined dates and graph-order sensors; no error-based cherry picking.
    ids=np.flatnonzero((dates[:-11]>=pd.Timestamp('2021-04-12')) & (dates[:-11]<pd.Timestamp('2021-04-14')))
    target_dates=dates[ids+8]
    x=np.stack([data['test'][i:i+8] for i in ids])[...,None]
    x=torch.tensor((x-mu.reshape(1,1,-1,1))/std.reshape(1,1,-1,1),device='xpu')
    mean_tensor=torch.tensor(mu,device='xpu');std_tensor=torch.tensor(std,device='xpu')
    target=data['test'][ids+8]
    valid=data['test_mask'][ids+8]
    colors={'temporal':'#2878b5','fixed':'#d65f00','adaptive':'#8e44ad','hybrid':'#00866d'}
    predictions={}
    graph_results={}
    latency={}
    dist=haversine(meta)
    off=~np.eye(716,dtype=bool)
    graph_results['reference']=dict(road_nonself_density=float((road>0).sum()/off.sum()),
        all_pairs_mean_distance_km=float(dist[off].mean()),
        all_pairs_under_5km_fraction=float((dist[off]<5).mean()),
        road_edges_mean_distance_km=float(dist[road>0].mean()))
    learned={}
    for mode in colors:
        model=Forecaster(mode,fixed).to('xpu').eval()
        ckpt=torch.load(ROOT/'checkpoints'/f'{mode}.pt',map_location='xpu',weights_only=True)
        assert np.array_equal(np.asarray(ckpt['config']['mean'],np.float32),mu)
        assert np.array_equal(np.asarray(ckpt['config']['std'],np.float32),std)
        model.load_state_dict(ckpt['state_dict'])
        predictions[mode]=torch.cat([model(x[j:j+32])[:,0,:,0]*std_tensor+mean_tensor for j in range(0,len(x),32)]).cpu().numpy()
        b=x[:16]
        for _ in range(10):
            model(b)
        torch.xpu.synchronize()
        samples=[]
        for _ in range(30):
            t=time.perf_counter(); model(b); torch.xpu.synchronize(); samples.append((time.perf_counter()-t)*1000)
        latency[mode]=dict(batch=16,median_ms=float(np.median(samples)),samples_ms=samples,
                           protocol='10 warmups, 30 synchronized forwards, resident XPU input, excludes transfer')
        if mode in ['adaptive','hybrid']:
            a=torch.softmax(torch.relu(model.E1@model.E2.T),dim=1).cpu().numpy()
            learned[mode]=a
            np.save(ROOT/'results'/f'{mode}_learned_component.npy',a)
            graph_results[mode]=graph_stats(a,road,dist)
            entropy=-(a*np.log(np.clip(a,1e-30,1))).sum(1)
            graph_results[mode]['mean_row_entropy']=float(entropy.mean())
            graph_results[mode]['uniform_entropy']=float(np.log(716))
            graph_results[mode]['mean_self_weight']=float(np.diag(a).mean())
            graph_results[mode]['embedding_std']=[float(model.E1.std().cpu()),float(model.E2.std().cpu())]
            graph_results[mode]['max_weight']=float(a.max())
            graph_results[mode]['max_abs_departure_from_uniform']=float(np.abs(a-1/len(a)).max())
    save_json('graph_analysis.json',graph_results)
    save_json('gpu_latency.json',latency)
    np.savez_compressed(ROOT/'results'/'plot_predictions.npz',dates=target_dates.to_numpy().astype('datetime64[ns]').astype(np.int64),
                        target=target,mask=valid,**predictions)
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
    for ax,node in zip(axes,prep['selected_plot_nodes']):
        truth=target[:,node].copy(); truth[~valid[:,node]]=np.nan
        ax.plot(target_dates,truth,color='black',lw=1.5,label='Actual')
        for mode,c in colors.items():
            ax.plot(target_dates,predictions[mode][:,node],lw=.9,alpha=.85,color=c,label=mode.title())
        ax.set(title=f'Sensor {int(meta.ID.iloc[node])}: 15-minute-ahead forecast',ylabel='Vehicles / 15 min')
    axes[0].legend(ncol=5,fontsize=8)
    figure('q6_predictions.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,(mode,a) in zip(axes,[('Fixed (row normalized)',fixed.cpu().numpy()),('Adaptive',learned['adaptive']),('Hybrid',np.load(ROOT/'results'/'hybrid_adj.npy'))]):
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
        im=ax.imshow(np.ma.masked_less_equal(a,0),cmap=cmap,norm=LogNorm(vmin=1e-5,vmax=1),aspect='auto')
        fig.colorbar(im,ax=ax,fraction=.046,ticks=[1e-5,1e-3,1e-1,1])
        ax.set(title=mode,xlabel='Source node j',ylabel='Receiving node i')
    figure('q5_adjacencies.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for mode,c in colors.items():
        history=json.loads((ROOT/'results'/f'{mode}_history.json').read_text())
        axes[0].plot([h['epoch'] for h in history],[h['val_mae'] for h in history],label=mode,color=c)
        axes[1].plot([15,30,45,60],metrics[mode]['test']['horizon_mae'],marker='o',color=c,label=mode)
    axes[0].set(xlabel='Epoch',ylabel='Validation MAE',title='Checkpoint selection');axes[0].legend()
    axes[1].set(xlabel='Forecast horizon (minutes)',ylabel='Test MAE',title='Error versus forecast horizon')
    figure('q6_learning.png')
    # Spatial plot: strongest non-self learned edge per node, deterministic 60-node subsample.
    a=learned['adaptive'].copy(); np.fill_diagonal(a,-np.inf)
    top=a.argmax(1)
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    ax.scatter(meta.Lng,meta.Lat,s=4,color='#888888',alpha=.5)
    for i in np.linspace(0,715,60,dtype=int):
        j=top[i]; color='#00866d' if road[i,j]>0 else '#bd4557'
        ax.plot([meta.Lng.iloc[i],meta.Lng.iloc[j]],[meta.Lat.iloc[i],meta.Lat.iloc[j]],color=color,alpha=.5,lw=.7)
    ax.set(xlabel='Longitude',ylabel='Latitude',title='Strongest adaptive links for 60 evenly spaced nodes\nGreen: road edge exists; red: absent from road graph')
    figure('q6_learned_map.png')
    print(json.dumps(graph_results,indent=2))

if __name__=='__main__':
    main()
