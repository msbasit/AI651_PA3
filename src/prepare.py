"""Select District 11 by ID, sum 5-minute counts, split before any imputation."""
import hashlib
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from common import ROOT, figure, save_json

def main():
    meta=pd.read_csv(ROOT/'data'/'ca_meta.csv')
    sd=meta.loc[meta.District.eq(11)].reset_index(drop=True)
    assert len(sd)==716 and sd.ID.is_unique and sd.ID2.is_unique
    sd.to_csv(ROOT/'data'/'sd_meta.csv',index=False)
    adj=np.load(ROOT/'data'/'ca_rn_adj.npy',mmap_mode='r')
    assert adj.shape==(8600,8600)
    A=np.asarray(adj[np.ix_(sd.ID2,sd.ID2)],dtype=np.float32)
    assert np.isfinite(A).all() and A.min()>=0
    np.save(ROOT/'data'/'sd_adj.npy', A)
    frames=[]
    raw_stats={}
    for year in [2020,2021]:
        with h5py.File(ROOT/'data'/f'ca_his_raw_{year}.h5','r') as h:
            columns=[x.decode() for x in h['t/axis0'][:]]
            assert columns == meta.ID.astype(str).tolist(), 'Full metadata/traffic order mismatch'
            idx=[columns.index(str(i)) for i in sd.ID]
            assert idx == sorted(idx)
            # h5py reads the selected columns without allocating the full CA matrix.
            values=h['t/block0_values'][:,idx].astype(np.float32)
            dates=pd.to_datetime(h['t/axis1'][:])
        expected=pd.date_range(f'{year}-01-01', f'{year+1}-01-01',freq='5min',inclusive='left')
        df=pd.DataFrame(values,index=dates,columns=sd.ID.astype(str))
        assert df.index.is_unique and df.index.is_monotonic_increasing
        df=df.reindex(expected)
        bad=~np.isfinite(df.to_numpy()) | (df.to_numpy()<0)
        raw_stats[str(year)]=dict(rows=len(df), missing_or_negative=int(bad.sum()),
                                zero_readings=int((df.to_numpy()==0).sum()), total=int(df.size))
        df=df.mask(bad)
        # Count units: three 5-minute vehicle counts give vehicles per 15-minute bin.
        # A partial bin is not a complete count; retain its missingness for masking.
        agg=df.resample('15min',label='left',closed='left').sum(min_count=3)
        frames.append(agg)
    full=pd.concat(frames)
    parts={'train':full.loc['2020'], 'val':full.loc['2021-01-01':'2021-03-31'],
           'test':full.loc['2021-04-01':'2021-12-31']}
    fallback=parts['train'].median(axis=0).fillna(0)
    train_filled=parts['train'].ffill().fillna(fallback)
    mu=float(train_filled.to_numpy(dtype=np.float64).mean())
    std=float(train_filled.to_numpy(dtype=np.float64).std())
    assert std>0
    payload={'mean':np.float32(mu),'std':np.float32(std),'ids':sd.ID.to_numpy()}
    stats={}
    for name,df in parts.items():
        mask=np.isfinite(df.to_numpy())
        filled=df.ffill().fillna(fallback).to_numpy(dtype=np.float32)
        assert np.isfinite(filled).all()
        payload[name]=filled
        payload[name+'_mask']=mask
        payload[name+'_dates']=df.index.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        stats[name]=dict(timestamps=len(df),windows=len(df)-11,start=str(df.index[0]),end=str(df.index[-1]),
                         missing_bins=int((~mask).sum()),valid_target_fraction=float(mask.mean()))
    np.savez_compressed(ROOT/'data'/'sd_prepared.npz',**payload)
    np.save(ROOT/'data'/'training_fallback.npy',fallback.to_numpy(np.float32))
    selected=[0, len(sd)//2, len(sd)-1]
    fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True,layout='constrained')
    week=full.loc['2020-02-03':'2020-02-09']
    for ax,i in zip(axes,selected):
        ax.plot(week.index,week.iloc[:,i],lw=.8)
        ax.set(ylabel='Vehicles / 15 min',title=f'Sensor {sd.ID.iloc[i]} (node {i})')
    figure('q3_week.png')
    off=A.copy(); np.fill_diagonal(off,0)
    binary=off>0
    degree=binary.sum(1)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    im=axes[0].imshow(A,cmap='viridis',aspect='auto'); fig.colorbar(im,ax=axes[0],label='Road weight')
    axes[0].set(xlabel='Source node j',ylabel='Receiving row i',title='Provided SD adjacency')
    axes[1].hist(degree,bins=25,color='#007f86',edgecolor='white')
    axes[1].set(xlabel='Non-self row neighbors',ylabel='Number of sensors',title='Degree distribution')
    figure('q3_graph.png')
    # Select node using graph only, without target-dependent selection.
    node=int(np.argmin(np.abs(degree-np.median(degree))))
    one=set(np.flatnonzero(binary[node]).tolist())
    two=one.union(*[set(np.flatnonzero(binary[j]).tolist()) for j in one]) if one else set()
    two.discard(node)
    save_json('preparation.json',dict(nodes=716, raw=raw_stats,splits=stats,mean=mu,std=std,
        aggregation='sum of three 5-minute counts; vehicles per 15 minutes',
        selected_plot_nodes=selected,selected_plot_ids=sd.ID.iloc[selected].tolist(),
        nonzero_including_diagonal=int(np.count_nonzero(A)),self_loops=int(np.count_nonzero(np.diag(A))),
        directed_nonself_edges=int(binary.sum()),symmetric=bool(np.allclose(A,A.T)),
        max_asymmetry=float(np.abs(A-A.T).max()),degree_min=int(degree.min()),degree_max=int(degree.max()),
        degree_mean=float(degree.mean()),neighborhood_node=node,neighborhood_id=int(sd.ID.iloc[node]),
        one_hop_nodes=sorted(one),one_hop_ids=sd.ID.iloc[sorted(one)].tolist(),
        at_most_two_hop_nodes=sorted(two),at_most_two_hop_ids=sd.ID.iloc[sorted(two)].tolist(),
        ordering_check='All 8600 metadata IDs equal raw HDF axis0 IDs; adjacency sliced with SD ID2 in same order'))
    print((ROOT/'results'/'preparation.json').read_text())

if __name__=='__main__':
    main()
