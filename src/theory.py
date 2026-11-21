"""Numerical verification of Q1 and Q2, independent of LargeST."""
import numpy as np
import matplotlib.pyplot as plt
from common import figure, save_json

def main():
    K, base = 8, 9
    triples = np.array([(a,b,c) for a in range(K+1) for b in range(K+1)
                        for c in range(K+1) if 1 <= a+b+c <= K])
    codes = triples @ np.array([1, base, base**2])
    decoded = np.stack([codes%base, codes//base%base, codes//base**2%base], axis=1)
    assert len(np.unique(codes)) == len(triples) == 164
    assert np.array_equal(decoded, triples)
    # Exhaustively check Yes rules over every legal count triple.
    mean = triples/triples.sum(1, keepdims=True)
    maximum = (triples>0).astype(int)
    assert np.array_equal(mean[:,0]>.5, triples[:,0] > triples.sum(1)/2)
    assert np.array_equal(maximum[:,0]>0, triples[:,0]>0)
    x = np.exp(-(np.arange(40)-20)**2/8)
    true, symmetric = [x.copy()], [x.copy()]
    for _ in range(20):
        true.append(np.roll(true[-1], -1))
        symmetric.append(.5*(np.roll(symmetric[-1], 1)+np.roll(symmetric[-1], -1)))
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax, data, title in zip(axes,[true,symmetric],['True backward motion','Equal-neighbor layer: a=0, b=1/2']):
        im=ax.imshow(data,origin='upper',aspect='auto',vmin=0,vmax=1,extent=[-.5,39.5,20.5,-.5],cmap='magma')
        ax.set(xlabel='Sensor index',ylabel='Time step',title=title)
    fig.colorbar(im,ax=axes,label='Slowdown')
    figure('q2_rollouts.png')
    rng=np.random.default_rng(25280025)
    errors=[]
    for direction in [-1,1]:
        current=rng.normal(size=40)
        previous=np.roll(current,-direction)
        predicted=np.roll(current,1)+np.roll(current,-1)-previous
        errors.append(float(np.max(np.abs(predicted-np.roll(current,direction)))))
    assert max(errors)<1e-14
    save_json('theory_checks.json',dict(K=K,base=base,triples=len(triples),unique_codes=len(np.unique(codes)),
                                     exact_decode=True,history_max_errors=errors,
                                     symmetric_single_impulse_sse=.5))
    print('Verified 164 encodings and exact bidirectional history rule; saved Q2 heatmaps.')

if __name__=='__main__':
    main()
