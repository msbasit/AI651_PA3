"""Persistence sanity checks on exactly the learned models' splits and masks."""
import numpy as np
import torch
from torch import nn
from common import ROOT,load_data,save_json
from models import normalize_fixed
from train import Windows,evaluate,REVISION

class Persistence(nn.Module):
    def forward(self,x):
        return x[:,-1:].expand(-1,4,-1,-1)

class GraphPersistence(nn.Module):
    def __init__(self,a,mean,std):
        super().__init__()
        self.register_buffer('a',a)
        self.register_buffer('mean',torch.tensor(mean,device=a.device))
        self.register_buffer('std',torch.tensor(std,device=a.device))
    def forward(self,x):
        raw=x[:,-1,:,0]*self.std+self.mean
        averaged=torch.matmul(self.a,raw.unsqueeze(-1)).squeeze(-1)
        return ((averaged-self.mean)/self.std)[:,None,:,None].expand(-1,4,-1,-1)

def main():
    torch.set_num_threads(8)
    assert torch.xpu.is_available()
    d=load_data();device='xpu'
    fixed=normalize_fixed(torch.tensor(np.load(ROOT/'data'/'sd_adj.npy'),device=device))
    methods={'persistence':Persistence(),'fixed_graph_persistence':GraphPersistence(fixed,d['mean'],d['std'])}
    results={}
    for name,model in methods.items():
        results[name]={'revision':REVISION,'training_seconds':0.}
        for split in ['val','test']:
            windows=Windows(d[split],d[split+'_mask'],d['mean'],d['std'],device)
            results[name]['validation' if split=='val' else 'test']=evaluate(model,windows,d['std'],256)
        print(name,results[name],flush=True)
    save_json('baselines.json',results)

if __name__=='__main__':main()
