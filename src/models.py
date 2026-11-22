"""The same lightweight temporal encoder, projection, and head for all four models."""
import torch
from torch import nn

class Forecaster(nn.Module):
    def __init__(self, mode, fixed, hidden=16, embedding=10, alpha=.5):
        super().__init__()
        self.mode=mode
        self.alpha=alpha
        self.temporal=nn.Sequential(nn.Conv2d(1,hidden,(1,3),padding=(0,1)),nn.ReLU(),
                                    nn.Conv2d(hidden,hidden,(1,3),padding=(0,1)),nn.ReLU())
        self.pool=nn.Linear(hidden*8,hidden)
        # Initialize all shared components before graph-only parameters, keeping
        # their initial weights identical for the temporal and graph models.
        self.local=nn.Linear(hidden,hidden,bias=False)
        self.head=nn.Linear(hidden,4)
        self.spatial=nn.Linear(hidden,hidden,bias=False) if mode!='temporal' else None
        self.register_buffer('fixed',fixed.clone())
        if mode in ['adaptive','hybrid']:
            self.E1=nn.Parameter(torch.randn(len(fixed),embedding))
            self.E2=nn.Parameter(torch.randn(len(fixed),embedding))
        self.register_buffer('frozen_graph',None)

    def adjacency(self):
        if self.frozen_graph is not None:
            return self.frozen_graph
        if self.mode=='fixed':
            return self.fixed
        if self.mode in ['adaptive','hybrid']:
            adaptive=torch.softmax(torch.relu(self.E1 @ self.E2.T),dim=1)
            return adaptive if self.mode=='adaptive' else self.alpha*self.fixed+(1-self.alpha)*adaptive
        return None

    def forward(self,x):
        # Input B,8,N,1; conv kernels never mix distinct sensors.
        z=self.temporal(x.permute(0,3,2,1))
        z=self.pool(z.permute(0,2,1,3).flatten(2))
        a=self.adjacency()
        local=self.local(z)
        if a is not None:
            local=local+self.spatial(torch.matmul(a,z))
        h=torch.relu(local)
        return self.head(h).permute(0,2,1).unsqueeze(-1)

def normalize_fixed(a):
    # Directed row aggregation: row i collects feature j with weight A[i,j].
    # The provided matrix already contains self-loops. Add exactly one unit
    # self-loop after clearing its diagonal, then row-normalize the directed graph.
    a=a.clone()
    a.fill_diagonal_(0)
    a=a+torch.eye(len(a),dtype=a.dtype,device=a.device)
    return a/a.sum(dim=1,keepdim=True).clamp_min(1e-12)
