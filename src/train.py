"""Complete split-local windows, masked original-unit MAE, validation-only selection."""
import argparse
import copy
import importlib.metadata as metadata
import json
import platform
import random
import sys
import time
import numpy as np
import torch
from common import ROOT,load_data,save_json
from models import Forecaster,normalize_fixed

SEED=25280025
REVISION='review-2026-11-22-v2'

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.xpu.manual_seed_all(SEED)

class Windows:
    def __init__(self,data,mask,mean,std,device):
        self.x=torch.as_tensor((data-mean)/std,device=device,dtype=torch.float32)
        self.mask=torch.as_tensor(mask,device=device,dtype=torch.bool)
        self.size=len(data)-11
        self.history=torch.arange(8,device=device)
        self.future=torch.arange(8,12,device=device)
        self.device=device

    def batch(self,ids):
        x=self.x[ids[:,None]+self.history[None,:]].unsqueeze(-1)
        y=self.x[ids[:,None]+self.future[None,:]].unsqueeze(-1)
        mask=self.mask[ids[:,None]+self.future[None,:]].unsqueeze(-1)
        return x,y,mask

@torch.inference_mode()
def evaluate(model,windows,std,batch=64):
    model.eval()
    scale=torch.as_tensor(std,device=windows.device,dtype=torch.float32).reshape(1,1,-1,1)
    # Arc has no native FP64: reduce FP32 batches on XPU, accumulate FP64 on CPU.
    ae=np.zeros(4,dtype=np.float64)
    se=np.zeros_like(ae); count=np.zeros_like(ae)
    for start in range(0,windows.size,batch):
        ids=torch.arange(start,min(start+batch,windows.size),device=windows.device)
        x,y,m=windows.batch(ids)
        err=(model(x)-y)*scale
        ae+=(err.abs()*m).sum((0,2,3)).cpu().numpy()
        se+=(err.square()*m).sum((0,2,3)).cpu().numpy()
        count+=m.sum((0,2,3)).cpu().numpy()
    return dict(mae=float(ae.sum()/count.sum()),rmse=float(np.sqrt(se.sum()/count.sum())),
                horizon_mae=(ae/count).tolist(),horizon_rmse=np.sqrt(se/count).tolist(),
                evaluated_values=int(count.sum()))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--modes',nargs='+',default=['temporal','fixed','adaptive','hybrid'])
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args()
    assert torch.xpu.is_available(),'Intel XPU is required for these experiments'
    torch.set_num_threads(8)
    device=torch.device('xpu')
    data=load_data(); mean=data['mean']; std=data['std']; loss_scale=float(data['loss_scale'])
    sets={name:Windows(data[name],data[name+'_mask'],mean,std,device) for name in ['train','val','test']}
    fixed=normalize_fixed(torch.tensor(np.load(ROOT/'data'/'sd_adj.npy'),device=device))
    config=dict(revision=REVISION,seed=SEED,hidden=16,embedding=10,embedding_init_std=1.,alpha=.5,batch=args.batch,max_epochs=args.epochs,
                patience=args.patience,optimizer='Adam',lr=.002,weight_decay=.0001,clip_norm=5.,
                loss='masked original-unit MAE divided by a single training-only global scale',selection='minimum full validation MAE',
                precision='FP32',fixed_normalization='clear supplied diagonal, add one unit self-loop, row-normalize',
                adaptive='dense row-softmax(ReLU(E1 @ E2.T)); no top-k sparsification',
                architecture='ReLU(local(Z) + spatial(A @ Z)); temporal-only omits graph term',
                normalization='per-sensor training-only',mean=mean.tolist(),std=std.tolist(),loss_scale=loss_scale)
    if not args.evaluate_only:
        save_json('config.json',config)
    env=dict(python=sys.version,python_executable=sys.executable,platform=platform.platform(),
             cpu=platform.processor(),threads=torch.get_num_threads(),gpu=torch.xpu.get_device_name(0),
             torch=torch.__version__,packages={k:metadata.version(k) for k in ['numpy','pandas','h5py','matplotlib','openvino','nbformat','reportlab']},
             timing='XPU synchronized wall time of training epochs including validation; excludes data loading, final evaluation and first warmup',
             deterministic_note='Fixed seeds and batch order; bitwise XPU determinism is not guaranteed.')
    save_json('environment.json',env)
    result_path=ROOT/'results'/'metrics.json'
    results=json.loads(result_path.read_text()) if result_path.exists() else {}
    results={k:v for k,v in results.items() if v.get('revision')==REVISION}
    scale=torch.tensor(std,device=device).reshape(1,1,-1,1)
    for mode in args.modes:
        seed_all()
        model=Forecaster(mode,fixed).to(device)
        parameters=sum(p.numel() for p in model.parameters())
        initial_entropy=None
        if mode in ['adaptive','hybrid']:
            with torch.no_grad():
                initial_a=torch.softmax(torch.relu(model.E1@model.E2.T),dim=1)
                initial_entropy=float(-(initial_a*initial_a.clamp_min(1e-30).log()).sum(1).mean())
        if args.evaluate_only:
            ckpt=torch.load(ROOT/'checkpoints'/f'{mode}.pt',map_location=device,weights_only=True)
            model.load_state_dict(ckpt['state_dict'])
            print(mode,evaluate(model,sets['test'],std,args.batch),flush=True)
            continue
        opt=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=.0001)
        # Exclude one untimed forward warmup, with no optimizer update.
        with torch.no_grad():
            model(sets['train'].batch(torch.arange(args.batch,device=device))[0])
        torch.xpu.synchronize()
        start=time.perf_counter()
        history=[]; best=float('inf'); stale=0; best_epoch=0; best_state=None
        for epoch in range(1,args.epochs+1):
            epoch_start=time.perf_counter()
            model.train()
            # Independent generator keeps batch order identical across graph modes.
            order=torch.randperm(sets['train'].size,generator=torch.Generator().manual_seed(SEED+epoch)).to(device)
            total=torch.zeros(2,device=device)
            for j in range(0,len(order),args.batch):
                x,y,mask=sets['train'].batch(order[j:j+args.batch])
                opt.zero_grad(set_to_none=True)
                err=((model(x)-y)*scale).abs()*mask
                loss=err.sum()/mask.sum().clamp_min(1)/loss_scale
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step()
                total[0]+=err.detach().sum(); total[1]+=mask.sum()
            validation=evaluate(model,sets['val'],std,args.batch)
            torch.xpu.synchronize()
            record=dict(epoch=epoch,train_mae=float(total[0]/total[1]),val_mae=validation['mae'],
                        val_rmse=validation['rmse'],seconds=time.perf_counter()-epoch_start)
            history.append(record)
            if validation['mae'] < best:
                best=validation['mae']; best_epoch=epoch; stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale+=1
            print(mode,record,flush=True)
            save_json(f'{mode}_history.json',history)
            if stale>=args.patience:
                break
        torch.xpu.synchronize()
        elapsed=time.perf_counter()-start
        model.load_state_dict(best_state)
        torch.save(dict(state_dict=best_state,mode=mode,config=config,best_epoch=best_epoch),ROOT/'checkpoints'/f'{mode}.pt')
        validation=evaluate(model,sets['val'],std,args.batch)
        # First test evaluation occurs only after checkpoint selection has ended.
        test=evaluate(model,sets['test'],std,args.batch)
        graph=model.adjacency()
        if graph is not None:
            np.save(ROOT/'results'/f'{mode}_adj.npy',graph.detach().cpu().numpy())
        torch.xpu.synchronize()
        results[mode]=dict(revision=REVISION,parameters=parameters,epochs_run=len(history),best_epoch=best_epoch,
                           initial_adaptive_entropy=initial_entropy,training_seconds=elapsed,validation=validation,test=test)
        save_json('metrics.json',results)
        print('FINAL',mode,results[mode],flush=True)
        del model,opt
        torch.xpu.empty_cache()

if __name__=='__main__':
    main()
