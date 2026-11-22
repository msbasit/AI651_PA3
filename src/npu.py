"""Static OpenVINO export and explicit Intel NPU inference; no AUTO/CPU fallback."""
import argparse
import json
import time
import numpy as np
import torch
import openvino as ov
from common import ROOT,load_data,save_json
from models import Forecaster,normalize_fixed

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--batch',type=int,default=1)
    args=parser.parse_args()
    torch.set_num_threads(8)
    core=ov.Core()
    print('OpenVINO devices:',core.available_devices,flush=True)
    assert 'NPU' in core.available_devices
    print('NPU:',core.get_property('NPU','FULL_DEVICE_NAME'),flush=True)
    records={}
    modes=['fixed'] if args.smoke else ['temporal','fixed','adaptive','hybrid']
    if args.smoke:
        torch.manual_seed(1)
        fixed=normalize_fixed(torch.rand(716,716))
        data=None
    else:
        fixed=normalize_fixed(torch.tensor(np.load(ROOT/'data'/'sd_adj.npy')))
        data=load_data()
    for mode in modes:
        m=Forecaster(mode,fixed).eval()
        if not args.smoke:
            checkpoint=torch.load(ROOT/'checkpoints'/f'{mode}.pt',map_location='cpu',weights_only=True)
            assert np.array_equal(np.asarray(checkpoint['config']['mean'],np.float32),data['mean'])
            assert np.array_equal(np.asarray(checkpoint['config']['std'],np.float32),data['std'])
            m.load_state_dict(checkpoint['state_dict'])
        graph=m.adjacency()
        if graph is not None:
            m.frozen_graph=graph.detach().clone()
        for p in m.parameters():
            p.requires_grad_(False)
        example=torch.zeros(args.batch,8,716,1)
        start=time.perf_counter()
        converted=ov.convert_model(m,example_input=example,input=[list(example.shape)])
        compiled=core.compile_model(converted,'NPU',{'PERFORMANCE_HINT':'LATENCY'})
        compile_seconds=time.perf_counter()-start
        print('Compiled',mode,'in',compile_seconds,'seconds; execution devices:',compiled.get_property('EXECUTION_DEVICES'),flush=True)
        req=compiled.create_infer_request()
        if args.smoke:
            x=np.random.default_rng(1).normal(size=tuple(example.shape)).astype(np.float32)
            pred=req.infer({0:x})[compiled.output(0)].copy()
            with torch.no_grad():
                reference=m(torch.from_numpy(x)).numpy()
            print('NPU output:',pred.shape,'max discrepancy:',float(np.max(np.abs(pred-reference))),flush=True)
            return
        mean=data['mean']; std=data['std']
        output_mean=mean.reshape(1,1,-1,1);output_std=std.reshape(1,1,-1,1)
        xdata=(data['test']-mean)/std
        mask=data['test_mask']
        n=len(xdata)-11
        warm=np.stack([xdata[i:i+8] for i in range(args.batch)])[...,None].astype(np.float32)
        for _ in range(5):
            req.infer({0:warm})
        with torch.no_grad():
            reference=m(torch.from_numpy(warm)).numpy()*output_std+output_mean
        npu_warm=req.infer({0:warm})[compiled.output(0)].copy()*output_std+output_mean
        delta=float(np.max(np.abs(reference-npu_warm)))
        ae=np.zeros(4,np.float64); se=np.zeros(4,np.float64); counts=np.zeros(4,np.float64)
        latencies=[]
        start=time.perf_counter()
        for j in range(0,n,args.batch):
            idx=np.arange(j,min(j+args.batch,n))
            x=np.zeros(tuple(example.shape),np.float32)
            x[:len(idx),:,:,0]=xdata[idx[:,None]+np.arange(8)]
            t=time.perf_counter()
            pred=req.infer({0:x})[compiled.output(0)][:len(idx),:,:,0].copy()*std+mean
            latencies.append(time.perf_counter()-t)
            target_ids=idx[:,None]+np.arange(8,12)
            target=data['test'][target_ids]
            valid=mask[target_ids]
            e=(pred-target).astype(np.float64)
            ae+=(np.abs(e)*valid).sum((0,2)); se+=(e**2*valid).sum((0,2)); counts+=valid.sum((0,2))
        elapsed=time.perf_counter()-start
        records[mode]=dict(revision=checkpoint['config']['revision'],device=str(core.get_property('NPU','FULL_DEVICE_NAME')),
            execution_devices=compiled.get_property('EXECUTION_DEVICES'),batch=args.batch,windows=n,
            compile_seconds=compile_seconds,test_seconds=elapsed,median_batch_ms=float(np.median(latencies)*1000),
            mae=float(ae.sum()/counts.sum()),rmse=float(np.sqrt(se.sum()/counts.sum())),
            horizon_mae=(ae/counts).tolist(),max_cpu_fp32_vs_npu_difference_first_batch=delta,
            evaluated_values=int(counts.sum()),precision='OpenVINO NPU device default (FP16 graph execution)')
        save_json('npu_metrics.json',records)
        print('NPU FINAL',mode,records[mode],flush=True)
        ov.save_model(converted,ROOT/'checkpoints'/f'{mode}_openvino.xml')

if __name__=='__main__':
    main()
