"""Create notebooks with actual executable experiment cells and explanatory prose."""
from pathlib import Path
import nbformat as nbf
from common import ROOT

def notebook(name,cells):
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec={'display_name':'Python (Intel Arc XPU - AI651)','language':'python','name':'ai651-xpu'}
    nbf.write(nb,ROOT/'notebooks'/name)

def main():
    md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
    setup=code("from pathlib import Path\nimport sys, subprocess, json\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nassert (ROOT/'src').is_dir()\ndef run(script,*args):\n    command=[sys.executable,str(ROOT/'src'/script),*args]\n    log=ROOT/'results'/(script.replace('.py','')+('_'+args[-1] if args else '')+'.log')\n    with log.open('w',encoding='utf-8') as f:\n        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)\n    print(log.read_text(encoding='utf-8')[-16000:])\n    result.check_returncode()\ndef show(name):\n    display(Image(filename=str(ROOT/'figures'/name)))")
    notebook('01_theory.ipynb',[
        md('# Q1-Q2: aggregation and directional motion\nMuhammad Sami Ul Basit | 25280025\n\nThese cells run the exhaustive NumPy checks and generate the ring-road heatmap. Full proofs and the nine-row aggregation table are in the report.'),setup,
        md('## Q1: scalar count encoding\nAn affine message only retains $wS+cn$. For nonlinear encoding take base $B=K+1$ and messages $1,B,B^2$; the three counts are its digits. There are $\\binom{11}{3}-1=164$ legal triples when $K=8$. The message $x^2$ counts non-steady neighbors.'),
        code("run('theory.py')\nprint((ROOT/'results'/'theory_checks.json').read_text())"),
        md('## Q2: direction\nThe equal-neighbor least-squares fit minimizes $a^2+(b-1)^2+b^2$, giving $a=0,b=1/2$. Directional weights $[0,0,1]$ exactly shift against traffic. With history, $[a,b,c]=[0,1,-1]$ predicts either direction because the previous local value equals the current value at the already-visited neighbor.'),
        code("show('q2_rollouts.png')")])
    notebook('02_data_and_checks.ipynb',[
        md('# Q3-Q4(a): data preparation and verification\nRun the downloader first. All 716 District-11 sensors are retained in metadata order. Each complete group of three five-minute vehicle counts is summed. Partial bins remain masked targets; inputs use split-local forward fill and a training-only median fallback.'),setup,
        code("run('prepare.py')"),
        md('## Leakage and implementation checks\nChronological boundaries are applied before filling. Windows contain eight inputs followed by four targets and cannot cross those boundaries. The checks also verify graph gradients and identical shared initialization.'),
        code("run('checks.py')"),code("show('q3_week.png')\nshow('q3_graph.png')")])
    cells=[md('# Q4-Q6: controlled Intel Arc XPU experiments\nReview revision: local skip path in every model; graph models add a separate graph term. FP32; hidden dimension 16; node embeddings 10 with standard-normal initialization; Adam 0.002; batch 64; maximum 40 epochs; patience 5; seed 25280025. Per-sensor training-only scaling; masked original-unit MAE loss divided by one common training scale. Every epoch uses all training windows. Full validation MAE selects checkpoints; final test evaluation follows selection.'),setup,
           code("import torch\nprint(torch.__version__,torch.xpu.is_available(),torch.xpu.get_device_name())\nassert torch.xpu.is_available()"),
           md('## Model implementation\nAll models retain each sensor\'s local representation through a shared local projection. The graph models add a separate projection of adjacency-mixed features before ReLU. The temporal baseline omits that graph term. Fixed adjacency clears the supplied diagonal, adds exactly one self-loop, and row normalizes. Adaptive adjacency uses dense row softmax of ReLU of standard-normal embedding products; hybrid mixing uses alpha=0.5. No top-k training sparsification or test-dependent graph construction occurs.'),
           code("print((ROOT/'src'/'models.py').read_text())")]
    cells.extend([md('## Persistence sanity checks\nRepeat each sensor\'s last observation for all four horizons, using the identical target masks. Also measure last values mixed in original units through the fixed graph as an aggregation diagnostic.'),code("run('baselines.py')\nprint((ROOT/'results'/'baselines.json').read_text())")])
    for mode in ['temporal','fixed','adaptive','hybrid']:
        cells.extend([md('## '+mode.title()),code(f"run('train.py','--modes','{mode}')")])
    cells.extend([md('## Saved-checkpoint analysis\nPredictions use graph-order sensors 0,358,715 and predetermined April dates. Top-10 graph analysis excludes self-links and compares with the physical graph and a uniform-pair geographic reference.'),code("run('analyze.py')\nprint((ROOT/'results'/'metrics.json').read_text())"),code("show('q5_adjacencies.png')\nshow('q6_predictions.png')\nshow('q6_learning.png')\nshow('q6_learned_map.png')")])
    notebook('03_forecasting_experiments.ipynb',cells)
    notebook('04_npu_inference.ipynb',[
        md('# Intel AI Boost NPU deployment\nTraining uses PyTorch XPU autograd. OpenVINO is used for NPU inference, with explicit NPU selection and no AUTO/CPU fallback. Static batch one avoids the NPU compiler striding limitation encountered at batch 16. Learned adjacency is frozen only for export after training. Each saved model is evaluated on every test window.'),setup,
        code("run('npu.py')\nprint((ROOT/'results'/'npu_metrics.json').read_text())")])
    print('Created four executable notebooks.')

if __name__=='__main__':
    main()
