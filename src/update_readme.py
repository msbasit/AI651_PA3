"""Build the repository report from saved measurements and existing experiment figures."""
import ast
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT

LABELS={'temporal':'Temporal-only','fixed':'Fixed graph','adaptive':'Adaptive graph','hybrid':'Hybrid graph'}
COLORS={'temporal':'#2878b5','fixed':'#d65f00','adaptive':'#8e44ad','hybrid':'#00866d','persistence':'#7b8794'}

def read(name):
    return json.loads((ROOT/'results'/name).read_text(encoding='utf-8-sig'))

def comparison_chart(metrics,baseline):
    modes=['adaptive','hybrid','fixed','temporal','persistence']
    names=[LABELS.get(m,'Persistence') for m in modes]
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    for ax,key,title in zip(axes,['mae','rmse'],['Mean absolute error','Root mean squared error']):
        values=[(baseline if m=='persistence' else metrics[m])['test'][key] for m in modes]
        bars=ax.barh(names,values,color=[COLORS[m] for m in modes],height=.58)
        ax.invert_yaxis();ax.set_xlim(0,max(values)*1.19)
        ax.bar_label(bars,fmt='%.2f',padding=6,fontsize=11)
        ax.set_title(title,loc='left',fontweight='bold',pad=14)
        ax.set_xlabel('Vehicles / 15-minute interval · lower is better')
        ax.set_axisbelow(True);ax.xaxis.grid(True,color='#e6eaf0')
        ax.tick_params(axis='both',length=0,labelsize=10)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Full-test forecasting comparison',fontsize=17,fontweight='bold')
    fig.savefig(ROOT/'figures'/'readme_results.png',dpi=170,facecolor='white',bbox_inches='tight')
    plt.close(fig)

def main():
    metrics=read('metrics.json');base=read('baselines.json')['persistence'];cfg=read('config.json')
    graphs=read('graph_analysis.json');npu=read('npu_metrics.json');prep=read('preparation.json')
    assert set(metrics)==set(LABELS) and all(v['revision']==cfg['revision'] for v in metrics.values())
    assert all(v['revision']==cfg['revision'] for v in npu.values())
    best=min(metrics,key=lambda m:metrics[m]['test']['mae'])
    gain_temporal=100*(1-metrics[best]['test']['mae']/metrics['temporal']['test']['mae'])
    gain_persistence=100*(1-metrics[best]['test']['mae']/base['test']['mae'])
    comparison_chart(metrics,base)
    rows=['| Model | Validation MAE | Test MAE | Test RMSE | Training seconds | Run / selected epoch |',
          '|---|---:|---:|---:|---:|---:|']
    for mode,label in LABELS.items():
        r=metrics[mode];b='**' if mode==best else ''
        rows.append(f'| {b}{label}{b} | {r["validation"]["mae"]:.2f} | {b}{r["test"]["mae"]:.2f}{b} | {r["test"]["rmse"]:.2f} | {r["training_seconds"]:.1f} | {r["epochs_run"]} / {r["best_epoch"]} |')
    rows.append(f'| Persistence | {base["validation"]["mae"]:.2f} | {base["test"]["mae"]:.2f} | {base["test"]["rmse"]:.2f} | 0 (untrained) | — |')
    graph_rows=['| Learned component | Road-edge overlap | Mean distance | Median distance | Links within 5 km | Row entropy |',
                '|---|---:|---:|---:|---:|---:|']
    for mode in ['adaptive','hybrid']:
        r=graphs[mode]
        graph_rows.append(f'| {LABELS[mode]} | {100*r["top10_road_overlap"]:.2f}% | {r["top10_mean_distance_km"]:.2f} km | {r["top10_median_distance_km"]:.2f} km | {100*r["top10_under_5km_fraction"]:.2f}% | {r["mean_row_entropy"]:.3f} nats |')
    ref=graphs['reference']
    graph_rows.append(f'| All non-self pairs (reference) | {100*ref["road_nonself_density"]:.2f}% | {ref["all_pairs_mean_distance_km"]:.2f} km | — | {100*ref["all_pairs_under_5km_fraction"]:.2f}% | {graphs["adaptive"]["uniform_entropy"]:.3f} nats (uniform) |')
    npu_rows=['| Model | GPU test MAE | NPU test MAE | NPU test RMSE | Full-test seconds | Median batch ms |',
              '|---|---:|---:|---:|---:|---:|']
    for mode,label in LABELS.items():
        r=npu[mode]
        npu_rows.append(f'| {label} | {metrics[mode]["test"]["mae"]:.2f} | {r["mae"]:.2f} | {r["rmse"]:.2f} | {r["test_seconds"]:.1f} | {r["median_batch_ms"]:.3f} |')
    split_rows=['| Split | Period | 15-minute bins | 8-in / 4-out windows |','|---|---|---:|---:|']
    for key,label,period in [('train','Training','Jan–Dec 2020'),('val','Validation','Jan–Mar 2021'),('test','Testing','Apr–Dec 2021')]:
        r=prep['splits'][key]
        split_rows.append(f'| {label} | {period} | {r["timestamps"]:,} | {r["windows"]:,} |')
    tree=ast.parse((ROOT/'src/report_content.py').read_text(encoding='utf-8'))
    ack=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('AI was used in finding the right dataset'))
    replacements={'@@BEST@@':LABELS[best],'@@BEST_MAE@@':f'{metrics[best]["test"]["mae"]:.2f}',
                  '@@GAIN_TEMPORAL@@':f'{gain_temporal:.2f}','@@GAIN_PERSISTENCE@@':f'{gain_persistence:.2f}',
                  '@@COUNT@@':f'{base["test"]["evaluated_values"]:,}','@@RESULTS@@':'\n'.join(rows),
                  '@@GRAPHS@@':'\n'.join(graph_rows),'@@NPU@@':'\n'.join(npu_rows),
                  '@@SPLITS@@':'\n'.join(split_rows),'@@ACK@@':ack}
    text=TEMPLATE
    for key,value in replacements.items():text=text.replace(key,value)
    assert '@@' not in text
    (ROOT/'README.md').write_text(text.strip()+'\n',encoding='utf-8')
    print('Built README report and comparison chart from measured records.')

TEMPLATE=r'''
# Traffic Forecasting on Graphs

### AI651 · Deep Learning for Space, Time and Graphs · Assignment 3

**Muhammad Sami Ul Basit · Roll No. 25280025 · Lahore University of Management Sciences**

Four lightweight forecasting models learn from **716 San Diego traffic sensors**. This report moves from aggregation and direction on graphs to full-year traffic forecasting, learned connectivity, and deployment on Intel hardware.

[**PDF report**](PA3_25280025_Muhammad_Sami_Ul_Basit.pdf) · [Executed notebooks](notebooks/) · [Source code](src/) · [Saved measurements](results/) · [LaTeX source](report/report.tex)

| Best test MAE | Improvement over temporal-only | Improvement over persistence | Forecast task |
|:---:|:---:|:---:|:---:|
| **@@BEST_MAE@@** · @@BEST@@ | **@@GAIN_TEMPORAL@@%** lower MAE | **@@GAIN_PERSISTENCE@@%** lower MAE | **2 hours → next 1 hour** |

[Results](#measured-results) · [Forecasts](#forecast-quality) · [Learning curves](#training-and-selection) · [Dataset](#dataset-and-road-graph) · [Architecture](#forecasting-models) · [Learned graphs](#learned-connectivity) · [Theory](#aggregation-and-direction) · [Intel NPU](#intel-npu-deployment) · [Reproduce](#reproduce-the-experiments)

## Measured results

![Full-test MAE and RMSE for four trained models and the persistence baseline](figures/readme_results.png)

@@RESULTS@@

Errors are **vehicles per 15-minute interval**, pooled over all four forecast horizons. Every method uses **26,389 test windows** and **@@COUNT@@ valid sensor–horizon targets** from April–December 2021. Checkpoints are selected by **validation MAE**, then scored on the common test period. The best result in this single-seed comparison is **@@BEST@@**.

<details>
<summary><strong>What is persistence, and how are the metrics computed?</strong></summary>

Persistence repeats each sensor's last observed count for all four future steps. If the last count is 500, the forecast is **500, 500, 500, 500**. It requires no training and provides a sanity check for the learned models.

```math
\widehat y^{\mathrm{persist}}_{s,h,i}=x_{s+7,i},\qquad h\in\{0,1,2,3\}.
```

Let $m_{s,h,i}$ mark an originally valid target and let $C=\sum_{s,h,i}m_{s,h,i}$. Predictions are returned to each sensor's original units before evaluation:

```math
\operatorname{MAE}=\frac{1}{C}\sum_{s,h,i}m_{s,h,i}\left|\widehat y_{s,h,i}-y_{s,h,i}\right|,
\qquad
\operatorname{RMSE}=\sqrt{\frac{1}{C}\sum_{s,h,i}m_{s,h,i}\left(\widehat y_{s,h,i}-y_{s,h,i}\right)^2}.
```

RMSE penalizes large failures more strongly. Overlapping-window forecasts count separately. XPU reductions use FP32, with batch totals accumulated in CPU FP64. Missing observations are excluded rather than treated as zero labels. [Raw model metrics](results/metrics.json) · [Baseline measurements](results/baselines.json)

</details>

## Forecast quality

![Actual traffic and temporal, fixed, adaptive, and hybrid forecasts at three predetermined sensors](figures/q6_predictions.png)

These are **15-minute-ahead** forecasts for sensors selected by graph-order indices **0, 358, 715**, using input starts on **12–13 April 2021**. The sensors and dates were chosen independently of prediction error. Missing observed values remain gaps. The curves show how closely each model follows daily volume changes; the next figure also compares errors at **15, 30, 45, and 60 minutes** over the entire test period.

## Training and selection

![Validation MAE learning curves and test MAE versus forecast horizon](figures/q6_learning.png)

The left panel shows **full-validation MAE per epoch**. The right panel uses each validation-selected checkpoint on the common test set. Errors increase with forecast horizon. All three graph models outperform temporal-only and persistence in this run.

Training uses **Adam, learning rate 0.002, batch 64, hidden dimension 16, FP32, seed 25280025**, a maximum of **40 epochs**, and **patience 5**. The fixed model selected epoch 38 of 40; full convergence is not established. Additional parameters and a single seed limit claims about a universal ranking.

<details>
<summary><strong>Training loss, timing, and optimizer checks</strong></summary>

Per-sensor normalization must not accidentally give low-volume sensors greater weight in the physical-unit loss. For a batch with $C_{\mathcal B}$ valid targets, normalized predictions are weighted by each training standard deviation $\sigma_i$:

```math
\mathcal L_{\mathcal B}=\frac{1}{S_{\mathrm{train}}C_{\mathcal B}}
\sum_{(s,h,i)\in\mathcal B}m_{s,h,i}\sigma_i
\left|\widehat{\widetilde y}_{s,h,i}-\widetilde y_{s,h,i}\right|.
```

$S_{\mathrm{train}}$ is one common training-only scale for numerical conditioning, so this is proportional to original-unit MAE. Gradients are clipped to norm 5. Forecasting weights use weight decay 0.0001; **graph embeddings use zero weight decay**. An intermediate diagnostic found that coupled decay otherwise erased the weak-gradient embeddings, producing a uniform graph. Adaptive/hybrid training was repeated after that structural check. [Diagnostic record](results/optimizer_diagnostic.json)

All shared encoder, local projection, and forecast-head weights have identical initial values across models; independent epoch-seeded shuffles preserve batch order. Fixed seeds do not guarantee bitwise XPU determinism. Training times include executed epochs and validation, with synchronized XPU wall timing; preparation, initial warmup, and final evaluation are excluded. Each total is independently measured and rounded. [Exact configuration](results/config.json) · [Per-epoch histories](results/)

</details>

## Dataset and road graph

**LargeST-SD** contains 716 District-11 highway sensors. Eight 15-minute observations form a **two-hour input**, and the next four intervals form a **one-hour forecast**.

@@SPLITS@@

```math
X_s=[x_s,\ldots,x_{s+7}]\in\mathbb R^{8\times716\times1},\qquad
Y_s=[x_{s+8},\ldots,x_{s+11}]\in\mathbb R^{4\times716\times1}.
```

![Provided directed adjacency matrix and non-self row degree distribution](figures/q3_graph.png)

The supplied directed graph has **18,035 nonzero entries**, including **716 self-loops** and **17,319 non-self ordered edges**. Its non-self mean row degree is **24.19**. Graph rows and columns follow verified sensor metadata order. In this implementation, row $i$ aggregates information from source sensor $j$ with weight $A_{ij}$.

<details>
<summary><strong>See a week of traffic and the preprocessing details</strong></summary>

![Complete week of traffic at three predetermined sensors during February 2020](figures/q3_week.png)

Three complete five-minute count intervals are **summed** to preserve volume:

```math
x_t^{(15\mathrm{min})}=\sum_{k=0}^{2}x_{3t+k}^{(5\mathrm{min})}.
```

This differs from the official benchmark's mean-and-round convention, so the numerical error scale is not directly comparable with that benchmark. Nonfinite and negative raw values are invalid; true zeros remain valid. A partial 15-minute bin is not a valid target.

Chronological splits are applied **before input filling**. Gaps are forward-filled within their own split, then use a per-sensor training median if needed. Labels retain the original validity masks. Each sensor's mean and standard deviation come only from the filled **2020 training series**:

```math
\widetilde x_{t,i}=\frac{x_{t,i}-\mu_i}{\sigma_i},\qquad
\widehat y_{s,h,i}=\sigma_i\widehat{\widetilde y}_{s,h,i}+\mu_i.
```

A constant training series uses unit scale. No window crosses a date boundary, giving $T-11$ windows per split. All 8,600 raw HDF column IDs are checked against metadata before selecting the 716 SD sensors. [Preparation statistics and sensor neighborhoods](results/preparation.json) · [Implementation checks](results/checks.json)

</details>

<details>
<summary><strong>Dataset source, download integrity, and raw-file format</strong></summary>

The [official LargeST project](https://github.com/liuxu77/LargeST) links to Kaggle, whose API returned empty HTTP 404 responses during this run. The downloader therefore uses a [public mirror](https://huggingface.co/datasets/ttydsc30/TrafficForecast-LargeST-Project-Dataset), pinned to revision `fcdd669cb606db00ddb42d526fdde4fb482b2d32`. SHA-256 LFS digests verify transfer integrity against the mirror; equivalence to an independently downloaded Kaggle copy was not established. [Download provenance](results/download_provenance.json)

Raw files occupy about 15 GB. Range downloads resume with `.ranges.json` markers and verify complete hashes. If the official files are already available, place `ca_his_raw_2020.h5`, `ca_his_raw_2021.h5`, `ca_meta.csv`, and `ca_rn_adj.npy` under `data/` and skip the downloader. The HDF datasets are `t/axis0`, `t/axis1`, and `t/block0_values`. Use the original five-minute files. Raw data and the large prepared cache are excluded from Git.

</details>

## Forecasting models

```mermaid
flowchart LR
    X["8 traffic observations per sensor"] --> T["Two temporal convolutions"]
    T --> Z["16 features per sensor"]
    Z --> L["Local projection"]
    Z --> G["Graph aggregation and projection"]
    A["Fixed / adaptive / hybrid adjacency"] --> G
    L --> S["Add and ReLU"]
    G --> S
    S --> H["Shared forecast head"]
    H --> Y["Next 4 traffic counts per sensor"]
```

The **local path** preserves each sensor's representation. Graph models add information from neighbors; temporal-only omits the graph branch. Two temporal convolutions use 16 channels, width 3, padding 1, and ReLU; the flattened eight-step feature map is projected from 128 to 16 features. The final head predicts all four future intervals directly.

```math
H=\operatorname{ReLU}\!\left(ZW_{\mathrm{local}}+\widehat A ZW_{\mathrm{graph}}\right),\qquad
\widehat Y=\operatorname{Head}(H).
```

| Variant | Adjacency | Trainable parameters |
|---|---|---:|
| Temporal-only | No graph term | 3,236 |
| Fixed | Directed road graph, one unit self-loop | 3,492 |
| Adaptive | Learned dense row-stochastic graph | 17,812 |
| Hybrid | 50% fixed + 50% learned graph | 17,812 |

### Fixed, adaptive, and hybrid adjacency

For the fixed graph, clear the provided diagonal, add **exactly one unit self-loop**, then row normalize:

```math
A_{\mathrm{off}}=A-\operatorname{diag}(\operatorname{diag}(A)),\qquad
\widehat A_{\mathrm{fixed}}=D^{-1}(A_{\mathrm{off}}+I),\qquad
D_{ii}=\sum_j(A_{\mathrm{off}}+I)_{ij}.
```

Two independent $716\times10$ standard-normal embedding matrices define the adaptive graph:

```math
E_1,E_2\sim\mathcal N(0,1),\qquad
\widehat A_{\mathrm{adaptive}}=\operatorname{softmax}_{\mathrm{row}}\!\left(\operatorname{ReLU}(E_1E_2^\top)\right).
```

```math
\widehat A_{\mathrm{hybrid}}=0.5\widehat A_{\mathrm{fixed}}+0.5\widehat A_{\mathrm{adaptive}}.
```

All rows sum to one. The adaptive graph remains **dense and differentiable through softmax and positive ReLU logits**; negative logits have zero ReLU gradient at that step. No top-k training sparsification is used. Top-ten links are selected only for analysis after training.

![Fixed, adaptive, and hybrid adjacency matrices on a shared logarithmic scale](figures/q5_adjacencies.png)

All panels share a logarithmic weight scale from $10^{-5}$ to $1$. Gray indicates exact zeros; smaller positive values use the lowest display color. This display clipping removes no model weights. [Model implementation](src/models.py) · [Saved graph arrays](results/)

## Learned connectivity

For each sensor, the **ten strongest learned non-self links** give 7,160 ordered edges. Their overlap with road edges and great-circle distance provide a quantitative comparison with physical connectivity. The hybrid row analyzes its **learned component**, not the mixed matrix that already contains fixed-road edges.

@@GRAPHS@@

![Strongest learned non-self adaptive connections for 60 predetermined sensors on geographic coordinates](figures/q6_learned_map.png)

The map uses 60 evenly spaced graph-order indices. Green lines coincide with provided road edges; red lines are absent from that graph. Lines show learned associations, not road routes. Adaptive links are slightly more local on average than uniform pairs, but their mean separation remains much larger than the **2.28 km** mean for road edges.

**Interpretation.** A useful forecasting graph need not recover the road map. Common commuting demand can connect distant sensors; correlated demand, confounding, or overfitting can produce similar links. Lower error supports predictive usefulness, not causal or physical connectivity. Hybrid mixing can combine a structural prior with learned dependencies, but it is not guaranteed to outperform either component. [Exact graph diagnostics](results/graph_analysis.json)

**Limits.** These are single-seed, small-model, fixed-budget results. Several revisions were made together, so their individual effects are not isolated. Static adjacency misses time-varying incidents and propagation delays; the 2020–2021 split also includes demand shift. Missingness may be nonrandom, so scores describe observed targets only.

## Aggregation and direction

The theory in Q1–Q2 explains why the information retained by a graph layer matters for traffic forecasting.

<details>
<summary><strong>Q1 · What can sum, mean, and max remember? Open the derivations and comparison.</strong></summary>

An affine scalar message preserves only the neighborhood size $n_v$ and total state $S_v$:

```math
\sum_{u\in\mathcal N(v)}\phi(x_u)=\sum_{u\in\mathcal N(v)}(wx_u+c)=wS_v+cn_v.
```

Neighborhoods $\{-1,+1\}$ and $\{0,0\}$ therefore remain indistinguishable for every $w,c$, although their number of steady sensors differs.

A nonlinear scalar can encode all three counts for neighborhoods of size at most $K$. Let $B=K+1$ and choose:

```math
\phi(-1)=1,\quad\phi(0)=B,\quad\phi(+1)=B^2,\qquad
q=n_-+Bn_0+B^2n_+.
```

The counts are base-$B$ digits, so there are no carries. For $K=8$, $B=9$ and all **164 nonempty count triples** have distinct codes. The simpler message $\phi(x)=x^2$ counts non-steady neighbors.

For one-hot messages ordered as slowing, steady, speeding up, write the sum as $(s,z,p)$ and $n=s+z+p>0$:

| Task | Sum | Mean | Max |
|---|---|---|---|
| Count steady | Yes: return $z$ | No: degree information is lost | No: multiplicity is lost |
| Slowing majority | Yes: test $2s>n$ | Yes: test first coordinate $>1/2$ | No: only presence is retained |
| Any slowing | Yes: test $s>0$ | Yes: test $s/n>0$ | Yes: test first coordinate $=1$ |

For counting, $\{0\}$ and $\{0,0\}$ have equal means/maxima but different answers. For majority under max, $\{-1,-1,+1\}$ and $\{-1,+1,+1\}$ have the same presence vector but opposite answers. [Exhaustive numerical checks](results/theory_checks.json) · [Theory notebook](notebooks/01_theory.ipynb)

</details>

<details>
<summary><strong>Q2 · Why symmetric neighbors diffuse motion, and how direction or history resolves it</strong></summary>

For an impulse at sensor 20, a symmetric layer predicts $(b,a,b)$ at sensors $(19,20,21)$, but backward motion requires $(1,0,0)$:

```math
\mathcal L(a,b)=a^2+(b-1)^2+b^2,\qquad
(a^*,b^*)=(0,\tfrac12),\qquad\mathcal L_{\min}=\tfrac12.
```

![True backward motion compared with the recursively applied symmetric neighbor layer](figures/q2_rollouts.png)

The true pattern translates without losing its shape; periodic wrapping occurs in the **true-motion panel**. Equal neighbor weights spread the signal both ways. Sensors 19 and 21 are constrained to the same prediction $b$, so the asymmetric target cannot be fitted.

Direction labels allow separate weights:

```math
\widehat x_i(t+1)=a x_i(t)+w_-x_{i-1}(t)+w_+x_{i+1}(t),\qquad
(a,w_-,w_+)=(0,0,1).
```

This exactly gives $x_{i+1}(t)$ for backward motion. With PyTorch cross-correlation, use kernel **[0, 0, 1]**, circular padding 1, and no bias.

One past local value also resolves the ambiguity:

```math
\widehat x_i(t+1)=x_{i-1}(t)+x_{i+1}(t)-x_i(t-1).
```

For backward motion, $x_i(t-1)=x_{i-1}(t)$, leaving $x_{i+1}(t)$. For forward motion, $x_i(t-1)=x_{i+1}(t)$, leaving $x_{i-1}(t)$. Thus the same coefficients **$(a,b,c)=(0,1,-1)$** work in both directions under the stated constant-speed, shape-preserving assumptions. Numerical errors are at floating-point precision. [Implementation](src/theory.py)

</details>

## Intel NPU deployment

Training and primary inference use **Intel Arc 140T with native PyTorch XPU FP32 kernels**. The selected checkpoints are also evaluated on **Intel AI Boost NPU using OpenVINO**, explicit device selection, and no AUTO/CPU fallback.

@@NPU@@

Every NPU run evaluates all **26,389 test windows** with static **batch 1** and device-default FP16 execution. Its MAE closely matches the GPU result. Full-test seconds include input packing, inference, and metric accumulation; compilation and warmups are excluded.

<details>
<summary><strong>Hardware, precision, and deployment constraints</strong></summary>

| Component | Measured configuration |
|---|---|
| CPU | Intel Core Ultra 7 255H |
| System RAM | 46.9 GiB |
| Training GPU | Intel Arc 140T integrated GPU |
| Additional inference | Intel AI Boost NPU |
| PyTorch | 2.14.1+xpu · FP32 |
| OpenVINO | 2026.4.1 · NPU device-default FP16 |
| GPU driver | 32.0.101.9175 |
| NPU driver | 32.0.100.5600 |

The GPU's reported 23 GB refers to shared integrated-GPU memory, not dedicated VRAM. This PyTorch installation has no NPU autograd backend, so training uses XPU and NPU deployment uses OpenVINO. Learned adjacency is frozen only after checkpoint selection for export. Batch 16 failed with a compiler striding-layout error; static batch 1 succeeded.

NPU batch latency includes its synchronous call and output copy. GPU latency uses resident batch-16 inputs, so these are **not direct speedup comparisons**. FP16 slightly changes predictions. [Environment](results/environment.json) · [Hardware](results/hardware.json) · [NPU measurements](results/npu_metrics.json) · [GPU latency](results/gpu_latency.json)

</details>

## Reproduce the experiments

Use Python 3.12 and an Intel XPU-enabled PyTorch environment. Select the same interpreter for terminal commands and notebooks. GPU stages should run sequentially for meaningful timing.

<details>
<summary><strong>1 · Install dependencies and register the notebook kernel</strong></summary>

Keep an existing working XPU installation. For a new environment, install the matching wheel before the remaining packages:

```powershell
python -m pip install torch==2.14.1+xpu --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__,torch.xpu.is_available()); print(torch.xpu.get_device_name())"
python -m ipykernel install --user --name ai651-xpu --display-name 'Python (Intel Arc XPU - AI651)'
```

Training does not require PyTorch Geometric or Intel Extension for PyTorch. NPU compilation requires the Windows NPU driver and compatible OpenVINO runtime.

</details>

<details>
<summary><strong>2 · Execute the full workflow, including report generation</strong></summary>

Run from the repository root:

```powershell
python src/download_data.py
python src/execute_notebook.py 01_theory.ipynb
python src/execute_notebook.py 02_data_and_checks.ipynb
python src/execute_notebook.py 03_forecasting_experiments.ipynb
python src/execute_notebook.py 04_npu_inference.ipynb
python src/update_readme.py
python src/build_report.py
python src/render_report.py
python src/verify_submission.py
python src/package_submission.py
```

Standard Jupyter **Run All** also works. Notebook execution saves outputs after each cell. `src/make_notebooks.py` creates fresh unexecuted templates; running it over saved notebooks requires executing them again.

Report generation needs `pdflatex` from MiKTeX or TeX Live, with Latin Modern, AMS math, booktabs, tabularx, graphicx, float, placeins, microtype, caption, hyperref, and xurl. The builder regenerates the editable `report/report.tex` and compiles twice. To edit that TeX directly, compile it twice with `pdflatex -output-directory=tmp/latex report/report.tex` and copy the resulting PDF to `PA3_25280025_Muhammad_Sami_Ul_Basit.pdf`; rerunning the Python builder replaces manual TeX edits.

</details>

<details>
<summary><strong>3 · Run individual source stages or evaluate existing checkpoints</strong></summary>

The equivalent source stages are:

```powershell
python src/theory.py
python src/prepare.py
python src/checks.py
python src/baselines.py
python src/train.py
python src/analyze.py
python src/npu.py
python src/update_readme.py
python src/build_report.py
```

To evaluate the saved models without retraining, download and prepare the data, then run:

```powershell
python src/train.py --evaluate-only
python src/npu.py
```

Retraining replaces the chosen modes' checkpoints and metrics. Regenerate analysis, NPU results, README, and report afterwards. No replacement values are supplied for missing experiments.

</details>

### Files and notebooks

| Artifact | Contents |
|---|---|
| [PDF report](PA3_25280025_Muhammad_Sami_Ul_Basit.pdf) | Full written solutions, results, and discussion |
| [01 · Theory](notebooks/01_theory.ipynb) | Q1 encodings and Q2 direction/motion checks |
| [02 · Data and checks](notebooks/02_data_and_checks.ipynb) | SD extraction, preprocessing, graph, and verification |
| [03 · Forecasting](notebooks/03_forecasting_experiments.ipynb) | Four executed GPU runs, baselines, and saved-checkpoint analysis |
| [04 · NPU inference](notebooks/04_npu_inference.ipynb) | Explicit NPU inference over the full test period |
| [Source](src/) | Reusable implementations and report generation |
| [Results](results/) | Metrics, histories, configuration, hashes, graph arrays, predictions, and logs |
| [Checkpoints](checkpoints/) | Four validation-selected FP32 model states |
| [Figures](figures/) | Report figures and the README comparison chart |
| [LaTeX source](report/report.tex) | Editable conventional report |
| [Derived data](data/) | Small SD graph, metadata, and training fallback values |

## AI Acknowledgement

<details>
<summary>Read the acknowledgement included in the PDF report</summary>

@@ACK@@

</details>

## References and attribution

- Liu et al., *LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting*, NeurIPS Datasets and Benchmarks 2023. [Paper](https://arxiv.org/abs/2306.08259) · [Official project](https://github.com/liuxu77/LargeST).
- [Pinned raw-data mirror](https://huggingface.co/datasets/ttydsc30/TrafficForecast-LargeST-Project-Dataset) · [Caltrans PeMS](https://pems.dot.ca.gov/?dnode=Clearinghouse).
- [PyTorch on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) · [OpenVINO NPU device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html).

LargeST data are [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Derived sensor metadata and adjacency retain that license and attribution. Large raw files are not redistributed here. Code is provided for academic reproducibility.

[Back to top](#traffic-forecasting-on-graphs)
'''

if __name__=='__main__':
    main()
