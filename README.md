# AI651 Assignment 3: traffic forecasting on graphs

**Muhammad Sami Ul Basit | Roll No. 25280025**

The report answers Questions 1-7. Four executed notebooks contain numerical checks, dataset preparation, all four GPU training runs, saved-checkpoint analysis, and full-test Intel NPU inference. Generative AI assistance is described in the AI Acknowledgement at the end of the report.

Public repository: [msbasit/AI651_PA3](https://github.com/msbasit/AI651_PA3).

## Deliverables

- `PA3_25280025_Muhammad_Sami_Ul_Basit.pdf`: technical report and written solutions, with the repository link on its first page.
- `notebooks/01_theory.ipynb`: exhaustive Q1 encodings and Q2 motion verification/heatmaps.
- `notebooks/02_data_and_checks.ipynb`: LargeST-SD extraction, plots and meaningful implementation checks.
- `notebooks/03_forecasting_experiments.ipynb`: temporal-only, fixed, adaptive and hybrid models, their real training outputs and analysis.
- `notebooks/04_npu_inference.ipynb`: full-test explicit Intel AI Boost NPU inference.
- `src/`: runnable, reusable source code underlying the notebooks and report.
- `report/report.tex`: editable native LaTeX report source, with numbered equations and conventional academic typography.
- `results/`: per-epoch histories, original-unit metrics, exact configuration, download hashes, graph diagnostics, prediction arrays and logs.
- `figures/`: all report figures.
- `checkpoints/{temporal,fixed,adaptive,hybrid}.pt`: validation-selected FP32 model weights.
- `data/sd_meta.csv`, `data/sd_adj.npy`, `data/training_fallback.npy`: small derived artifacts. Large raw files and the prepared cache are excluded from Git.

## Measured results

| Model | Validation MAE | Test MAE | Test RMSE | Training seconds | Epochs / selected |
|---|---:|---:|---:|---:|---:|
| Temporal-only | 53.21 | 61.75 | 97.28 | 780.0 | 39 / 34 |
| Fixed graph | 49.85 | 58.88 | 92.48 | 930.9 | 40 / 38 |
| Adaptive graph | 43.99 | 52.72 | 83.58 | 970.8 | 39 / 34 |
| Hybrid graph | 47.12 | 55.11 | 87.19 | 609.7 | 24 / 19 |
| Persistence | 81.18 | 87.11 | 132.88 | 0 (untrained) | -- |

Errors are vehicles per 15-minute interval. Every method uses the same 26,389 test windows and 71,023,748 valid sensor-horizon targets. The lowest test MAE among the trained models is Adaptive graph (52.72). Checkpoints are selected by validation only. This is a single-seed comparison, and graph variants have additional parameters.

Training times are synchronized XPU wall times including epoch validation, independently measured and rounded. Exact totals and per-epoch durations are retained in `results/metrics.json` and history files. Full-test NPU deployment metrics are in `results/npu_metrics.json`.

Following independent human review, this revision adds a local path, per-sensor scaling, stronger adaptive initialization, a longer budget, and persistence checks. Private review notes are excluded from the public repository and submission. These changes were made together; the rerun is not an ablation isolating a single cause.

## Environment

Report generation requires a LaTeX distribution (MiKTeX or TeX Live) providing `pdflatex`, Latin Modern, AMS math, booktabs, tabularx, graphicx, float, placeins, microtype, caption, hyperref and xurl. `python src/build_report.py` generates `report/report.tex` from measured records and compiles it twice. To edit TeX directly, run `pdflatex -output-directory=tmp/latex report/report.tex` twice from the project root and copy the resulting PDF to the deliverable filename; the Python builder will regenerate the TeX source when rerun.

Use **Python 3.12 with Intel XPU-enabled PyTorch**. The measured machine uses a Core Ultra 7 255H, Intel Arc 140T integrated GPU and Intel AI Boost NPU. Exact versions and driver details are saved in `results/environment.json` and `results/hardware.json`.

Keep a working Intel XPU installation. For a new environment, install the matching XPU wheel explicitly before the remaining packages:

```powershell
python -m pip install torch==2.14.1+xpu --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__,torch.xpu.is_available()); print(torch.xpu.get_device_name())"
python -m ipykernel install --user --name ai651-xpu --display-name 'Python (Intel Arc XPU - AI651)'
```

Training uses native PyTorch XPU kernels in FP32. It does not need PyTorch Geometric or Intel Extension for PyTorch. OpenVINO handles NPU inference. The installed PyTorch has no NPU training/autograd backend. NPU compilation requires its Windows driver and an OpenVINO-supported Intel NPU.

## Reproduce the complete submission

Run from the project root. Select the same XPU interpreter for terminal and notebooks.

```powershell
python src/download_data.py
python src/execute_notebook.py 01_theory.ipynb
python src/execute_notebook.py 02_data_and_checks.ipynb
python src/execute_notebook.py 03_forecasting_experiments.ipynb
python src/execute_notebook.py 04_npu_inference.ipynb
python src/build_report.py
python src/render_report.py
python src/verify_submission.py
python src/package_submission.py
```

Notebook execution saves actual outputs after each cell. Standard Jupyter **Run All** also works. `src/make_notebooks.py` creates fresh, unexecuted notebook templates; do not run it over saved outputs unless you intend to regenerate and execute them again.

The individual source stages are equivalent:

```powershell
python src/theory.py
python src/prepare.py
python src/checks.py
python src/baselines.py
python src/train.py
python src/analyze.py
python src/npu.py
python src/build_report.py
```

To evaluate the saved checkpoints without retraining, first download/prepare data, then run:

```powershell
python src/train.py --evaluate-only
python src/npu.py
```

GPU stages should run sequentially for meaningful timings. Checkpoints are selected by full validation MAE, then evaluated on the full common test period. Re-running training replaces checkpoints and metrics for its selected modes; after any change, regenerate analysis, NPU results and the report. The report builder requires all four measured model results and does not supply substitute values.

## Dataset and preprocessing

The [official LargeST project](https://github.com/liuxu77/LargeST) hosts its raw data on Kaggle. Its API returned empty HTTP 404 responses during this run. The downloader uses this [public mirror](https://huggingface.co/datasets/ttydsc30/TrafficForecast-LargeST-Project-Dataset), pinned to revision `fcdd669cb606db00ddb42d526fdde4fb482b2d32`. SHA-256 LFS digests verify transfer integrity against the mirror; equivalence to a separately downloaded Kaggle copy was not independently established. `results/download_provenance.json` records exact URLs, sizes and hashes. Raw files consume about 15 GB. Range downloads resume using `.ranges.json` markers and verify complete hashes before preparation.

If official Kaggle files are already available, place them under `data/` as `ca_his_raw_2020.h5`, `ca_his_raw_2021.h5`, `ca_meta.csv`, `ca_rn_adj.npy`, and skip the downloader. Expected raw HDF datasets are `t/axis0`, `t/axis1` and `t/block0_values`. Use original five-minute files, not the benchmark's already-resampled files.

The code selects all 716 District-11 sensors. Every HDF column ID is checked against metadata; graph rows/columns are selected using the matching `ID2` values. [PeMS](https://pems.dot.ca.gov/?dnode=Clearinghouse) defines total flow as vehicles per five-minute interval. Three complete count intervals are **summed**, yielding vehicles per 15 minutes. This differs from the official benchmark notebook's mean-and-round convention, so numerical errors use a different physical scale.

NaN, infinite and negative raw observations are invalid. Partial 15-minute bins remain missing labels. After the required date splits, input gaps are forward-filled within each split, then filled with per-sensor training medians if needed. True zeros are retained. Missing-label masks apply consistently to training and all metrics. Each sensor uses its own mean/std from the filled 2020 series, with unit scale for a constant training series. All predictions are inverted per sensor before scoring. Windows do not cross date boundaries.

| Split | Dates | 15-minute bins | Eight-in/four-out windows |
|---|---|---:|---:|
| Training | January-December 2020 | 35,136 | 35,125 |
| Validation | January-March 2021 | 8,640 | 8,629 |
| Testing | April-December 2021 | 26,400 | 26,389 |

## Experimental choices

Each sensor receives two temporal convolutions (16 channels, width 3), a flattened-history projection to 16 features, and a bias-free 16-to-16 local projection. Graph models add a separate bias-free 16-to-16 projection of adjacency-mixed features before the common ReLU and 16-to-4 head: `H = relu(local(Z) + spatial(A @ Z))`. Temporal-only omits the graph term. Shared encoder, local projection and head initial weights are identical across experiments; graph-only parameters are initialized afterwards.

Fixed adjacency first clears the provided diagonal, then adds exactly one unit self-loop and row normalizes. Adaptive adjacency is `softmax(relu(E1 @ E2.T), dim=1)` with two standard-normal 716-by-10 embedding matrices (std 1). It is dense, with no top-k training sparsification. Hybrid adjacency is `0.5*fixed + 0.5*adaptive`, combining row-stochastic terms. Top-ten links are selected only for post-training analysis.

All runs use seed 25280025, batch 64, Adam at 0.002, weight decay 0.0001 on forecasting weights and zero on graph embeddings, norm clipping at 5, maximum 40 epochs and patience 5. The loss is masked original-unit MAE divided by one fixed training-only global scale, preserving equal physical-unit weighting despite per-sensor normalization. Every epoch uses every training window. Independent epoch-seeded permutations keep batch order identical. Node embeddings are excluded from coupled weight decay because an intermediate diagnostic showed that their small gradients otherwise allowed the regularizer to erase them. The adaptive/hybrid runs were repeated after that graph diagnostic; the temporal/fixed runs have no embeddings and are unaffected. No features, graph dependencies or normalization statistics are fitted on validation/test data. XPU seeds are fixed, but bitwise determinism across devices/releases is not guaranteed.

NPU exports freeze the learned adjacency after training and use static batch one, explicit `NPU` compilation and device-default FP16. Batch 16 failed with a measured compiler striding-layout error. The NPU notebook measures full-test accuracy for every model; it does not silently fall back to CPU. GPU and NPU latency configurations differ, so the reported per-batch times are not direct speedup comparisons.

## Attribution and license

Liu et al., *LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting*, NeurIPS Datasets and Benchmarks 2023. [Paper](https://arxiv.org/abs/2306.08259).

LargeST data are [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Derived sensor metadata and adjacency retain that license and attribution. Large raw files are not redistributed here. The forecasting and experiment code is implemented for this assignment; generative AI assistance is explicitly disclosed.
