# Generative AI assistance

OpenAI Codex generated the mathematical explanations, implementation, data pipeline, experiments, plots, notebooks, README and report, then executed and checked the artifacts on the specified machine. Measured results come from saved outputs of the actual runs.

## User prompt

> You are an excellent standford Deep learning instructor who teaches Deel Learning for space time and graph, complete this assignment as per instructions and attached assignment document. For training and inference use intel Arc 140T gpu on this machine using pytorch optimizied libraries for intel - they are already installed and available Use Intel builtin NPU where you can for again training and inference. Upload the source code on github I have already logged in to the builtin browser https://github.com/msbasit Name: Muhammad Sami Ul Basit Roll No: 25280025.

## Output and edits

The complete authored source, four executed notebooks, report, plots and saved measurements are AI-assisted outputs. Agent-led corrections included selecting the installed Intel XPU interpreter instead of CPU-only default Python, using a pinned dataset mirror after Kaggle API failures, accumulating metric totals in CPU FP64 after XPU FP32 reductions, and reducing the static NPU inference batch to one after a compiler error at batch 16. No claim is made that these corrections were human edits.

On 22 November 2026, the user requested: "I dont really like the style of the report. I like conventional report style with Latex based equations and expressions". Codex retypeset the report in native LaTeX with serif text, numbered equations, conventional tables and references. Experimental measurements and trained models were retained, with no retraining for the style revision. The editable `report/report.tex` source is included.

The user supplied this style preference; no direct human edits to the code or solutions are recorded. The student should independently review and understand all submitted material and be ready to explain it. The report repeats this disclosure as required by the supplied assignment; additional course-outline AI rules were not available for inspection.
