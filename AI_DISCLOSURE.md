# Generative AI assistance

OpenAI Codex generated the mathematical explanations, implementation, data pipeline, experiments, plots, notebooks, README and report, then executed and checked the artifacts on the specified machine. Measured results come from saved outputs of the actual runs.

## User prompt

> You are an excellent standford Deep learning instructor who teaches Deel Learning for space time and graph, complete this assignment as per instructions and attached assignment document. For training and inference use intel Arc 140T gpu on this machine using pytorch optimizied libraries for intel - they are already installed and available Use Intel builtin NPU where you can for again training and inference. Upload the source code on github I have already logged in to the builtin browser https://github.com/msbasit Name: Muhammad Sami Ul Basit Roll No: 25280025.

## Output and edits

The authored source, four executed notebooks, report, plots and saved measurements are AI-assisted outputs. Codex made the initial runtime corrections: selecting the installed Intel XPU interpreter, using a pinned dataset mirror after Kaggle API failures, accumulating metric totals in CPU FP64 after XPU FP32 reductions, and reducing static NPU inference to batch one after a compiler error. The subsequent model and report corrections were directed by the independent human review documented below.

On 22 November 2026, the user requested: "I dont really like the style of the report. I like conventional report style with Latex based equations and expressions". Codex retypeset the report in native LaTeX with serif text, numbered equations, conventional tables and references. Experimental measurements and trained models were retained, with no retraining for the style revision. The editable `report/report.tex` source is included.

## Independent review and issue-fix request

The user independently reviewed the earlier solution, confirming theory/data handling and identifying problems with the graph architecture, initialization, training budget and reporting. The correction request was to review and fix all documented problems, discussing any disagreements first. The user subsequently requested that the review notes remain private and that the AI acknowledgement accurately credit human input and intervention. The private notes are not included in the report, public repository or submission archive.

Codex agreed with the substantive findings and implemented a separate local path, 10-dimensional standard-normal graph embeddings, per-sensor training-only scaling with original-unit loss weighting, exactly one model self-loop, a 40-epoch maximum with patience 5, persistence measurements and report corrections. All four models were retrained. A final graph diagnostic caught embedding collapse caused by coupled weight decay; Codex excluded the embeddings from weight decay and repeated the adaptive/hybrid training and downstream checks. Temporal/fixed optimizer updates were unchanged. Their GPU/NPU measurements, plots, graph diagnostics, conclusions, notebooks and LaTeX report regenerated. The optional training top-k was not used; the revised adaptive graph remains dense and differentiable through its softmax logits.

## First-person review statement

I independently reviewed the earlier solution against the assignment. I checked the Q1-Q2 derivations and data/split conventions, identified weaknesses in model design, initialization and the training budget, and supplied specific corrections and report revisions. I directed these changes and reviewed the implementation choices. Codex assisted with implementation, retraining, automated checks and artifact regeneration. This work includes substantive human review and intervention alongside AI assistance. This statement records the review described in the chat and does not claim an additional human review of the new runs. Additional course-outline AI rules were not supplied.
