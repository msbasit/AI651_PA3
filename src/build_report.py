"""Build a paginated PDF using only measured experiment artifacts."""
import html
import json
import statistics
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from common import ROOT

REPO='https://github.com/msbasit/AI651_PA3'
OUT=ROOT/'AI651_PA3_Muhammad_Sami_Ul_Basit_25280025.pdf'
STORY=[]
styles=getSampleStyleSheet()

def read(name):
    return json.loads((ROOT/'results'/name).read_text(encoding='utf-8-sig'))

def p(text,style='Body'):
    STORY.append(Paragraph(text,styles[style]))

def h(text): p(text,'Heading2')
def page(title):
    if STORY: STORY.append(PageBreak())
    p(title,'Heading1')

def table(rows,widths=None,small=False):
    cells=[[Paragraph(str(c),styles['TableHead'] if ri==0 else styles['Small' if small else 'Table']) for c in row] for ri,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#133e52')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f7'),colors.white]),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#133e52'))]))
    STORY.extend([t,Spacer(1,8)])

def fig(name,caption,width=475,maxheight=270):
    path=ROOT/'figures'/name
    w,h=ImageReader(str(path)).getSize()
    scale=min(width/w,maxheight/h)
    STORY.append(KeepTogether([Image(str(path),width=w*scale,height=h*scale,hAlign='CENTER'),
                              Paragraph(caption,styles['Caption'])]))

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#c7d3d8'));canvas.line(45,38,550,38)
    canvas.setFont('BodyFont',8);canvas.setFillColor(colors.HexColor('#50616c'))
    canvas.drawString(45,25,'AI651 | Assignment 3 | Muhammad Sami Ul Basit | 25280025')
    canvas.drawRightString(550,25,str(doc.page))

def main():
    pdfmetrics.registerFont(TTFont('BodyFont','C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('BodyBold','C:/Windows/Fonts/arialbd.ttf'))
    pdfmetrics.registerFontFamily('BodyFont',normal='BodyFont',bold='BodyBold',italic='BodyFont',boldItalic='BodyBold')
    styles.add(ParagraphStyle(name='Body',fontName='BodyFont',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#20333e')))
    styles.add(ParagraphStyle(name='Table',parent=styles['Body'],fontSize=8.5,leading=11,spaceAfter=0))
    styles.add(ParagraphStyle(name='Small',parent=styles['Table'],fontSize=7.7,leading=10))
    styles.add(ParagraphStyle(name='TableHead',parent=styles['Table'],fontName='BodyBold',textColor=colors.white))
    styles.add(ParagraphStyle(name='Caption',parent=styles['Body'],fontSize=8,leading=10.5,textColor=colors.HexColor('#50616c'),spaceAfter=10))
    for name in ['Title','Heading1','Heading2']:
        styles[name].fontName='BodyBold';styles[name].textColor=colors.HexColor('#133e52')
    styles['Heading1'].fontSize=18;styles['Heading1'].leading=22;styles['Heading1'].spaceAfter=14
    styles['Heading2'].fontSize=11.5;styles['Heading2'].leading=15;styles['Heading2'].spaceAfter=7
    prep=read('preparation.json');metrics=read('metrics.json');cfg=read('config.json')
    graphs=read('graph_analysis.json');npu=read('npu_metrics.json');hw=read('hardware.json')
    env=read('environment.json');checks=read('checks.json');gpu=read('gpu_latency.json')
    best=min(metrics,key=lambda k:metrics[k]['test']['mae'])
    improvement=(metrics['temporal']['test']['mae']-metrics[best]['test']['mae'])/metrics['temporal']['test']['mae']*100

    page('Traffic forecasting on graphs')
    p('AI651 - Deep Learning for Space, Time and Graphs','Heading2')
    p('Assignment 3 | Fall 2026 | Lahore University of Management Sciences')
    STORY.append(Spacer(1,14))
    p('<b>Muhammad Sami Ul Basit</b><br/>Roll No: <b>25280025</b>')
    p(f'<b>Public source repository:</b><br/><link href="{REPO}" color="#007b8a">{REPO}</link>')
    p('Prepared 21 November 2026. Full AI-use disclosure appears at the end of this report.')
    h('Scope and outcome')
    p('Questions 1-2 establish what aggregation preserves and how direction can enter a graph layer. Questions 3-7 implement four lightweight forecasters using all 716 San Diego sensors, 2020 training data, January-March 2021 validation data, and April-December 2021 test data. Each prediction uses eight 15-minute observations and forecasts four further intervals.')
    outcome=('None of the graph variants improves on the temporal-only test MAE under this protocol.' if best=='temporal' else f'This is a {improvement:.2f}% relative MAE reduction versus temporal-only.')
    p(f'The lowest measured test MAE is from <b>{best}</b>: <b>{metrics[best]["test"]["mae"]:.2f}</b> vehicles per 15 minutes. {outcome} These are single-seed, finite-budget comparisons; they do not establish a universal ranking or statistical significance.')
    h('Final test comparison (all four horizons pooled)')
    rows=[['Model','Adjacency','MAE','RMSE','Train time (s)']]
    for mode,label,adj in [('temporal','Temporal-only','None'),('fixed','STGNN','Fixed'),('adaptive','STGNN','Adaptive'),('hybrid','STGNN','Hybrid')]:
        r=metrics[mode]; rows.append([label,adj,f'{r["test"]["mae"]:.2f}',f'{r["test"]["rmse"]:.2f}',f'{r["training_seconds"]:.1f}'])
    table(rows,[110,90,75,75,130])
    p('Errors are in vehicles per 15-minute interval. Metrics pool every valid sensor/horizon target over the same full test period. Training times include all executed epochs and epoch validation, with synchronized XPU wall time; preparation, warmup and final evaluation are excluded. Early stopping can give different realized durations under the same rule.','Caption')
    for mode in metrics:
        hist=read(f'{mode}_history.json')
        median=statistics.median(r['seconds'] for r in hist)
        unusual=[r for r in hist if r['seconds']>2*median]
        for r in unusual:
            p(f'Timing note: {mode} epoch {r["epoch"]} took {r["seconds"]:.1f}s versus a median {median:.1f}s. This wall-time outlier is retained in the reported total; training times are not isolated kernel-throughput measurements.','Caption')
    h('Hardware')
    p(f'{html.escape(hw["cpu"])}; {hw["ram_bytes"]/2**30:.1f} GiB system RAM. Training and primary inference: {html.escape(env["gpu"])} using PyTorch {env["torch"]}. Additional full-test inference: Intel AI Boost through OpenVINO {env["packages"]["openvino"]}. The GPU memory label describes shared integrated-GPU memory, not dedicated VRAM.')

    page('Q1 - What does a sum remember?')
    h('(a) Affine messages retain only degree and total trend')
    p('For scalar or vector w,c, the sum is <b>Σ(wx<sub>u</sub>+c)=wS<sub>v</sub>+cn<sub>v</sub></b>. Equal n and S therefore give the same aggregated message, and applying any function ρ cannot recover a distinction that has already vanished. For example, {-1,+1} and {0,0} both have n=2 and S=0, but contain zero and two steady neighbors respectively.')
    h('(b) A single nonlinear scalar can encode all three counts')
    p('Set B=K+1 and choose <b>ϕ(-1)=1, ϕ(0)=B, ϕ(+1)=B²</b>. If the counts are (n<sub>-</sub>,n<sub>0</sub>,n<sub>+</sub>), the sum q=n<sub>-</sub>+Bn<sub>0</sub>+B²n<sub>+</sub>. Each count lies in [0,K], so each is a base-B digit and no carry occurs. Decode using n<sub>-</sub>=q mod B, n<sub>0</sub>=floor(q/B) mod B, and n<sub>+</sub>=floor(q/B²) mod B.')
    p('For K=8, B=9. Exhaustive NumPy enumeration verifies all <b>164</b> nonempty count triples with total at most eight: every encoded sum is distinct and every triple decodes exactly. The number is C(11,3)-1=164. To count only non-steady neighbors at any degree, choose <b>ϕ(x)=x²</b>, equivalently outputs (1,0,1).')
    h('(c) Fixed one-hot messages: complete comparison')
    p('Write the count vector as (s,z,p), n=s+z+p. The coordinates always refer to slowing, steady and speeding up in that order. All counterexamples below obey the 1-8 neighbor bound.')
    rows=[['Task','Method','Possible?','Explicit calculation or indistinguishable pair'],
    ['Count steady','Sum','Yes','ρ(s,z,p)=z.'],
    ['Count steady','Mean','No','{0} and {0,0} both aggregate to (0,1,0), but their correct counts are 1 and 2.'],
    ['Count steady','Max','No','{0} and {0,0} both aggregate to (0,1,0), but their correct counts are 1 and 2.'],
    ['Slowing majority','Sum','Yes','Test 2s &gt; s+z+p.'],
    ['Slowing majority','Mean','Yes','Test first coordinate &gt; 1/2.'],
    ['Slowing majority','Max','No','{-1,-1,+1} and {-1,+1,+1} both aggregate to (1,0,1); correct answers are True and False.'],
    ['Any slowing','Sum','Yes','Test first coordinate s &gt; 0.'],
    ['Any slowing','Mean','Yes','Test first coordinate s/n &gt; 0 (n is nonzero).'],
    ['Any slowing','Max','Yes','Test first coordinate = 1.']]
    table(rows,[86,48,61,285],small=True)

    page('Q2 - Which way is the slowdown moving?')
    h('(a) Equal weights diffuse the pattern')
    p('For an impulse at sensor 20, predictions at sensors (19,20,21) are (b,a,b), whereas targets are (1,0,0). All other predictions and targets are zero. Thus L=(b-1)²+a²+b². Setting its derivatives to zero gives <b>a=0, b=1/2</b>; the Hessian diag(2,4) is positive definite, so this is the unique minimum, with SSE=1/2.')
    fig('q2_rollouts.png','Figure 1. Both sequences start from exp(-(i-20)²/8), use the same [0,1] color scale and show t=0,...,20. The fitted symmetric rule is applied recursively without teacher forcing.',maxheight=240)
    p('The true bump moves one sensor against traffic at every step and preserves its height and shape. The equal-neighbor layer spreads mass both ways, lowering and widening the peak rather than translating it; periodic wrapping is visible near the boundary.')
    h('(b) Give the layer direction labels')
    p('Choose <b>a=0, w<sub>-</sub>=0, w<sub>+</sub>=1</b>. Substitution gives x̂<sub>i</sub>(t+1)=x<sub>i+1</sub>(t), exactly the true rule for every input. A PyTorch cross-correlation Conv1d uses kernel <b>[0,0,1]</b> on [x<sub>i-1</sub>,x<sub>i</sub>,x<sub>i+1</sub>], with circular padding of one and no bias. A mathematical convolution reverses its stored kernel convention. Labeling ahead and behind breaks the reflection ambiguity that persists through arbitrary stacks of unlabeled symmetric layers.')
    h('(c) History supplies the missing directional information')
    p('Choose <b>a=0, b=1, c=-1</b>. For backward motion, x<sub>i-1</sub>(t)=x<sub>i</sub>(t-1), so x̂<sub>i</sub>(t+1)=x<sub>i-1</sub>(t)+x<sub>i+1</sub>(t)-x<sub>i</sub>(t-1)=x<sub>i+1</sub>(t). For forward motion, x<sub>i+1</sub>(t)=x<sub>i</sub>(t-1), so the same expression equals x<sub>i-1</sub>(t).')
    p('The past local value identifies the spatial term associated with where the pattern has already passed. Subtracting it leaves the opposite neighbor, even though both spatial coefficients remain equal. This exact identity assumes the stated constant-speed, shape-preserving motion; noise or changing speed need not satisfy it. Numerical checks verify both directions to floating-point precision.')

    page('Q3(a) - Data extraction and resampling')
    h('Source, sensor identity and units')
    p('The official LargeST repository [1] specifies District 11 for SD and maps sensor IDs to adjacency indices using metadata ID2. Kaggle API requests returned empty HTTP 404 responses on this machine. Raw 2020/2021 HDF5 files, metadata and adjacency were therefore obtained from a public mirror [2]. Downloads use a pinned mirror revision and verify LFS SHA-256 digests. These checks establish transfer integrity against the mirror; equivalence to an independently downloaded Kaggle archive was not verified.')
    p('All 8,600 HDF column IDs are checked against the complete metadata order. The 716 District-11 rows then select both the traffic columns by ID and the graph by ID2. The raw five-minute grid is reindexed to the full calendar year, preserving missing timestamps explicitly.')
    p('PeMS defines five-minute total flow in vehicles per five minutes [3]. Consequently <b>X<sup>15min</sup><sub>t</sub>=X<sup>5min</sup><sub>3t</sub>+X<sup>5min</sup><sub>3t+1</sub>+X<sup>5min</sup><sub>3t+2</sub></b>, yielding vehicles per 15-minute interval. This follows the assignment count-volume option. The official benchmark notebook averages and rounds [1]; our sum preserves interval volume, so these errors should not be compared directly to its reported scale.')
    h('Missing values and split-local preprocessing')
    raw=prep['raw']
    p(f'Nonfinite or negative five-minute values are invalid: {raw["2020"]["missing_or_negative"]:,} in 2020 and {raw["2021"]["missing_or_negative"]:,} in 2021. A 15-minute target is valid only if all three inputs are valid (sum with min_count=3). Zeros remain valid; no observation-quality flags are available to distinguish genuine zero flow from undocumented outage zeros.')
    rows=[['Split','15-min bins','Windows','Missing sensor-bins']]
    for name,label in [('train','2020 training'),('val','Jan-Mar 2021 validation'),('test','Apr-Dec 2021 testing')]:
        r=prep['splits'][name];rows.append([label,f'{r["timestamps"]:,}',f'{r["windows"]:,}',f'{r["missing_bins"]:,}'])
    table(rows,[190,90,90,110])
    p('Within each split, input gaps are forward-filled, with each sensor\'s training median as the fallback at the beginning of a split (zero only if that sensor has no training observations). Validation/test values never enter training statistics. Missing targets are excluded from the loss and every reported error; filled targets are never treated as labels.')
    fig('q3_week.png','Figure 2. Complete calendar week, 3-9 February 2020. Sensors are predetermined nodes 0,358,715; gaps remain visible in the observed series.',maxheight=180)

    page('Q3(b-c) - Road graph and neighborhoods')
    h('Graph construction and direction')
    p(f'The SD matrix is 716×716. It contains <b>{prep["nonzero_including_diagonal"]:,}</b> nonzero ordered entries including <b>{prep["self_loops"]}</b> diagonal entries, or <b>{prep["directed_nonself_edges"]:,}</b> non-self directed entries. It is <b>{"symmetric" if prep["symmetric"] else "not symmetric"}</b>; max |A-Aᵀ|={prep["max_asymmetry"]:.6f}. Non-self row degree ranges from {prep["degree_min"]} to {prep["degree_max"]}, with mean {prep["degree_mean"]:.2f}. Ordered entries are reported rather than silently halving a directed edge count.')
    p('A nonzero A<sub>ij</sub> is a retained weighted road-distance relationship. LargeST uses a thresholded Gaussian of shortest driving distance [4]; it is a strength, rather than the distance itself. Our multiplication convention is that row i receives the feature of node j with this weight. This convention need not mean vehicles travel from j to i.')
    fig('q3_graph.png','Figure 3. Adjacency in the verified sensor order and distribution of non-self row neighbors. Raw self-loops are included in the heatmap and excluded from the degree count.',maxheight=225)
    h(f'Example: sensor {prep["neighborhood_id"]} (node {prep["neighborhood_node"]})')
    p('The example is selected using only the road graph: the first node whose non-self degree is closest to the median. The lists below follow the assignment literally: N(i)={j:Aij≠0}. Thus a provided self-loop includes the focal sensor itself; non-self degrees in the histogram exclude it.')
    p('<b>One-hop sensor IDs:</b> '+', '.join(map(str,prep['literal_one_hop_ids']))+'.')
    p('<b>At-most-two-hop sensor IDs:</b> '+', '.join(map(str,prep['literal_at_most_two_hop_ids']))+'.')
    p('Nearby sensors can reveal upstream congestion, downstream bottlenecks and common demand changes before the local series fully reflects them. One layer combines immediate row neighbors. Two layers can propagate information along paths of length two, while repeated self-loops also retain shorter paths. More layers can blend distinct local states (oversmoothing), compress many distant signals (oversquashing), increase noise and overfit; greater depth is therefore not automatically better.')

    page('Q4 - Spatio-temporal forecasting')
    h('(a) Forecast windows and normalization')
    p('Each split of length T supplies exactly T-11 starts s=0,...,T-12. Inputs are indices s,...,s+7 and targets s+8,...,s+11, with shapes B×8×716×1 and B×4×716×1. No input or target crosses a split boundary; even validation/test history begins inside its own split.')
    p(f'A single global mean <b>{prep["mean"]:.5f}</b> and standard deviation <b>{prep["std"]:.5f}</b> are computed from the filled 2020 input series only. All models receive (x-mean)/std. Predictions are inverted before MAE and RMSE; using one global scale keeps every valid sensor-count error equally weighted in the loss.')
    h('(b-c) Architecture and controlled baseline')
    table([['Stage','Operation / shape'],['Temporal encoder','Shared Conv2d(1,16,kernel=(1,3),padding=(0,1)), ReLU; Conv2d(16,16,(1,3),same padding), ReLU. Convolution spans time only.'],['Temporal representation','Flatten each sensor\'s 16×8 feature map, then Linear(128,16), yielding Z in B×716×16.'],['Spatial layer','H=ReLU(Â Z Wg), Wg in 16×16 with no bias. One node-mixing layer.'],['Forecast head','Linear(16,4) per sensor, reshaped to B×4×716×1. Predict all horizons directly.'],['Temporal-only','Same encoder, channel projection Wg, nonlinearity and head, with identity node mixing. No neighbor exchange.']],[110,370])
    p(f'Temporal/fixed: <b>{metrics["fixed"]["parameters"]:,}</b> parameters. Adaptive/hybrid: <b>{metrics["adaptive"]["parameters"]:,}</b>, including two 716×8 embeddings. Shared encoder, projection and head initial weights are identical across all experiments.')
    h('Training and model selection')
    p('All runs use seed 25280025, FP32, Adam (lr 0.002, weight decay 0.0001), batch 64, norm clipping at 5, maximum 20 epochs and patience 5. Every training window is used, with identical epoch-seeded shuffles. Minimum full-validation MAE selects the checkpoint. Test data never select learning rates or architecture. Fixed seeds do not guarantee bitwise XPU determinism.')
    p('Loss is masked MAE in standardized units. Original-unit metrics pool errors over valid targets, excluding filled labels and retaining real zeros. Overlapping-window horizon forecasts count separately, with the same mask for all models. Every final metric uses the validation-selected checkpoint.')
    fig('q6_learning.png','Figure 4. Validation learning curves and test MAE at 15,30,45,60 minutes. The latter uses each model\'s validation-selected checkpoint.',maxheight=200)

    page('Q5 - Fixed, adaptive and hybrid adjacency')
    h('(a) Fixed graph')
    p('We use <b>Â<sub>fixed</sub>=D<sup>-1</sup>(A+I)</b>, where D contains row sums. This directed normalization makes each receiving row a nonnegative weighted average. It avoids applying a symmetric-graph formula to a potentially asymmetric graph. The supplied diagonal is retained and I is added literally: an existing unit self-loop therefore becomes weight two before normalization. The fixed matrix is a nontrainable buffer and never changes during optimization.')
    h('(b) Learnable graph')
    p('<b>Â<sub>adaptive</sub>=rowSoftmax(ReLU(E<sub>1</sub>E<sub>2</sub>ᵀ))</b>, with two independently initialized 716×8 embeddings (normal, std=0.1). Gradients flow from the forecasting loss through softmax, positive ReLU logits and the embedding product. Negative logits have zero ReLU gradient on that step. All 716 neighbors are retained; no top-k sparsification or discontinuous neighbor selection is used in training.')
    h('(c) Hybrid graph')
    p('<b>Â<sub>hybrid</sub>=0.5Â<sub>fixed</sub>+0.5Â<sub>adaptive</sub></b>. Both terms are already row stochastic, so their convex combination is normalized consistently and also sums to one per row. The hybrid graph has its own learned embeddings, trained under the same protocol. Alpha is fixed at 0.5 rather than selected using test performance.')
    fig('q5_adjacencies.png','Figure 5. Final node-mixing matrices for the validation-selected checkpoints. Each panel has its own colorbar so weak adaptive weights remain visible; visual brightness is not directly comparable across panels.',maxheight=180)
    rows=[['Mode','Val MAE','Val RMSE','Test MAE','Test RMSE','Epochs / best']]
    for mode in ['temporal','fixed','adaptive','hybrid']:
        r=metrics[mode];rows.append([mode.title(),f'{r["validation"]["mae"]:.2f}',f'{r["validation"]["rmse"]:.2f}',f'{r["test"]["mae"]:.2f}',f'{r["test"]["rmse"]:.2f}',f'{r["epochs_run"]} / {r["best_epoch"]}'])
    table(rows,[85,72,80,72,80,91])
    baseline=metrics['temporal']['test']['mae']
    deltas={mode:(baseline-metrics[mode]['test']['mae'])/baseline*100 for mode in ['fixed','adaptive','hybrid']}
    p('Relative test-MAE reductions versus temporal-only: '+', '.join(f'<b>{mode} {delta:+.2f}%</b>' for mode,delta in deltas.items())+'. A negative reduction means higher error. Useful neighbor information must outweigh mixing losses and graph-estimation error. Keeping the local temporal signal only can win when this single graph layer mixes unlike sensor states too strongly.')

    page('Q6(a) - Forecasting comparison')
    p('The final comparison table on page 1 reports original-unit MAE, RMSE and synchronized training time. Figure 6 compares actual measurements with all four saved models for the same three predetermined sensors and dates. These are 15-minute-ahead forecasts; the table below also reports longer horizons over the entire test period.')
    fig('q6_predictions.png','Figure 6. Predicted and observed volume at sensors chosen by graph order (0,358,715), using input starts on 12-13 April 2021. Every line uses the same timestamps; missing observed labels are left as gaps.',maxheight=350)
    rows=[['Model','15-min MAE','30-min MAE','45-min MAE','60-min MAE']]
    for mode in ['temporal','fixed','adaptive','hybrid']:
        rows.append([mode.title()]+[f'{v:.2f}' for v in metrics[mode]['test']['horizon_mae']])
    table(rows,[100,95,95,95,95])
    p('MAE=(Σ m|ŷ-y|)/(Σm); RMSE=sqrt((Σm(ŷ-y)²)/(Σm)). Here m is the original-label validity mask, and sums include every permitted window, sensor and forecast horizon. Per-batch sums use FP32 on Arc and accumulate in CPU FP64. The same dates, masks and normalization are used for every model.')
    p('RMSE gives greater influence to large failures than MAE. Missingness may be nonrandom, so these scores characterize the observed subset of sensor-bins and cannot measure performance on unavailable ground truth. One seed and one small architecture limit any claim about the general superiority of spatial modeling.')

    page('Q6(b) - What did the learned graph capture?')
    p('For each node, select its ten strongest learned non-self edges, giving 7,160 ordered edges. This is a post-training diagnostic only. Compare exact edge overlap with the original non-self road graph and great-circle separation from sensor coordinates. Distances are geographic, not road-network path distances.')
    ref=graphs['reference']
    rows=[['Graph / reference','Road overlap','Mean km','Median km','Within 5 km']]
    for mode in ['adaptive','hybrid']:
        r=graphs[mode];rows.append([mode.title()+' learned component',f'{100*r["top10_road_overlap"]:.2f}%',f'{r["top10_mean_distance_km"]:.2f}',f'{r["top10_median_distance_km"]:.2f}',f'{100*r["top10_under_5km_fraction"]:.2f}%'])
    rows.append(['Uniform non-self pair reference',f'{100*ref["road_nonself_density"]:.2f}%',f'{ref["all_pairs_mean_distance_km"]:.2f}','-',f'{100*ref["all_pairs_under_5km_fraction"]:.2f}%'])
    table(rows,[176,76,67,71,90],small=True)
    p('The hybrid row in this table analyzes its learned component alone, rather than the mixed matrix that mechanically contains fixed-road edges. The uniform-pair reference describes all 716×715 off-diagonal possibilities; it is not a separately fitted random graph. Selection ties are resolved deterministically by NumPy index sorting.')
    r=graphs['adaptive']
    p(f'The adaptive top-ten links overlap the road graph at {100*r["top10_road_overlap"]:.2f}%, versus its density of {100*ref["road_nonself_density"]:.2f}%. Their mean geographic separation is {r["top10_mean_distance_km"]:.2f} km, versus {ref["all_pairs_mean_distance_km"]:.2f} km across all non-self pairs. They are {"more local on average" if r["top10_mean_distance_km"]<ref["all_pairs_mean_distance_km"] else "not more local on average"} than uniform pairs, but much less local than road edges (mean {ref["road_edges_mean_distance_km"]:.2f} km). The map shows many strongest links converging on a common eastern anchor rather than tracing individual road segments.')
    p(f'Mean adaptive row entropy is {r["mean_row_entropy"]:.3f} nats (uniform would be {r["uniform_entropy"]:.3f}); mean self-weight is {r["mean_self_weight"]:.5f}. Ranking small differences in a diffuse row can overstate link importance, so top-ten membership alone should not be interpreted as strong evidence of physical connectivity.')
    fig('q6_learned_map.png','Figure 7. Strongest non-self adaptive link for each of 60 evenly spaced node indices, plotted at actual coordinates. Straight segments show learned associations, not road routes.',maxheight=255)
    p('Connections absent from the road graph are possible because every pair has a trainable embedding-product logit. They can represent common commuting demand or synchronized usage patterns. They can also reflect confounding, noise or overfitting. Better held-out forecasts would support their predictive usefulness, while causal or physical meaning needs independent evidence.')

    page('Q6(c) and Q7 - Discussion and reproducibility')
    h('Interpretation and limitations')
    p('Lower forecasting error does not establish actual road connectivity: many graphs can generate similar predictions, node embeddings are not uniquely identifiable, and correlated demand can connect distant roads. A fixed graph supplies a useful structural prior and may generalize with fewer parameters. An adaptive graph can discover missing associations but may overfit or average away distinctive sensor behavior. The hybrid can balance those effects; it is not guaranteed to dominate either term.')
    p('Global scaling can mix sensors with different lane capacities and flow levels; the single graph layer has no separate local residual path. Training-only per-sensor scaling and a local skip connection are reasonable follow-ups, not explanations established by this experiment. A static graph also misses rush-hour changes, incidents and propagation delays. Future work should test time-dependent or delayed messages, additional seeds, persistence and stronger temporal baselines. The required 2020-2021 split includes substantial demand shift; results are not a shuffled-split benchmark.')
    h('Hardware execution and measured NPU inference')
    p('All four networks are trained on Intel Arc through native PyTorch XPU kernels. This PyTorch installation has no Intel NPU autograd backend; OpenVINO supplies deployment inference. The NPU is selected explicitly, without AUTO or CPU fallback. Batch 16 failed NPU compilation with "Max striding levels exceeded"; static batch one succeeded. The trained learned graph is frozen only when exporting for inference.')
    rows=[['Mode','NPU MAE','NPU RMSE','Full-test s','Median batch ms']]
    for mode in ['temporal','fixed','adaptive','hybrid']:
        r=npu[mode];rows.append([mode.title(),f'{r["mae"]:.2f}',f'{r["rmse"]:.2f}',f'{r["test_seconds"]:.1f}',f'{r["median_batch_ms"]:.3f}'])
    table(rows,[100,90,90,95,105])
    p('Every NPU run evaluates the complete test set, at batch one and device-default FP16 execution. Full-test time includes window packing, inference and metric accumulation but excludes compilation and five warmups. Batch latency times the synchronous inference call and output copy. GPU latency uses batch 16 and resident inputs, so those latency measurements are not directly comparable. FP16 can slightly change errors; exact raw values and first-batch FP32 differences are retained in results/npu_metrics.json.','Caption')
    drivers='; '.join(f'{x["DeviceName"]}: {x["DriverVersion"]}' for x in hw['drivers'])
    p('Drivers: '+html.escape(drivers)+'. Python '+env['python'].split()[0]+'. Exact package versions, CPU/GPU descriptions, random seed, per-epoch histories, validity checks, graph arrays, predictions and checkpoints accompany the code.')
    h('Reproduction')
    p('Use an XPU-enabled PyTorch environment. Install requirements.txt without replacing the XPU wheel, then register its Jupyter kernel as ai651-xpu. Run src/download_data.py, execute notebooks 01 through 04 in order, and run src/build_report.py. The README includes exact commands, evaluation-only instructions and the pinned dataset provenance. Raw CA files and the large prepared cache are excluded from Git; download_data.py recreates them.')
    p('Checks verify sensor identity/order, complete date grids, exact split sizes, first/last window alignment, target masks, training-only normalization, row-stochastic matrices, nonzero adaptive gradients and identical shared initial weights. Executed notebooks and saved checkpoints make the measured outputs inspectable. No LMS upload was performed.')

    page('AI-use disclosure and references')
    h('Disclosure required by the assignment')
    p('<b>Tool:</b> OpenAI Codex. Generative AI was used extensively to produce the mathematical explanations, source implementation, data pipeline, experiments, plots, notebook structure and this report. This submission should not be represented as unaided student work.')
    p('<b>User prompt (verbatim):</b>')
    prompt='You are an excellent standford Deep learning instructor who teaches Deel Learning for space time and graph, complete this assignment as per instructions and attached assignment document. For training and inference use intel Arc 140T gpu on this machine using pytorch optimizied libraries for intel - they are already installed and available Use Intel builtin NPU where you can for again training and inference. Upload the source code on github I have already logged in to the builtin browser https://github.com/msbasit Name: Muhammad Sami Ul Basit Roll No: 25280025.'
    p(html.escape(prompt))
    p('<b>AI output:</b> the complete source in src/, four executed notebooks, measured experiment artifacts and checkpoints, seven figures, README reproduction instructions, and this PDF. The generated code was executed on the stated hardware; tabulated results come from saved measurements rather than invented examples.')
    p('<b>Edits and verification:</b> during agent-led development, Intel XPU execution replaced the default CPU-only interpreter; dataset downloads used a pinned public mirror after Kaggle 404 responses; metrics use FP32 device reductions with FP64 CPU accumulation because Arc lacks native FP64; NPU batch size was reduced from 16 to one after a measured compiler failure. Automated checks and PDF rendering were used to verify the final artifacts. These are AI-agent edits, not claimed human corrections.')
    p('<b>Human edits:</b> none recorded in this chat. The student should read, understand and independently review the derivations, code and results before submitting and be able to explain them in the course viva. This statement records authorship assistance; it does not assert compliance with additional course-outline rules that were not supplied.')
    h('References and data provenance')
    refs=[('[1] Official LargeST repository, SD extraction and preprocessing notebooks','https://github.com/liuxu77/LargeST'),
          ('[2] Public raw-data mirror (pinned revision and SHA-256 in download_provenance.json)','https://huggingface.co/datasets/ttydsc30/TrafficForecast-LargeST-Project-Dataset'),
          ('[3] Caltrans PeMS Traffic Data Downloads: five-minute station Total Flow definition','https://pems.dot.ca.gov/?dnode=Clearinghouse'),
          ('[4] Liu et al. (2023), LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting, NeurIPS Datasets and Benchmarks','https://arxiv.org/abs/2306.08259'),
          ('[5] PyTorch: Getting Started on Intel GPU','https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html'),
          ('[6] OpenVINO: NPU Device','https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html')]
    for label,url in refs:
        p(f'{html.escape(label)}<br/><link href="{html.escape(url)}" color="#007b8a">{html.escape(url)}</link>','Caption')
    p('LargeST is released under CC BY-NC 4.0 [1]. Raw dataset files are not redistributed in this repository. The small derived SD adjacency and sensor metadata retain their source attribution and noncommercial data license. Source code is provided for academic reproducibility.')
    doc=SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),rightMargin=50,leftMargin=50,topMargin=43,bottomMargin=50,
                         title='AI651 Assignment 3 - Muhammad Sami Ul Basit (25280025)',author='Muhammad Sami Ul Basit; AI assistance disclosed')
    doc.build(STORY,onFirstPage=footer,onLaterPages=footer)
    print(OUT)

if __name__=='__main__':
    main()
