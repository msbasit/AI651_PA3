"""Generate a conventional, native LaTeX report from measured experiment records."""
import html
from html.parser import HTMLParser
import json
import re
import shutil
import subprocess
from pathlib import Path
from common import ROOT
from report_content import content

REPORT = ROOT/'report'
BUILD = ROOT/'tmp'/'latex'
NAME = 'PA3_25280025_Muhammad_Sami_Ul_Basit.pdf'
REFS = ['largest-repo','mirror','pems','largest-paper','pytorch','openvino']

def escape(text):
    substitutions={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
        '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
        '×':r'\ensuremath{\times}','Σ':r'\ensuremath{\sum}','ρ':r'\ensuremath{\rho}',
        'ϕ':r'\ensuremath{\phi}','φ':r'\ensuremath{\phi}','α':r'\ensuremath{\alpha}',
        '²':r'\ensuremath{{}^2}','ᵀ':r'\ensuremath{{}^{\mathsf T}}',
        '≠':r'\ensuremath{\ne}','∈':r'\ensuremath{\in}','Â':r'\ensuremath{\widehat A}',
        'ŷ':r'\ensuremath{\widehat y}','x̂':r'\ensuremath{\widehat x}'}
    result=''.join(substitutions.get(c,c) for c in text)
    result=result.replace('–','--').replace('—','---').replace('−','-').replace('…',r'\ldots{}')
    result=result.replace('\u0302','')
    for i,key in enumerate(REFS,1):
        result=result.replace(f'[{i}]',r'\cite{'+key+'}')
    return result

class HtmlToTex(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in ['b','strong']: self.parts.append(r'\textbf{')
        elif tag=='sub': self.parts.append(r'\textsubscript{')
        elif tag=='sup': self.parts.append(r'\textsuperscript{')
        elif tag=='br': self.parts.append(r'\newline ')
        elif tag=='link': self.parts.append(r'\href{'+dict(attrs)['href']+'}{')
    def handle_endtag(self,tag):
        if tag in ['b','strong','sub','sup','link']: self.parts.append('}')
    def handle_data(self,data): self.parts.append(escape(data))

def convert(text):
    parser=HtmlToTex();parser.feed(str(text));return ''.join(parser.parts)

# Main mathematical arguments are native TeX, with display equations and labels.
# The remaining prose/tables come from the same measured records as the original.
def math_paragraph(text):
    overrides=[
    ('For scalar or vector w,c',r'''Let \(n_v=|\mathcal N(v)|\) and \(S_v=\sum_{u\in\mathcal N(v)}x_u\). For scalar or vector parameters \(w,c\),
\begin{equation}\label{eq:affine}
\sum_{u\in\mathcal N(v)}\phi(x_u)
=\sum_{u\in\mathcal N(v)}(wx_u+c)=wS_v+cn_v.
\end{equation}
Equal \(n_v\) and \(S_v\) therefore force the same aggregate for every \(w,c\); applying any function \(\rho\) preserves this equality. For example, the neighborhoods \(\{-1,+1\}\) and \(\{0,0\}\) both have \(n_v=2\) and \(S_v=0\), but contain zero and two steady neighbors, respectively.'''),
    ('Set B=K+1',r'''Choose base \(B=K+1\) and assign
\begin{equation}
\phi(-1)=1,\qquad \phi(0)=B,\qquad \phi(+1)=B^2.
\end{equation}
If the three counts are \((n_-,n_0,n_+)\), their aggregate is
\begin{equation}q=n_-+Bn_0+B^2n_+.\end{equation}
Each count is at most \(K<B\), so no carries occur. The counts are the base-\(B\) digits:
\begin{align}
n_-&=q\bmod B, & n_0&=\left\lfloor\frac qB\right\rfloor\bmod B,
& n_+&=\left\lfloor\frac q{B^2}\right\rfloor\bmod B.
\end{align}'''),
    ('For K=8, B=9',r'''For \(K=8\), the base is \(B=9\). Exhaustive NumPy enumeration checks
\begin{equation}\binom{K+3}{3}-1=\binom{11}{3}-1=164\end{equation}
nonempty count triples with total at most eight. Every code is distinct and every triple decodes exactly. To count only non-steady neighbors at any degree, choose
\begin{equation}\phi(x)=x^2\quad\Longrightarrow\quad\sum_{u\in\mathcal N(v)}\phi(x_u)=n_-+n_+.\end{equation}'''),
    ('Write the count vector',r'''Write the count vector as \((s,z,p)\), with \(n=s+z+p\). Coordinates refer to slowing, steady and speeding up, respectively. All counterexamples below satisfy \(1\le n\le8\).'''),
    ('For an impulse at sensor 20',r'''For an impulse at sensor 20, the predictions at sensors \((19,20,21)\) are \((b,a,b)\), while the targets are \((1,0,0)\). All other predictions and targets vanish. Hence
\begin{align}
\mathcal L(a,b)&=(b-1)^2+a^2+b^2,\\
\frac{\partial\mathcal L}{\partial a}&=2a=0,
&\frac{\partial\mathcal L}{\partial b}&=4b-2=0.
\end{align}
The unique minimizer is \(a=0\), \(b=\tfrac12\), with \(\mathcal L_{\min}=\tfrac12\). Its Hessian \(\operatorname{diag}(2,4)\) is positive definite.'''),
    ('Choose <b>a=0, w',r'''Choose
\begin{equation}a=0,\qquad w_-=0,\qquad w_+=1.\end{equation}
Then \(\widehat x_i(t+1)=x_{i+1}(t)\), exactly the required motion for every input. A PyTorch \texttt{Conv1d} cross-correlation uses
\begin{equation}\underbrace{[\,0,\ 0,\ 1\,]}_{\text{kernel}}\quad\text{on}\quad[x_{i-1},x_i,x_{i+1}],\end{equation}
with circular padding of one and no bias. A mathematical convolution reverses its stored-kernel convention. In part (a), sensors 19 and 21 are forced to receive the same prediction \(b\), so the asymmetric target \((1,0)\) cannot be fitted. Separate \(w_-\) and \(w_+\) remove this constraint. Direction labels therefore break the reflection ambiguity that persists through stacks of unlabeled symmetric layers.'''),
    ('Choose <b>a=0, b=1, c=-1',r'''Choose \(a=0\), \(b=1\), \(c=-1\), so that
\begin{equation}\widehat x_i(t+1)=x_{i-1}(t)+x_{i+1}(t)-x_i(t-1).\end{equation}
For backward motion, \(x_{i-1}(t)=x_i(t-1)\); substitution gives
\begin{equation}\widehat x_i(t+1)=\cancelterm{x_{i-1}(t)}+x_{i+1}(t)-\cancelterm{x_i(t-1)}=x_{i+1}(t).\end{equation}
For forward motion, \(x_{i+1}(t)=x_i(t-1)\), so the same coefficients give
\begin{equation}\widehat x_i(t+1)=x_{i-1}(t)+\cancelterm{x_{i+1}(t)}-\cancelterm{x_i(t-1)}=x_{i-1}(t).\end{equation}
Since \(x_{i-1}(t)\) is exactly the forward-motion target, the same coefficients \((a,b,c)=(0,1,-1)\) still work.'''),
    ('PeMS defines five-minute',r'''PeMS defines five-minute total flow in vehicles per five minutes~\cite{pems}. We therefore preserve interval volume by summing:
\begin{equation}\label{eq:resample}
X^{(15\,\mathrm{min})}_t=\sum_{k=0}^{2}X^{(5\,\mathrm{min})}_{3t+k}.
\end{equation}
The resulting unit is vehicles per 15-minute interval. This follows the assignment's count-volume option. The official benchmark notebook averages and rounds~\cite{largest-repo}; our errors consequently use a different physical scale and should not be compared directly with that benchmark's reported values.'''),
    ('A nonzero A<sub>ij</sub>',r'''A nonzero \(A_{ij}\) is a retained road-distance weight. LargeST constructs strengths using a thresholded Gaussian of shortest driving distance~\cite{largest-paper}:
\begin{equation}
A_{ij}=\begin{cases}\exp(-d_{ij}^2/\sigma^2),&\exp(-d_{ij}^2/\sigma^2)\ge r,\\0,&\text{otherwise}.\end{cases}
\end{equation}
The entry is a strength, not the distance itself. In our multiplication convention, row \(i\) receives node \(j\)'s features with this weight; this does not necessarily mean that vehicles travel from \(j\) to \(i\).'''),
    ('The example is selected using only',r'''The example uses only the road graph: the first node whose non-self degree is closest to the median. The following lists implement the assignment's definition literally,
\begin{equation}\mathcal N(i)=\{j:A_{ij}\ne0\}.\end{equation}
A provided self-loop therefore includes the focal sensor itself; the degree histogram excludes self-loops.'''),
    ('Each split of length T supplies',r'''A split of length \(T\) supplies exactly \(T-11\) windows, indexed by \(s=0,\ldots,T-12\):
\begin{align}
X_s&=[x_s,x_{s+1},\ldots,x_{s+7}],\\
Y_s&=[x_{s+8},x_{s+9},x_{s+10},x_{s+11}].
\end{align}
Thus \(X_s\in\mathbb R^{8\times716\times1}\), \(Y_s\in\mathbb R^{4\times716\times1}\). Batched tensors add a leading dimension \(B\). No input or target crosses a split boundary; validation and test history also begins within its own split.'''),
    ('We use <b>Â',r'''The fixed graph uses directed row normalization:
\begin{equation}\label{eq:fixed}
\widehat A_{\mathrm{fixed}}=D^{-1}(A_{\mathrm{off}}+I),\qquad D_{ii}=\sum_j(A_{\mathrm{off}}+I)_{ij}.
\end{equation}
Here \(A_{\mathrm{off}}=A-\operatorname{diag}(\operatorname{diag}A)\). Clearing the supplied diagonal before adding \(I\) gives exactly one unit self-loop. Each receiving row is a nonnegative weighted average; directed normalization avoids assuming symmetry. The raw graph remains unchanged for Section~\ref{sec:q3}; the normalized model graph is a nontrainable buffer throughout optimization.'''),
    ('<b>Â<sub>adaptive</sub>',r'''Introduce two trainable matrices \(E_1,E_2\in\mathbb R^{716\times10}\), independently initialized from \(\mathcal N(0,1)\). Define
\begin{align}
L&=\operatorname{ReLU}(E_1E_2^{\mathsf T}),\\
(\widehat A_{\mathrm{adaptive}})_{ij}&=\frac{\exp L_{ij}}{\sum_{k=1}^{716}\exp L_{ik}}.
\end{align}
The standard-normal initialization produces nontrivial logits rather than an almost uniform initial graph. Gradients flow through row softmax, positive ReLU logits and the embedding product; negative logits have zero ReLU gradient on that step. All 716 neighbors are retained, with no top-\(k\) training sparsification.'''),
    ('<b>Â<sub>hybrid</sub>',r'''Combine the consistently normalized matrices using
\begin{equation}\label{eq:hybrid}
\widehat A_{\mathrm{hybrid}}=\alpha\widehat A_{\mathrm{fixed}}+(1-\alpha)\widehat A_{\mathrm{adaptive}},\qquad\alpha=0.5.
\end{equation}
Both terms are row stochastic, so their convex combination remains normalized. The hybrid has its own learned embeddings under the same training protocol. The mixing coefficient is fixed, rather than selected using test performance.'''),
    ('MAE=(Σ',r'''Let \(m_{s,h,i}\) indicate an originally valid target, and let \(C=\sum_{s,h,i}m_{s,h,i}\). With predictions returned to physical units,
\begin{align}
\operatorname{MAE}&=\frac{1}{C}\sum_{s,h,i}m_{s,h,i}\left|\widehat y_{s,h,i}-y_{s,h,i}\right|,\\
\operatorname{RMSE}&=\sqrt{\frac{1}{C}\sum_{s,h,i}m_{s,h,i}\left(\widehat y_{s,h,i}-y_{s,h,i}\right)^2}.
\end{align}
The sums cover every permitted window, sensor and forecast horizon. Per-batch sums use FP32 on Arc and accumulate in CPU FP64. All models share dates, masks and normalization.'''),
    ]
    for prefix,result in overrides:
        if text.startswith(prefix): return result
    return None

def paragraph(text,style='Body'):
    special=math_paragraph(text)
    if special: return special+'\n\n'
    # Native notation for normalization, retaining the measured values in prose.
    result=convert(text)
    result=result.replace(r'max |A-A\ensuremath{{}^{\mathsf T}}|=1.000000',r'\(\max_{i,j}|A_{ij}-A_{ji}|=1.000000\)')
    if text.startswith('Per-sensor training statistics'):
        result+='\n'+r'\begin{equation}\widetilde x_{t,i}=\frac{x_{t,i}-\mu_i}{\sigma_i},\qquad \widehat y_{s,h,i}=\sigma_i\widehat{\widetilde y}_{s,h,i}+\mu_i.\end{equation}'
    if text.startswith('The loss is masked MAE after per-sensor'):
        result+='\n'+r'\begin{equation}\mathcal L_{\mathcal B}=\frac{1}{s_{\mathrm{train}}C_{\mathcal B}}\sum_{(s,h,i)\in\mathcal B}m_{s,h,i}\,\sigma_i\left|\widehat{\widetilde y}_{s,h,i}-\widetilde y_{s,h,i}\right|.\end{equation}'+'\n'+r'Here \(C_{\mathcal B}\) counts valid batch targets, and \(s_{\mathrm{train}}\) is one scalar standard deviation of the filled training counts, shared by every model.'
    if style=='Caption': return '{\\small '+result+'}\n\n'
    return result+'\n\n'

CELL_MATH={
    'ρ(s,z,p)=z.':r'\(\rho(s,z,p)=z\).',
    'Test 2s &gt; s+z+p.':r'Test \(2s>s+z+p\).',
    'Test first coordinate &gt; 1/2.':r'Test \(s/n>\tfrac12\).',
    'Test first coordinate s &gt; 0.':r'Test \(s>0\).',
    'Test first coordinate s/n &gt; 0 (n is nonzero).':r'Test \(s/n>0\), since \(n>0\).',
    'Test first coordinate = 1.':r'Test whether the first coordinate equals \(1\).',
    'Epochs / best':r'Run / best',
    'Median batch ms':r'Median batch (ms)',
    'Shared Conv2d(1,16,kernel=(1,3),padding=(0,1)), ReLU; Conv2d(16,16,(1,3),same padding), ReLU. Convolution spans time only.':r'Two temporal convolutions, \(1\to16\) then \(16\to16\) channels, each with a \(1\times3\) kernel, \((0,1)\) padding and ReLU. Node positions are not mixed.',
    'Temporal representation': 'Temporal representation',
    'Flatten each sensor\'s 16×8 feature map, then Linear(128,16), yielding Z in B×716×16.':r'Flatten each sensor\textquotesingle s \(16\times8\) feature map; apply \(\operatorname{Linear}(128,16)\), giving \(Z\in\mathbb R^{B\times716\times16}\).',
    'H=ReLU(Z Wlocal + Â Z Wg). The local signal is retained; both 16×16 projections have no bias.':r'\(H=\operatorname{ReLU}(ZW_{\mathrm{local}}+\widehat A ZW_g)\). Both \(16\times16\) projections have no bias. The local representation survives independently of graph averaging.',
    'Same temporal encoder, local projection and forecasting head. Omit the graph term entirely.':r'Same temporal encoder, local projection and head, with \(H=\operatorname{ReLU}(ZW_{\mathrm{local}})\). No graph term.',
    'Linear(16,4) per sensor, reshaped to B×4×716×1. Predict all horizons directly.':r'\(\operatorname{Linear}(16,4)\) per sensor, yielding \(\widehat Y\in\mathbb R^{B\times4\times716\times1}\). All horizons are predicted directly.'}

CAPTIONS=['Final test comparison in vehicles per 15-minute interval.',
    'Information retained by one-hot sum, mean and max aggregation.',
    'Chronological splits and valid-label availability.',
    'Shared forecasting architecture.',
    'Validation and test errors for validation-selected checkpoints.',
    'Test MAE by forecasting horizon.',
    'Learned top-ten non-self links and geographic reference.',
    'Full-test Intel AI Boost NPU inference (static batch one).']

def make_table(event,index):
    rows=event['rows'];n=len(rows[0])
    ragged=r'>{\raggedright\arraybackslash}'
    if n==2: spec='@{}'+ragged+r'p{.21\linewidth}'+ragged+'X@{}'
    elif n==4 and rows[0][0]=='Task': spec='@{}'+ragged+r'p{.16\linewidth}'+ragged+r'p{.10\linewidth}'+ragged+r'p{.11\linewidth}'+ragged+'X@{}'
    elif n==5 and 'Graph / reference' in rows[0]:spec='@{}'+ragged+r'p{.30\linewidth}'+(ragged+'X')*4+'@{}'
    else: spec='@{}'+(ragged+'X')*n+'@{}'
    result=[r'\begin{table}[H]',r'\centering\small',r'\caption{'+CAPTIONS[index]+'}',
            r'\renewcommand{\arraystretch}{1.16}',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule']
    for i,row in enumerate(rows):
        cells=[CELL_MATH.get(str(c),convert(str(c))) for c in row]
        if i==0:cells=[r'\textbf{'+c+'}' for c in cells]
        result.append(' & '.join(cells)+r' \\')
        if i==0:result.append(r'\midrule')
    result.extend([r'\bottomrule',r'\end{tabularx}',r'\end{table}'])
    return '\n'.join(result)+'\n\n'

PREAMBLE=r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=25mm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx,booktabs,tabularx,array,float,placeins}
\usepackage{microtype}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\graphicspath{{figures/}}
\setlength{\parindent}{1.2em}
\setlength{\parskip}{3pt}
\setlength{\emergencystretch}{2em}
\newcommand{\cancelterm}[1]{\underbrace{#1}_{\text{equal terms}}}
\hypersetup{pdftitle={AI651 Assignment 3: Traffic Forecasting on Graphs},pdfauthor={Muhammad Sami Ul Basit}}
\title{\textbf{Traffic Forecasting on Graphs}\\[4pt]\large AI651: Deep Learning for Space, Time and Graphs\\Assignment 3, Fall 2026}
\author{Muhammad Sami Ul Basit\\Roll No. 25280025\\Lahore University of Management Sciences}
\date{22 November 2026}
\begin{document}
\maketitle
\begin{center}\small Public repository: \url{https://github.com/msbasit/AI651_PA3}\end{center}
'''

def build_tex():
    events=content();out=[PREAMBLE];section='';table_index=0;refs=False;hardware=False
    for event in events:
        if event['kind']=='section':
            title=event['title'];section=title
            if title=='Traffic forecasting on graphs':continue
            if title.startswith('Q1 '):out.append(r'\FloatBarrier\section{What does a sum remember?}\label{sec:q1}')
            elif title.startswith('Q2 '):out.append(r'\FloatBarrier\section{Which way is the slowdown moving?}\label{sec:q2}')
            elif title.startswith('Q3(a)'):out.append(r'\FloatBarrier\section{Dataset preparation and graph construction}\label{sec:q3}\subsection{Data extraction and resampling}')
            elif title.startswith('Q3(b-c)'):out.append(r'\FloatBarrier\subsection{Road graph and neighborhoods}')
            elif title.startswith('Q4 '):out.append(r'\FloatBarrier\section{Spatio-temporal forecasting}\label{sec:q4}')
            elif title.startswith('Q5 '):out.append(r'\FloatBarrier\section{Fixed, adaptive and hybrid adjacency}\label{sec:q5}')
            elif title.startswith('Q6(a)'):out.append(r'\FloatBarrier\section{Experimental analysis and interpretation}\label{sec:q6}\subsection{Forecasting comparison}')
            elif title.startswith('Q6(b)'):out.append(r'\FloatBarrier\subsection{Learned graph structure}')
            elif title.startswith('Q6(c)'):out.append(r'\FloatBarrier\subsection{Critical discussion}')
            elif title.startswith('AI-use'):out.append(r'\FloatBarrier\appendix\section{AI Acknowledgement}')
        elif event['kind']=='paragraph':
            text=event['text'];style=event['style']
            if section=='Traffic forecasting on graphs':
                if text.startswith(('AI651 -','Assignment 3 |','<b>Muhammad','<b>Public source','Prepared 21','Revised 22','Scope and outcome','Final test comparison','Hardware')):continue
                if text.startswith('Questions 1-2'):out.append(r'\begin{abstract}'+convert(text)+'\n')
                elif text.startswith('The lowest measured'):out.append(convert(text)+r'\end{abstract}'+'\n')
                else:out.append(paragraph(text,style))
                continue
            if style=='Heading2':
                if text=='References and data provenance':refs=True;continue
                if text=='Disclosure required by the assignment':continue
                if section.startswith('Q6(c)') and text=='Interpretation and limitations':continue
                if section.startswith('Q6(c)') and text=='Hardware execution and measured NPU inference':
                    out.append(r'\section{Reproducibility and hardware execution}\label{sec:q7}\subsection{GPU training and NPU deployment}');hardware=True
                else:
                    heading=re.sub(r'^\([a-c](?:-[a-c])?\)\s*','',text)
                    level='subsubsection' if section.startswith('Q3') else 'subsection'
                    out.append('\\'+level+'{'+convert(heading)+'}\n')
                continue
            if refs:
                if text.startswith('LargeST is released'):out.append(paragraph(text))
                continue
            out.append(paragraph(text,style))
        elif event['kind']=='table':
            out.append(make_table(event,table_index));table_index+=1
        elif event['kind']=='figure':
            caption=re.sub(r'^Figure \d+\.\s*','',event['caption'])
            caption=caption.replace('exp(-(i-20)²/8)',r'INITIALBUMP')
            cap=convert(caption).replace('INITIALBUMP',r'\(\exp(-(i-20)^2/8)\)')
            height='.44' if 'predictions' in event['name'] else '.34'
            out.append(r'\begin{figure}[H]\centering'+'\n'+
                r'\includegraphics[width=\linewidth,height='+height+r'\textheight,keepaspectratio]{'+event['name']+'}\n'+
                r'\caption{'+cap+'}\n'+r'\end{figure}'+'\n')
    out.append(r'''\FloatBarrier
\begingroup\small
\begin{thebibliography}{9}\setlength{\itemsep}{2pt}\setlength{\parskip}{0pt}
\bibitem{largest-repo} X. Liu et al. \emph{LargeST: official repository and preprocessing notebooks}. \url{https://github.com/liuxu77/LargeST}.
\bibitem{mirror} \emph{Public LargeST raw-data mirror}. Pinned revision and SHA-256 values appear in \texttt{results/download\_provenance.json}. \url{https://huggingface.co/datasets/ttydsc30/TrafficForecast-LargeST-Project-Dataset}.
\bibitem{pems} Caltrans. \emph{PeMS Traffic Data Downloads: five-minute station total flow}. \url{https://pems.dot.ca.gov/?dnode=Clearinghouse}.
\bibitem{largest-paper} X. Liu et al. ``LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting.'' \emph{NeurIPS Datasets and Benchmarks}, 2023. \url{https://arxiv.org/abs/2306.08259}.
\bibitem{pytorch} PyTorch. \emph{Getting Started on Intel GPU}. \url{https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html}.
\bibitem{openvino} Intel. \emph{OpenVINO: NPU Device}. \url{https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html}.
\end{thebibliography}
\endgroup
\end{document}
''')
    return '\n'.join(out)

def main():
    REPORT.mkdir(exist_ok=True);BUILD.mkdir(parents=True,exist_ok=True)
    tex=REPORT/'report.tex';tex.write_text(build_tex(),encoding='utf-8')
    compiler=shutil.which('pdflatex')
    if not compiler:raise RuntimeError('Install a LaTeX distribution providing pdflatex; report/report.tex is ready to compile.')
    command=[compiler,'--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error',
             '-output-directory=tmp/latex','report/report.tex']
    for _ in range(2):
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (BUILD/'compiler_output.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:
            print((result.stdout+'\n'+result.stderr)[-7000:]);raise SystemExit(result.returncode)
    shutil.copyfile(BUILD/'report.pdf',ROOT/NAME)
    log=(BUILD/'report.log').read_text(encoding='utf-8',errors='replace')
    warnings=[line for line in log.splitlines() if any(x in line for x in ['Overfull','undefined references','Missing character'])]
    (ROOT/'results'/'latex_build.json').write_text(json.dumps(dict(engine='pdfLaTeX',passes=2,source='report/report.tex',warnings=warnings),indent=2))
    print(ROOT/NAME)
    print('LaTeX warnings:',warnings)

if __name__=='__main__':main()
