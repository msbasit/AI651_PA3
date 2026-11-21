import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 160, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

def save_json(name, value):
    (ROOT/'results'/name).write_text(json.dumps(value, indent=2, allow_nan=False))

def figure(name):
    plt.savefig(ROOT/'figures'/name, bbox_inches='tight')
    plt.close()

def load_data():
    return np.load(ROOT/'data'/'sd_prepared.npz')
