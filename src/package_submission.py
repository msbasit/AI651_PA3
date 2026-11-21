"""Bundle report, notebooks, source and reproducibility evidence without raw CA data."""
import zipfile
from common import ROOT

def main():
    target=ROOT.parent/'AI651_PA3_Muhammad_Sami_Ul_Basit_25280025.zip'
    excluded={'.git','tmp','__pycache__','.ipynb_checkpoints'}
    big={'ca_his_raw_2020.h5','ca_his_raw_2021.h5','ca_rn_adj.npy','ca_meta.csv','sd_prepared.npz'}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(ROOT.rglob('*')):
            rel=file.relative_to(ROOT)
            if not file.is_file() or any(x in excluded for x in rel.parts): continue
            if file.name in big or file.name.endswith('.ranges.json') or '_openvino.' in file.name: continue
            z.write(file,PathName(rel))
    print(target,'bytes:',target.stat().st_size)

def PathName(rel):
    return 'AI651_PA3_25280025/'+rel.as_posix()

if __name__=='__main__':
    main()
