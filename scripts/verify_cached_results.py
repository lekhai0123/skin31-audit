"""Image-free, CPU-only verification of released cached results.

This checks cached arrays and recomputes paired Derm7pt contrasts from their logits.
It does not train models, perform inference, download anything, or change archival outputs.
Requirements: NumPy and pandas. Outputs are written only when --out is provided.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2024]
DX = {'BCC':'basal cell carcinoma', 'MEL':'melanoma', 'NEV':'nevus',
      'SEK':'pigmented benign keratosis', 'DF':'dermatofibroma', 'VASC':'vascular lesion'}
ISIC = {'actinic keratosis','dermatofibroma','nevus','pigmented benign keratosis',
        'squamous cell carcinoma','vascular lesion'}
LABELS = ['E1','E2','E3','clin_recall_target','P1_all','untreated_derm_to_MEL',
          'NEV_derm_recall','MEL_derm_recall','macro_recall_derm','accuracy_derm']
ARMS = {'A':'milk_eval/logits/derm7pt_A', 'B':'milk_eval/logits/derm7pt_B',
        'C':'milk_eval/logits/derm7pt_C', 'H':'deconf/logits/derm7pt_deconf',
        'R':'relabel_eval/logits/derm7pt_R', 'vB':'vit_milk/logits/derm7pt_B',
        'vC':'vit_milk/logits/derm7pt_C'}

def stats(pc, pd_, cls, names):
    group = np.isin(names, list(ISIC))
    target = np.isin(cls,['BCC','MEL'])
    untreated = np.isin(cls,['NEV','SEK','DF','VASC'])
    def recall(pred,d):return (names[pred[cls==d]]==DX[d]).mean()
    e1=np.mean([recall(pd_,d) for d in ['BCC','MEL']])
    e2=group[pd_[target]].mean()-group[pc[target]].mean()
    e3=group[pd_[untreated]].mean()-group[pc[untreated]].mean()
    clinic=np.mean([recall(pc,d) for d in ['BCC','MEL']])
    p1=group[pd_].mean()-group[pc].mean()
    extra=[(names[pd_[untreated]]=='melanoma').mean(),recall(pd_,'NEV'),recall(pd_,'MEL')]
    return np.array([e1,e2,e3,clinic,p1]+extra+[np.mean([recall(pd_,d) for d in DX]),
        (names[pd_]==np.array([DX[d] for d in cls])).mean()])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,help='Write a verification summary to this directory')
    ap.add_argument('--bootstrap',action='store_true',help='Also recompute 2,000 lesion-bootstrap CIs')
    args=ap.parse_args()
    names=np.array([n.lower() for n in json.loads((ROOT/'lists/class_mapping_clean.json').read_text())['class_names']])
    cls=pd.read_csv(ROOT/'prereg/derm7pt_manifest.csv').cls.to_numpy()
    assert len(cls)==960 and len(names)==30
    checks=[]
    # Check all numeric caches without pickle; prohibit image-shaped arrays or object payloads.
    for p in sorted((ROOT/'outputs').rglob('*')):
        if p.suffix not in {'.npy','.npz'}:continue
        loaded=np.load(p,allow_pickle=False)
        if isinstance(loaded,np.ndarray):arrays={'array':loaded}
        else:arrays={k:loaded[k] for k in loaded.files}
        for key,z in arrays.items():
            assert z.ndim<=2 and z.dtype.kind in 'biuf', (p,key,z.shape,z.dtype)
            assert np.isfinite(z).all(), (p,key,'nonfinite')
        checks.append({'path':p.relative_to(ROOT).as_posix(),
                       'arrays':{k:{'shape':list(z.shape),'dtype':str(z.dtype)} for k,z in arrays.items()}})
        if not isinstance(loaded,np.ndarray):loaded.close()
    preds={};rows=[]
    for a,prefix in ARMS.items():
        for s in SEEDS:
            with np.load(ROOT/'outputs'/f'{prefix}_seed{s}.npz',allow_pickle=False) as z:
                assert z['clinic'].shape==z['derm'].shape==(960,30)
                pc,pd_=z['clinic'].argmax(1),z['derm'].argmax(1)
            preds[(a,s)]=(pc,pd_)
            values=stats(pc,pd_,cls,names)
            rows.append({'arm':a,'seed':s,**dict(zip(LABELS,values))})
    per_seed=pd.DataFrame(rows)
    contrasts=[]
    for tag,a,b,source,source_row in [('R-A','R','A','relabel_eval/derm7pt_results.csv','R-A'),
        ('H-A','H','A','deconf/derm7pt_primary.csv',None),
        ('C-B (B0)','C','B','milk_eval/derm7pt_results.csv','C-B'),
        ('C-B (ViT)','vC','vB','vit_milk/derm7pt_results.csv','C-B')]:
        values=np.array([stats(*preds[(a,s)],cls,names)-stats(*preds[(b,s)],cls,names) for s in SEEDS])
        lo=hi=np.full(len(LABELS),np.nan)
        if args.bootstrap:
            rng=np.random.default_rng(20260930);boots=[]
            while len(boots)<2000:
                idx=rng.integers(0,len(cls),len(cls))
                if any(not np.any(cls[idx]==d) for d in DX):continue
                boots.append(np.mean([stats(preds[(a,s)][0][idx],preds[(a,s)][1][idx],cls[idx],names)
                    -stats(preds[(b,s)][0][idx],preds[(b,s)][1][idx],cls[idx],names) for s in SEEDS],axis=0))
            lo,hi=np.percentile(boots,[2.5,97.5],axis=0)
        for k,label in enumerate(LABELS):
            contrasts.append({'contrast':tag,'endpoint':label,'delta':values[:,k].mean(),
                'seed42':values[0,k],'seed123':values[1,k],'seed2024':values[2,k],
                'ci_low':lo[k],'ci_high':hi[k]})
        archived=pd.read_csv(ROOT/'outputs'/source)
        if source_row is not None:archived=archived[archived.row==source_row]
        labels={'E1':'E1_derm_recall_target','E2':'E2_P1_target','E3':'E3_P1_control','P1_all':'S_P1_all'} if a=='C' else {}
        for k,label in enumerate(LABELS[:3]):
            archived_row=archived[archived.estimand==labels.get(label,label)]
            if len(archived_row)==0:
                # Historical HAM and R tables use the same definitions under longer labels.
                archived_row=archived[archived.estimand=={'E1':'E1_derm_recall_target','E2':'E2_P1_target','E3':'E3_P1_control'}[label]]
            assert len(archived_row)==1,(tag,label)
            assert np.isclose(float(archived_row.iloc[0]['delta']),values[:,k].mean(),rtol=0,atol=1e-12),(tag,label)
            if args.bootstrap:
                assert np.isclose(float(archived_row.iloc[0].ci_low),lo[k],rtol=0,atol=1e-12),(tag,label,'ci_low')
                assert np.isclose(float(archived_row.iloc[0].ci_high),hi[k],rtol=0,atol=1e-12),(tag,label,'ci_high')
    # Historical bundles remain bit-for-bit copies, including their hash lists.
    historical=[]
    for p in (ROOT/'prereg/registered').rglob('SHA256SUMS.txt'):
        for line in p.read_text().splitlines():
            digest,name=line.split(maxsplit=1);target=p.parent/name.lstrip('*')
            assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==digest,target
            historical.append(target.relative_to(ROOT).as_posix())
    summary={'numeric_cache_files':len(checks),'historical_hashes_checked':len(historical),
             'paired_contrasts':4,'primary_points_verified':12,'primary_cis_verified':12 if args.bootstrap else 0,
             'numpy_version':np.__version__,'pandas_version':pd.__version__}
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(contrasts).query("endpoint in ['E1','E2','E3','P1_all']").round(4).to_string(index=False))
    if args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (args.out/'numeric_cache_inventory.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        per_seed.to_csv(args.out/'derm7pt_per_seed.csv',index=False)
        pd.DataFrame(contrasts).to_csv(args.out/'derm7pt_contrasts.csv',index=False)

if __name__=='__main__':main()
