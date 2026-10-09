"""Post hoc sensitivity of archived test predictions only; no model inference or training."""
import csv, json
from pathlib import Path, PurePosixPath
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/sensitivity'
OUT.mkdir(parents=True,exist_ok=True)
def portable_path(value):
    return PurePosixPath(str(value).replace('\\','/'))
read=lambda p:list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig',newline='')))
flags=read('outputs/repro/skin31_test_flags.csv')
strict=np.array([r['strict']=='True' for r in flags])
strictrows=[r for r in flags if r['strict']=='True']
unknown=np.array([r['id'].startswith('ISIC_') and not r['lesion_id'] for r in strictrows])
assert len(strictrows)==863 and unknown.sum()==50
assert all(r['folder']=='nevus' for r,u in zip(strictrows,unknown) if u)
truth=np.array([portable_path(r['path']).parent.name.lower() for r in strictrows])
keys=[(portable_path(r['path']).parent.name.lower(),portable_path(r['path']).name) for r in strictrows]
origmapping=json.loads((ROOT/'prereg/class_mapping.json').read_text())['class_to_idx']
orig=np.array([k.lower() for k,v in sorted(origmapping.items(),key=lambda kv:kv[1])])
clean=np.array([name for name in orig if name!='seborrheic keratosis'])
ctest=[portable_path(r['dst']) for r in sorted(read('lists/skin31_clean/manifest.csv'),key=lambda r:r['dst']) if r['split']=='test' and r['dst']]
ckeys={(p.parent.name.lower(),p.name):i for i,p in enumerate(ctest)}
order=np.array([ckeys[k] for k in keys])
SEEDS=[42,123,2024]
correct={}
for c in ['1_b2_baseline','2_b2_trainopt','3_b2_fusion_teacher','4_b0_baseline','5_b0_trainopt','6_b0_kd_nonfusion','7_b0_kd_fusion']:
    correct[c]=np.stack([orig[np.load(ROOT/f'outputs/repro/{c}/seed{s}/skin31_test_fp32.npy').argmax(1)[strict]]==truth for s in SEEDS])
authrows=[r for r in read('outputs/authors_model/predictions.csv') if r['split']=='test']
authpred={(portable_path(r['path']).parent.name.lower(),portable_path(r['path']).name):r['pred'].lower() for r in authrows}
correct['authors']=np.stack([np.array([authpred[k] for k in keys])==truth])
correct['vit_original']=np.stack([orig[np.load(ROOT/f'outputs/vit/seed{s}.npz')['test'].argmax(1)[strict]]==truth for s in SEEDS])
for c in ['4_b0_baseline','5_b0_trainopt']:
    correct['clean_'+c]=np.stack([clean[np.load(ROOT/f'outputs/clean_eval/{c}_seed{s}_fp32.npy').argmax(1)[order]]==truth for s in SEEDS])
arms={'A':'outputs/milk_eval/logits/test_A','R':'outputs/relabel_eval/logits/test_R',
      'H':'outputs/deconf/logits/test_deconf','B':'outputs/milk_eval/logits/test_B',
      'C':'outputs/milk_eval/logits/test_C','vB':'outputs/vit_milk/logits/test_B','vC':'outputs/vit_milk/logits/test_C'}
for a,stem in arms.items():
    correct[a]=np.stack([clean[np.load(ROOT/(stem+f'_seed{s}.npy')).argmax(1)[order]]==truth for s in SEEDS])
assert np.array_equal(correct['A'],correct['clean_5_b0_trainopt'])
rows=[]
for k,c in correct.items():
    rows.append({'model':k,'n_checkpoints':len(c),'acc_all_863':float(c.mean()),
        'acc_excluding_unknown_ID_813':float(c[:,~unknown].mean()),
        'acc_unknown_ID_50':float(c[:,unknown].mean()),
        'per_seed_remaining':c[:,~unknown].mean(1).tolist()})
with (OUT/'isic_unknown_id_exclusion_levels.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
contrast_rows=[]
for x,y in [('clean_4_b0_baseline','4_b0_baseline'),('A','5_b0_trainopt'),('R','A'),('H','A'),('B','A'),('C','A'),('C','B'),('vC','vB')]:
    diff=correct[x].astype(float)-correct[y]
    for label,m in [('all_863',np.ones(863,dtype=bool)),('excluding_unknown_ID_813',~unknown)]:
        d=diff[:,m].mean(0)
        rng=np.random.default_rng(20260930)
        boots=np.stack([rng.integers(0,len(d),len(d)) for _ in range(2000)])
        lo,hi=np.percentile(d[boots].mean(1),[2.5,97.5])
        contrast_rows.append({'contrast':x+' - '+y,'cohort':label,'n_images':int(m.sum()),
            'delta':float(d.mean()),'per_seed':diff[:,m].mean(1).tolist(),'ci_low':float(lo),'ci_high':float(hi)})
with (OUT/'isic_unknown_id_exclusion_contrasts.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(contrast_rows[0]));writer.writeheader();writer.writerows(contrast_rows)
print(json.dumps({'levels':rows,'contrasts':contrast_rows},indent=2))
