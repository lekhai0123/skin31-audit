"""Build a blind review page for the 96 pHash candidate pairs between skin31 and the external sets (Derm7pt, PAD).
The earlier AI screening verdict is not shown. Output: outputs/dup_check/review_external/review.html"""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dup_check" / "review_external"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for ext in ["derm7pt", "pad"]:
    c = pd.read_csv(ROOT / "outputs" / "dup_check" / f"candidates_{ext}.csv")
    for r in c.itertuples():
        rows.append({"pair": len(rows), "set": ext, "ham": int(r.hamming),
                     "tl": f"{r.tgt_split} / {r.tgt_label}", "rl": f"skin31 {r.src_split} / {r.src_label}",
                     "tf": Path(r.tgt_path).name, "rf": Path(r.src_path).name,
                     "timg": os.path.relpath(r.tgt_path, OUT).replace("\\", "/").replace(" ", "%20"),
                     "rimg": os.path.relpath(r.src_path, OUT).replace("\\", "/").replace(" ", "%20")})
pd.DataFrame(rows).drop(columns=["timg", "rimg"]).to_csv(OUT / "pairs.csv", index=False)

html = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>External duplicate review</title><style>
:root{--bg:#fafafa;--card:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1e1e1e;--fg:#eee;--mut:#aaa;--line:#333}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--acc);color:#fff;cursor:pointer}
main{max-width:1100px;margin:0 auto;padding:12px 16px 80px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}
.c.done{border-left:5px solid var(--acc)} .meta{color:var(--mut);font-size:13px;margin-bottom:8px}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px} .imgs figure{margin:0}
.imgs img{width:100%;max-height:340px;object-fit:contain;background:#000;border-radius:6px}
figcaption{font-size:12px;color:var(--mut)} .opts{display:flex;gap:18px;margin-top:10px;flex-wrap:wrap} .opts label{cursor:pointer}
@media (max-width:640px){.imgs{grid-template-columns:1fr}}
</style></head><body>
<header><b>96 cặp ứng viên trùng: skin31 với Derm7pt/PAD</b><span id="prog"></span>
<input id="who" placeholder="Người chấm" size="12">
<button onclick="exp()">Tải kết quả (CSV)</button>
<span class="meta">Trái: ảnh external · Phải: ảnh skin31. Chọn <b>same</b> chỉ khi là cùng một ảnh hoặc cùng tổn thương của cùng người.</span></header>
<main id="list"></main>
<script>
const D=__DATA__;
const K='ext_review_v1';let S={};try{S=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
function save(){try{localStorage.setItem(K,JSON.stringify(S))}catch(e){};prog()}
function prog(){document.getElementById('prog').textContent=Object.keys(S).length+'/'+D.length+' đã chấm'}
const L=document.getElementById('list');
D.forEach(d=>{const c=document.createElement('div');c.className='c'+(S[d.pair]?' done':'');c.id='p'+d.pair;
c.innerHTML='<div class="meta">#'+d.pair+' · '+d.set+' · pHash Hamming '+d.ham+'</div><div class="imgs">'+
'<figure><img loading="lazy" src="'+d.timg+'"><figcaption>'+d.set+': '+d.tf+' ('+d.tl+')</figcaption></figure>'+
'<figure><img loading="lazy" src="'+d.rimg+'"><figcaption>'+d.rf+' ('+d.rl+')</figcaption></figure></div>'+
'<div class="opts">'+['same','different','uncertain'].map(v=>'<label><input type="radio" name="r'+d.pair+'" value="'+v+'"'+(S[d.pair]===v?' checked':'')+'> '+v+'</label>').join('')+'</div>';
c.querySelectorAll('input').forEach(i=>i.onchange=()=>{S[d.pair]=i.value;c.classList.add('done');save()});L.appendChild(c)});
prog();
function exp(){const who=document.getElementById('who').value||'';const t=new Date().toISOString().slice(0,10);
const lines=['pair,set,ext_file,skin31_file,rating,rater,date'].concat(D.map(d=>[d.pair,d.set,d.tf,d.rf,S[d.pair]||'',who,t].join(',')));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{type:'text/csv'}));a.download='external_review_rating.csv';a.click()}
</script></body></html>"""
(OUT / "review.html").write_text(html.replace("__DATA__", json.dumps(rows, ensure_ascii=False)), encoding="utf-8")
print(len(rows), "pairs ->", OUT / "review.html")
