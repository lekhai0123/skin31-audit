"""Build paper/refs.bib from Crossref / DataCite metadata (not from doi.org BibTeX, which is inconsistent).

Rules: year/month of the print issue when it exists (else the issued date); article numbers in `pages`;
page ranges with `--`; DOI only (no URL) for DOI-registered works; LaTeX-escaped accents; entries that have no DOI
(conference proceedings without Crossref records, Kaggle pages) come from paper/refs_manual.bib.
Entries that fail are reported, never invented."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFS = {
    "rafay2023": "10.1016/j.bspc.2023.104869",
    "pacheco2020": "10.1016/j.dib.2020.106221",
    "kawahara2019": "10.1109/JBHI.2018.2824327",
    "tschandl2018": "10.1038/sdata.2018.161",
    "milk10k_paper": "10.1016/j.jid.2025.06.1594",
    "abhishek2025": "10.1038/s41597-025-04382-5",
    "vollmer2025": "10.1111/ddg.15900",
    "degrave2021": "10.1038/s42256-021-00338-7",
    "degrave2023": "10.1038/s41551-023-01160-9",
    "zech2018": "10.1371/journal.pmed.1002683",
    "roberts2021": "10.1038/s42256-021-00307-0",
    "cassidy2022": "10.1016/j.media.2021.102305",
    "nauta2022": "10.3390/diagnostics12010040",
    "varoquaux2022": "10.1038/s41746-022-00592-y",
    "geirhos2020": "10.1038/s42256-020-00257-z",
    "winkler2019": "10.1001/jamadermatol.2019.1735",
    "daneshjou2022": "10.1126/sciadv.abq6147",
    "oakdenrayner2020": "10.1145/3368555.3384468",
    "media2026": "10.1016/j.media.2026.104258",
    "copycats2024": "10.52202/079017-3602",
    "bissoto2019": "10.1109/cvprw.2019.00335",
    "bissoto2020": "10.1109/cvprw50498.2020.00378",
    "cibm31": "10.1016/j.compbiomed.2025.110007",
    "jaad2026": "10.1016/j.jaad.2026.03.069",
}
PREPRINTS = {"mietkiewicz2025": "10.48550/arXiv.2502.02500"}  # only version; uses the same 31-class benchmark (checked 08/10/2026)
# datasets: DataCite metadata as registered
DATASETS = {"mendeley32": "10.17632/pgd42j3h5c.2", "pad_data": "10.17632/zr7vgbcyr2.1",
            "milk10k": "10.34970/648456", "ham_data": "10.7910/DVN/DBW86T"}
# Corrections where Crossref lacks print-issue data, verified against PubMed / publisher pages (04/10/2026)
OVERRIDES = {
    "degrave2023": {"year": "2025", "month": "mar"},          # PubMed: Nat Biomed Eng 2025 Mar;9(3):294-306
    "nauta2022": {"year": "2022", "month": None},              # MDPI: Diagnostics 2022, 12(1), 40 (online 24 Dec 2021)
    "winkler2019": {"pages": "1135--1141"},                    # PubMed: JAMA Dermatol 2019;155(10):1135-1141
    "bissoto2019": {"title": "{(De)Constructing Bias on Skin Lesion Datasets}"},  # CVF title
    'roberts2021': {'author': 'Roberts, Michael and Driggs, Derek and Thorpe, Matthew and Gilbey, Julian and Yeung, Michael and Ursprung, Stephan and Aviles-Rivero, Angelica I. and Etmann, Christian and McCague, Cathal and Beer, Lucian and Weir-McCall, Jonathan R. and Teng, Zhongzhao and Gkrania-Klotsas, Effrossyni and {AIX-COVNET} and Rudd, James H. F. and Sala, Evis and Sch{\\"o}nlieb, Carola-Bibiane'},  # primary-source preflight correction, 09/10/2026
    'copycats2024': {'author': "Jim{\\'e}nez-S{\\'a}nchez, Amelia and Avlona, Natalia-Rozalia and Juodelyte, Dovile and Sourget, Th{\\'e}o and Vang-Larsen, Caroline and Rogers, Anna and Zaj{\\k{a}}c, Hubert Dariusz and Cheplygina, Veronika"},  # primary-source preflight correction, 09/10/2026
    'oakdenrayner2020': {'author': "Oakden-Rayner, Luke and Dunnmon, Jared and Carneiro, Gustavo and R{\\'e}, Christopher"},  # primary-source preflight correction, 09/10/2026
    'mendeley32': {'version': '2', 'note': 'Version 2; [dataset]'},  # primary-source preflight correction, 09/10/2026
    'pad_data': {'version': '1', 'note': 'Version 1; [dataset]'},  # primary-source preflight correction, 09/10/2026
    'ham_data': {'note': 'Version 4.0; [dataset]'},  # primary-source preflight correction, 09/10/2026
}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
ACCENTS = {"á": "{\\'a}", "à": "{\\`a}", "ã": "{\\~a}", "â": "{\\^a}", "ä": '{\\"a}', "é": "{\\'e}", "è": "{\\`e}",
           "ê": "{\\^e}", "ë": '{\\"e}', "í": "{\\'i}", "ó": "{\\'o}", "ô": "{\\^o}", "ö": '{\\"o}', "õ": "{\\~o}",
           "ú": "{\\'u}", "ü": '{\\"u}', "ç": "{\\c{c}}", "ñ": "{\\~n}", "ø": "{\\o}", "å": "{\\aa}", "ł": "{\\l}",
           "Ö": '{\\"O}', "Ü": '{\\"U}', "Á": "{\\'A}", "É": "{\\'E}", "š": "{\\v{s}}", "ž": "{\\v{z}}",
           "č": "{\\v{c}}", "ć": "{\\'c}", "ś": "{\\'s}", "ń": "{\\'n}", "ğ": "{\\u{g}}", "ı": "{\\i}", "ę": "{\\k{e}}", "ą": "{\\k{a}}", "Ł": "{\\L}"}
DASHES = {"‐": "-", "‑": "-", "‒": "--", "–": "--", "—": "---", " ": " ", " ": " ",
          "’": "'", "‘": "`", "“": "``", "”": "''"}


def tex(s):
    s = re.sub(r"<[^>]+>", "", s or "")  # Crossref titles may contain JATS tags
    for a, b in {**DASHES, **ACCENTS}.items():
        s = s.replace(a, b)
    s = s.replace("&amp;", "\\&").replace("&", "\\&").replace("%", "\\%")
    return re.sub(r"\s+", " ", s).strip()


def get(url, accept="application/json"):
    r = subprocess.run(["curl", "-sL", "-H", f"Accept: {accept}", url], capture_output=True, text=True, encoding="utf-8")
    return r.stdout


def person(a):
    if "family" in a:
        given = re.sub(r"(?<=\.)(?=[A-Z]\.)", " ", a.get("given", ""))
        return f"{tex(a['family'])}, {tex(given)}".rstrip(", ")
    return "{" + tex(a.get("name", "")) + "}"  # group author


def crossref(key, doi):
    m = json.loads(get(f"https://api.crossref.org/works/{doi}"))["message"]
    date = (m.get("journal-issue", {}).get("published-print") or m.get("published-print") or m.get("issued"))["date-parts"][0]
    typ = {"journal-article": "article", "proceedings-article": "inproceedings"}.get(m["type"], "misc")
    f = {"author": " and ".join(person(a) for a in m.get("author", [])), "title": "{" + tex(m["title"][0]) + "}",
         "year": str(date[0])}
    if len(date) > 1:
        f["month"] = MONTHS[date[1] - 1]
    cont = tex((m.get("container-title") or [""])[0])
    if typ == "article":
        f["journal"] = cont
    elif typ == "inproceedings":
        f["booktitle"] = cont
    for src, dst in [("volume", "volume"), ("issue", "number")]:
        if m.get(src):
            f[dst] = tex(m[src])
    if m.get("page"):
        f["pages"] = tex(m["page"]).replace("--", "-").replace("-", "--")
    elif m.get("article-number"):
        f["pages"] = tex(m["article-number"])
    f["doi"] = doi
    return typ, f


def datacite(key, doi):
    m = json.loads(get(f"https://api.datacite.org/dois/{doi}"))["data"]["attributes"]
    authors = [
        (f"{tex(c['familyName'])}, {tex(c.get('givenName', ''))}" if c.get("familyName") else "{" + tex(c["name"]) + "}")
        for c in m["creators"]]
    f = {"author": " and ".join(authors), "title": "{" + tex(m["titles"][0]["title"]) + "}",
         "howpublished": tex(m.get("publisher") if isinstance(m.get("publisher"), str) else m["publisher"]["name"]),
         "year": str(m["publicationYear"]), "doi": doi, "note": "[dataset]"}
    if m.get("version"):
        f["version"] = tex(str(m["version"]))
    return "misc", f


def fmt(typ, key, f):
    for k, v in OVERRIDES.get(key, {}).items():
        if v is None:
            f.pop(k, None)
        else:
            f[k] = v
    order = ["author", "title", "journal", "booktitle", "howpublished", "year", "month", "volume", "number", "pages",
             "version", "doi", "note"]
    body = ",\n".join(f"  {k:<12} = {v if k == 'month' else '{' + v + '}'}" for k in order if (v := f.get(k)))
    return f"@{typ}{{{key},\n{body}\n}}"


out, failed = [], []
for group, fn in [(REFS, crossref), (DATASETS, datacite)]:
    for key, doi in group.items():
        try:
            typ, f = fn(key, doi)
            out.append(fmt(typ, key, f))
        except Exception as e:  # noqa: BLE001
            failed.append((key, doi, repr(e)[:80]))
for key, doi in PREPRINTS.items():
    m = json.loads(get(f"https://api.datacite.org/dois/{doi}"))["data"]["attributes"]
    f = {"author": " and ".join(f"{tex(c['familyName'])}, {tex(c.get('givenName', ''))}" for c in m["creators"]),
         "title": "{" + tex(m["titles"][0]["title"]) + "}", "howpublished": "arXiv preprint arXiv:" + doi.split("arXiv.")[1],
         "year": str(m["publicationYear"]), "doi": doi}
    out.append(fmt("misc", key, f))
manual = (ROOT / "paper" / "refs_manual.bib").read_text(encoding="utf-8")
(ROOT / "paper" / "refs.bib").write_text("\n\n".join(out) + "\n\n" + manual, encoding="utf-8")
print("ok", len(out), "failed", failed)
