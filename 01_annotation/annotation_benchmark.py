# BLAST performed as supplementary information
# Version used for the BLAST ncbi-blast-2.12.0+-win64.exe
# Check the offical ncbi page: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/

import itertools
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# --- Paths (change to your folders) -----------------------------------------
base          = r"D:/All results/ESM-2 embeddings-20260310T003306Z-3-001/ESM-2 embeddings"
annotation    = base + "/Results/annotation_GO_full.csv"           # step 1d output (368 proteins)
embeddings    = base + "/Results/embeddings_mean.npz"              # step 1a output
kw_cache      = base + "/Results/swissprot_go_cache.json"          # keywords of the ESM-2 neighbours
swissprot_fa  = base + "/SwissProt/swissprot_viridiplantae.fasta"  # Swiss-Prot reference (42,002)
chickpea_fa   = r"D:/P2/DIA/uniprotkb_taxonomy_id_3827_2026_02_09.fasta"
supp          = r"D:/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/SECOND PAPER/First round/foods-4574474-supplementary/Supplementary Material"
dep_cosine    = supp + "/Supplementary Table S7.csv"                # 21 x 21 DEP cosine matrix
stress_table  = supp + "/Supplementary Table S3.csv"                # 55 stress proteins (current Table S3)
output_dir    = r"D:/P2/VETEALV/Results/2/Annotation_benchmark"     # keep this path WITHOUT spaces (BLAST)
BLAST_BIN     = r""   # empty = blastp/makeblastdb are in PATH; otherwise e.g. r"C:/Program Files/NCBI/blast-2.12.0+/bin"
KEYWORDS_FALLBACK = os.path.join(output_dir, "blast_hit_keywords.tsv")   # accession <tab> keywords

EVALUE     = 1e-5
STRESS_KW  = ["stress response", "chaperone", "antioxidant", "cold shock"]   # rule of step 1d
os.makedirs(output_dir, exist_ok=True)
work = lambda f: os.path.join(output_dir, f)


def run(cmd):
    cmd = [os.path.join(BLAST_BIN, cmd[0]) if BLAST_BIN else cmd[0]] + cmd[1:]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


if " " in output_dir:
    raise SystemExit("output_dir must not contain spaces (BLAST databases fail with spaces on Windows)")
blastp_exe = os.path.join(BLAST_BIN, "blastp") if BLAST_BIN else "blastp"
print(subprocess.run([blastp_exe, "-version"], capture_output=True, text=True).stdout.splitlines()[0])


# --- inputs -------------------------------------------------------------------
ann = pd.read_csv(annotation)
ids = list(ann["UniProt_ID"])
print(f"Identified proteins: {len(ids)} | stress-related: {int(ann['Is_Stress_GO'].sum())} "
      f"| proteins with GO terms read: {int((ann['N_GO_Terms'] > 0).sum())}")

# query sequences
seqs, cur = {}, None
with open(chickpea_fa) as f:
    for line in f:
        if line.startswith(">"):
            acc = line.split("|")[1] if "|" in line else line[1:].split()[0]
            cur = acc if acc in set(ids) else None
            if cur:
                seqs[cur] = []
        elif cur:
            seqs[cur].append(line.strip())
with open(work("queries.fasta"), "w") as f:
    for acc in ids:
        f.write(f">{acc}\n{''.join(seqs[acc])}\n")
print(f"Sequences written: {len(seqs)}")

# Swiss-Prot names
sp_name = {}
with open(swissprot_fa) as f:
    for line in f:
        if line.startswith(">"):
            sp_name[line.split("|")[1]] = re.sub(r" OS=.*", "", line.split(" ", 1)[1]).strip()

# --- 1. BLASTp against Swiss-Prot ----------------------------------------------
shutil.copy(swissprot_fa, work("swissprot_viridiplantae.fasta"))     # copy to a path without spaces
run(["makeblastdb", "-in", work("swissprot_viridiplantae.fasta"), "-dbtype", "prot", "-parse_seqids", "-out", work("sp_plants")])
run(["blastp", "-query", work("queries.fasta"), "-db", work("sp_plants"), "-evalue", str(EVALUE),
     "-max_target_seqs", "500", "-outfmt", "6 qseqid sacc pident length qlen slen evalue bitscore qcovs",
     "-out", work("blast_vs_swissprot.tsv")])
bl = pd.read_csv(work("blast_vs_swissprot.tsv"), sep="\t", header=None,
                 names=["q", "s", "pident", "alen", "qlen", "slen", "evalue", "bits", "qcovs"])
bl = bl.sort_values(["q", "bits"], ascending=[True, False])

# keywords: ESM-2 neighbours (cache of step 1d) + BLASTp best hits (UniProt)
kw = {k: v["keywords"] for k, v in json.load(open(kw_cache)).items()}
best_hits = set(bl.groupby("q").head(1)["s"])
missing = sorted(best_hits - set(kw))
try:
    for i in range(0, len(missing), 50):
        q = " OR ".join(f"accession:{a}" for a in missing[i:i + 50])
        url = ("https://rest.uniprot.org/uniprotkb/stream?query=" + urllib.parse.quote(q)
               + "&fields=accession,keyword&format=tsv")
        with urllib.request.urlopen(url, timeout=60) as r:
            for line in r.read().decode().splitlines()[1:]:
                a = line.split("\t")
                kw[a[0]] = a[1] if len(a) > 1 else ""
    pd.Series({a: kw.get(a, "") for a in missing}).to_csv(KEYWORDS_FALLBACK, sep="\t", header=False)
except Exception as e:
    print("  UniProt not reachable (", e, ") -> reading", KEYWORDS_FALLBACK)
    for line in open(KEYWORDS_FALLBACK):
        a = line.rstrip("\n").split("\t")
        kw[a[0]] = a[1] if len(a) > 1 else ""
is_stress = lambda acc: any(k in str(kw.get(acc, "")).lower() for k in STRESS_KW)

rows = []
for _, r in ann.iterrows():
    q, esm = r["UniProt_ID"], r["SwissProt_ID"]
    h = bl[bl.q == q]
    top = h.iloc[0] if len(h) else None
    top_set = set(h[h.bits == top.bits].s) if top is not None else set()
    pair = h[h.s == esm]
    rows.append({
        "UniProt_ID": q, "Description": r["Description"],
        "ESM2_neighbour": esm, "ESM2_neighbour_name": sp_name.get(esm, r["SwissProt_Description"]),
        "ESM2_cosine": r["Cosine_Similarity"], "ESM2_confidence": r["ESM2_Confidence"],
        "BLASTp_best_hit": top.s if top is not None else "no hit",
        "BLASTp_best_hit_name": sp_name.get(top.s, "") if top is not None else "",
        "BLASTp_identity_%": round(top.pident, 1) if top is not None else np.nan,
        "BLASTp_Evalue": top.evalue if top is not None else np.nan,
        "Same_top_hit": "Yes" if esm in top_set else "No",
        "ESM2_neighbour_in_BLASTp_top5": "Yes" if esm in list(h.s.drop_duplicates()[:5]) else "No",
        "Identity_to_ESM2_neighbour_%": round(pair.pident.max(), 1) if len(pair) else np.nan,
        "Stress_ESM2_rule": "Yes" if is_stress(esm) else "No",
        "Stress_BLASTp_rule": ("Yes" if is_stress(top.s) else "No") if top is not None else "No",
    })
bench = pd.DataFrame(rows)
assert (bench["Stress_ESM2_rule"].eq("Yes") == ann["Is_Stress_GO"]).all(), "stress rule mismatch"

hit = bench["BLASTp_best_hit"] != "no hit"
a = bench["Stress_ESM2_rule"].eq("Yes"); b = bench["Stress_BLASTp_rule"].eq("Yes")
po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
kappa = (po - pe) / (1 - pe)
print(f"\nBLASTp hit (E <= {EVALUE}): {hit.sum()} of {len(bench)}")
print(f"Same top hit: {(bench.Same_top_hit == 'Yes').sum()} of {hit.sum()} | ESM-2 neighbour in BLASTp top 5: "
      f"{(bench.ESM2_neighbour_in_BLASTp_top5 == 'Yes').sum()}")
print(f"Stress classification, ESM-2 vs BLASTp: agree {int((a == b).sum())} of {len(bench)} "
      f"({100 * po:.1f}%), Cohen's kappa = {kappa:.2f}")
print(f"  stress by ESM-2 only: {int((a & ~b).sum())} (without BLASTp hit: {int((a & ~hit).sum())}) "
      f"| stress by BLASTp only: {int((~a & b).sum())}")

tiers = (bench.assign(Tier=pd.cut(bench.ESM2_cosine, [0, 0.8, 0.9, 1.0001], right=False,
                                  labels=["0.70-0.80", "0.80-0.90", ">=0.90"]))
              .groupby("Tier", observed=True)
              .agg(Proteins=("UniProt_ID", "size"),
                   ESM2_neighbour_supported_by_BLASTp=("Identity_to_ESM2_neighbour_%", lambda v: int(v.notna().sum())),
                   Median_identity_to_neighbour=("Identity_to_ESM2_neighbour_%", "median"))
              .reset_index())
print(tiers.to_string(index=False))

# --- 2. Null distribution of cosine similarity -----------------------------------
z = np.load(embeddings, allow_pickle=True)
eids = [str(x) for x in z["protein_ids"]]
E = z["embeddings"].astype(np.float64)
E /= np.linalg.norm(E, axis=1, keepdims=True)
C = E @ E.T
run(["makeblastdb", "-in", work("queries.fasta"), "-dbtype", "prot", "-out", work("chickpea")])
run(["blastp", "-query", work("queries.fasta"), "-db", work("chickpea"), "-evalue", str(EVALUE),
     "-max_target_seqs", "1000", "-outfmt", "6 qseqid sseqid pident evalue bitscore",
     "-out", work("blast_chickpea_all_vs_all.tsv")])
aa = pd.read_csv(work("blast_chickpea_all_vs_all.tsv"), sep="\t", header=None, names=["q", "s", "pid", "e", "bits"])
hom = set(map(frozenset, zip(aa.q, aa.s))) - {frozenset([x]) for x in eids}
iu = np.triu_indices(len(eids), 1)
is_hom = np.array([frozenset((eids[i], eids[j])) in hom for i, j in zip(*iu)])
cos = C[iu]
null, homc = cos[~is_hom], cos[is_hom]
pc = lambda v: 100 * (null < v).mean()
print(f"\nNull (pairs without detectable homology, n = {len(null)}): median {np.median(null):.3f}, "
      f"95th {np.percentile(null, 95):.3f}, 99th {np.percentile(null, 99):.3f}")
print(f"Homologous pairs (n = {len(homc)}): median {np.median(homc):.3f}, 5th {np.percentile(homc, 5):.3f}")

summary = [
    ["Identified proteins annotated", len(bench)],
    ["Stress-related (ESM-2 neighbour keyword rule)", int(a.sum())],
    [f"Proteins with a BLASTp hit (E <= {EVALUE})", int(hit.sum())],
    ["Same top hit (ESM-2 vs BLASTp)", int((bench.Same_top_hit == "Yes").sum())],
    ["ESM-2 neighbour among BLASTp top 5", int((bench.ESM2_neighbour_in_BLASTp_top5 == "Yes").sum())],
    ["Stress classification agreement, n (%)", f"{int((a == b).sum())} ({100 * po:.1f}%)"],
    ["Cohen's kappa", round(kappa, 2)],
    ["Stress by ESM-2 only (without BLASTp hit)", f"{int((a & ~b).sum())} ({int((a & ~hit).sum())})"],
    ["Stress by BLASTp only", int((~a & b).sum())],
    ["", ""],
    ["Cosine null: chickpea protein pairs without detectable homology, n", len(null)],
    ["  median", round(float(np.median(null)), 3)],
    ["  95th percentile", round(float(np.percentile(null, 95)), 3)],
    ["  99th percentile", round(float(np.percentile(null, 99)), 3)],
    ["Homologous pairs, n", len(homc)],
    ["  median", round(float(np.median(homc)), 3)],
]
if os.path.exists(dep_cosine):
    S = pd.read_csv(dep_cosine, index_col=0)
    dv = np.array([S.loc[x, y] for x, y in itertools.combinations(S.index, 2)])
    erd = S.loc["A0A1S2Y7V6"].drop("A0A1S2Y7V6") if "A0A1S2Y7V6" in S.index else None
    p95 = np.percentile(null, 95)
    summary += [["", ""],
                ["DEP pairs (Table S7), n", len(dv)],
                ["  median (percentile of the null)", f"{np.median(dv):.3f} ({pc(np.median(dv)):.0f}th)"],
                ["  pairs above the 95th percentile of the null", int((dv > p95).sum())]]
    if erd is not None:
        summary += [["  dehydrin ERD14-like, mean vs other DEPs (percentile)", f"{erd.mean():.3f} ({pc(erd.mean()):.0f}th)"]]
    print(f"DEP median {np.median(dv):.3f} ({pc(np.median(dv)):.0f}th percentile); "
          f"pairs > 95th: {(dv > p95).sum()} of {len(dv)}")

# --- Table S3 ------------------------------------------------------------------
if os.path.exists(stress_table):          # keep the current Table S3 and add the BLASTp columns
    s3a = pd.read_csv(stress_table)
    s3a = s3a.drop(columns=[c for c in ["Is_Stress_GO", "Primary_Category_GO", "All_Categories_GO"] if c in s3a.columns])
else:
    keep = ["UniProt_ID", "Gene_Name", "Description", "SwissProt_ID", "SwissProt_Description", "SwissProt_Organism",
            "Cosine_Similarity", "ESM2_Confidence", "UniProt_Keywords"]
    s3a = ann[ann["Is_Stress_GO"]][keep].copy()
s3a = s3a.merge(bench[["UniProt_ID", "BLASTp_best_hit", "BLASTp_best_hit_name", "BLASTp_identity_%", "BLASTp_Evalue",
                       "Same_top_hit", "Stress_BLASTp_rule"]], on="UniProt_ID", how="left")
xlsx = work("Table_S3_stress_annotation.xlsx")
with pd.ExcelWriter(xlsx) as w:
    s3a.to_excel(w, sheet_name="S3a Stress proteins", index=False)
    bench.to_excel(w, sheet_name="S3b ESM-2 vs BLASTp", index=False)
    pd.DataFrame(summary, columns=["Measure", "Value"]).to_excel(w, sheet_name="S3c Summary", index=False)
    tiers.to_excel(w, sheet_name="S3c Summary", index=False, startrow=len(summary) + 3)
print("\nSaved:", xlsx)
