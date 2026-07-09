

import torch, os, sys, time, subprocess

print("=" * 72)
print("  GPU AND ENVIRONMENT CHECK")
print("=" * 72)

if not torch.cuda.is_available():
    raise RuntimeError("NO GPU DETECTED. Select A100 GPU in Runtime settings.")

gpu_name = torch.cuda.get_device_name(0)
gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU: {gpu_name}  ({gpu_mem:.1f} GB)")

if gpu_mem >= 75:
    MAX_SEQ_LEN_FOLD, MAX_SEQ_LEN_EMB = 1000, 1022
    ESM2_MODEL_NAME = "esm2_t48_15B_UR50D"
elif gpu_mem >= 38:
    MAX_SEQ_LEN_FOLD, MAX_SEQ_LEN_EMB = 800, 1022
    ESM2_MODEL_NAME = "esm2_t36_3B_UR50D"
else:
    raise RuntimeError(f"Insufficient GPU memory: {gpu_mem:.0f} GB")

print(f"  ESMFold max seq: {MAX_SEQ_LEN_FOLD} aa | ESM-2: {ESM2_MODEL_NAME}")
print("=" * 72)

print("\nInstalling dependencies...\n")

def run_install(cmd, label):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"  [{label}] {'OK' if r.returncode==0 else 'FAILED'}")
    return r.returncode == 0

run_install("pip install -q transformers accelerate", "transformers+accelerate")
run_install("pip install -q fair-esm", "fair-esm")
run_install("pip install -q pydssp", "pydssp")
run_install("pip install -q pyarrow", "pyarrow")
run_install("pip install -q biopython matplotlib seaborn pandas requests tqdm numpy scipy", "science")

print("\nVerifying imports...")
for m in ["transformers","esm","pydssp","pandas","pyarrow","matplotlib","scipy"]:
    try: __import__(m); print(f"  {m} -- OK")
    except ImportError as e: print(f"  {m} -- FAILED: {e}")
print("\nDependencies ready.")

import datetime, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DRIVE_BASE = "/content/drive/MyDrive/ESM-Fold"

if os.path.isdir(DRIVE_BASE):
    print(f"  Drive accessible: {DRIVE_BASE}")
elif os.path.isdir("/content/drive/MyDrive"):
    print(f"  Drive mounted but folder not found: {DRIVE_BASE}")
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception as e:
        print(f"  Drive mount skipped ({type(e).__name__})")
        if os.path.isdir(DRIVE_BASE):
            print(f"  Drive accessible anyway.")

BASE_DIR   = "/content/pipeline_outputs"
PHASE1_DIR = os.path.join(BASE_DIR, "Phase1_Structure")
AF_DIR     = os.path.join(BASE_DIR, "AlphaFold_Comparison")
TREAT_DIR  = os.path.join(BASE_DIR, "Treatment_Analysis")
PHASE2_DIR = os.path.join(BASE_DIR, "Phase2_ESM2_15B")
PERRES_DIR = os.path.join(PHASE2_DIR, "per_residue_embeddings")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
TEMP_DIR   = "/content/temp_pdb"

for d in [PHASE1_DIR,AF_DIR,TREAT_DIR,PHASE2_DIR,PERRES_DIR,PLOTS_DIR,TEMP_DIR]:
    os.makedirs(d, exist_ok=True)
print(f"  Output: {BASE_DIR}/")

print("=" * 72)
print("  LOADING DIA-NN REPORT (FULL PROTEOME)")
print("=" * 72)

parquet_path = os.path.join(DRIVE_BASE, "report.parquet")
if not os.path.exists(parquet_path):
    parquet_path = "/content/report.parquet"
if not os.path.exists(parquet_path):
    raise FileNotFoundError("report.parquet not found in Drive or /content/")

report_df = pd.read_parquet(parquet_path)
print(f"  Loaded: {report_df.shape}")
print(f"  Columns: {list(report_df.columns)[:12]}...")

PROTEIN_COL = next((c for c in ['Protein.Ids','Protein.Group','Protein.Id']
                     if c in report_df.columns), None)
if PROTEIN_COL is None:
    pcols = [c for c in report_df.columns if 'protein' in c.lower()]
    PROTEIN_COL = pcols[0] if pcols else None
FILE_COL = next((c for c in ['File.Name','Run','FileName']
                  if c in report_df.columns), None)
QUANT_COL = next((c for c in ['PG.MaxLFQ','PG.Quantity','Precursor.Quantity',
                                'Genes.MaxLFQ','Genes.Quantity']
                   if c in report_df.columns), None)
if QUANT_COL is None:
    qcols = [c for c in report_df.columns if 'quant' in c.lower() or 'intens' in c.lower()]
    QUANT_COL = qcols[0] if qcols else None
GENE_COL = next((c for c in ['Genes','Gene.Names'] if c in report_df.columns), None)

print(f"  Protein col: {PROTEIN_COL}")
print(f"  File col:    {FILE_COL}")
print(f"  Quant col:   {QUANT_COL}")
print(f"  Gene col:    {GENE_COL}")

TREATMENTS = {'CD':'Coffee Drought','CS':'Coffee Salt',
              'GH':'Greenhouse Heat','RH':'Recovery/Hydrated'}

def parse_treatment(fp):
    if pd.isna(fp): return 'Unknown'
    s = str(fp).replace('\\','/').split('/')[-1]
    for code in TREATMENTS:
        if s.startswith(code+'_') or s.startswith(code+'.'): return code
    return 'Unknown'

all_entries = report_df[PROTEIN_COL].dropna().unique()
unique_ids = set()
for entry in all_entries:
    for sub in str(entry).split(';'):
        sub = sub.strip()
        if sub: unique_ids.add(sub)
all_protein_ids = sorted(unique_ids)
print(f"\n  Unique proteins: {len(all_protein_ids)}")

treat_summary_df = pd.DataFrame()
if FILE_COL and QUANT_COL:
    report_df['_Treatment'] = report_df[FILE_COL].apply(parse_treatment)
    print(f"\n  Treatment row counts:")
    for t in TREATMENTS:
        print(f"    {t}: {(report_df['_Treatment']==t).sum()}")

    quant_agg = (report_df.groupby([PROTEIN_COL, FILE_COL, '_Treatment'])[QUANT_COL]
                 .max().reset_index())

    rows = []
    for prot_id in all_entries:
        sub = quant_agg[quant_agg[PROTEIN_COL] == prot_id]
        primary = str(prot_id).split(';')[0].strip()
        gene = ''
        if GENE_COL:
            gv = report_df.loc[report_df[PROTEIN_COL]==prot_id, GENE_COL].dropna()
            gene = str(gv.iloc[0]) if len(gv)>0 else ''
        r = {'Protein_Group': str(prot_id), 'Primary_ID': primary, 'Gene': gene}
        for t in TREATMENTS:
            t_sub = sub[sub['_Treatment'] == t]
            vals = t_sub[QUANT_COL].values
            vals = vals[~np.isnan(vals)] if len(vals) > 0 else np.array([])
            pos = vals[vals > 0]
            r[f'{t}_Mean'] = round(float(pos.mean()), 1) if len(pos) > 0 else 0.0
            r[f'{t}_Detected'] = len(pos) >= 2
            r[f'{t}_N_Replicates'] = int(len(pos))
        rows.append(r)

    treat_summary_df = pd.DataFrame(rows)
    treat_summary_df.to_csv(os.path.join(TREAT_DIR, "protein_treatment_quantification.csv"),
                            index=False)
    print(f"\n  Proteins detected (>=2/3 replicates):")
    for t in TREATMENTS:
        print(f"    {t}: {treat_summary_df[f'{t}_Detected'].sum()}")

CSV_FILENAME = "stress_proteins_GO_classified.csv"
stress_csv = os.path.join(DRIVE_BASE, CSV_FILENAME)
if not os.path.exists(stress_csv):
    stress_csv = f"/content/{CSV_FILENAME}"
stress_df = pd.read_csv(stress_csv) if os.path.exists(stress_csv) else pd.DataFrame()
if len(stress_df) > 0:
    stress_df.columns = [c.strip() for c in stress_df.columns]
stress_id_set = set(stress_df['UniProt_ID'].dropna().str.strip()) if 'UniProt_ID' in stress_df.columns else set()
print(f"\n  Stress proteins: {len(stress_id_set)}")

ptm_records = []
for ptm_type, ptm_fn in {'Acetylation':'report_UniMod_1_sites_99.tsv',
                           'Oxidation':'report_UniMod_35_sites_99.tsv'}.items():
    pp = os.path.join(DRIVE_BASE, ptm_fn)
    if not os.path.exists(pp): pp = f"/content/{ptm_fn}"
    if not os.path.exists(pp): continue
    pdf = pd.read_csv(pp, sep='\t')
    pdf.columns = [c.strip() for c in pdf.columns]
    icols = [c for c in pdf.columns if 'mzML' in c]
    for _, row in pdf.iterrows():
        conds = set()
        tot, nd = 0, 0
        for c in icols:
            v = row.get(c, 0)
            if pd.notna(v) and float(v) > 0:
                tot += float(v); nd += 1
                conds.add(c.replace('\\','/').split('/')[-1].split('_')[0])
        ptm_records.append({
            'UniProt_ID': str(row.get('Protein','')).strip(),
            'Gene_Name': str(row.get('Gene.Names','')).strip(),
            'PTM_Type': ptm_type,
            'Residue': str(row.get('Residue','')).strip(),
            'Site': int(row.get('Site', 0)),
            'Flanking': str(row.get('Sequence','')).strip(),
            'Total_Intensity': round(tot,1), 'N_Detected': nd,
            'Conditions': ';'.join(sorted(conds)),
        })
    print(f"  {ptm_type}: {len(pdf)} sites")
ptm_all_df = pd.DataFrame(ptm_records) if ptm_records else pd.DataFrame()

print(f"\n  TOTAL proteins for structure analysis: {len(all_protein_ids)}")

import requests
from tqdm.notebook import tqdm

def fetch_seq(uid, retries=3):
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    for a in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                lines = r.text.strip().split('\n')
                return ''.join(lines[1:])
            if r.status_code == 404: return None
            time.sleep(2**a)
        except: time.sleep(2**a)
    return None

print(f"\nFetching {len(all_protein_ids)} sequences...\n")
sequences, failed_fetch = {}, []
for uid in tqdm(all_protein_ids, desc="UniProt"):
    s = fetch_seq(uid)
    if s: sequences[uid] = s
    else: failed_fetch.append(uid)
    time.sleep(0.15)

print(f"\n  Retrieved: {len(sequences)} / {len(all_protein_ids)}")
if failed_fetch:
    print(f"  Failed ({len(failed_fetch)}): {failed_fetch[:10]}...")

with open(os.path.join(PHASE1_DIR, "all_sequences.fasta"), 'w') as f:
    for uid, seq in sequences.items():
        f.write(f">{uid}\n{seq}\n")

from transformers import AutoTokenizer, EsmForProteinFolding

print("=" * 72)
print("  LOADING ESMFold")
print("=" * 72)

esmfold_tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esmfold_mdl = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True).eval().cuda()
esmfold_mdl.trunk.set_chunk_size(128)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} / {gpu_mem:.1f} GB")

import pydssp
from Bio.PDB import PDBParser
pdb_parser = PDBParser(QUIET=True)

def assign_ss(backbone):
    if isinstance(backbone, np.ndarray):
        t = torch.tensor(backbone, dtype=torch.float32)
    else: t = backbone.float()
    if t.dim() == 3: t = t.unsqueeze(0)
    try:
        ss = pydssp.assign(t, out_type='c3')
        return ['C' if s=='-' else s for s in ss[0]]
    except: return ['C'] * t.shape[1]

def ss_fracs(ss):
    n = len(ss)
    if n == 0: return 0,0,0
    return (round(sum(s=='H' for s in ss)/n*100,1),
            round(sum(s=='E' for s in ss)/n*100,1),
            round(sum(s=='C' for s in ss)/n*100,1))

def classify_plddt(v):
    if v>=90: return "structured_high"
    if v>=70: return "structured"
    if v>=50: return "flexible"
    return "disordered"

def find_idrs(residues, thr=50, minl=5):
    regs, st, run = [], None, []
    for r in residues:
        if r['pLDDT'] < thr:
            if st is None: st = r['residue_index']
            run.append(r)
        else:
            if st and len(run) >= minl:
                regs.append((st, run[-1]['residue_index'], len(run)))
            st, run = None, []
    if st and len(run) >= minl:
        regs.append((st, run[-1]['residue_index'], len(run)))
    return regs

def pdb_plddt(path):
    struct = pdb_parser.get_structure("p", path)
    res = []
    for mdl in struct:
        for ch in mdl:
            for r in ch:
                if r.get_id()[0] == ' ':
                    res.append({'residue_index': r.get_id()[1],
                                'pLDDT': round(float(np.mean([a.get_bfactor() for a in r])),2)})
        break
    return res

def pdb_backbone(path):
    struct = pdb_parser.get_structure("p", path)
    coords = []
    for mdl in struct:
        for ch in mdl:
            for r in ch:
                if r.get_id()[0] != ' ': continue
                try:
                    coords.append([r['N'].get_vector().get_array(),
                                   r['CA'].get_vector().get_array(),
                                   r['C'].get_vector().get_array(),
                                   r['O'].get_vector().get_array()])
                except KeyError: pass
        break
    return np.array(coords) if coords else None

def get_val(df, uid, col):
    if col in df.columns and uid in df['UniProt_ID'].values:
        v = df.loc[df['UniProt_ID']==uid, col].values[0]
        return '' if pd.isna(v) else str(v)
    return ''

print("=" * 72)
print(f"  PHASE 1B -- ESMFold ({len(sequences)} proteins)")
print("=" * 72)

all_plddt, all_ss, esm_rows, failed_fold = {}, {}, [], []
total = len(sequences)
t0 = time.time()

for i, (uid, seq) in enumerate(sequences.items(), 1):
    slen = len(seq)
    print(f"  [{i:>3}/{total}] {uid} ({slen} aa) ", end="", flush=True)
    try:
        eseq = seq[:MAX_SEQ_LEN_FOLD] if slen > MAX_SEQ_LEN_FOLD else seq
        trunc = slen > MAX_SEQ_LEN_FOLD
        elen = len(eseq)

        inp = esmfold_tok([eseq], return_tensors="pt", add_special_tokens=False)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            out = esmfold_mdl(**inp)

        pr = out['plddt'][0]
        ps = (pr.mean(dim=-1) if pr.dim()==2 else pr).cpu().numpy()[:elen]
        if ps.max() <= 1.0: ps *= 100.0

        bb = out['positions'][-1, 0][:elen, :4, :].cpu()
        ssc = assign_ss(bb)

        rd = [{'residue_index':j+1, 'residue_aa':eseq[j], 'pLDDT':round(float(ps[j]),2),
               'SS':ssc[j] if j<len(ssc) else 'C'} for j in range(elen)]
        all_plddt[uid] = rd
        all_ss[uid] = ssc[:elen]

        pa = np.array([r['pLDDT'] for r in rd])
        pst = float(np.sum(pa>=70)/elen*100)
        pdi = float(np.sum(pa<50)/elen*100)
        h,e,c = ss_fracs(ssc[:elen])
        is_str = uid in stress_id_set
        has_ptm = uid in set(ptm_all_df['UniProt_ID']) if len(ptm_all_df)>0 else False

        esm_rows.append({
            'UniProt_ID':uid, 'Gene_Name':get_val(stress_df,uid,'Gene_Name'),
            'Sequence_Length':slen, 'Effective_Length':elen, 'Truncated':trunc,
            'Is_Stress':is_str, 'Has_PTM':has_ptm,
            'Mean_pLDDT':round(float(pa.mean()),2),
            'Median_pLDDT':round(float(np.median(pa)),2),
            'Std_pLDDT':round(float(pa.std()),2),
            'Pct_Structured':round(pst,1), 'Pct_Disordered':round(pdi,1),
            'Pct_Helix':h, 'Pct_Sheet':e, 'Pct_Coil':c,
            'Classification':('Mostly structured' if pst>70 else 'Mixed' if pst>40 else 'Mostly disordered'),
        })

        del out, inp, bb; torch.cuda.empty_cache()
        eta = (time.time()-t0)/i*(total-i)/60
        print(f"pLDDT={pa.mean():.1f} H={h:.0f}% E={e:.0f}% C={c:.0f}% ETA {eta:.1f}m")
        if i%20==0: torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print("OOM"); failed_fold.append(uid); torch.cuda.empty_cache()
    except Exception as ex:
        print(f"ERR: {ex}"); failed_fold.append(uid); torch.cuda.empty_cache()

print(f"\n  ESMFold done: {len(esm_rows)}/{total} in {(time.time()-t0)/60:.1f} min")
if failed_fold: print(f"  Failed: {failed_fold}")

esmfold_df = pd.DataFrame(esm_rows)
esmfold_df.to_csv(os.path.join(PHASE1_DIR, "ESMFold_summary_all_proteins.csv"), index=False)

pr_rows = []
for uid, rd in all_plddt.items():
    for r in rd:
        pr_rows.append({'UniProt_ID':uid, 'Idx':r['residue_index'],
                         'AA':r['residue_aa'], 'pLDDT':r['pLDDT'],
                         'Class':classify_plddt(r['pLDDT']), 'SS':r['SS']})
pd.DataFrame(pr_rows).to_csv(os.path.join(PHASE1_DIR,"ESMFold_per_residue.csv"),index=False)

dr = []
for uid, rd in all_plddt.items():
    for s,e,l in find_idrs(rd):
        dr.append({'UniProt_ID':uid,'Start':s,'End':e,'Length':l})
if dr: pd.DataFrame(dr).to_csv(os.path.join(PHASE1_DIR,"disordered_regions.csv"),index=False)

print(f"  Saved to {PHASE1_DIR}/")
print(f"  {esmfold_df['Classification'].value_counts().to_dict()}")

print("=" * 72)
print("  PHASE 1C -- ALPHAFOLD DB COMPARISON")
print("=" * 72)

AF_TMP = os.path.join(TEMP_DIR, "af_pdb")
os.makedirs(AF_TMP, exist_ok=True)

af2_res, af2_ss, af2_fail = {}, {}, []

def fetch_af(uid, outd, retries=2):
    for ver in ['v4','v3']:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_{ver}.pdb"
        for a in range(retries):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    p = os.path.join(outd, f"AF-{uid}.pdb")
                    with open(p,'w') as f: f.write(r.text)
                    return p
                if r.status_code == 404: break
                time.sleep(1)
            except: time.sleep(2)
    return None

print(f"  Fetching AF2 structures for {len(sequences)} proteins...\n")
af_t0 = time.time()

for uid in tqdm(sequences.keys(), desc="AlphaFold DB"):
    p = fetch_af(uid, AF_TMP)
    if p is None:
        af2_fail.append(uid); continue
    try:
        af2_res[uid] = pdb_plddt(p)
        bb = pdb_backbone(p)
        af2_ss[uid] = assign_ss(bb) if bb is not None and len(bb)>=6 else ['C']*len(af2_res[uid])
        os.remove(p)
    except:
        af2_fail.append(uid)
        if os.path.exists(p): os.remove(p)
    time.sleep(0.1)

print(f"\n  AF2 retrieved: {len(af2_res)} / {len(sequences)}")
print(f"  Not found: {len(af2_fail)}")
print(f"  Time: {(time.time()-af_t0)/60:.1f} min")

from scipy.stats import pearsonr
comp_rows = []
for uid in sequences:
    r = {'UniProt_ID': uid}
    if uid in all_plddt:
        ep = np.array([x['pLDDT'] for x in all_plddt[uid]])
        r['ESM_pLDDT'] = round(float(ep.mean()),2)
        eh,ee,ec = ss_fracs(all_ss.get(uid,[]))
        r['ESM_H'],r['ESM_E'],r['ESM_C'] = eh,ee,ec
    if uid in af2_res:
        ap = np.array([x['pLDDT'] for x in af2_res[uid]])
        r['AF2_pLDDT'] = round(float(ap.mean()),2)
        ah,ae,ac = ss_fracs(af2_ss.get(uid,[]))
        r['AF2_H'],r['AF2_E'],r['AF2_C'] = ah,ae,ac
        r['AF2_Found'] = True
    else:
        r['AF2_Found'] = False

    if 'ESM_pLDDT' in r and 'AF2_pLDDT' in r:
        r['D_pLDDT'] = round(r['ESM_pLDDT']-r['AF2_pLDDT'],2)
        r['D_H'] = round(r.get('ESM_H',0)-r.get('AF2_H',0),1)
        r['D_E'] = round(r.get('ESM_E',0)-r.get('AF2_E',0),1)
        ev = np.array([x['pLDDT'] for x in all_plddt[uid]])
        av = np.array([x['pLDDT'] for x in af2_res[uid]])
        ml = min(len(ev),len(av))
        if ml > 5:
            cr, _ = pearsonr(ev[:ml], av[:ml])
            r['Pearson_r'] = round(float(cr),4)
    comp_rows.append(r)

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(AF_DIR,"ESMFold_vs_AF2.csv"),index=False)

vc = comp_df.dropna(subset=['D_pLDDT'])
if len(vc)>0:
    print(f"\n  Comparison (n={len(vc)}):")
    print(f"    Mean delta pLDDT: {vc['D_pLDDT'].mean():.2f}")
    if 'Pearson_r' in vc: print(f"    Mean correlation: {vc['Pearson_r'].mean():.4f}")

print("=" * 72)
print("  PER-TREATMENT STRUCTURAL ANALYSIS")
print("=" * 72)

treat_struct_rows = []
if len(treat_summary_df) > 0 and len(esmfold_df) > 0:
    esm_ids = set(esmfold_df['UniProt_ID'])

    for tc, tn in TREATMENTS.items():
        dc = f'{tc}_Detected'
        if dc not in treat_summary_df.columns: continue
        det = treat_summary_df[treat_summary_df[dc]==True]

        for _, gr in det.iterrows():
            matched = None
            for sub in str(gr['Protein_Group']).split(';'):
                sub = sub.strip()
                if sub in esm_ids: matched = sub; break
            if not matched: continue

            er = esmfold_df[esmfold_df['UniProt_ID']==matched].iloc[0]
            treat_struct_rows.append({
                'Treatment':tc, 'Treatment_Name':tn,
                'UniProt_ID':matched,
                'Gene':get_val(stress_df,matched,'Gene_Name'),
                'Intensity':gr.get(f'{tc}_Mean',0),
                'Is_Stress':matched in stress_id_set,
                'Mean_pLDDT':er['Mean_pLDDT'],
                'Pct_Helix':er['Pct_Helix'], 'Pct_Sheet':er['Pct_Sheet'],
                'Pct_Coil':er['Pct_Coil'], 'Class':er['Classification'],
            })

    if treat_struct_rows:
        tdf = pd.DataFrame(treat_struct_rows)
        tdf.to_csv(os.path.join(TREAT_DIR,"treatment_structural_profiles.csv"),index=False)

        for tc in TREATMENTS:
            td = tdf[tdf['Treatment']==tc]
            if len(td)==0: continue
            ns = td['Is_Stress'].sum()
            print(f"\n  {tc} ({TREATMENTS[tc]}): {len(td)} proteins ({ns} stress)")
            print(f"    pLDDT={td['Mean_pLDDT'].mean():.1f}  "
                  f"H={td['Pct_Helix'].mean():.1f}%  "
                  f"E={td['Pct_Sheet'].mean():.1f}%  "
                  f"C={td['Pct_Coil'].mean():.1f}%")

        tsets = {t: set(tdf[tdf['Treatment']==t]['UniProt_ID']) for t in TREATMENTS}
        shared = set.intersection(*tsets.values()) if all(tsets.values()) else set()
        print(f"\n  Shared all treatments: {len(shared)}")
        for t in TREATMENTS:
            others = set.union(*[s for k,s in tsets.items() if k!=t]) if len(tsets)>1 else set()
            excl = tsets[t] - others
            print(f"  Exclusive {t}: {len(excl)}")
else:
    print("  Insufficient data for treatment analysis.")

ptm_ctx = []
if len(ptm_all_df) > 0:
    for _, pr in ptm_all_df.iterrows():
        uid, site = pr['UniProt_ID'], int(pr['Site'])
        ctx = dict(pr)
        ctx['In_Stress'] = uid in stress_id_set
        if uid in all_plddt and 1<=site<=len(all_plddt[uid]):
            r = all_plddt[uid][site-1]
            ctx['Site_pLDDT'] = r['pLDDT']
            ctx['Site_SS'] = r['SS']
            ctx['FTIR'] = '1648-1658' if r['SS']=='H' else '1620-1640' if r['SS']=='E' else '1640-1650'
            if uid in af2_res and site<=len(af2_res[uid]):
                ctx['AF2_Site_pLDDT'] = af2_res[uid][site-1]['pLDDT']
        ptm_ctx.append(ctx)
    pd.DataFrame(ptm_ctx).to_csv(os.path.join(PHASE1_DIR,"PTM_context.csv"),index=False)
    print(f"\n  PTM sites mapped: {len(ptm_ctx)}")

ftir = [{'UniProt_ID':r['UniProt_ID'],'Is_Stress':r['Is_Stress'],
         'H':r['Pct_Helix'],'E':r['Pct_Sheet'],'C':r['Pct_Coil'],
         'pLDDT':r['Mean_pLDDT']} for _,r in esmfold_df.iterrows()]
pd.DataFrame(ftir).to_csv(os.path.join(PHASE1_DIR,"FTIR_comparison.csv"),index=False)

import matplotlib.pyplot as plt, matplotlib.patches as mpatches

plt.rcParams.update({'figure.dpi':150,'font.size':9,'font.family':'sans-serif'})
SC = {'H':'#E31A1C','E':'#1F78B4','C':'#A6CEE3'}
TC = {'CD':'#D62728','CS':'#FF7F0E','GH':'#2CA02C','RH':'#1F77B4'}

vc = comp_df.dropna(subset=['ESM_pLDDT','AF2_pLDDT'])
if len(vc) > 5:
    fig, ax = plt.subplots(figsize=(7,7))
    cols = ['#E31A1C' if u in stress_id_set else '#A6CEE3' for u in vc['UniProt_ID']]
    ax.scatter(vc['AF2_pLDDT'], vc['ESM_pLDDT'], c=cols, alpha=0.6,
               edgecolors='black', linewidth=0.3, s=30)
    ax.plot([20,100],[20,100],'k--',alpha=0.4)
    ax.set_xlabel('AlphaFold2 Mean pLDDT'); ax.set_ylabel('ESMFold Mean pLDDT')
    ax.set_title('ESMFold vs AlphaFold2 (red=stress)', fontweight='bold')
    r,_ = pearsonr(vc['AF2_pLDDT'],vc['ESM_pLDDT'])
    ax.text(25,95,f'r={r:.3f}, n={len(vc)}',fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR,"ESMFold_vs_AF2_pLDDT.png"),dpi=300,bbox_inches='tight')
    plt.show()

if len(vc) > 5:
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for idx,(st,lb) in enumerate([('H','Helix'),('E','Sheet'),('C','Coil')]):
        ax = axes[idx]
        ec, ac = f'ESM_{st}', f'AF2_{st}'
        v2 = comp_df.dropna(subset=[ec,ac])
        ax.scatter(v2[ac],v2[ec],alpha=0.5,s=20,c=SC[st])
        ax.plot([0,100],[0,100],'k--',alpha=0.4)
        ax.set_xlabel(f'AF2 {lb}%'); ax.set_ylabel(f'ESM {lb}%')
        ax.set_title(lb,fontweight='bold')
        if len(v2)>5: r,_=pearsonr(v2[ac],v2[ec]); ax.text(5,90,f'r={r:.3f}',fontsize=8)
    plt.suptitle('Secondary Structure: ESMFold vs AlphaFold2',fontweight='bold')
    plt.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR,"SS_ESM_vs_AF2.png"),dpi=300,bbox_inches='tight'); plt.show()

if treat_struct_rows:
    tdf = pd.DataFrame(treat_struct_rows)
    fig, axes = plt.subplots(1,4,figsize=(16,5),sharey=True)
    for idx, tc in enumerate(TREATMENTS):
        ax = axes[idx]
        td = tdf[tdf['Treatment']==tc]
        if len(td)==0: continue
        mh,me,mc = td['Pct_Helix'].mean(),td['Pct_Sheet'].mean(),td['Pct_Coil'].mean()
        bars = ax.bar(['H','E','C'],[mh,me,mc],color=[SC['H'],SC['E'],SC['C']],edgecolor='black',linewidth=0.5)
        for b,v in zip(bars,[mh,me,mc]): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f'{v:.1f}%',ha='center',fontsize=8)
        ax.set_title(f'{tc}\n({TREATMENTS[tc]})\nn={len(td)}',fontweight='bold',fontsize=9,color=TC[tc])
        ax.set_ylim(0,max(mh,me,mc)*1.3+5)
        if idx==0: ax.set_ylabel('Mean SS %')
    plt.suptitle('Secondary Structure by Treatment\n(H~1648-1658 | E~1620-1640 | C~1640-1650 cm-1)',fontweight='bold')
    plt.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR,"SS_per_treatment.png"),dpi=300,bbox_inches='tight'); plt.show()

if treat_struct_rows:
    fig, ax = plt.subplots(figsize=(10,5))
    for tc in TREATMENTS:
        td = tdf[tdf['Treatment']==tc]
        if len(td)>0: ax.hist(td['Mean_pLDDT'],bins=20,alpha=0.5,label=tc,color=TC[tc],edgecolor='black',linewidth=0.3)
    ax.set_xlabel('Mean pLDDT'); ax.set_ylabel('Count')
    ax.set_title('pLDDT by Treatment',fontweight='bold'); ax.legend()
    plt.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR,"pLDDT_by_treatment.png"),dpi=300,bbox_inches='tight'); plt.show()

sdf = esmfold_df[esmfold_df['Is_Stress']==True]
if len(sdf)>0:
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
    mh,me,mc=sdf['Pct_Helix'].mean(),sdf['Pct_Sheet'].mean(),sdf['Pct_Coil'].mean()
    bs=a1.bar(['Helix\n1648-1658','Sheet\n1620-1640','Coil\n1640-1650'],[mh,me,mc],
              color=[SC['H'],SC['E'],SC['C']],edgecolor='black')
    for b,v in zip(bs,[mh,me,mc]): a1.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{v:.1f}%',ha='center',fontweight='bold')
    a1.set_ylabel('Mean %'); a1.set_title('Stress Proteins SS (FTIR bands, cm-1)',fontweight='bold')
    a2.hist(sdf['Pct_Helix'],bins=15,alpha=0.7,color=SC['H'],edgecolor='black',label='Helix')
    a2.hist(sdf['Pct_Sheet'],bins=15,alpha=0.7,color=SC['E'],edgecolor='black',label='Sheet')
    a2.set_xlabel('SS %'); a2.set_ylabel('Count'); a2.set_title('SS Distribution (Stress)',fontweight='bold'); a2.legend()
    plt.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR,"FTIR_SS.png"),dpi=300,bbox_inches='tight'); plt.show()

print(f"\n  Plots saved to {PLOTS_DIR}/")

import gc
del esmfold_mdl, esmfold_tok; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
print(f"  VRAM free: {(torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_allocated())/1e9:.1f} GB")

import esm

print("=" * 72)
print(f"  PHASE 2 -- ESM-2 15B ({len(stress_id_set)} stress proteins)")
print("=" * 72)

sseqs = {u:s for u,s in sequences.items() if u in stress_id_set}
print(f"  Loading {ESM2_MODEL_NAME}...\n")

m2, alph = getattr(esm.pretrained, ESM2_MODEL_NAME)()
m2 = m2.eval().cuda()
bc = alph.get_batch_converter()
NL, ED = m2.num_layers, m2.embed_dim
print(f"  Loaded. Layers={NL}, Dim={ED}, VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB\n")

membs, emeta, efail = {}, [], []
et0 = time.time(); tot = len(sseqs)

for i,(uid,seq) in enumerate(sseqs.items(),1):
    sl = len(seq)
    print(f"  [{i:>3}/{tot}] {uid} ({sl} aa) ", end="", flush=True)
    try:
        es = seq[:MAX_SEQ_LEN_EMB] if sl>MAX_SEQ_LEN_EMB else seq
        tr = sl>MAX_SEQ_LEN_EMB
        _,_,toks = bc([(uid,es)]); toks=toks.cuda()
        with torch.no_grad():
            res = m2(toks, repr_layers=[NL], return_contacts=False)
        rr = res["representations"][NL][0,1:len(es)+1,:].cpu().numpy()
        me = rr.mean(axis=0); membs[uid]=me
        np.save(os.path.join(PERRES_DIR,f"{uid}_emb.npy"),rr)
        emeta.append({'UniProt_ID':uid,'Len':sl,'Dim':ED,'L2':round(float(np.linalg.norm(me)),4)})
        del toks,res,rr; torch.cuda.empty_cache()
        eta=(time.time()-et0)/i*(tot-i)/60
        print(f"dim={ED} ETA {eta:.1f}m")
    except torch.cuda.OutOfMemoryError:
        print("OOM"); efail.append(uid); torch.cuda.empty_cache()
    except Exception as ex:
        print(f"ERR: {ex}"); efail.append(uid)

print(f"\n  Done: {len(membs)}/{tot} in {(time.time()-et0)/60:.1f} min")
np.savez_compressed(os.path.join(PHASE2_DIR,"stress_embeddings.npz"),**membs)
pd.DataFrame(emeta).to_csv(os.path.join(PHASE2_DIR,"embedding_meta.csv"),index=False)

print("\nBuilding master summary...")
mdf = esmfold_df.copy()
if len(comp_df)>0:
    ac=[c for c in ['UniProt_ID','AF2_pLDDT','AF2_H','AF2_E','AF2_C','D_pLDDT','Pearson_r'] if c in comp_df.columns]
    mdf = mdf.merge(comp_df[ac], on='UniProt_ID', how='left')
if len(treat_summary_df)>0:
    tc2 = treat_summary_df[['Primary_ID']+[c for c in treat_summary_df.columns if '_Detected' in c or '_Mean' in c]].rename(columns={'Primary_ID':'UniProt_ID'})
    mdf = mdf.merge(tc2, on='UniProt_ID', how='left')
mdf.to_csv(os.path.join(BASE_DIR,"master_summary.csv"),index=False)
print(f"  master_summary.csv: {mdf.shape}")

print(f"""
{'='*72}
  PIPELINE COMPLETE
{'='*72}
  Proteins folded (ESMFold):  {len(esm_rows)}
  AlphaFold2 compared:        {len(af2_res)}
  Stress embedded (ESM-2):    {len(membs)}
  Treatments analysed:        {', '.join(TREATMENTS.keys())}
  PTM sites mapped:           {len(ptm_ctx)}

  pLDDT: >=90 structured | 70-90 confident | 50-70 flexible | <50 disordered
  FTIR:  H~1648-1658 | E~1620-1640 | C~1640-1650 cm-1
{'='*72}
""")

import zipfile, shutil
zn = f"pipeline_{TIMESTAMP}.zip"; zp = f"/content/{zn}"
print(f"Creating {zn}...")
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for rt,ds,fs in os.walk(BASE_DIR):
        ds[:] = [d for d in ds if not d.startswith('.')]
        for f in fs: z.write(os.path.join(rt,f), os.path.relpath(os.path.join(rt,f),"/content"))
print(f"  Size: {os.path.getsize(zp)/1e6:.1f} MB")
try: shutil.copy(zp, os.path.join(DRIVE_BASE, zn)); print(f"  Saved to Drive.")
except: pass
try:
    from google.colab import files as dl; dl.download(zp); print("  Download started.")
except: print(f"  ZIP at: {zp}")
print(f"\n{'='*72}\n  DONE\n{'='*72}")
