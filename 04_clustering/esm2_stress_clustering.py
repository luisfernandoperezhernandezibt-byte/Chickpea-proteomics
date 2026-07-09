import os, sys, subprocess, warnings, time, gc, re

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

import torch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="n_jobs value")


def ensure_package(import_name, pip_name=None):
    if pip_name is None:
        pip_name = import_name
    try:
        return __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return __import__(import_name)


requests = ensure_package("requests")

PARQUET_PATH   = "/content/drive/MyDrive/ESM-2/report.parquet"
FASTA_PATH     = "/content/drive/MyDrive/ESM-2/uniprotkb_taxonomy_id_3827_2026_02_09.fasta"
ESM_MODEL_NAME = "esm2_t48_15B_UR50D"
MAX_SEQ_LEN    = 1022
BATCH_SIZE     = 2
HALF_PRECISION = False

PTM_FILES = {
    "Acetylation": "/content/drive/MyDrive/ESM-2/report.UniMod_1_sites_99.tsv",
    "Oxidation":   "/content/drive/MyDrive/ESM-2/report.UniMod_35_sites_99.tsv",
}

PTM2_MANUAL_IDS = [
    "A0A1S2XLJ8", "A0A1S2Y867", "A0A1S2YCI1",
    "A0A1S2YHI3", "A0A1S2Z0P8", "O49817",
]

STRESS_DEP_CSV   = "/content/drive/MyDrive/ESM-2/Supplementary_stress_DEP_unique.csv"
STRESS_ANNOT_CSV = "/content/drive/MyDrive/ESM-2/stress_proteins_annotated.csv"
BASE_OUTPUT      = "/content/drive/MyDrive/ESM-2"

TREATMENTS = ["RH", "GH", "CD", "CS"]
CONTROL    = "CS"

df = pd.read_parquet(PARQUET_PATH)

for qcol in ["Global.Q.Value", "Global.PG.Q.Value", "Lib.Q.Value"]:
    if qcol in df.columns:
        df = df[df[qcol] <= 0.01]

df["Protein_ID"] = df["Protein.Group"].astype(str).str.split(";").str[0].str.strip()

runs = df["Run"].unique()
sample_info = pd.DataFrame({"Run": runs})
sample_info["BaseName"]  = sample_info["Run"].apply(lambda x: os.path.basename(x))
sample_info["BaseName"]  = sample_info["BaseName"].str.replace(
    r"\.(mzML|raw|d|wiff)$", "", regex=True)
sample_info["Treatment"] = sample_info["BaseName"].str.extract(r"^([A-Za-z]+)")[0]
sample_info["Replicate"] = sample_info["BaseName"].str.extract(r"(\d+)$")[0].astype(int)
sample_info["Treatment"] = sample_info["Treatment"].str.upper()
sample_info["SampleID"]  = sample_info["Treatment"] + "_R" + sample_info["Replicate"].astype(str)

df = df.merge(sample_info[["Run", "SampleID", "Treatment"]], on="Run", how="left")

quant_col = None
for candidate in ["PG.MaxLFQ", "PG.Normalised", "PG.Quantity"]:
    if candidate in df.columns:
        quant_col = candidate
        break
assert quant_col is not None, "No suitable quantity column found!"

pg_long = (
    df[["Protein_ID", "SampleID", "Treatment", quant_col]]
    .rename(columns={quant_col: "Intensity"})
    .drop_duplicates(subset=["Protein_ID", "SampleID"])
)
pg_long = pg_long.groupby(["Protein_ID", "SampleID"], as_index=False)["Intensity"].max()
pg_wide = pg_long.pivot(index="Protein_ID", columns="SampleID", values="Intensity")
pg_wide = pg_wide.replace(0, np.nan)

keep = []
for prot in pg_wide.index:
    for treat in TREATMENTS:
        cols = [c for c in pg_wide.columns if c.startswith(treat + "_")]
        if pg_wide.loc[prot, cols].notna().sum() >= 2:
            keep.append(prot)
            break
pg_filt_all = pg_wide.loc[keep]

gene_name_map    = {}
protein_name_map = {}
fasta_sequences  = {}

if FASTA_PATH and os.path.exists(FASTA_PATH):
    current_acc = None
    current_seq = []
    with open(FASTA_PATH, "r") as f:
        for line in f:
            line_s = line.strip()
            if line_s.startswith(">"):
                if current_acc and current_seq:
                    fasta_sequences[current_acc] = "".join(current_seq)
                parts = line_s.split("|")
                acc = parts[1] if len(parts) >= 2 else line_s[1:].split()[0]
                current_acc = acc
                current_seq = []
                gn_match = re.search(r"GN=(\S+)", line_s)
                if gn_match:
                    gene_name_map[acc] = gn_match.group(1)
                if "|" in line_s:
                    after_pipe = line_s.split("|")[-1]
                    pn2 = re.search(r"^\S+\s+(.+?)(?:\sOS=)", after_pipe)
                    if pn2:
                        protein_name_map[acc] = pn2.group(1).strip()
                else:
                    pn_match = re.search(r"(?<=\s).+?(?=\sOS=)", line_s)
                    if pn_match:
                        protein_name_map[acc] = pn_match.group(0).strip()
            else:
                current_seq.append(line_s)
    if current_acc and current_seq:
        fasta_sequences[current_acc] = "".join(current_seq)

ptm_proteins_original = set()
ptm_type_map_original = {}
ptm_site_details_all  = []

for ptm_name, ptm_path in PTM_FILES.items():
    if not os.path.exists(ptm_path):
        continue
    ptm_df = pd.read_csv(ptm_path, sep="\t")
    protein_col = ptm_df.columns[0]
    unique_prots = ptm_df[protein_col].unique()
    for acc in unique_prots:
        acc = str(acc).strip()
        ptm_proteins_original.add(acc)
        if acc not in ptm_type_map_original:
            ptm_type_map_original[acc] = set()
        ptm_type_map_original[acc].add(ptm_name)
    for _, row in ptm_df.iterrows():
        detail = {"Protein_ID": str(row[protein_col]).strip(), "PTM_type": ptm_name,
                  "Residue": row.get("Residue", ""), "Site": row.get("Site", ""),
                  "Sequence": row.get("Sequence", "")}
        if "Protein.Names" in ptm_df.columns:
            detail["Protein_Name"] = row["Protein.Names"]
        if "Gene.Names" in ptm_df.columns:
            detail["Gene_Name"] = row["Gene.Names"]
        ptm_site_details_all.append(detail)

ptm_type_map_v2 = {k: set(v) for k, v in ptm_type_map_original.items()}
ptm_proteins_v2 = set(ptm_proteins_original)
for acc in PTM2_MANUAL_IDS:
    ptm_proteins_v2.add(acc)
    if acc not in ptm_type_map_v2:
        ptm_type_map_v2[acc] = set()
    ptm_type_map_v2[acc].add("Curated_PTM")

model, alphabet = torch.hub.load("facebookresearch/esm:main", ESM_MODEL_NAME)
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if HALF_PRECISION and device.type == "cuda":
    model = model.half()
model = model.to(device)

n_layers = model.num_layers


def generate_embeddings(protein_list, sequences_dict):
    emb = {}
    total = len(protein_list)
    for i in range(0, total, BATCH_SIZE):
        batch_prots = protein_list[i:i + BATCH_SIZE]
        batch_data = [(p, sequences_dict[p][:MAX_SEQ_LEN]) for p in batch_prots]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[n_layers], return_contacts=False)
        token_reps = results["representations"][n_layers]
        for j, prot in enumerate(batch_prots):
            seq_len = min(len(sequences_dict[prot]), MAX_SEQ_LEN)
            prot_rep = token_reps[j, 1:seq_len + 1, :].float().mean(dim=0).cpu().numpy()
            emb[prot] = prot_rep
        del batch_tokens, results, token_reps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return emb


def get_ptm_label(prot, ptm_map):
    if prot not in ptm_map:
        return "No PTM"
    types = ptm_map[prot]
    has_acet = "Acetylation" in types
    has_oxid = "Oxidation" in types
    has_cur  = "Curated_PTM" in types
    if has_acet and has_oxid:
        return "Both"
    elif has_acet:
        return "Acetylation"
    elif has_oxid:
        return "Oxidation"
    elif has_cur:
        return "Curated PTM"
    return "PTM"


def run_stress_analysis(
    analysis_name, stress_df, protein_id_col, description_col, category_col,
    output_dir,
    ptm_mode,
    label_mode,
    pg_filt, fasta_seqs,
    ptm_proteins_set, ptm_type_map, ptm_site_details,
    cached_data=None,
):
    for sub in ["Embeddings", "Tables"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    stress_ids = stress_df[protein_id_col].astype(str).str.strip().unique().tolist()
    available = [p for p in stress_ids if p in pg_filt.index]

    if len(available) < 3:
        return False

    desc_map = {}
    for _, row in stress_df.iterrows():
        pid = str(row[protein_id_col]).strip()
        desc = str(row[description_col]).strip() if pd.notna(row[description_col]) else ""
        if desc and desc.lower() != "nan":
            clean = desc[:42] + "..." if len(desc) > 45 else desc
            desc_map[pid] = clean

    def get_protein_name_label(acc):
        if acc in desc_map and desc_map[acc]:
            return desc_map[acc]
        if acc in protein_name_map and protein_name_map[acc]:
            pn = protein_name_map[acc]
            return pn[:42] + "..." if len(pn) > 45 else pn
        return acc

    def get_label(acc):
        if label_mode == "protein_id":
            return acc
        return get_protein_name_label(acc)

    cat_map = {}
    if category_col and category_col in stress_df.columns:
        for _, row in stress_df.iterrows():
            pid = str(row[protein_id_col]).strip()
            cat = str(row[category_col]).strip() if pd.notna(row[category_col]) else "Other"
            if cat.lower() == "nan":
                cat = "Other"
            cat_map[pid] = cat

    if cached_data is not None:
        proteins_with_seq = cached_data["proteins_with_seq"]
        sequences      = cached_data["sequences"]
        embed_matrix   = cached_data["embed_matrix"]
        umap_results   = cached_data["umap_results"]
        umap_df        = cached_data["umap_df"]
        pg_log2        = cached_data["pg_log2"]
        treat_means_df = cached_data["treat_means_df"]
        log2fc_df      = cached_data["log2fc_df"]
        dominant_fc    = cached_data["dominant_fc"]
        max_absFC      = cached_data["max_absFC"]
        most_abundant_treat = cached_data["most_abundant_treat"]
        seq_lens       = cached_data["seq_lens"]
    else:
        pg_sub = pg_filt.loc[available]
        pg_log2 = np.log2(pg_sub).astype(np.float64)
        np.random.seed(42)
        for col in pg_log2.columns:
            mask = pg_log2[col].isna()
            if mask.sum() > 0:
                col_mean = pg_log2[col].dropna().mean()
                col_std  = pg_log2[col].dropna().std()
                if pd.isna(col_std) or col_std == 0:
                    col_std = 1.0
                pg_log2.loc[mask, col] = np.random.normal(
                    col_mean - 1.8 * col_std, 0.3 * col_std, size=mask.sum())

        treat_means = {}
        for treat in TREATMENTS:
            cols = [c for c in pg_log2.columns if c.startswith(treat + "_")]
            treat_means[treat] = pg_log2[cols].mean(axis=1)
        treat_means_df = pd.DataFrame(treat_means)

        log2fc = {}
        for treat in TREATMENTS:
            if treat != CONTROL:
                log2fc[f"{treat}_vs_{CONTROL}"] = treat_means_df[treat] - treat_means_df[CONTROL]
        log2fc_df = pd.DataFrame(log2fc)

        dominant_contrast = log2fc_df.abs().idxmax(axis=1)
        dominant_fc = pd.Series(
            [log2fc_df.loc[p, dominant_contrast.loc[p]] for p in log2fc_df.index],
            index=log2fc_df.index, name="dominant_log2FC")
        max_absFC = log2fc_df.abs().max(axis=1)
        most_abundant_treat = treat_means_df.idxmax(axis=1)

        sequences = {}
        for prot in available:
            if prot in fasta_seqs:
                sequences[prot] = fasta_seqs[prot]
        missing = [p for p in available if p not in sequences]
        if missing:
            UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
            for i in range(0, len(missing), 50):
                batch = missing[i:i + 50]
                query = " OR ".join([f"accession:{acc}" for acc in batch])
                params = {"query": query, "format": "fasta", "size": len(batch)}
                try:
                    resp = requests.get(UNIPROT_URL, params=params, timeout=60)
                    if resp.status_code == 200:
                        cur_acc = None; cur_seq = []
                        for line in resp.text.strip().split("\n"):
                            if line.startswith(">"):
                                if cur_acc and cur_seq:
                                    sequences[cur_acc] = "".join(cur_seq)
                                parts = line.split("|")
                                cur_acc = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                                cur_seq = []
                            else:
                                cur_seq.append(line.strip())
                        if cur_acc and cur_seq:
                            sequences[cur_acc] = "".join(cur_seq)
                except Exception:
                    pass
                time.sleep(0.5)

        proteins_with_seq = [p for p in available if p in sequences]

        if len(proteins_with_seq) < 3:
            return False

        seq_lens = {p: len(sequences[p]) for p in proteins_with_seq}

        embeddings = generate_embeddings(proteins_with_seq, sequences)
        embed_matrix = np.vstack([embeddings[p] for p in proteins_with_seq])

        umap_mod = ensure_package("umap", "umap-learn")
        umap_configs = [
            {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
            {"n_neighbors": 10, "min_dist": 0.2, "metric": "cosine"},
            {"n_neighbors": 20, "min_dist": 0.05, "metric": "cosine"},
        ]
        umap_results = {}
        for idx, cfg in enumerate(umap_configs):
            nn = min(cfg["n_neighbors"], len(proteins_with_seq) - 1)
            reducer = umap_mod.UMAP(n_components=2, n_neighbors=nn,
                                    min_dist=cfg["min_dist"], metric=cfg["metric"],
                                    random_state=42)
            coords = reducer.fit_transform(embed_matrix)
            umap_results[idx] = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"],
                                             index=proteins_with_seq)
        umap_df = umap_results[0].copy()

    embed_df_save = pd.DataFrame(embed_matrix, index=proteins_with_seq)
    embed_df_save.to_parquet(os.path.join(output_dir, "Embeddings",
                                          "esm2_stress_embeddings.parquet"))

    include_ptm = ptm_mode != "none"
    active_ptm_set = ptm_proteins_set
    active_ptm_map = ptm_type_map

    ptm_proteins_sub = active_ptm_set.intersection(set(proteins_with_seq))
    ptm_status = pd.Series([get_ptm_label(p, active_ptm_map) for p in proteins_with_seq],
                           index=proteins_with_seq, name="PTM_status")
    ptm_binary = pd.Series(["PTM" if p in ptm_proteins_sub else "No PTM"
                            for p in proteins_with_seq],
                           index=proteins_with_seq, name="PTM_binary")

    master = umap_df.copy()
    master["Protein_ID"]          = master.index
    master["Protein_Name"]        = [get_protein_name_label(p) for p in master.index]
    master["Display_Label"]       = [get_label(p) for p in master.index]
    master["Gene_Name"]           = [gene_name_map.get(p, "") for p in master.index]
    master["Functional_Category"] = [cat_map.get(p, "Other") for p in master.index]
    master["dominant_log2FC"]     = dominant_fc.reindex(master.index).values
    master["max_abs_log2FC"]      = max_absFC.reindex(master.index).values
    master["mean_intensity"]      = pg_log2.loc[master.index].mean(axis=1).values
    master["seq_length"]          = [seq_lens.get(p, np.nan) for p in master.index]
    master["most_abundant_treat"] = most_abundant_treat.reindex(master.index).values

    if include_ptm:
        master["PTM_status"] = ptm_status.reindex(master.index).fillna("No PTM").values
        master["PTM_binary"] = ptm_binary.reindex(master.index).fillna("No PTM").values

    for contrast in log2fc_df.columns:
        master[contrast] = log2fc_df[contrast].reindex(master.index).values

    master.to_csv(os.path.join(output_dir, "Tables", "stress_master_annotation.csv"))

    if include_ptm and ptm_site_details:
        sub_details = [d for d in ptm_site_details
                       if d["Protein_ID"] in set(proteins_with_seq)]
        if sub_details:
            pd.DataFrame(sub_details).to_csv(
                os.path.join(output_dir, "Tables", "PTM_site_details_stress.csv"),
                index=False)

    cos_sim = cosine_similarity(embed_matrix)
    cos_sim_df = pd.DataFrame(cos_sim, index=proteins_with_seq,
                              columns=proteins_with_seq)

    if include_ptm:
        ptm_list = [p for p in proteins_with_seq if p in ptm_proteins_sub]
        n_ptm = len(ptm_list)
        if n_ptm >= 3:
            ptm_idx   = [proteins_with_seq.index(p) for p in ptm_list]
            ptm_embed = embed_matrix[ptm_idx]
            ptm_cos   = cosine_similarity(ptm_embed)
            iu        = np.triu_indices(n_ptm, k=1)
            observed_mean = ptm_cos[iu].mean()

            N_PERM = 10000
            np.random.seed(42)
            perm_means = np.zeros(N_PERM)
            n_total = len(proteins_with_seq)
            for perm in range(N_PERM):
                rand_idx = np.random.choice(n_total, size=n_ptm, replace=False)
                rand_cos = cosine_similarity(embed_matrix[rand_idx])
                perm_means[perm] = rand_cos[np.triu_indices(n_ptm, k=1)].mean()
            p_value = (np.sum(perm_means >= observed_mean) + 1) / (N_PERM + 1)

            pd.DataFrame({
                "metric": ["observed_mean_cosine_sim", "permutation_mean",
                           "permutation_sd", "p_value", "n_ptm_proteins",
                           "n_permutations", "n_total_stress_proteins", "model"],
                "value": [observed_mean, perm_means.mean(), perm_means.std(),
                          p_value, n_ptm, N_PERM, n_total, ESM_MODEL_NAME],
            }).to_csv(os.path.join(output_dir, "Tables",
                                   "PTM_permutation_test_results.csv"), index=False)

    cos_out = cos_sim_df.copy()
    cos_out.index   = [get_label(p) for p in cos_out.index]
    cos_out.columns = [get_label(p) for p in cos_out.columns]
    cos_out.to_csv(os.path.join(output_dir, "Tables", "cosine_similarity_matrix.csv"))

    return {
        "proteins_with_seq": proteins_with_seq,
        "sequences": sequences,
        "embed_matrix": embed_matrix,
        "umap_results": umap_results,
        "umap_df": umap_df,
        "pg_log2": pg_log2,
        "treat_means_df": treat_means_df,
        "log2fc_df": log2fc_df,
        "dominant_fc": dominant_fc,
        "max_absFC": max_absFC,
        "most_abundant_treat": most_abundant_treat,
        "seq_lens": seq_lens,
    }


stress_dep_df = pd.read_csv(STRESS_DEP_CSV)

out_a1 = os.path.join(BASE_OUTPUT, "Results_StressDEP_withPTM")
cache_a = run_stress_analysis(
    analysis_name="Stress DEP (core)", stress_df=stress_dep_df,
    protein_id_col="Protein.ID", description_col="Description",
    category_col="Category", output_dir=out_a1,
    ptm_mode="original", label_mode="protein_name",
    pg_filt=pg_filt_all, fasta_seqs=fasta_sequences,
    ptm_proteins_set=ptm_proteins_original, ptm_type_map=ptm_type_map_original,
    ptm_site_details=ptm_site_details_all, cached_data=None)

if cache_a:
    out_a2 = os.path.join(BASE_OUTPUT, "Results_StressDEP_noPTM")
    _ = run_stress_analysis(
        analysis_name="Stress DEP (core)", stress_df=stress_dep_df,
        protein_id_col="Protein.ID", description_col="Description",
        category_col="Category", output_dir=out_a2,
        ptm_mode="none", label_mode="protein_name",
        pg_filt=pg_filt_all, fasta_seqs=fasta_sequences,
        ptm_proteins_set=ptm_proteins_original, ptm_type_map=ptm_type_map_original,
        ptm_site_details=ptm_site_details_all, cached_data=cache_a)

    out_a3 = os.path.join(BASE_OUTPUT, "Results_StressDEP_withPTM2")
    _ = run_stress_analysis(
        analysis_name="Stress DEP (core)", stress_df=stress_dep_df,
        protein_id_col="Protein.ID", description_col="Description",
        category_col="Category", output_dir=out_a3,
        ptm_mode="ptm2", label_mode="protein_name",
        pg_filt=pg_filt_all, fasta_seqs=fasta_sequences,
        ptm_proteins_set=ptm_proteins_v2, ptm_type_map=ptm_type_map_v2,
        ptm_site_details=ptm_site_details_all, cached_data=cache_a)

    out_a4 = os.path.join(BASE_OUTPUT, "Results_StressDEP_ProteinID")
    _ = run_stress_analysis(
        analysis_name="Stress DEP (core)", stress_df=stress_dep_df,
        protein_id_col="Protein.ID", description_col="Description",
        category_col="Category", output_dir=out_a4,
        ptm_mode="original", label_mode="protein_id",
        pg_filt=pg_filt_all, fasta_seqs=fasta_sequences,
        ptm_proteins_set=ptm_proteins_original, ptm_type_map=ptm_type_map_original,
        ptm_site_details=ptm_site_details_all, cached_data=cache_a)

stress_annot_df = pd.read_csv(STRESS_ANNOT_CSV)

out_b1 = os.path.join(BASE_OUTPUT, "Results_StressAnnotated_withPTM")
cache_b = run_stress_analysis(
    analysis_name="Stress Annotated (expanded)", stress_df=stress_annot_df,
    protein_id_col="UniProt_ID", description_col="Description",
    category_col="Functional_Category", output_dir=out_b1,
    ptm_mode="original", label_mode="protein_name",
    pg_filt=pg_filt_all, fasta_seqs=fasta_sequences,
    ptm_proteins_set=ptm_proteins_original, ptm_type_map=ptm_type_map_original,
    ptm_site_details=ptm_site_details_all, cached_data=None)

if cache_b:
    out_b2 = os.path.join(BASE_OUTPUT, "Results_StressAnnotated_noPTM")
    _ = run_stress_analysis(
        analysis_name="Stress Annotated (expanded)", stress_df=stress_annot_df,
        protein_id_col="UniProt_ID", description_col="Description",
        category_col="Functional_Category", output_dir=out_b2,
        ptm_mode="none", label_mode="protein_name",
        pg_filt=pg_filt_all, fasta_seqs=fasta_sequences,
        ptm_proteins_set=ptm_proteins_original, ptm_type_map=ptm_type_map_original,
        ptm_site_details=ptm_site_details_all, cached_data=cache_b)

    out_b3 = os.path.join(BASE_OUTPUT, "Results_StressAnnotated_withPTM2")
    _ = run_stress_analysis(
        analysis_name="Stress Annotated (expanded)", stress_df=stress_annot_df,
        protein_id_col="UniProt_ID", description_col="Description",
        category_col="Functional_Category", output_dir=out_b3,
        ptm_mode="ptm2", label_mode="protein_name",
        pg_filt=pg_filt_all, fasta_seqs=fasta_sequences,
        ptm_proteins_set=ptm_proteins_v2, ptm_type_map=ptm_type_map_v2,
        ptm_site_details=ptm_site_details_all, cached_data=cache_b)

    out_b4 = os.path.join(BASE_OUTPUT, "Results_StressAnnotated_ProteinID")
    _ = run_stress_analysis(
        analysis_name="Stress Annotated (expanded)", stress_df=stress_annot_df,
        protein_id_col="UniProt_ID", description_col="Description",
        category_col="Functional_Category", output_dir=out_b4,
        ptm_mode="original", label_mode="protein_id",
        pg_filt=pg_filt_all, fasta_seqs=fasta_sequences,
        ptm_proteins_set=ptm_proteins_original, ptm_type_map=ptm_type_map_original,
        ptm_site_details=ptm_site_details_all, cached_data=cache_b)

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
