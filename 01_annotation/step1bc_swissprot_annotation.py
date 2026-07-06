## Secoond part of data processing
RESULTS_DIR    = "/content/drive/MyDrive/ESM-2 embeddings/Results"
SWISSPROT_DIR  = "/content/drive/MyDrive/ESM-2 embeddings/SwissProt"

ESM_MODEL_NAME = "esm2_t48_15B_UR50D"
EMBEDDING_DIM  = 5120
MAX_SEQ_LEN    = 1022
BATCH_SIZE     = 2
HALF_PRECISION = True

TAXONOMY_ID    = 33090
UNIPROT_QUERY  = f"(taxonomy_id:{TAXONOMY_ID}) AND (reviewed:true)"

TOP_K          = 5

CHECKPOINT_EVERY = 500

import os
import sys
import re
import csv
import time
import json
import logging
from datetime import datetime

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SWISSPROT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "step1BC_log.txt"), mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

print("=" * 70)
print("STEPS 1B + 1C: Swiss-Prot Embeddings & Annotation Transfer")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model: {ESM_MODEL_NAME}")
print()

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

def download_swissprot_plants():
    """Download reviewed plant proteins from UniProt REST API."""
    import urllib.request
    import gzip

    fasta_path = os.path.join(SWISSPROT_DIR, "swissprot_viridiplantae.fasta")

    if os.path.exists(fasta_path):

        n = sum(1 for line in open(fasta_path) if line.startswith('>'))
        if n > 10000:
            logging.info(f"Swiss-Prot FASTA already exists: {n} proteins")
            return fasta_path
        else:
            logging.info(f"Existing FASTA has only {n} entries — re-downloading")

    logging.info("Downloading Swiss-Prot Viridiplantae from UniProt...")
    logging.info(f"  Query: {UNIPROT_QUERY}")

    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": UNIPROT_QUERY,
        "format": "fasta",
        "compressed": "false",
    }

    query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{base_url}?{query_string}"
    logging.info(f"  URL: {url[:100]}...")

    try:
        import urllib.parse
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Python/ESM2-annotation')

        logging.info("  Downloading (this may take 2-5 minutes)...")
        t0 = time.time()

        with urllib.request.urlopen(req, timeout=600) as response:
            data = response.read().decode('utf-8')

        with open(fasta_path, 'w') as f:
            f.write(data)

        n_proteins = data.count('>')
        elapsed = time.time() - t0
        file_size_mb = len(data) / 1e6
        logging.info(f"  Downloaded {n_proteins} proteins ({file_size_mb:.1f} MB) in {elapsed:.0f}s")

    except Exception as e:
        logging.error(f"Download failed: {e}")
        logging.info("FALLBACK: Trying alternative URL format...")

        url2 = (f"https://rest.uniprot.org/uniprotkb/search?"
                f"query={urllib.parse.quote(UNIPROT_QUERY)}"
                f"&format=fasta&size=500")

        all_data = []
        cursor = None
        page = 0

        while True:
            page_url = url2 if cursor is None else cursor
            try:
                req = urllib.request.Request(page_url)
                req.add_header('User-Agent', 'Python/ESM2-annotation')
                with urllib.request.urlopen(req, timeout=300) as response:
                    page_data = response.read().decode('utf-8')
                    all_data.append(page_data)
                    page += 1

                    link_header = response.headers.get('Link', '')
                    if 'rel="next"' in link_header:
                        cursor = link_header.split('<')[1].split('>')[0]
                        if page % 10 == 0:
                            logging.info(f"  Page {page}: {sum(d.count('>') for d in all_data)} proteins so far...")
                    else:
                        break
            except Exception as e2:
                logging.error(f"  Page {page} failed: {e2}")
                break

        if all_data:
            full_data = ''.join(all_data)
            with open(fasta_path, 'w') as f:
                f.write(full_data)
            n_proteins = full_data.count('>')
            logging.info(f"  Downloaded {n_proteins} proteins via pagination ({page} pages)")
        else:
            sys.exit("ERROR: Could not download Swiss-Prot data. "
                     "Please download manually from uniprot.org")

    return fasta_path

def parse_swissprot_fasta(fasta_path):
    """Parse Swiss-Prot FASTA into sequences and metadata."""
    logging.info(f"Parsing Swiss-Prot FASTA: {fasta_path}")

    sequences = {}
    metadata = {}
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):

                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_seq = []

                header = line[1:]
                parts = header.split('|')
                if len(parts) >= 3:
                    db = parts[0]
                    accession = parts[1]
                    rest = parts[2]

                    entry_parts = rest.split(' ', 1)
                    entry_name = entry_parts[0]
                    full_desc = entry_parts[1] if len(entry_parts) > 1 else ''

                    description = re.sub(r'\s*OS=.*$', '', full_desc)
                    organism = re.search(r'OS=(.+?)(?:\s+OX=|\s+GN=|\s+PE=|\s*$)', full_desc)
                    gene = re.search(r'GN=(\S+)', full_desc)
                    tax_id = re.search(r'OX=(\d+)', full_desc)

                    current_id = accession
                    metadata[accession] = {
                        'entry_name': entry_name,
                        'description': description,
                        'organism': organism.group(1) if organism else '',
                        'gene': gene.group(1) if gene else '',
                        'tax_id': tax_id.group(1) if tax_id else '',
                    }
                else:
                    current_id = header.split()[0]
                    metadata[current_id] = {
                        'entry_name': current_id,
                        'description': header,
                        'organism': '', 'gene': '', 'tax_id': ''
                    }
            else:
                current_seq.append(line)

    if current_id:
        sequences[current_id] = ''.join(current_seq)

    lengths = [len(s) for s in sequences.values()]
    logging.info(f"  Total proteins: {len(sequences)}")
    logging.info(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
                 f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")
    logging.info(f"  Proteins > {MAX_SEQ_LEN} aa (will truncate): "
                 f"{sum(1 for l in lengths if l > MAX_SEQ_LEN)}")

    from collections import Counter
    org_counts = Counter(m['organism'] for m in metadata.values())
    logging.info(f"  Top organisms:")
    for org, count in org_counts.most_common(10):
        logging.info(f"    {org}: {count}")

    return sequences, metadata

def compute_swissprot_embeddings(sequences, metadata):
    """Compute ESM-2 embeddings for all Swiss-Prot plant proteins."""
    import esm

    checkpoint_path = os.path.join(SWISSPROT_DIR, "swissprot_embeddings_checkpoint.pt")
    final_path = os.path.join(RESULTS_DIR, "swissprot_plant_embeddings.pt")

    if os.path.exists(final_path):
        logging.info(f"Loading pre-computed Swiss-Prot embeddings: {final_path}")
        saved = torch.load(final_path, map_location='cpu', weights_only=False)
        logging.info(f"  Loaded {saved['n_proteins']} protein embeddings")
        return saved['protein_ids'], saved['embeddings']

    completed_ids = set()
    completed_embeddings = {}
    if os.path.exists(checkpoint_path):
        logging.info("Found checkpoint — resuming from last save...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        completed_ids = set(ckpt['protein_ids'])
        for pid, emb in zip(ckpt['protein_ids'], ckpt['embeddings']):
            completed_embeddings[pid] = emb
        logging.info(f"  Resumed with {len(completed_ids)} already computed")

    all_ids = sorted(sequences.keys())
    remaining_ids = [pid for pid in all_ids if pid not in completed_ids]
    logging.info(f"\nProteins to embed: {len(remaining_ids)} "
                 f"(of {len(all_ids)} total, {len(completed_ids)} already done)")

    if not remaining_ids:

        protein_ids = sorted(completed_embeddings.keys())
        embedding_matrix = torch.stack([completed_embeddings[pid] for pid in protein_ids])
        torch.save({
            'protein_ids': protein_ids,
            'embeddings': embedding_matrix,
            'model': ESM_MODEL_NAME,
            'embedding_dim': EMBEDDING_DIM,
            'n_proteins': len(protein_ids),
        }, final_path)
        return protein_ids, embedding_matrix

    logging.info(f"Loading ESM-2 model: {ESM_MODEL_NAME}")
    t0 = time.time()
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    if HALF_PRECISION and device.type == "cuda":
        model = model.half()
    n_layers = model.num_layers
    logging.info(f"  Model loaded in {time.time()-t0:.1f}s, device={device}")

    n_batches = (len(remaining_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()
    newly_computed = 0
    errors = []

    for batch_idx in tqdm(range(n_batches), desc="Swiss-Prot embeddings"):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(remaining_ids))
        batch_ids = remaining_ids[start:end]

        batch_data = []
        for pid in batch_ids:
            seq = sequences[pid]
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]
            seq_clean = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', seq.upper())
            if len(seq_clean) > 0:
                batch_data.append((pid, seq_clean))

        if not batch_data:
            continue

        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[n_layers], return_contacts=False)

            representations = results["representations"][n_layers]

            for i, (label, seq_str) in enumerate(batch_data):
                seq_len = len(seq_str)
                per_residue = representations[i, 1:seq_len + 1, :].cpu().float()
                mean_emb = per_residue.mean(dim=0)
                completed_embeddings[label] = mean_emb
                newly_computed += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()

                for pid, seq_str in batch_data:
                    try:
                        _, _, tokens = batch_converter([(pid, seq_str)])
                        tokens = tokens.to(device)
                        with torch.no_grad():
                            res = model(tokens, repr_layers=[n_layers], return_contacts=False)
                        rep = res["representations"][n_layers]
                        per_res = rep[0, 1:len(seq_str)+1, :].cpu().float()
                        completed_embeddings[pid] = per_res.mean(dim=0)
                        newly_computed += 1
                    except RuntimeError:
                        errors.append(pid)
                        torch.cuda.empty_cache()
            else:
                raise

        if newly_computed > 0 and newly_computed % CHECKPOINT_EVERY == 0:
            ckpt_ids = sorted(completed_embeddings.keys())
            ckpt_matrix = torch.stack([completed_embeddings[pid] for pid in ckpt_ids])
            torch.save({
                'protein_ids': ckpt_ids,
                'embeddings': ckpt_matrix,
                'model': ESM_MODEL_NAME,
            }, checkpoint_path)
            elapsed = time.time() - t_start
            rate = newly_computed / elapsed
            remaining = len(remaining_ids) - newly_computed
            eta = remaining / rate if rate > 0 else 0
            logging.info(f"  Checkpoint: {len(ckpt_ids)} total, "
                         f"{newly_computed} new, ETA: {eta/60:.0f} min")

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    logging.info(f"\nSwiss-Prot embedding complete:")
    logging.info(f"  Newly computed: {newly_computed}")
    logging.info(f"  Errors: {len(errors)}")
    logging.info(f"  Total: {len(completed_embeddings)}")
    logging.info(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    protein_ids = sorted(completed_embeddings.keys())
    embedding_matrix = torch.stack([completed_embeddings[pid] for pid in protein_ids])

    torch.save({
        'protein_ids': protein_ids,
        'embeddings': embedding_matrix,
        'model': ESM_MODEL_NAME,
        'embedding_dim': embedding_matrix.shape[1],
        'n_proteins': len(protein_ids),
        'timestamp': datetime.now().isoformat(),
    }, final_path)
    logging.info(f"  Saved: {final_path} — shape {embedding_matrix.shape}")

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "swissprot_plant_embeddings.npz"),
        protein_ids=protein_ids,
        embeddings=embedding_matrix.numpy(),
    )

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logging.info("  Removed checkpoint file (no longer needed)")

    return protein_ids, embedding_matrix

def find_nearest_neighbors(chickpea_ids, chickpea_emb,
                           swissprot_ids, swissprot_emb,
                           swissprot_meta, top_k=TOP_K):
    """Find top-K Swiss-Prot nearest neighbors for each chickpea protein."""
    logging.info(f"\n{'='*70}")
    logging.info("STEP 1C: FAISS Nearest-Neighbor Search")
    logging.info(f"{'='*70}")
    logging.info(f"  Query proteins (chickpea): {len(chickpea_ids)}")
    logging.info(f"  Reference (Swiss-Prot): {len(swissprot_ids)}")
    logging.info(f"  Top-K: {top_k}")

    try:
        import faiss
    except ImportError:
        logging.info("  Installing faiss-cpu...")
        os.system("pip install faiss-cpu -q")
        import faiss

    chickpea_np = chickpea_emb.numpy().astype(np.float32)
    swissprot_np = swissprot_emb.numpy().astype(np.float32)

    faiss.normalize_L2(chickpea_np)
    faiss.normalize_L2(swissprot_np)

    d = swissprot_np.shape[1]
    logging.info(f"  Building FAISS index (dim={d})...")
    index = faiss.IndexFlatIP(d)
    index.add(swissprot_np)
    logging.info(f"  Index built: {index.ntotal} vectors")

    logging.info(f"  Searching...")
    t0 = time.time()
    similarities, indices = index.search(chickpea_np, top_k)
    logging.info(f"  Search completed in {time.time()-t0:.2f}s")

    results = []
    for i, chickpea_id in enumerate(chickpea_ids):
        for rank in range(top_k):
            sp_idx = indices[i, rank]
            sp_id = swissprot_ids[sp_idx]
            sim = similarities[i, rank]
            meta = swissprot_meta.get(sp_id, {})

            results.append({
                'Chickpea_ID': chickpea_id,
                'Rank': rank + 1,
                'SwissProt_ID': sp_id,
                'Cosine_Similarity': round(float(sim), 4),
                'SwissProt_EntryName': meta.get('entry_name', ''),
                'SwissProt_Description': meta.get('description', ''),
                'SwissProt_Organism': meta.get('organism', ''),
                'SwissProt_Gene': meta.get('gene', ''),
            })

    df_results = pd.DataFrame(results)
    logging.info(f"\n  Results: {len(df_results)} rows ({len(chickpea_ids)} × {top_k})")

    top1_sims = similarities[:, 0]
    logging.info(f"\n  Top-1 similarity statistics:")
    logging.info(f"    Min:    {top1_sims.min():.4f}")
    logging.info(f"    Max:    {top1_sims.max():.4f}")
    logging.info(f"    Mean:   {top1_sims.mean():.4f}")
    logging.info(f"    Median: {np.median(top1_sims):.4f}")
    logging.info(f"    > 0.90: {(top1_sims > 0.90).sum()} proteins")
    logging.info(f"    > 0.80: {(top1_sims > 0.80).sum()} proteins")
    logging.info(f"    > 0.70: {(top1_sims > 0.70).sum()} proteins")
    logging.info(f"    < 0.50: {(top1_sims < 0.50).sum()} proteins (poorly matched)")

    return df_results, similarities

def transfer_annotations(df_nn, chickpea_metadata_path):
    """Combine nearest-neighbor results with chickpea metadata."""
    logging.info(f"\n{'='*70}")
    logging.info("ANNOTATION TRANSFER & QUALITY ASSESSMENT")
    logging.info(f"{'='*70}")

    df_meta = pd.read_csv(chickpea_metadata_path)
    logging.info(f"  Loaded chickpea metadata: {len(df_meta)} proteins")

    df_top1 = df_nn[df_nn['Rank'] == 1].copy()

    df_merged = df_meta.merge(
        df_top1[['Chickpea_ID', 'SwissProt_ID', 'Cosine_Similarity',
                 'SwissProt_EntryName', 'SwissProt_Description',
                 'SwissProt_Organism', 'SwissProt_Gene']],
        left_on='UniProt_ID',
        right_on='Chickpea_ID',
        how='left'
    )

    def classify_confidence(row):
        sim = row.get('Cosine_Similarity', 0)
        if pd.isna(sim) or sim == 0:
            return 'No match'
        elif sim >= 0.90:
            return 'Very high confidence'
        elif sim >= 0.80:
            return 'High confidence'
        elif sim >= 0.70:
            return 'Moderate confidence'
        elif sim >= 0.60:
            return 'Low confidence'
        else:
            return 'Very low confidence'

    df_merged['ESM2_Confidence'] = df_merged.apply(classify_confidence, axis=1)

    def classify_improvement(row):
        current = str(row.get('Description', '')).lower()
        transferred = str(row.get('SwissProt_Description', '')).lower()
        sim = row.get('Cosine_Similarity', 0)

        if pd.isna(sim) or sim < 0.60:
            return 'Insufficient similarity'
        elif 'uncharacterized' in current or current.strip() == '':
            return 'NEW ANNOTATION (was uncharacterized)'
        elif any(kw in current for kw in ['putative', 'probable', 'predicted',
                                           'hypothetical', '-like', 'family protein']):
            return 'IMPROVED (was partial annotation)'
        elif sim >= 0.80:
            return 'VALIDATED (high-confidence match)'
        else:
            return 'Consistent'

    df_merged['Annotation_Status'] = df_merged.apply(classify_improvement, axis=1)

    stress_keywords = [
        'stress', 'heat shock', 'hsp', 'dehydrin', 'lea', 'late embryogenesis',
        'peroxidase', 'superoxide', 'catalase', 'glutathione', 'thioredoxin',
        'redox', 'drought', 'cold', 'defense', 'pathogen', 'reactive oxygen',
        'antioxid', 'chaperone', 'protease inhibitor', 'universal stress',
        'osmotic', 'desiccation', 'senescence', 'ascorbate', 'metallothionein',
        'aquaporin', 'calmodulin', 'annexin', 'germin', 'chitinase'
    ]

    def check_stress(row):
        """Check both original AND transferred description for stress keywords."""
        original = str(row.get('Description', '')).lower()
        transferred = str(row.get('SwissProt_Description', '')).lower()
        combined = original + ' ' + transferred

        for kw in stress_keywords:
            if kw in combined:
                return kw.title()
        return 'No'

    df_merged['Stress_Related_ESM2'] = df_merged.apply(check_stress, axis=1)

    logging.info(f"\n  Annotation Status:")
    for status, count in df_merged['Annotation_Status'].value_counts().items():
        logging.info(f"    {status}: {count}")

    logging.info(f"\n  ESM2 Confidence:")
    for conf, count in df_merged['ESM2_Confidence'].value_counts().items():
        logging.info(f"    {conf}: {count}")

    original_stress = df_merged[
        df_merged['Description'].str.lower().str.contains(
            '|'.join(stress_keywords), na=False
        )
    ]
    esm2_stress = df_merged[df_merged['Stress_Related_ESM2'] != 'No']
    new_stress = esm2_stress[~esm2_stress['UniProt_ID'].isin(original_stress['UniProt_ID'])]

    logging.info(f"\n  Stress-related proteins:")
    logging.info(f"    Original annotation: {len(original_stress)}")
    logging.info(f"    After ESM-2 transfer: {len(esm2_stress)}")
    logging.info(f"    NEW stress proteins discovered: {len(new_stress)}")

    if len(new_stress) > 0:
        logging.info(f"\n  Newly identified stress proteins:")
        for _, row in new_stress.iterrows():
            logging.info(f"    {row['UniProt_ID']}: "
                         f"was '{row.get('Description', 'N/A')[:40]}' → "
                         f"matched '{row.get('SwissProt_Description', 'N/A')[:40]}' "
                         f"(sim={row.get('Cosine_Similarity', 0):.3f}, "
                         f"kw={row.get('Stress_Related_ESM2', '')})")

    return df_merged

def create_visualizations(df_merged, similarities):
    """Create summary plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        top1_sims = similarities[:, 0]
        axes[0].hist(top1_sims, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(x=0.80, color='red', linestyle='--', label='High conf. threshold')
        axes[0].axvline(x=0.60, color='orange', linestyle='--', label='Low conf. threshold')
        axes[0].set_xlabel('Cosine Similarity (Top-1 Match)')
        axes[0].set_ylabel('Number of Proteins')
        axes[0].set_title('Distribution of Nearest-Neighbor Similarities')
        axes[0].legend()

        status_counts = df_merged['Annotation_Status'].value_counts()
        colors = {
            'VALIDATED (high-confidence match)': '#2ecc71',
            'NEW ANNOTATION (was uncharacterized)': '#e74c3c',
            'IMPROVED (was partial annotation)': '#f39c12',
            'Consistent': '#3498db',
            'Insufficient similarity': '#95a5a6',
        }
        bar_colors = [colors.get(s, '#7f8c8d') for s in status_counts.index]
        axes[1].barh(range(len(status_counts)), status_counts.values, color=bar_colors)
        axes[1].set_yticks(range(len(status_counts)))
        axes[1].set_yticklabels(status_counts.index, fontsize=8)
        axes[1].set_xlabel('Number of Proteins')
        axes[1].set_title('Annotation Improvement Status')

        conf_order = ['Very high confidence', 'High confidence',
                      'Moderate confidence', 'Low confidence',
                      'Very low confidence', 'No match']
        conf_counts = df_merged['ESM2_Confidence'].value_counts()
        conf_vals = [conf_counts.get(c, 0) for c in conf_order if conf_counts.get(c, 0) > 0]
        conf_labels = [c for c in conf_order if conf_counts.get(c, 0) > 0]
        conf_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6']
        axes[2].pie(conf_vals, labels=conf_labels, colors=conf_colors[:len(conf_vals)],
                    autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
        axes[2].set_title('ESM-2 Annotation Confidence')

        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, "annotation_transfer_summary.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"\n  Saved figure: {fig_path}")

    except ImportError:
        logging.info("  matplotlib not available — skipping plots")

def save_final_results(df_nn, df_merged):
    """Save all output files."""

    nn_path = os.path.join(RESULTS_DIR, "annotation_transfer_all_neighbors.csv")
    df_nn.to_csv(nn_path, index=False)
    logging.info(f"  Saved: {nn_path}")

    summary_path = os.path.join(RESULTS_DIR, "annotation_transfer_summary.csv")
    cols_to_save = [
        'UniProt_ID', 'Gene_Name', 'Database', 'Description',
        'Sequence_Length', 'SwissProt_ID', 'SwissProt_Description',
        'SwissProt_Organism', 'SwissProt_Gene', 'Cosine_Similarity',
        'ESM2_Confidence', 'Annotation_Status', 'Stress_Related_ESM2'
    ]
    cols_available = [c for c in cols_to_save if c in df_merged.columns]
    df_merged[cols_available].to_csv(summary_path, index=False)
    logging.info(f"  Saved: {summary_path}")

    df_stress = df_merged[df_merged['Stress_Related_ESM2'] != 'No']
    stress_path = os.path.join(RESULTS_DIR, "stress_proteins_esm2.csv")
    df_stress[cols_available].to_csv(stress_path, index=False)
    logging.info(f"  Saved: {stress_path} ({len(df_stress)} stress proteins)")

    return summary_path, stress_path

def main():
    t_total = time.time()

    logging.info("=" * 70)
    logging.info("STEP 1B: Swiss-Prot Viridiplantae Embeddings")
    logging.info("=" * 70)

    swissprot_fasta = download_swissprot_plants()

    sp_sequences, sp_metadata = parse_swissprot_fasta(swissprot_fasta)

    sp_ids, sp_embeddings = compute_swissprot_embeddings(sp_sequences, sp_metadata)

    logging.info(f"\nLoading Step 1A results from: {RESULTS_DIR}")
    step1a = torch.load(
        os.path.join(RESULTS_DIR, "embeddings_mean.pt"),
        map_location='cpu',
        weights_only=False
    )
    ck_ids = step1a['protein_ids']
    ck_embeddings = step1a['embeddings']
    logging.info(f"  Chickpea: {len(ck_ids)} proteins × {ck_embeddings.shape[1]} dims")
    logging.info(f"  Swiss-Prot: {len(sp_ids)} proteins × {sp_embeddings.shape[1]} dims")

    assert ck_embeddings.shape[1] == sp_embeddings.shape[1], (
        f"Embedding dimension mismatch! Chickpea={ck_embeddings.shape[1]}, "
        f"Swiss-Prot={sp_embeddings.shape[1]}. Both must use the same ESM-2 model."
    )

    df_nn, similarities = find_nearest_neighbors(
        ck_ids, ck_embeddings,
        sp_ids, sp_embeddings,
        sp_metadata
    )

    metadata_csv = os.path.join(RESULTS_DIR, "protein_metadata.csv")
    df_merged = transfer_annotations(df_nn, metadata_csv)

    create_visualizations(df_merged, similarities)

    summary_path, stress_path = save_final_results(df_nn, df_merged)

    elapsed = time.time() - t_total
    logging.info(f"\n{'='*70}")
    logging.info(f"STEPS 1B + 1C COMPLETE")
    logging.info(f"{'='*70}")
    logging.info(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logging.info(f"  Swiss-Prot proteins embedded: {len(sp_ids)}")
    logging.info(f"  Chickpea proteins annotated: {len(ck_ids)}")
    logging.info(f"\n  Output files:")
    logging.info(f"    • annotation_transfer_summary.csv    — Main results table")
    logging.info(f"    • annotation_transfer_all_neighbors.csv — All top-{TOP_K} matches")
    logging.info(f"    • stress_proteins_esm2.csv           — Stress protein subset")
    logging.info(f"    • annotation_transfer_summary.png    — Summary figure")
    logging.info(f"    • swissprot_plant_embeddings.pt      — Swiss-Prot embeddings")
    logging.info(f"\n  Next: Use annotation_transfer_summary.csv to revise your")
    logging.info(f"        stress protein list, then proceed to Phase 2.")

if __name__ == "__main__":
    main()
