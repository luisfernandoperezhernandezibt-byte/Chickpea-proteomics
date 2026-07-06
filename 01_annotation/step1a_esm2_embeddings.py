#!/usr/bin/env python3
import os
import sys
import re
import csv
import time
import logging
from pathlib import Path
from datetime import datetime

USE_COLAB = True

TREATMENT_CODES = ["CS1", "CS2", "CS3", "CD1", "CD2", "CD3", "RH1", "RH2", "RH3"]

if USE_COLAB:

    TSV_DIR        = "/content/drive/MyDrive/ESM-2 embeddings/All the reports"
    FASTA_PATH     = "/content/drive/MyDrive/ESM-2 embeddings/uniprotkb_taxonomy_id_3827_2026_02_09.fasta"
    OUTPUT_DIR     = "/content/drive/MyDrive/ESM-2 embeddings/Results"
    ESM_MODEL_NAME = "esm2_t48_15B_UR50D"
    EMBEDDING_DIM  = 5120
    MAX_SEQ_LEN    = 1022
    BATCH_SIZE     = 2
    HALF_PRECISION = True
    TOP_N_PROTEINS = None
else:

    TSV_DIR        = r"D:\ESM-2 embeddings\All the reports"
    FASTA_PATH     = r"D:\ESM-2 embeddings\uniprotkb_taxonomy_id_3827_2026_02_09.fasta"
    OUTPUT_DIR     = r"D:\ESM-2 embeddings\Results"
    ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
    EMBEDDING_DIM  = 320
    MAX_SEQ_LEN    = 1022
    BATCH_SIZE     = 8
    HALF_PRECISION = False
    TOP_N_PROTEINS = 30

def setup_environment():
    """Import libraries and configure output directories."""
    global torch, esm, pd, tqdm

    print("=" * 70)
    print("STEP 1A v2: ESM-2 Embedding Computation for Chickpea Proteome")
    print("           (multi-TSV input mode)")
    print("=" * 70)
    print(f"Start time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model      : {ESM_MODEL_NAME}")
    print(f"TSV dir    : {TSV_DIR}")
    print(f"Output dir : {OUTPUT_DIR}\n")

    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU device:      {torch.cuda.get_device_name(0)}")
            print(f"  VRAM:            "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠ Running on CPU — this will be slow for large models")
    except ImportError:
        sys.exit("ERROR: PyTorch not installed. Run: pip install torch")

    try:
        import esm as esm_module
        esm = esm_module
        print(f"  ESM version:     "
              f"{esm.__version__ if hasattr(esm, '__version__') else 'installed'}")
    except ImportError:
        sys.exit("ERROR: fair-esm not installed. Run: pip install fair-esm")

    try:
        import pandas as pd_module
        pd = pd_module
    except ImportError:
        sys.exit("ERROR: pandas not installed. Run: pip install pandas")

    try:
        from tqdm import tqdm as tqdm_module
        tqdm = tqdm_module
    except ImportError:
        def tqdm(iterable, **kwargs):
            total = kwargs.get('total', None)
            desc  = kwargs.get('desc', '')
            for i, item in enumerate(iterable):
                if total and (i % max(1, total // 20) == 0):
                    print(f"  {desc}: {i}/{total} ({100*i/total:.0f}%)")
                yield item

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "embeddings_per_residue_2"), exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, "embedding_log_2.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Configuration: model={ESM_MODEL_NAME}, batch={BATCH_SIZE}, "
                 f"fp16={HALF_PRECISION}, top_n={TOP_N_PROTEINS}")
    print()

def load_protein_ids_from_tsv_files(tsv_dir, treatment_codes):
    all_ids    = set()
    names_map  = {}
    genes_map  = {}
    source_map = {}

    for code in treatment_codes:
        tsv_path = os.path.join(tsv_dir, f"report_{code}.tsv")

        if not os.path.isfile(tsv_path):
            logging.warning(f"  [SKIP] TSV file not found: {tsv_path}")
            continue

        logging.info(f"  Reading TSV: {tsv_path}")
        try:
            df = pd.read_csv(
                tsv_path,
                sep="\t",
                usecols=["Protein.Ids", "Protein.Names", "Genes"],
                dtype=str,
                low_memory=False,
            )
        except ValueError:

            existing_cols = pd.read_csv(tsv_path, sep="\t", nrows=0).columns.tolist()
            use_cols = [c for c in ["Protein.Ids", "Protein.Names", "Genes"]
                        if c in existing_cols]
            df = pd.read_csv(tsv_path, sep="\t", usecols=use_cols,
                             dtype=str, low_memory=False)

            for col in ["Protein.Ids", "Protein.Names", "Genes"]:
                if col not in df.columns:
                    df[col] = float("nan")

        n_before = len(all_ids)

        for _, row in df[["Protein.Ids", "Protein.Names", "Genes"]].drop_duplicates().iterrows():
            ids   = str(row["Protein.Ids"]).split(";") if pd.notna(row["Protein.Ids"]) else []
            names = (str(row["Protein.Names"]).split(";")
                     if pd.notna(row.get("Protein.Names")) else [])
            genes = (str(row["Genes"]).split(";")
                     if pd.notna(row.get("Genes")) else [])

            for i, pid in enumerate(ids):
                pid = pid.strip()
                if not pid or pid.lower() == "nan":
                    continue
                all_ids.add(pid)

                if pid not in names_map and i < len(names) and names[i].strip():
                    names_map[pid] = names[i].strip()
                if pid not in genes_map and i < len(genes) and genes[i].strip():
                    genes_map[pid] = genes[i].strip()

                source_map.setdefault(pid, [])
                if code not in source_map[pid]:
                    source_map[pid].append(code)

        n_new = len(all_ids) - n_before
        logging.info(f"    {code}: {n_new} new proteins "
                     f"(running total: {len(all_ids)})")

    logging.info(f"\n  ➜  Total unique protein IDs across all TSVs: {len(all_ids)}")
    return sorted(all_ids), names_map, genes_map, source_map

def load_sequences_from_fasta(fasta_path, target_ids):
    """Parse UniProt FASTA file and extract sequences for target proteins."""
    logging.info(f"Reading FASTA: {fasta_path}")
    target_set   = set(target_ids)
    sequences    = {}
    descriptions = {}
    db_types     = {}

    current_id  = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_id in target_set:
                    sequences[current_id] = ''.join(current_seq)
                current_seq = []
                current_id  = None

                parts = line[1:].split('|')
                if len(parts) >= 3:
                    db        = parts[0]
                    accession = parts[1]
                    rest      = parts[2]
                    desc      = rest.split(' ', 1)[1] if ' ' in rest else ''
                    desc_clean = re.sub(r'\s*OS=.*$', '', desc)
                    current_id = accession
                    descriptions[accession] = desc_clean
                    db_types[accession]     = db
            else:
                current_seq.append(line)

        if current_id and current_id in target_set:
            sequences[current_id] = ''.join(current_seq)

    found   = len(sequences)
    missing = target_set - set(sequences.keys())
    logging.info(f"  Matched {found}/{len(target_set)} proteins to FASTA sequences")
    if missing:
        logging.warning(f"  {len(missing)} proteins not found in FASTA: "
                        f"{sorted(list(missing))[:5]}...")
    return sequences, descriptions, db_types

def load_esm_model(model_name):
    """Load ESM-2 model and alphabet."""
    logging.info(f"Loading ESM-2 model: {model_name}")
    t0 = time.time()

    model, alphabet  = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter  = alphabet.get_batch_converter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    if HALF_PRECISION and device.type == "cuda":
        model = model.half()
        logging.info("  Applied FP16 half precision")

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  Model loaded in {time.time()-t0:.1f}s "
                 f"({n_params/1e6:.0f}M parameters)")
    logging.info(f"  Device: {device}")

    n_layers = model.num_layers
    logging.info(f"  Layers: {n_layers}")
    return model, alphabet, batch_converter, device, n_layers

def compute_embeddings(model, alphabet, batch_converter, device, n_layers,
                       sequences, protein_ids):
    """
    Compute ESM-2 embeddings for all proteins.

    Per-residue files are saved to  embeddings_per_residue_2/
    Mean embeddings are returned as a dict {protein_id: tensor(D,)}.
    """
    logging.info(f"\nComputing embeddings for {len(protein_ids)} proteins...")
    logging.info(f"  Batch size: {BATCH_SIZE}")
    logging.info(f"  Max sequence length in dataset: "
                 f"{max(len(sequences[pid]) for pid in protein_ids)}")

    mean_embeddings  = {}
    per_residue_dir  = os.path.join(OUTPUT_DIR, "embeddings_per_residue_2")
    skipped          = []

    data     = [(pid, sequences[pid]) for pid in protein_ids]
    n_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()

    for batch_idx in tqdm(range(n_batches), desc="Computing embeddings",
                          total=n_batches):
        start      = batch_idx * BATCH_SIZE
        end        = min(start + BATCH_SIZE, len(data))
        batch_data = data[start:end]

        valid_batch = []
        for label, seq in batch_data:
            if len(seq) > MAX_SEQ_LEN:
                logging.warning(f"  Truncating {label}: {len(seq)} → {MAX_SEQ_LEN} aa")
                seq = seq[:MAX_SEQ_LEN]
            if len(seq) == 0:
                logging.warning(f"  Skipping {label}: empty sequence")
                skipped.append(label)
                continue
            seq_clean = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', seq.upper())
            valid_batch.append((label, seq_clean))

        if not valid_batch:
            continue

        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(valid_batch)
            batch_tokens = batch_tokens.to(device)
            if HALF_PRECISION and device.type == "cuda":
                batch_tokens = batch_tokens.long()

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[n_layers],
                                return_contacts=False)

            representations = results["representations"][n_layers]

            for i, (label, seq_str) in enumerate(valid_batch):
                seq_len    = len(seq_str)
                per_residue = representations[i, 1:seq_len + 1, :].cpu().float()
                mean_emb    = per_residue.mean(dim=0)

                mean_embeddings[label] = mean_emb

                per_res_path = os.path.join(per_residue_dir, f"{label}.pt")
                torch.save({
                    'protein_id':            label,
                    'sequence':              seq_str,
                    'per_residue_embedding': per_residue,
                    'mean_embedding':        mean_emb,
                    'model':                 ESM_MODEL_NAME,
                    'embedding_dim':         per_residue.shape[1],
                }, per_res_path)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"  OOM at batch {batch_idx}! Sequences: "
                              f"{[len(s) for _, s in valid_batch]}")
                logging.error("  → Reduce BATCH_SIZE or use a smaller model")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                for label, seq_str in valid_batch:
                    try:
                        single_batch = [(label, seq_str)]
                        _, _, tokens = batch_converter(single_batch)
                        tokens = tokens.to(device)
                        with torch.no_grad():
                            res = model(tokens, repr_layers=[n_layers],
                                        return_contacts=False)
                        rep         = res["representations"][n_layers]
                        seq_len     = len(seq_str)
                        per_residue = rep[0, 1:seq_len + 1, :].cpu().float()
                        mean_emb    = per_residue.mean(dim=0)
                        mean_embeddings[label] = mean_emb

                        per_res_path = os.path.join(per_residue_dir, f"{label}.pt")
                        torch.save({
                            'protein_id':            label,
                            'sequence':              seq_str,
                            'per_residue_embedding': per_residue,
                            'mean_embedding':        mean_emb,
                            'model':                 ESM_MODEL_NAME,
                            'embedding_dim':         per_residue.shape[1],
                        }, per_res_path)
                    except RuntimeError:
                        logging.error(f"  Failed even individually: "
                                      f"{label} ({len(seq_str)} aa)")
                        skipped.append(label)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            else:
                raise

    elapsed = time.time() - t0
    logging.info(f"\n  Embeddings computed in {elapsed:.1f}s "
                 f"({elapsed/len(protein_ids):.2f}s per protein)")
    logging.info(f"  Successful: {len(mean_embeddings)}")
    logging.info(f"  Skipped:    {len(skipped)}")
    if skipped:
        logging.info(f"  Skipped IDs: {skipped}")
    return mean_embeddings, skipped

def save_results(mean_embeddings, sequences, descriptions, db_types,
                 names_map, genes_map, source_map, skipped):
    """Save mean embeddings matrix and protein metadata CSV (v2 filenames)."""

    protein_ids      = sorted(mean_embeddings.keys())
    embedding_matrix = torch.stack([mean_embeddings[pid] for pid in protein_ids])

    emb_path = os.path.join(OUTPUT_DIR, "embeddings_mean_2.pt")
    torch.save({
        'protein_ids':   protein_ids,
        'embeddings':    embedding_matrix,
        'model':         ESM_MODEL_NAME,
        'embedding_dim': embedding_matrix.shape[1],
        'n_proteins':    len(protein_ids),
        'timestamp':     datetime.now().isoformat(),
        'source_tsv':    TREATMENT_CODES,
    }, emb_path)
    logging.info(f"\nSaved mean embeddings: {emb_path}")
    logging.info(f"  Shape: {embedding_matrix.shape}")

    try:
        import numpy as np
        np_path = os.path.join(OUTPUT_DIR, "embeddings_mean_2.npz")
        np.savez_compressed(
            np_path,
            protein_ids=protein_ids,
            embeddings=embedding_matrix.numpy(),
            model=ESM_MODEL_NAME,
        )
        logging.info(f"  Also saved as numpy: {np_path}")
    except ImportError:
        pass

    csv_path = os.path.join(OUTPUT_DIR, "protein_metadata_2.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'UniProt_ID', 'Entry_Name', 'Database', 'Gene_Name',
            'Description', 'Sequence_Length', 'Embedding_Computed',
            'Model_Used', 'Embedding_Dim', 'Found_In_Treatments'
        ])
        for pid in sorted(set(list(mean_embeddings.keys()) + skipped)):
            treatments = ";".join(sorted(source_map.get(pid, [])))
            writer.writerow([
                pid,
                names_map.get(pid, ''),
                'Swiss-Prot' if db_types.get(pid) == 'sp' else 'TrEMBL',
                genes_map.get(pid, ''),
                descriptions.get(pid, ''),
                len(sequences.get(pid, '')),
                'Yes' if pid in mean_embeddings else 'No (skipped)',
                ESM_MODEL_NAME,
                mean_embeddings[pid].shape[0] if pid in mean_embeddings else 'N/A',
                treatments,
            ])
    logging.info(f"  Metadata CSV: {csv_path}")
    return protein_ids, embedding_matrix

def validate_embeddings(protein_ids, embedding_matrix):
    """Run basic sanity checks on the computed embeddings."""
    logging.info("\n" + "=" * 70)
    logging.info("VALIDATION")
    logging.info("=" * 70)

    n, d = embedding_matrix.shape
    logging.info(f"  Proteins embedded: {n}")
    logging.info(f"  Embedding dimension: {d}")
    logging.info(f"  Expected dimension:  {EMBEDDING_DIM}")

    if d != EMBEDDING_DIM:
        logging.warning(f"  ⚠ Dimension mismatch! Got {d}, expected {EMBEDDING_DIM}")

    has_nan = torch.isnan(embedding_matrix).any().item()
    has_inf = torch.isinf(embedding_matrix).any().item()
    logging.info(f"  Contains NaN: {has_nan}")
    logging.info(f"  Contains Inf: {has_inf}")
    if has_nan or has_inf:
        logging.error("  ⚠ CRITICAL: Embeddings contain NaN/Inf values!")

    norms = torch.norm(embedding_matrix, dim=1)
    logging.info(f"  L2 norm — min: {norms.min():.3f}, max: {norms.max():.3f}, "
                 f"mean: {norms.mean():.3f}")

    if n >= 5:
        from torch.nn.functional import cosine_similarity
        logging.info(f"\n  Cosine similarity sample (first 5 proteins):")
        for i in range(min(5, n)):
            for j in range(i + 1, min(5, n)):
                sim = cosine_similarity(
                    embedding_matrix[i].unsqueeze(0),
                    embedding_matrix[j].unsqueeze(0)
                ).item()
                logging.info(f"    {protein_ids[i]} vs {protein_ids[j]}: {sim:.4f}")

    logging.info("\n✅ Validation complete")

def main():
    t_start = time.time()

    setup_environment()

    protein_ids, names_map, genes_map, source_map = load_protein_ids_from_tsv_files(
        TSV_DIR, TREATMENT_CODES
    )

    sequences, descriptions, db_types = load_sequences_from_fasta(FASTA_PATH, protein_ids)

    valid_ids = [pid for pid in protein_ids if pid in sequences]
    logging.info(f"\nProteins with sequences: {len(valid_ids)}")

    if TOP_N_PROTEINS is not None:
        valid_ids = valid_ids[:TOP_N_PROTEINS]
        logging.info(f"  Limited to {TOP_N_PROTEINS} proteins (local testing mode)")

    model, alphabet, batch_converter, device, n_layers = load_esm_model(ESM_MODEL_NAME)

    mean_embeddings, skipped = compute_embeddings(
        model, alphabet, batch_converter, device, n_layers,
        sequences, valid_ids
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    protein_ids_saved, embedding_matrix = save_results(
        mean_embeddings, sequences, descriptions, db_types,
        names_map, genes_map, source_map, skipped
    )

    validate_embeddings(protein_ids_saved, embedding_matrix)

    elapsed = time.time() - t_start
    logging.info(f"\n{'=' * 70}")
    logging.info(f"STEP 1A v2 COMPLETE")
    logging.info(f"{'=' * 70}")
    logging.info(f"  Total time:          {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logging.info(f"  TSV files processed: {len(TREATMENT_CODES)}")
    logging.info(f"  Proteins embedded:   {len(protein_ids_saved)}")
    logging.info(f"  Embedding dimension: {embedding_matrix.shape[1]}")
    logging.info(f"  Model:               {ESM_MODEL_NAME}")
    logging.info(f"  Output directory:    {OUTPUT_DIR}")
    logging.info(f"\n  Files created (all with _2 suffix):")
    logging.info(f"    • embeddings_mean_2.pt              — mean-pooled embeddings")
    logging.info(f"    • embeddings_mean_2.npz             — numpy format (compatibility)")
    logging.info(f"    • embeddings_per_residue_2/*.pt     — per-residue embeddings")
    logging.info(f"    • protein_metadata_2.csv            — protein metadata table")
    logging.info(f"    • embedding_log_2.txt               — this log file")
    logging.info(f"\n  Next step: Run Step 1B (Swiss-Prot plant embeddings) on Colab")

if __name__ == "__main__":
    main()
