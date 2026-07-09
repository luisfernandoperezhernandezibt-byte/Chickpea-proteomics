
!rm -rf /content/drive

from google.colab import drive
drive.mount('/content/drive')

import os, sys, time, requests
import pandas as pd
import numpy as np
import torch
import subprocess

print("Installing stable dependencies...")
subprocess.run(
    "pip install -q transformers accelerate py3Dmol biopython",
    shell=True
)

from transformers import AutoTokenizer, EsmForProteinFolding
import py3Dmol

print("=" * 72)
if not torch.cuda.is_available():
    raise RuntimeError("NO GPU DETECTED. Please enable GPU in Colab Runtime.")
print(f"GPU Active: {torch.cuda.get_device_name(0)}")

BASE_DIR = "/content/drive/MyDrive/ESM-Fold"
PDB_DIR  = os.path.join(BASE_DIR, "Target_PDBs")
os.makedirs(PDB_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_DIR, "structural_analysis_notes.txt")

def log_note(message):
    """Prints to console and appends to the rigorous .txt log file."""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

with open(LOG_FILE, "w") as f:
    f.write("====================================================\n")
    f.write(f" STRUCTURAL ANALYSIS LOG (A100 RUN) - {time.ctime()}\n")
    f.write("====================================================\n")

log_note(f"Directories initialized. PDBs will be saved to: {PDB_DIR}")

TARGET_PROTEINS = [
    "A0A1S2YJZ0", "A0A3Q7XXP6", "A0A0X9LEN0", "A0A1S2XYP2", "A0A1S2YGY5",
    "A0A3Q7K7A2", "A0A1S2YEV0", "A0A1S2YL09", "A0A109PJV0", "A0A1S2XFJ7",
    "A0A1S2YJT8", "A0A1S2XXX3", "A0A1S3E0N0", "A0A1S2YP37", "A0A1S2XPL0",
    "A0A1S2Y7V6", "A0A1S2Z0V6", "A0A1S2XUK3", "A0A1S2YIG4", "Q9ZNQ4",
    "A0A1S2XXX7",
]

log_note("\nLoading Hugging Face ESMFold Model (Stable Port)...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model     = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True
)
model = model.eval().cuda()

model.trunk.set_chunk_size(128)

log_note("Model loaded successfully. Optimized for A100 80GB VRAM.")
log_note("=" * 72)

log_note("\nMapping PTMs from TSV files...")

def load_ptms(filename, ptm_name):
    """Load PTM sites from a TSV; searches /content/ then Google Drive."""
    paths_to_check = [
        f"/content/{filename}",
        os.path.join(BASE_DIR, filename),
    ]

    filepath = None
    for path in paths_to_check:
        if os.path.exists(path):
            filepath = path
            break

    if not filepath:
        log_note(f"  [WARNING] {filename} not found. Please upload to /content/")
        return {}

    df = pd.read_csv(filepath, sep="\t")
    ptm_dict = {}

    for _, row in df.iterrows():
        prot_str    = str(row.get("Protein", ""))
        matched_prot = next(
            (p for p in TARGET_PROTEINS if p in prot_str), None
        )
        if matched_prot:
            site = int(row.get("Site", 0))
            ptm_dict.setdefault(matched_prot, []).append(
                {"site": site, "type": ptm_name}
            )

    return ptm_dict

acetyl_data = load_ptms("report.UniMod_1_sites_99.tsv",  "Acetylation")
oxid_data   = load_ptms("report.UniMod_35_sites_99.tsv", "Oxidation")

all_ptms = {prot: [] for prot in TARGET_PROTEINS}
for prot in TARGET_PROTEINS:
    if prot in acetyl_data:
        all_ptms[prot].extend(acetyl_data[prot])
    if prot in oxid_data:
        all_ptms[prot].extend(oxid_data[prot])

log_note("\nPTM Mapping Complete. Target Modifications Identified:")
for prot, ptms in all_ptms.items():
    if ptms:

        labels = ["{} at {}".format(p["type"], p["site"]) for p in ptms]
        log_note(f"  {prot}: {labels}")
    else:
        log_note(f"  {prot}: No modifications found in provided TSVs.")

log_note("\n" + "=" * 72)
log_note(" PREDICTING 3D STRUCTURES & SAVING PDBs")
log_note("=" * 72)

def fetch_seq(uid):
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return "".join(r.text.strip().split("\n")[1:])
    except Exception as e:
        log_note(f"  [API ERROR] Could not fetch {uid}: {e}")
    return None

sequences = {}
for uid in TARGET_PROTEINS:
    seq = fetch_seq(uid)
    if seq:
        sequences[uid] = seq
    else:
        log_note(f"  [WARNING] Sequence for {uid} not found on UniProt.")

MAX_SEQ_LEN = 1200

for i, (uid, seq) in enumerate(sequences.items(), 1):
    pdb_path = os.path.join(PDB_DIR, f"{uid}.pdb")

    if len(seq) > MAX_SEQ_LEN:
        log_note(
            f"  [WARNING] {uid} is {len(seq)} aa. "
            f"Truncating to {MAX_SEQ_LEN} aa."
        )
        seq = seq[:MAX_SEQ_LEN]

    if os.path.exists(pdb_path):
        log_note(
            f"  [{i}/{len(sequences)}] {uid} - "
            "PDB already exists, skipping folding."
        )
        continue

    log_note(f"  [{i}/{len(sequences)}] Folding {uid} ({len(seq)} aa)...")

    inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    try:
        with torch.no_grad():
            output = model(**inputs)

        pdb_string = model.output_to_pdb(output)[0]

        with open(pdb_path, "w") as f:
            f.write(pdb_string)

        log_note(f"    Saved → {pdb_path}")

    except torch.cuda.OutOfMemoryError:
        log_note(f"  [ERROR] OOM on {uid}. GPU VRAM exceeded. Skipping.")
    except Exception as e:
        log_note(f"  [ERROR] Failed on {uid}: {e}")

    del inputs
    torch.cuda.empty_cache()

log_note(
    f"\nAll structural predictions completed. "
    f"PDBs safely exported to: {PDB_DIR}"
)

def visualize_protein(uid):
    pdb_path = os.path.join(PDB_DIR, f"{uid}.pdb")
    if not os.path.exists(pdb_path):
        return

    with open(pdb_path, "r") as f:
        pdb_data = f.read()

    print(f"\n--- 3D Visualization: {uid} ---")

    seq_len  = len(sequences.get(uid, ""))
    eff_len  = min(seq_len, MAX_SEQ_LEN)
    valid_ptms = [p for p in all_ptms.get(uid, []) if p["site"] <= eff_len]

    if valid_ptms:
        labels = ["{} at {}".format(p["type"], p["site"]) for p in valid_ptms]
        print(f"Mapped PTMs: {labels}")
    else:
        print("No valid PTMs in modeled region.")

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, "pdb")
    view.setStyle(
        {"cartoon": {
            "colorscheme": {
                "prop": "b",
                "gradient": "roygb",
                "min": 50,
                "max": 90,
            }
        }}
    )

    for ptm in valid_ptms:
        site  = ptm["site"]
        color = "red" if ptm["type"] == "Acetylation" else "yellow"
        view.addStyle(
            {"resi": str(site)},
            {"sphere": {"color": color, "radius": 1.5}},
        )
        view.addLabel(
            f"{ptm['type']} (Site {site})",
            {
                "position": {"resi": str(site)},
                "backgroundColor": "black",
                "fontColor": "white",
            },
        )

    view.zoomTo()
    view.show()

for uid in TARGET_PROTEINS:
    if all_ptms.get(uid):
        visualize_protein(uid)

pml_script_path = os.path.join(PDB_DIR, "render_ptms.pml")

with open(pml_script_path, "w") as f:
    f.write("# PyMOL Script for Stress Proteomics Visualization\n")
    f.write("bg_color white\n")
    f.write("set ray_shadows, 0\n\n")

    for uid in TARGET_PROTEINS:
        seq_len    = len(sequences.get(uid, ""))
        eff_len    = min(seq_len, MAX_SEQ_LEN)
        valid_ptms = [p for p in all_ptms.get(uid, []) if p["site"] <= eff_len]

        if valid_ptms:
            f.write(f"load {uid}.pdb\n")
            f.write(f"hide all, {uid}\n")
            f.write(f"show cartoon, {uid}\n")
            f.write(
                f"spectrum b, blue_cyan_green_yellow_orange_red, "
                f"{uid}, minimum=50, maximum=90\n"
            )
            for ptm in valid_ptms:
                site     = ptm["site"]
                color    = "red" if ptm["type"] == "Acetylation" else "yellow"
                sel_name = f"PTM_{uid}_{site}"
                f.write(f"select {sel_name}, {uid} and resi {site}\n")
                f.write(f"show spheres, {sel_name}\n")
                f.write(f"color {color}, {sel_name}\n")
            f.write("\n")

log_note("\n" + "=" * 72)
log_note(" PUBLICATION READY")
log_note("=" * 72)
log_note(f" PyMOL rendering script written to: {pml_script_path}")
log_note(f" Structural logs securely saved to:  {LOG_FILE}")

!rm -rf /content/drive

from google.colab import drive
drive.mount('/content/drive')

import os, sys, time, requests
import pandas as pd
import numpy as np
import torch
import subprocess

print("Installing stable dependencies...")
subprocess.run(
    "pip install -q transformers accelerate py3Dmol biopython",
    shell=True
)

from transformers import AutoTokenizer, EsmForProteinFolding
import py3Dmol
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

print("=" * 72)
if not torch.cuda.is_available():
    raise RuntimeError("NO GPU DETECTED. Please enable GPU in Colab Runtime.")
print(f"GPU Active: {torch.cuda.get_device_name(0)}")

BASE_DIR = "/content/drive/MyDrive/ESM-Fold"
PDB_DIR  = os.path.join(BASE_DIR, "Target_PDBs")
PTM_PDB_DIR = os.path.join(BASE_DIR, "PTM_PDBs")
os.makedirs(PDB_DIR,     exist_ok=True)
os.makedirs(PTM_PDB_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_DIR, "structural_analysis_notes.txt")

def log_note(message):
    """Prints to console and appends to the rigorous .txt log file."""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

with open(LOG_FILE, "w") as f:
    f.write("====================================================\n")
    f.write(" STRUCTURAL ANALYSIS LOG (A100 RUN) - {}\n".format(time.ctime()))
    f.write("====================================================\n")

log_note("Directories initialized.")
log_note("  Standard PDBs  → {}".format(PDB_DIR))
log_note("  PTM-protein PDBs → {}".format(PTM_PDB_DIR))

TARGET_PROTEINS = [
    "A0A1S2YJZ0", "A0A3Q7XXP6", "A0A0X9LEN0", "A0A1S2XYP2", "A0A1S2YGY5",
    "A0A3Q7K7A2", "A0A1S2YEV0", "A0A1S2YL09", "A0A109PJV0", "A0A1S2XFJ7",
    "A0A1S2YJT8", "A0A1S2XXX3", "A0A1S3E0N0", "A0A1S2YP37", "A0A1S2XPL0",
    "A0A1S2Y7V6", "A0A1S2Z0V6", "A0A1S2XUK3", "A0A1S2YIG4", "Q9ZNQ4",
    "A0A1S2XXX7",
]

log_note("\nLoading Hugging Face ESMFold Model (Stable Port)...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model     = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True
)
model = model.eval().cuda()
model.trunk.set_chunk_size(128)
log_note("Model loaded successfully. Optimized for A100 80GB VRAM.")
log_note("=" * 72)

log_note("\nMapping PTMs from TSV files...")

CONDITION_COLS = [
    "C:\\DIA\\mzML\\CD_1.mzML", "C:\\DIA\\mzML\\CD_2.mzML", "C:\\DIA\\mzML\\CD_3.mzML",
    "C:\\DIA\\mzML\\CS_1.mzML", "C:\\DIA\\mzML\\CS_2.mzML", "C:\\DIA\\mzML\\CS_3.mzML",
    "C:\\DIA\\mzML\\GH_1.mzML", "C:\\DIA\\mzML\\GH_2.mzML", "C:\\DIA\\mzML\\GH_3.mzML",
    "C:\\DIA\\mzML\\RH_1.mzML", "C:\\DIA\\mzML\\RH_2.mzML", "C:\\DIA\\mzML\\RH_3.mzML",
]
CONDITION_LABELS = ["CD", "CD", "CD", "CS", "CS", "CS",
                    "GH", "GH", "GH", "RH", "RH", "RH"]

def load_ptms_full(filename, ptm_name):
    """
    Load ALL PTM entries from a TSV regardless of TARGET_PROTEINS membership.
    Returns a list of dicts with full quantitative and site metadata.
    """
    paths_to_check = [
        "/content/{}".format(filename),
        os.path.join(BASE_DIR, filename),
    ]
    filepath = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not filepath:
        log_note("  [WARNING] {} not found. Please upload to /content/".format(filename))
        return []

    df = pd.read_csv(filepath, sep="\t")
    records = []
    for _, row in df.iterrows():
        prot = str(row["Protein"]).strip()
        site = int(row["Site"])
        res  = str(row["Residue"]).strip()
        seq_window = str(row.get("Sequence", "")).strip()
        gene = str(row.get("Gene.Names", "")).strip()

        condition_means = {}
        for cond in ["CD", "CS", "GH", "RH"]:
            cols  = [c for c, l in zip(CONDITION_COLS, CONDITION_LABELS) if l == cond]
            avail = [col for col in cols if col in df.columns]
            if avail:
                vals = pd.to_numeric(row[avail], errors="coerce").fillna(0)
                condition_means[cond] = float(vals.mean())
            else:
                condition_means[cond] = 0.0

        detected_in = [c for c, v in condition_means.items() if v > 0]

        records.append({
            "protein":        prot,
            "gene":           gene,
            "site":           site,
            "residue":        res,
            "ptm_type":       ptm_name,
            "seq_window":     seq_window,
            "condition_means": condition_means,
            "detected_in":    detected_in,
        })
    return records

acetyl_records = load_ptms_full("report.UniMod_1_sites_99.tsv",  "Acetylation")
oxid_records   = load_ptms_full("report.UniMod_35_sites_99.tsv", "Oxidation")
all_ptm_records = acetyl_records + oxid_records

ptm_by_protein = {}
for rec in all_ptm_records:
    ptm_by_protein.setdefault(rec["protein"], []).append(rec)

PTM_PROTEINS = sorted(ptm_by_protein.keys())

log_note("\nPTM Mapping Complete. PTM proteins derived from TSV files:")
log_note("  Acetylation entries : {}".format(len(acetyl_records)))
log_note("  Oxidation entries   : {}".format(len(oxid_records)))
log_note("  Unique PTM proteins : {}".format(len(PTM_PROTEINS)))
for uid in PTM_PROTEINS:
    recs = ptm_by_protein[uid]
    summary = ["{} {}{}".format(r["ptm_type"], r["residue"], r["site"]) for r in recs]
    log_note("    {} : {}".format(uid, ", ".join(summary)))

overlap = set(TARGET_PROTEINS) & set(PTM_PROTEINS)
log_note(
    "\n  [NOTE] Overlap between TARGET_PROTEINS and PTM_PROTEINS: {} protein(s).".format(
        len(overlap)
    )
)
if not overlap:
    log_note(
        "  [NOTE] Zero overlap — these are two independent protein sets. "
        "Cell 6 processes the PTM proteins as a separate folding & analysis universe."
    )

log_note("\n" + "=" * 72)
log_note(" PREDICTING 3D STRUCTURES — TARGET_PROTEINS")
log_note("=" * 72)

def fetch_seq(uid):
    url = "https://rest.uniprot.org/uniprotkb/{}.fasta".format(uid)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return "".join(r.text.strip().split("\n")[1:])
    except Exception as e:
        log_note("  [API ERROR] Could not fetch {}: {}".format(uid, e))
    return None

MAX_SEQ_LEN = 1200

sequences = {}
for uid in TARGET_PROTEINS:
    seq = fetch_seq(uid)
    if seq:
        sequences[uid] = seq
    else:
        log_note("  [WARNING] Sequence for {} not found on UniProt.".format(uid))

def fold_and_save(uid, seq, pdb_dir, index_label=""):
    """Fold a single sequence and write PDB. Returns pdb_path or None."""
    pdb_path = os.path.join(pdb_dir, "{}.pdb".format(uid))

    if len(seq) > MAX_SEQ_LEN:
        log_note("  [WARNING] {} is {} aa. Truncating to {}.".format(
            uid, len(seq), MAX_SEQ_LEN))
        seq = seq[:MAX_SEQ_LEN]

    if os.path.exists(pdb_path):
        log_note("  [{}] {} — PDB already exists, skipping.".format(index_label, uid))
        return pdb_path

    log_note("  [{}] Folding {} ({} aa)...".format(index_label, uid, len(seq)))
    inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    try:
        with torch.no_grad():
            output = model(**inputs)

        pdb_string = model.output_to_pdb(output)[0]
        with open(pdb_path, "w") as f:
            f.write(pdb_string)
        log_note("    Saved → {}".format(pdb_path))
        return pdb_path
    except torch.cuda.OutOfMemoryError:
        log_note("  [ERROR] OOM on {}. Skipping.".format(uid))
    except Exception as e:
        log_note("  [ERROR] Failed on {}: {}".format(uid, e))
    finally:
        del inputs
        torch.cuda.empty_cache()
    return None

for i, (uid, seq) in enumerate(sequences.items(), 1):
    fold_and_save(uid, seq, PDB_DIR, index_label="{}/{}".format(i, len(sequences)))

log_note("\nTarget protein folding complete. PDBs → {}".format(PDB_DIR))

def visualize_protein(uid, pdb_dir, ptms=None):
    pdb_path = os.path.join(pdb_dir, "{}.pdb".format(uid))
    if not os.path.exists(pdb_path):
        return
    with open(pdb_path, "r") as f:
        pdb_data = f.read()
    print("\n--- 3D Visualization: {} ---".format(uid))
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"colorscheme": {
        "prop": "b", "gradient": "roygb", "min": 50, "max": 90
    }}})
    for ptm in (ptms or []):
        site  = ptm["site"]
        color = "red" if ptm["ptm_type"] == "Acetylation" else "yellow"
        view.addStyle({"resi": str(site)}, {"sphere": {"color": color, "radius": 1.5}})
        view.addLabel(
            "{} (Site {})".format(ptm["ptm_type"], site),
            {"position": {"resi": str(site)},
             "backgroundColor": "black", "fontColor": "white"},
        )
    view.zoomTo()
    view.show()

for uid in TARGET_PROTEINS:
    visualize_protein(uid, PDB_DIR)

pml_script_path = os.path.join(PDB_DIR, "render_ptms.pml")
with open(pml_script_path, "w") as f:
    f.write("# PyMOL Script for Stress Proteomics Visualization\n")
    f.write("bg_color white\nset ray_shadows, 0\n\n")
    for uid in TARGET_PROTEINS:
        pdb_path = os.path.join(PDB_DIR, "{}.pdb".format(uid))
        if os.path.exists(pdb_path):
            f.write("load {}.pdb\nhide all, {}\nshow cartoon, {}\n".format(uid, uid, uid))
            f.write("spectrum b, blue_cyan_green_yellow_orange_red, {}, "
                    "minimum=50, maximum=90\n\n".format(uid))

log_note("\n" + "=" * 72)
log_note(" PUBLICATION READY — TARGET_PROTEINS")
log_note("=" * 72)
log_note(" PyMOL script → {}".format(pml_script_path))
log_note(" Log          → {}".format(LOG_FILE))

log_note("\n" + "=" * 72)
log_note(" CELL 6: PTM PROTEIN FOLDING & STRUCTURAL CONTEXT ANALYSIS")
log_note("=" * 72)

ptm_sequences = {}
for uid in PTM_PROTEINS:
    seq = fetch_seq(uid)
    if seq:
        ptm_sequences[uid] = seq
    else:
        log_note("  [WARNING] Sequence for {} not found on UniProt.".format(uid))

log_note("\nFolding PTM proteins...")
for i, uid in enumerate(PTM_PROTEINS, 1):
    if uid in ptm_sequences:
        fold_and_save(uid, ptm_sequences[uid], PTM_PDB_DIR,
                      index_label="{}/{}".format(i, len(PTM_PROTEINS)))

def extract_plddt_from_pdb(pdb_path):
    """
    Returns a dict {residue_number (int): mean_pLDDT (float)} by averaging
    B-factor across all heavy atoms of each residue.
    ESMFold writes pLDDT (0–100) into the B-factor field.
    """
    plddt_sum   = {}
    plddt_count = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                res_num  = int(line[22:26].strip())
                b_factor = float(line[60:66].strip())
            except ValueError:
                continue
            plddt_sum[res_num]   = plddt_sum.get(res_num, 0.0)   + b_factor
            plddt_count[res_num] = plddt_count.get(res_num, 0)   + 1
    return {r: plddt_sum[r] / plddt_count[r] for r in plddt_sum}

def infer_secondary_structure(pdb_path, target_residues):
    """
    Assigns secondary structure to each residue in target_residues using
    Cα–Cα distance geometry (no external DSSP binary required).

    Logic (simplified but consistent with standard definitions):
      - Helix  : if residue i and i+4 Cα < 6.5 Å  (α-helix rise ~5.4 Å)
      - Sheet  : if residue i and i+2 Cα > 6.5 Å  AND atoms are roughly
                 co-planar with neighbours (crude β-sheet indicator)
      - Loop   : all other cases

    Returns dict {residue_number: 'Helix' | 'Sheet' | 'Loop'}
    """
    ca_coords = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    res_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ca_coords[res_num] = np.array([x, y, z])
                except ValueError:
                    continue

    def dist(a, b):
        return float(np.linalg.norm(a - b)) if (a is not None and b is not None) else None

    ss = {}
    residues = sorted(ca_coords.keys())
    res_set  = set(residues)
    for r in target_residues:
        if r not in ca_coords:
            ss[r] = "Unknown"
            continue

        r4 = r + 4
        if r4 in res_set:
            d = dist(ca_coords[r], ca_coords[r4])
            if d is not None and d < 6.5:
                ss[r] = "Helix"
                continue

        r2 = r + 2
        if r2 in res_set:
            d = dist(ca_coords[r], ca_coords[r2])
            if d is not None and d > 6.5:
                ss[r] = "Sheet"
                continue
        ss[r] = "Loop"
    return ss

def plddt_label(val):
    if val >= 90: return "Very High (structured)"
    if val >= 70: return "High (confident)"
    if val >= 50: return "Low (disordered/flexible)"
    return "Very Low (IDR)"

log_note("\nAnalysing PTM structural context...")

annotation_rows = []

for uid in PTM_PROTEINS:
    pdb_path = os.path.join(PTM_PDB_DIR, "{}.pdb".format(uid))
    if not os.path.exists(pdb_path):
        log_note("  [SKIP] No PDB for {} — folding may have failed.".format(uid))
        continue

    recs       = ptm_by_protein[uid]
    target_sites = [r["site"] for r in recs]
    plddt_map  = extract_plddt_from_pdb(pdb_path)
    ss_map     = infer_secondary_structure(pdb_path, target_sites)
    seq_len    = len(ptm_sequences.get(uid, ""))

    log_note("\n  ── {} (gene: {}, {} aa) ──".format(
        uid, recs[0].get("gene", "?"), seq_len))

    for rec in recs:
        site      = rec["site"]
        ptm_type  = rec["ptm_type"]
        residue   = rec["residue"]
        window    = rec["seq_window"]
        conds     = rec["detected_in"]
        cond_vals = rec["condition_means"]

        plddt_val = plddt_map.get(site, None)
        ss_label  = ss_map.get(site, "Unknown")

        plddt_str = (
            "{:.1f} — {}".format(plddt_val, plddt_label(plddt_val))
            if plddt_val is not None else "N/A (site outside model)"
        )

        log_note("    PTM  : {} on {} at site {}".format(ptm_type, residue, site))
        log_note("    pLDDT: {}".format(plddt_str))
        log_note("    SS   : {}".format(ss_label))
        log_note("    Window (TSV): {}".format(window))
        log_note("    Detected in conditions: {}".format(
            ", ".join(conds) if conds else "None with intensity > 0"
        ))
        log_note("    Condition intensities (mean):")
        for c, v in cond_vals.items():
            log_note("      {} : {:.1f}".format(c, v))

        if plddt_val is not None:
            if plddt_val < 50:
                interp = (
                    "PTM in intrinsically disordered region — likely "
                    "accessible to modifying enzymes; regulatory role probable."
                )
            elif plddt_val < 70:
                interp = (
                    "PTM in flexible/loop region — moderate structural "
                    "impact expected; may alter local dynamics."
                )
            elif ss_label == "Helix":
                interp = (
                    "PTM in confident α-helix — modification may disrupt "
                    "helix dipole or packing; potential stability effect."
                )
            elif ss_label == "Sheet":
                interp = (
                    "PTM in confident β-strand — modification may affect "
                    "hydrogen bonding network or strand geometry."
                )
            else:
                interp = (
                    "PTM in well-structured loop — likely surface-exposed; "
                    "possible interaction interface involvement."
                )
        else:
            interp = "Site outside folded region — cannot assess structural context."

        log_note("    Interpretation: {}".format(interp))

        annotation_rows.append({
            "Protein":       uid,
            "Gene":          rec.get("gene", ""),
            "PTM_Type":      ptm_type,
            "Residue":       residue,
            "Site":          site,
            "pLDDT":         round(plddt_val, 2) if plddt_val is not None else "N/A",
            "pLDDT_Label":   plddt_label(plddt_val) if plddt_val is not None else "N/A",
            "SecondaryStr":  ss_label,
            "SeqWindow":     window,
            "Detected_CD":   round(cond_vals.get("CD", 0), 1),
            "Detected_CS":   round(cond_vals.get("CS", 0), 1),
            "Detected_GH":   round(cond_vals.get("GH", 0), 1),
            "Detected_RH":   round(cond_vals.get("RH", 0), 1),
            "Interpretation": interp,
        })

log_note("\nGenerating interactive 3D visualisations for PTM proteins...")
for uid in PTM_PROTEINS:
    ptms = ptm_by_protein.get(uid, [])

    seq_len  = len(ptm_sequences.get(uid, ""))
    eff_len  = min(seq_len, MAX_SEQ_LEN)
    valid    = [r for r in ptms if r["site"] <= eff_len]
    if valid:
        visualize_protein(uid, PTM_PDB_DIR, ptms=valid)

ptm_pml_path = os.path.join(PTM_PDB_DIR, "render_ptm_proteins.pml")
with open(ptm_pml_path, "w") as f:
    f.write("# PyMOL Script — PTM Protein Structural Context\n")
    f.write("bg_color white\nset ray_shadows, 0\n\n")
    for uid in PTM_PROTEINS:
        pdb_path = os.path.join(PTM_PDB_DIR, "{}.pdb".format(uid))
        if not os.path.exists(pdb_path):
            continue
        recs  = ptm_by_protein.get(uid, [])
        seq_l = len(ptm_sequences.get(uid, ""))
        eff_l = min(seq_l, MAX_SEQ_LEN)
        valid = [r for r in recs if r["site"] <= eff_l]
        if not valid:
            continue
        f.write("load {}.pdb\nhide all, {}\nshow cartoon, {}\n".format(uid, uid, uid))
        f.write("spectrum b, blue_cyan_green_yellow_orange_red, {}, "
                "minimum=50, maximum=90\n".format(uid))
        for rec in valid:
            site     = rec["site"]
            color    = "red" if rec["ptm_type"] == "Acetylation" else "yellow"
            sel_name = "PTM_{}_{}".format(uid, site)
            f.write("select {}, {} and resi {}\n".format(sel_name, uid, site))
            f.write("show spheres, {}\ncolor {}, {}\n".format(sel_name, color, sel_name))
            f.write("label {} and name CA, \"{} {}{}\"\n".format(
                sel_name, rec["ptm_type"][:3], rec["residue"], site))
        f.write("\n")

log_note("\nPTM structural analysis complete.")
log_note("  PyMOL script → {}".format(ptm_pml_path))

log_note("\n" + "=" * 72)
log_note(" CELL 7: EXPORTING PTM ANNOTATION TABLE")
log_note("=" * 72)

if annotation_rows:
    ann_df   = pd.DataFrame(annotation_rows)
    tsv_path = os.path.join(BASE_DIR, "ptm_structural_annotations.tsv")
    ann_df.to_csv(tsv_path, sep="\t", index=False)
    log_note(" Annotation table ({} PTM sites) → {}".format(len(ann_df), tsv_path))
    print("\nAnnotation Table Preview:")
    print(ann_df[["Protein", "PTM_Type", "Residue", "Site",
                  "pLDDT", "pLDDT_Label", "SecondaryStr",
                  "Detected_CD", "Detected_CS",
                  "Detected_GH", "Detected_RH"]].to_string(index=False))
else:
    log_note(" [WARNING] No annotation rows collected — check folding results.")

log_note("\n" + "=" * 72)
log_note(" PIPELINE COMPLETE")
log_note("=" * 72)
log_note(" Standard PDBs        → {}".format(PDB_DIR))
log_note(" PTM protein PDBs     → {}".format(PTM_PDB_DIR))
log_note(" PTM annotation table → {}".format(
    os.path.join(BASE_DIR, "ptm_structural_annotations.tsv")))
log_note(" PyMOL PTM script     → {}".format(ptm_pml_path))
log_note(" Full log             → {}".format(LOG_FILE))
