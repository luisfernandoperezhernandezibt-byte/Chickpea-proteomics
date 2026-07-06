#Final part of the annotation, step 01_annotation

import os, sys, csv, time, json, re, logging
from datetime import datetime
from collections import Counter, defaultdict

RESULTS_DIR = "/content/drive/MyDrive/ESM-2 embeddings/Results"

STRESS_GO_ROOTS = {
    "GO:0006950": "response to stress",
    "GO:0009628": "response to abiotic stimulus",
    "GO:0006979": "response to oxidative stress",
    "GO:0009266": "response to temperature stimulus",
    "GO:0009415": "response to water",
    "GO:0009414": "response to water deprivation",
    "GO:0009651": "response to salt stress",
    "GO:0009269": "response to desiccation",
    "GO:0009409": "response to cold",
    "GO:0009408": "response to heat",
    "GO:0006457": "protein folding",
    "GO:0034976": "response to endoplasmic reticulum stress",
}

CATEGORY_RULES = [

    ({"GO:0009408", "GO:0010286", "GO:0009266", "GO:0006457", "GO:0051082",
      "GO:0034976"}, "Chaperones and HSPs"),
    ({"GO:0006979", "GO:0098869", "GO:0004601", "GO:0004784", "GO:0004096",
      "GO:0004364", "GO:0045174", "GO:0016684", "GO:0055114"}, "Antioxidant defence"),
    ({"GO:0009415", "GO:0009414", "GO:0009269"}, "Dehydrins and LEA proteins"),
    ({"GO:0009409"}, "Cold stress response"),
    ({"GO:0009651"}, "Osmotic and salt stress"),
    ({"GO:0006952", "GO:0009620", "GO:0009617"}, "Defence and pathogenesis-related"),
    ({"GO:0005509", "GO:0009931"}, "Calcium-mediated signalling"),
    ({"GO:0016165", "GO:0031408"}, "Lipid stress signalling"),
]

DPI = 600
FIG_FORMAT = "tiff"

os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "step1D_log.txt"), mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
print("=" * 70)
print("STEP 1D: GO-Based Stress Protein Classification")
print("=" * 70)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Runtime: CPU (no GPU required)\n")

import pandas as pd
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

def fetch_go_annotations(swissprot_ids):
    """Fetch GO annotations for Swiss-Prot IDs via UniProt REST API."""
    import urllib.request, urllib.parse

    cache_path = os.path.join(RESULTS_DIR, "swissprot_go_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        if len(cached) >= len(swissprot_ids) * 0.8:
            logging.info(f"Loaded {len(cached)} cached GO annotations")
            return cached

    logging.info(f"Fetching GO annotations for {len(swissprot_ids)} Swiss-Prot IDs...")
    go_annotations = {}
    id_list = sorted(swissprot_ids)
    batch_size = 80

    for i in tqdm(range(0, len(id_list), batch_size), desc="Fetching GO terms"):
        batch = id_list[i:i+batch_size]
        accession_query = " OR ".join(f"accession:{acc}" for acc in batch)

        url = (f"https://rest.uniprot.org/uniprotkb/search?"
               f"query={urllib.parse.quote(f'({accession_query})')}"
               f"&fields=accession,go_id,go_p,go_f,go_c,keyword"
               f"&format=tsv&size=500")

        for attempt in range(3):
            try:
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Python/ESM2-GO')
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read().decode('utf-8')

                lines = data.strip().split('\n')
                if len(lines) > 1:
                    headers = lines[0].split('\t')
                    for line in lines[1:]:
                        fields = line.split('\t')
                        if len(fields) < 2:
                            continue
                        accession = fields[0]
                        go_terms = set()
                        keywords = ""
                        for j, h in enumerate(headers):
                            if j < len(fields):
                                if 'go' in h.lower() and h.lower() not in ('entry', 'entry name'):
                                    matches = re.findall(r'GO:\d{7}', fields[j])
                                    go_terms.update(matches)
                                if 'keyword' in h.lower():
                                    keywords = fields[j]
                        go_annotations[accession] = {
                            'go_terms': sorted(go_terms),
                            'keywords': keywords,
                        }
                break
            except Exception as e:
                if attempt == 2:
                    logging.warning(f"  Batch at {i} failed after 3 attempts: {e}")
                time.sleep(2 * (attempt + 1))

        time.sleep(0.5)

    with open(cache_path, 'w') as f:
        json.dump(go_annotations, f)

    n_with_go = sum(1 for v in go_annotations.values() if v['go_terms'])
    logging.info(f"  Retrieved: {len(go_annotations)} proteins, {n_with_go} with GO terms")
    return go_annotations

def fetch_stress_go_set():
    """
    For each stress root GO term, fetch all descendant terms from QuickGO.
    Returns a dict: {go_term: root_term_it_descends_from}
    Only ~12 API calls — fast and robust.
    """
    import urllib.request, urllib.parse

    cache_path = os.path.join(RESULTS_DIR, "stress_go_descendants_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        if len(cached) > 50:
            logging.info(f"Loaded {len(cached)} cached stress GO descendants")
            return cached

    logging.info(f"Fetching descendants for {len(STRESS_GO_ROOTS)} stress root terms...")
    stress_go_set = {}

    for root_id, root_name in tqdm(STRESS_GO_ROOTS.items(), desc="Fetching GO descendants"):

        stress_go_set[root_id] = root_name

        url = (f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"
               f"{urllib.parse.quote(root_id)}/descendants?relations=is_a,part_of")

        for attempt in range(3):
            try:
                req = urllib.request.Request(url)
                req.add_header('Accept', 'application/json')
                req.add_header('User-Agent', 'Python/ESM2-GO')
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode('utf-8'))

                if 'results' in data:
                    for result in data['results']:
                        desc_id = result.get('id', '')
                        if desc_id.startswith('GO:'):
                            stress_go_set[desc_id] = root_name
                break

            except Exception as e:
                if attempt == 2:
                    logging.warning(f"  {root_id} failed: {e}")
                time.sleep(2 * (attempt + 1))

        time.sleep(0.3)

    with open(cache_path, 'w') as f:
        json.dump(stress_go_set, f)

    logging.info(f"  Total stress-related GO terms (roots + descendants): {len(stress_go_set)}")
    return stress_go_set

def classify_proteins(df, go_annotations, stress_go_set):
    """Classify each protein as stress/non-stress using GO membership."""
    logging.info("\nClassifying proteins by GO stress membership...")

    stress_terms = set(stress_go_set.keys())

    category_go_sets = []
    for rule_terms, label in CATEGORY_RULES:
        category_go_sets.append((rule_terms, label))

    results = []
    for _, row in df.iterrows():
        sp_id = str(row.get('SwissProt_ID', ''))
        sp_go = go_annotations.get(sp_id, {})
        protein_go_terms = set(sp_go.get('go_terms', []))
        keywords = sp_go.get('keywords', '')

        matched_stress = protein_go_terms & stress_terms
        is_stress = len(matched_stress) > 0

        matched_roots = set()
        for go_term in matched_stress:
            root_name = stress_go_set.get(go_term, '')
            if root_name:
                matched_roots.add(root_name)

        primary_category = 'Non-stress'
        all_categories = set()

        if is_stress:

            for rule_terms, label in category_go_sets:
                if protein_go_terms & rule_terms:
                    all_categories.add(label)

            if not all_categories:
                for root_name in matched_roots:
                    rn = root_name.lower()
                    if 'oxidat' in rn or 'redox' in rn:
                        all_categories.add('Antioxidant defence')
                    elif 'heat' in rn or 'temperature' in rn or 'fold' in rn:
                        all_categories.add('Chaperones and HSPs')
                    elif 'cold' in rn:
                        all_categories.add('Cold stress response')
                    elif 'water' in rn or 'desicc' in rn:
                        all_categories.add('Dehydrins and LEA proteins')
                    elif 'salt' in rn:
                        all_categories.add('Osmotic and salt stress')
                    elif 'endoplasmic' in rn:
                        all_categories.add('Chaperones and HSPs')
                    else:
                        all_categories.add('General stress response')

            primary_category = sorted(all_categories)[0] if all_categories else 'General stress response'

        kw_stress = False
        stress_kw_list = ['Stress response', 'Chaperone', 'Antioxidant', 'Cold shock']
        for skw in stress_kw_list:
            if skw.lower() in keywords.lower():
                kw_stress = True
                if not is_stress:
                    is_stress = True
                    primary_category = 'General stress response'
                    all_categories.add('General stress response (UniProt keyword)')

        results.append({
            'UniProt_ID': row.get('UniProt_ID', ''),
            'Gene_Name': row.get('Gene_Name', ''),
            'Description': row.get('Description', ''),
            'SwissProt_ID': sp_id,
            'SwissProt_Description': row.get('SwissProt_Description', ''),
            'SwissProt_Organism': row.get('SwissProt_Organism', ''),
            'Cosine_Similarity': row.get('Cosine_Similarity', ''),
            'ESM2_Confidence': row.get('ESM2_Confidence', ''),
            'Annotation_Status': row.get('Annotation_Status', ''),
            'Database': row.get('Database', ''),
            'Sequence_Length': row.get('Sequence_Length', ''),
            'Is_Stress_GO': is_stress,
            'Stress_GO_Terms': '; '.join(sorted(matched_stress)),
            'Stress_Root_Categories': '; '.join(sorted(matched_roots)),
            'Primary_Category_GO': primary_category,
            'All_Categories_GO': '; '.join(sorted(all_categories)) if all_categories else 'Non-stress',
            'N_GO_Terms': len(protein_go_terms),
            'All_GO_Terms': '; '.join(sorted(protein_go_terms)),
            'UniProt_Keywords': keywords,
            'Keyword_Stress': kw_stress,
        })

    df_out = pd.DataFrame(results)

    n_stress = df_out['Is_Stress_GO'].sum()
    logging.info(f"  Stress (GO-based): {n_stress} / {len(df_out)} ({100*n_stress/len(df_out):.1f}%)")

    cat_counts = df_out[df_out['Is_Stress_GO']]['Primary_Category_GO'].value_counts()
    logging.info(f"\n  Categories:")
    for cat, cnt in cat_counts.items():
        logging.info(f"    {cat}: {cnt}")

    no_go = (df_out['N_GO_Terms'] == 0).sum()
    logging.info(f"\n  Proteins with no GO terms: {no_go}")

    return df_out

def create_figures(df_full):
    """Generate TIFF figures at 600 DPI with percentages."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    })

    n_total = len(df_full)
    df_stress = df_full[df_full['Is_Stress_GO']].copy()
    save_kw = {"format": FIG_FORMAT, "dpi": DPI, "bbox_inches": "tight",
               "pil_kwargs": {"compression": "tiff_lzw"}}

    paths = []

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    sims = pd.to_numeric(df_full['Cosine_Similarity'], errors='coerce').dropna().values
    if len(sims) > 0:
        weights = np.ones_like(sims) * 100 / len(sims)
        ax.hist(sims, bins=25, weights=weights, color='#4472C4', edgecolor='white', linewidth=0.5)
        pct90 = 100 * np.sum(sims >= 0.90) / len(sims)
        pct80 = 100 * np.sum(sims >= 0.80) / len(sims)
        ax.axvline(0.90, color='#C00000', ls='--', lw=0.8, label=f'≥0.90: {pct90:.0f}%')
        ax.axvline(0.80, color='#ED7D31', ls='--', lw=0.8, label=f'≥0.80: {pct80:.0f}%')
        ax.legend(frameon=False, loc='upper left', fontsize=6)
        ax.set_xlim(max(0.55, sims.min() - 0.05), 1.02)
    ax.set_xlabel('Cosine similarity (top-1 Swiss-Prot match)')
    ax.set_ylabel('Proportion of proteins (%)')
    ax.set_title('Distribution of nearest-neighbour similarities')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"Fig1_similarity_distribution.{FIG_FORMAT}")
    fig.savefig(p, **save_kw); plt.close(fig); paths.append(p)
    logging.info(f"  Saved: {os.path.basename(p)}")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    status_counts = df_full['Annotation_Status'].value_counts()
    status_pct = (status_counts / n_total * 100).sort_values()
    colors_map = {
        'VALIDATED (high-confidence match)': '#2E7D32',
        'IMPROVED (was partial annotation)': '#F57F17',
        'NEW ANNOTATION (was uncharacterized)': '#C62828',
        'Consistent': '#1565C0', 'Insufficient similarity': '#757575',
    }
    bc = [colors_map.get(s, '#757575') for s in status_pct.index]
    bars = ax.barh(range(len(status_pct)), status_pct.values, color=bc, height=0.6)
    ax.set_yticks(range(len(status_pct)))
    ax.set_yticklabels([s.replace(' (', '\n(') for s in status_pct.index], fontsize=6)
    ax.set_xlabel('Proportion of proteins (%)')
    ax.set_title('Annotation improvement status')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, status_pct.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=6)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"Fig2_annotation_status.{FIG_FORMAT}")
    fig.savefig(p, **save_kw); plt.close(fig); paths.append(p)
    logging.info(f"  Saved: {os.path.basename(p)}")

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    conf_order = ['Very high confidence', 'High confidence', 'Moderate confidence',
                  'Low confidence', 'Very low confidence']
    conf_counts = df_full['ESM2_Confidence'].value_counts()
    vals = [conf_counts.get(c, 0)/n_total*100 for c in conf_order if conf_counts.get(c, 0) > 0]
    labs = [c for c in conf_order if conf_counts.get(c, 0) > 0]
    cols = ['#2E7D32', '#66BB6A', '#FDD835', '#FB8C00', '#E53935']
    if vals:
        wedges, texts, autotexts = ax.pie(vals, labels=labs, colors=cols[:len(vals)],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 6}, pctdistance=0.75)
        for at in autotexts:
            at.set_fontsize(6); at.set_fontweight('bold')
    ax.set_title('ESM-2 annotation confidence')
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"Fig3_confidence_pie.{FIG_FORMAT}")
    fig.savefig(p, **save_kw); plt.close(fig); paths.append(p)
    logging.info(f"  Saved: {os.path.basename(p)}")

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    if len(df_stress) > 0:
        cat_counts = df_stress['Primary_Category_GO'].value_counts()
        cat_pct = (cat_counts / cat_counts.sum() * 100).sort_values()
        cat_colors = {
            'Antioxidant defence': '#C62828', 'Chaperones and HSPs': '#E65100',
            'Dehydrins and LEA proteins': '#1565C0', 'Defence and pathogenesis-related': '#2E7D32',
            'Calcium-mediated signalling': '#6A1B9A', 'Lipid stress signalling': '#F9A825',
            'Cold stress response': '#00838F', 'Temperature stress response': '#BF360C',
            'Osmotic and salt stress': '#4E342E', 'General stress response': '#757575',
        }
        bc = [cat_colors.get(c, '#757575') for c in cat_pct.index]
        bars = ax.barh(range(len(cat_pct)), cat_pct.values, color=bc, height=0.65)
        ax.set_yticks(range(len(cat_pct)))
        ax.set_yticklabels(cat_pct.index, fontsize=6)
        ax.set_xlabel('Proportion of stress proteome (%)')
        ax.set_title(f'GO-based stress protein classification (n = {len(df_stress)})')
        for bar, val in zip(bars, cat_pct.values):
            cnt = int(round(val * cat_counts.sum() / 100))
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}% (n={cnt})', va='center', fontsize=5.5)
    else:
        ax.text(0.5, 0.5, 'No stress proteins identified', ha='center', va='center')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"Fig4_stress_categories_GO.{FIG_FORMAT}")
    fig.savefig(p, **save_kw); plt.close(fig); paths.append(p)
    logging.info(f"  Saved: {os.path.basename(p)}")

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    s_sims = pd.to_numeric(df_full[df_full['Is_Stress_GO']]['Cosine_Similarity'], errors='coerce').dropna().values
    ns_sims = pd.to_numeric(df_full[~df_full['Is_Stress_GO']]['Cosine_Similarity'], errors='coerce').dropna().values

    plot_data = []
    plot_labels = []
    if len(ns_sims) > 0:
        plot_data.append(ns_sims)
        plot_labels.append(f'Non-stress\n(n={len(ns_sims)})')
    if len(s_sims) > 0:
        plot_data.append(s_sims)
        plot_labels.append(f'Stress-related\n(n={len(s_sims)})')

    if len(plot_data) >= 2:
        parts = ax.violinplot(plot_data, positions=list(range(len(plot_data))),
                              showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#4472C4'); pc.set_alpha(0.7)
        parts['cmeans'].set_color('#C00000')
        parts['cmedians'].set_color('black')
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_labels)
    elif len(plot_data) == 1:
        ax.hist(plot_data[0], bins=20, color='#4472C4', alpha=0.7)
        ax.set_xlabel(plot_labels[0])
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Embedding similarity: stress vs. non-stress')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"Fig5_stress_vs_nonstress.{FIG_FORMAT}")
    fig.savefig(p, **save_kw); plt.close(fig); paths.append(p)
    logging.info(f"  Saved: {os.path.basename(p)}")

    return paths

def main():
    t0 = time.time()

    summary_path = os.path.join(RESULTS_DIR, "annotation_transfer_summary.csv")
    logging.info(f"Loading: {summary_path}")
    df = pd.read_csv(summary_path)
    logging.info(f"  {len(df)} proteins loaded")

    sp_ids = set(df['SwissProt_ID'].dropna().unique())
    logging.info(f"  {len(sp_ids)} unique Swiss-Prot IDs")

    go_annotations = fetch_go_annotations(sp_ids)

    stress_go_set = fetch_stress_go_set()

    df_classified = classify_proteins(df, go_annotations, stress_go_set)

    full_path = os.path.join(RESULTS_DIR, "annotation_GO_full.csv")
    df_classified.to_csv(full_path, index=False)
    logging.info(f"\n  Saved: {full_path}")

    df_stress = df_classified[df_classified['Is_Stress_GO']].copy()
    stress_path = os.path.join(RESULTS_DIR, "stress_proteins_GO_classified.csv")
    df_stress.to_csv(stress_path, index=False)
    logging.info(f"  Saved: {stress_path} ({len(df_stress)} stress proteins)")

    logging.info("\nGenerating publication figures (600 DPI TIFF)...")
    fig_paths = create_figures(df_classified)

    elapsed = time.time() - t0
    n_stress = len(df_stress)
    logging.info(f"\n{'='*70}")
    logging.info(f"STEP 1D COMPLETE")
    logging.info(f"{'='*70}")
    logging.info(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logging.info(f"  Stress proteins (GO): {n_stress} / {len(df_classified)}")
    logging.info(f"  Method: Gene Ontology descendant membership")
    logging.info(f"  Root terms: {len(STRESS_GO_ROOTS)}")
    logging.info(f"  Total stress GO terms (incl. descendants): {len(stress_go_set)}")
    logging.info(f"  Citation: Ashburner et al. (2000); Gene Ontology Consortium (2021)")
    logging.info(f"\n  Files:")
    logging.info(f"    annotation_GO_full.csv")
    logging.info(f"    stress_proteins_GO_classified.csv")
    for fp in fig_paths:
        logging.info(f"    {os.path.basename(fp)}")

if __name__ == "__main__":
    main()
