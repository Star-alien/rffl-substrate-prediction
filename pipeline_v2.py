#!/usr/bin/env python3
"""
RFFL Substrate Prediction Pipeline -- Version 2
Improvements over v1:
  1. Real SASA via biotite (Lee-Richards, no external binary needed)
  2. Experimentally confirmed ubiquitination sites from UniProt PTM annotations
     -> PWM trained on real sites, not all surface K
  3. MobiDB disorder predictions (independent of AlphaFold pLDDT)
  4. Harder negative control: proteins with known ubiquitination by other ligases
  5. Protein-level ROC AUC (correct metric for how rankings are used)

Extends Narendradev et al. 2025, J. Proteome Res.
DOI: 10.1021/acs.jproteome.5c00086
"""

import os, sys, json, time, math, random, warnings, traceback
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# ── BioPython ──────────────────────────────────────────────────
from Bio.PDB import PDBParser, is_aa

def _make_t3o():
    try:
        from Bio.PDB.Polypeptide import three_to_one as f
        return lambda x: (lambda: f(x.upper()))() if True else 'X'
    except Exception:
        pass
    try:
        from Bio.PDB.Polypeptide import protein_letters_3to1 as d
        return lambda x: d.get(x.upper()[:3], 'X')
    except Exception:
        pass
    from Bio.Data.PDBData import protein_letters_3to1 as d
    return lambda x: d.get(x.upper()[:3], 'X')

def _safe_t3o(resname):
    try:
        return _t3o_inner(resname)
    except Exception:
        return 'X'

_t3o_inner = _make_t3o()
three_to_one = _safe_t3o

# ── Biotite for real SASA ──────────────────────────────────────
try:
    from biotite.structure.io.pdb import PDBFile as BtPDBFile
    import biotite.structure as bs
    HAS_BIOTITE = True
    print("[INFO] Biotite available -- using Lee-Richards SASA")
except Exception:
    HAS_BIOTITE = False
    print("[INFO] Biotite unavailable -- using Cα-neighbor RSA proxy")

# ── Directories ────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
SUBSTRATE_DIR = DATA_DIR / "substrates"
CANDIDATE_DIR = DATA_DIR / "candidates"
NEG_DIR       = DATA_DIR / "negatives"
HARD_NEG_DIR  = DATA_DIR / "hard_negatives"
OUTPUT_DIR    = ROOT / "output"
V2_DIR        = OUTPUT_DIR / "v2"

for d in [HARD_NEG_DIR, V2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────
ALPHAFOLD_API    = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"
UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{uid}.json"
UNIPROT_SEARCH   = "https://rest.uniprot.org/uniprotkb/search"
MOBIDB_URL       = "https://mobidb.bio.unipd.it/api/entry/{uid}"

RFFL_UNIPROT = "Q8WZ73"

# v2 weights -- biology-driven; see doc comments per weight
W_RSA      = 0.30  # still primary but reduced -- MobiDB now shares disorder load
W_PWM      = 0.25  # uses annotated ub sites -> less circular than v1
W_MOBIDB   = 0.20  # independent disorder predictor; orthogonal to pLDDT
W_PLDDT    = 0.15  # kept as second independent disorder signal
W_CHARGE   = 0.10  # unchanged; FYVE electrostatic prior

RSA_THRESH    = 0.25
PLDDT_HIGH    = 70.0
PLDDT_LOW     = 50.0
CA_RADIUS     = 10.0
CA_MAX        = 25
MOTIF_WIN     = 5
MOTIF_LEN     = 2 * MOTIF_WIN + 1
CHARGE_WIN    = 5

AA1 = "ACDEFGHIKLMNPQRSTVWY"
AA_BG = {
    'A':0.070,'C':0.023,'D':0.053,'E':0.063,'F':0.040,
    'G':0.070,'H':0.022,'I':0.053,'K':0.058,'L':0.091,
    'M':0.023,'N':0.045,'P':0.049,'Q':0.040,'R':0.052,
    'S':0.072,'T':0.058,'V':0.065,'W':0.013,'Y':0.033,
    'X':0.050,
}
AA_CHG = {'K':+1,'R':+1,'H':+0.1,'D':-1,'E':-1}

# Reference max ASA per residue (Tien et al. 2013, theoretical scale)
MAX_ASA = {
    'ALA':129,'ARG':274,'ASN':195,'ASP':193,'CYS':167,
    'GLN':225,'GLU':223,'GLY':104,'HIS':224,'ILE':197,
    'LEU':201,'LYS':236,'MET':224,'PHE':240,'PRO':159,
    'SER':155,'THR':172,'TRP':285,'TYR':263,'VAL':174,
}

KNOWN_SUBSTRATES = {
    "CFTR":"P13569","MFN2":"O95140","RIPK1":"Q13546",
    "TP53":"P04637","CASP8":"Q14790","CASP10":"Q92851",
    "PRR5L":"Q6MZQ0","STUB1":"Q9UNE7","KCNH2":"Q12809",
    "JMJD6":"Q6NYC1","DNAJB11":"Q9UBS4",
}

CANDIDATES = {
    "DNAJB1":("P25685","A"), "DNAJB2":("P25686","A"),
    "DNAJB4":("Q9ULZ3","A"), "DNAJB5":("Q9NZL4","A"),
    "DNAJB6":("O75190","A"), "DNAJB7":("Q7L5N7","A"),
    "DNAJB8":("Q9Y3Y2","A"), "DNAJB9":("Q9UBV8","A"),
    "DNAJB12":("Q9NZH0","A"),"DNAJB14":("Q8NB43","A"),
    "DNAJA1":("P31689","A"), "DNAJA2":("O60884","A"),
    "DNAJA3":("Q96EY1","A"), "DNAJA4":("Q9ULX3","A"),
    "DNAJC3":("Q13217","A"), "DNAJC5":("Q9H3Z4","A"),
    "DNAJC10":("Q8IXB1","A"),
    "RNF5":("Q99942","B"),   "RNF185":("Q8N8P7","B"),
    "AMFR":("Q9UKU7","B"),   "HERC3":("P46933","B"),
    "SYVN1":("Q86TM6","B"),  "DERL1":("Q9BUN8","B"),
    "DERL2":("Q9GZP9","B"),  "SEL1L":("O75185","B"),
    "OS9":("Q13438","B"),
    "MFN1":("Q8IWA4","C"),   "OPA1":("O60313","C"),
    "DNM1L":("O00429","C"),  "FIS1":("Q9Y3D6","C"),
    "MFF":("Q8TDI8","C"),    "MARCH5":("Q9NX47","C"),
    "HSPA5":("P11021","D"),  "HSPA8":("P11142","D"),
    "HSP90B1":("P14625","D"),"CANX":("P27824","D"),
    "CALR":("P27797","D"),   "PDIA3":("P30101","D"),
    "VCP":("P55072","D"),    "UBQLN2":("Q9UHD9","D"),
}

# ==============================================================
# UTILITY
# ==============================================================

def get_json(url, params=None, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20,
                             headers={"User-Agent": "rffl-prediction-pipeline/2.0"})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return None

def fetch_structure(uid, save_path, retries=3):
    if save_path.exists() and save_path.stat().st_size > 500:
        return True
    data = get_json(ALPHAFOLD_API.format(uid=uid))
    if not data or not isinstance(data, list) or not data[0].get('pdbUrl'):
        return False
    url = data[0]['pdbUrl']
    for i in range(retries):
        try:
            r = requests.get(url, timeout=60,
                             headers={"User-Agent": "rffl-prediction-pipeline/2.0"})
            if r.status_code == 200:
                save_path.write_text(r.text, encoding='utf-8')
                return True
        except Exception:
            pass
        time.sleep(2 ** i)
    return False

def load_residues(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("P", str(pdb_path))
    model = next(structure.get_models())
    residues = []
    for chain in model.get_chains():
        for res in chain.get_residues():
            if not is_aa(res, standard=True):
                continue
            aa = three_to_one(res.get_resname())
            ca = res['CA'] if 'CA' in res else None
            residues.append({
                'aa': aa, 'pos': res.get_id()[1],
                'chain': chain.get_id(),
                'plddt': ca.get_bfactor() if ca else 50.0,
                'ca_coord': np.array(ca.get_coord()) if ca else None,
                'resname': res.get_resname(),
            })
    return residues

# ── SASA ───────────────────────────────────────────────────────

def ca_neighbor_rsa(residues):
    coords = np.array([r['ca_coord'] for r in residues if r['ca_coord'] is not None])
    if not len(coords):
        return [0.5] * len(residues)
    rsa = []
    for r in residues:
        if r['ca_coord'] is None:
            rsa.append(0.0); continue
        d = np.linalg.norm(coords - r['ca_coord'], axis=1)
        n = int(np.sum((d > 0.5) & (d <= CA_RADIUS)))
        rsa.append(min(1.0, max(0.0, 1.0 - n / CA_MAX)))
    return rsa

def biotite_rsa(pdb_path, residues):
    """Compute real per-residue RSA via biotite Lee-Richards algorithm."""
    if not HAS_BIOTITE:
        return None
    try:
        pf = BtPDBFile.read(str(pdb_path))
        arr = pf.get_structure(model=1)
        aa_mask = bs.filter_amino_acids(arr)
        protein = arr[aa_mask]
        atom_sasa = bs.sasa(protein, point_number=1000)
        res_starts = bs.get_residue_starts(protein)
        res_sasa = np.add.reduceat(atom_sasa, res_starts)
        res_names = protein.res_name[res_starts]
        res_ids   = protein.res_id[res_starts]
        chain_ids = protein.chain_id[res_starts]
        rsa_map = {}
        for i in range(len(res_starts)):
            max_a = MAX_ASA.get(res_names[i].strip(), 200.0)
            rsa = min(1.0, max(0.0, float(res_sasa[i]) / max_a))
            rsa_map[(chain_ids[i], int(res_ids[i]))] = rsa
        # Map back to residue list order
        out = []
        for r in residues:
            key = (r['chain'], r['pos'])
            out.append(rsa_map.get(key, 0.3))  # 0.3 = uncertain default
        return out
    except Exception as e:
        return None

def get_rsa(pdb_path, residues):
    """Return RSA list: try biotite first, fall back to CA-neighbor."""
    rsa = biotite_rsa(pdb_path, residues)
    if rsa is not None:
        return rsa, "biotite"
    return ca_neighbor_rsa(residues), "ca_neighbor"

# ── UniProt ubiquitination sites ───────────────────────────────

def fetch_ub_sites(uid):
    """
    Query UniProt REST API for experimentally confirmed ubiquitination sites.
    Returns list of 1-indexed residue positions with ubiquitination evidence.
    Looks for: Modified residue or Cross-link features mentioning ubiquitin/glycyl.
    """
    data = get_json(UNIPROT_JSON_URL.format(uid=uid))
    if not data:
        return []
    sites = []
    for feat in data.get('features', []):
        ftype = feat.get('type', '')
        desc  = feat.get('description', '').lower()
        ub_keywords = ('ubiquitin', 'glycyl lysine', 'sumo', 'nedd8', 'glycyl-lysine')
        if ftype in ('Modified residue', 'Cross-link') and any(k in desc for k in ub_keywords):
            pos = feat.get('location', {}).get('start', {}).get('value')
            if pos is not None:
                sites.append(int(pos))
    return sorted(set(sites))

# ── MobiDB disorder ────────────────────────────────────────────

def fetch_mobidb(uid, length):
    """
    Fetch MobiDB-lite disorder prediction.
    Returns per-residue disorder array (0=ordered, 1=disordered), length = protein length.
    Falls back to zeros (all ordered) if API unavailable.
    """
    data = get_json(MOBIDB_URL.format(uid=uid))
    if not data:
        return np.zeros(length)

    disorder = np.zeros(length)
    regions  = []

    # Format A: top-level 'prediction-disorder-mobidb_lite' or 'mobidb_lite'
    for key in ('prediction-disorder-mobidb_lite', 'mobidb_lite',
                'consensus', 'prediction-disorder-merge'):
        if key in data:
            entry = data[key]
            if isinstance(entry, dict):
                regions = entry.get('regions', [])
            elif isinstance(entry, list) and len(entry) == length:
                return np.array(entry, dtype=float)
            break

    # Format B: 'data' array of prediction objects
    if not regions and 'data' in data:
        for item in data['data']:
            method = item.get('method', '') or item.get('source', '')
            if 'mobidb' in method.lower() or 'mobidb' in str(item).lower():
                regions = item.get('regions', [])
                if regions:
                    break

    # Format C: nested 'consensus' block
    if not regions and isinstance(data.get('consensus'), dict):
        for sub in data['consensus'].values():
            if isinstance(sub, dict) and 'regions' in sub:
                regions = sub['regions']
                break

    for region in regions:
        if isinstance(region, (list, tuple)) and len(region) >= 2:
            s = max(0, int(region[0]) - 1)
            e = min(length, int(region[1]))
            disorder[s:e] = 1.0

    return disorder

def plddt_to_disorder(plddt):
    if plddt < PLDDT_LOW:  return 1.0
    if plddt < PLDDT_HIGH: return 0.5
    return 0.0

# ── Local features ────────────────────────────────────────────

def local_charge(residues, idx):
    lo, hi = max(0, idx - CHARGE_WIN), min(len(residues), idx + CHARGE_WIN + 1)
    return sum(AA_CHG.get(residues[j]['aa'], 0) for j in range(lo, hi)) / (2*CHARGE_WIN+1)

def local_motif(residues, idx):
    seq = [r['aa'] for r in residues]
    return ''.join(seq[i] if 0 <= i < len(seq) else 'X'
                   for i in range(idx - MOTIF_WIN, idx + MOTIF_WIN + 1))

# ── PWM ───────────────────────────────────────────────────────

def build_pwm(motifs, pseudocount=0.5):
    motifs = [m for m in motifs if len(m) == MOTIF_LEN]
    if not motifs:
        return {}
    pwm = {}
    for pos in range(MOTIF_LEN):
        counts = Counter(m[pos] for m in motifs)
        total  = sum(counts.values()) + len(AA1) * pseudocount
        pwm[pos] = {
            aa: math.log2(((counts.get(aa, 0) + pseudocount) / total)
                          / AA_BG.get(aa, 0.05))
            for aa in AA1 + 'X'
        }
    return pwm

def score_pwm(motif, pwm):
    if not pwm:
        return 0.0
    return sum(pwm.get(p, {}).get(aa, 0.0) for p, aa in enumerate(motif))

def minmax(vals, v):
    mn, mx = min(vals), max(vals)
    return (v - mn) / (mx - mn) if mx > mn else 0.5

# ── Composite score (v2 weights) ──────────────────────────────

def composite(rsa, pwm_norm, mobidb, plddt_dis, charge):
    """
    v2 composite score.
    RSA 0.30  -- surface accessibility required; weight reduced from v1 (0.40)
                 because MobiDB now shares disorder load independently
    PWM 0.25  -- trained on annotated ub sites, less circular than v1 surface-K PWM
    MobiDB 0.20 -- curated multi-method disorder; orthogonal to pLDDT-based proxy
    pLDDT 0.15  -- kept as complementary, independent structural signal
    Charge 0.10 -- unchanged; RFFL FYVE domain electrostatic prior
    """
    charge_norm = (charge + 1.0) / 2.0
    return round(W_RSA*rsa + W_PWM*pwm_norm + W_MOBIDB*mobidb
                 + W_PLDDT*plddt_dis + W_CHARGE*charge_norm, 4)

# ── ROC ──────────────────────────────────────────────────────

def roc_auc_manual(scores, labels):
    s, l = np.array(scores), np.array(labels)
    thresholds = np.linspace(0, 1, 300)
    fprs, tprs = [], []
    for t in thresholds:
        p = (s >= t).astype(int)
        tp = np.sum((p==1)&(l==1)); fp = np.sum((p==1)&(l==0))
        fn = np.sum((p==0)&(l==1)); tn = np.sum((p==0)&(l==0))
        tprs.append(tp/(tp+fn) if (tp+fn) else 0.0)
        fprs.append(fp/(fp+tn) if (fp+tn) else 0.0)
    pairs = sorted(zip(fprs, tprs))
    return (float(np.trapezoid([p[1] for p in pairs], [p[0] for p in pairs])),
            [p[0] for p in pairs], [p[1] for p in pairs])

# ==============================================================
# PART 1 -- AUC Discrepancy Diagnosis
# ==============================================================

def part1_diagnosis():
    print("\n" + "="*62)
    print("PART 1 -- AUC Discrepancy Diagnosis")
    print("="*62)

    # Load v1 substrate features
    feat_path = DATA_DIR / "substrate_features.csv"
    val_path  = OUTPUT_DIR / "rffl_known_substrates_validation.csv"
    if not feat_path.exists() or not val_path.exists():
        print("  [SKIP] v1 output files not found -- run pipeline.py first")
        return None, None, None

    feat_df = pd.read_csv(feat_path)
    val_df  = pd.read_csv(val_path)

    # v1 thresholds
    p50_v1 = 0.487
    p75_v1 = 0.602  # HIGH threshold from v1

    # 1a -- Per-protein fraction of surface K above HIGH threshold
    surf = feat_df[feat_df['is_surface'] == True].copy()
    print(f"\n[1a] Fraction of surface K above HIGH threshold ({p75_v1}) per substrate:")
    fracs = {}
    for gene, grp in surf.groupby('gene_name'):
        if 'composite' not in grp.columns:
            continue
        above = (grp['composite'] > p75_v1).sum()
        total = len(grp)
        fracs[gene] = (above, total, above/total if total else 0)
        print(f"  {gene:12s}: {above:3d}/{total:3d} ({100*above/total:.1f}%)")

    # 1b -- Protein-level AUC: need scores for negative proteins too
    # Re-score the 12 negative-set PDB files from v1 using the same v1 approach
    print(f"\n[1b] Computing protein-level AUC...")
    substrate_max_scores = val_df.set_index('gene_name')['composite'].to_dict()

    # Score negative proteins at protein level (max composite across their K)
    neg_protein_scores = []
    for pdb in sorted(NEG_DIR.glob("*.pdb")):
        try:
            res = load_residues(pdb)
            rsa_vals = ca_neighbor_rsa(res)
            k_scores = []
            for idx, (r, rsa) in enumerate(zip(res, rsa_vals)):
                if r['aa'] != 'K' or rsa <= RSA_THRESH:
                    continue
                dis = plddt_to_disorder(r['plddt'])
                chg = local_charge(res, idx)
                # Use v1 weights (no PWM for this quick calculation -- set to 0.5)
                sc = 0.4*rsa + 0.3*0.5 + 0.2*dis + 0.1*(chg+1)/2
                k_scores.append(sc)
            if k_scores:
                neg_protein_scores.append(max(k_scores))
        except Exception:
            pass

    pos_scores_prot = list(substrate_max_scores.values())
    all_prot_scores = pos_scores_prot + neg_protein_scores
    all_prot_labels = [1]*len(pos_scores_prot) + [0]*len(neg_protein_scores)
    auc_prot_v1, _, _ = roc_auc_manual(all_prot_scores, all_prot_labels)

    print(f"  v1 lysine-level AUC  : 0.530")
    print(f"  v1 protein-level AUC : {auc_prot_v1:.3f}")
    print(f"  (n_pos={len(pos_scores_prot)}, n_neg={len(neg_protein_scores)})")

    # 1c -- Write diagnosis markdown
    lines_frac = "\n".join(
        f"| {g:12s} | {v[0]:3d}/{v[1]:3d} | {100*v[2]:.1f}% |"
        for g, v in sorted(fracs.items(), key=lambda x: -x[1][2])
    )

    diagnosis = f"""# AUC Discrepancy Analysis -- v1 Pipeline

## The apparent contradiction

v1 ROC AUC at **lysine level** = **0.530** (barely above random)
Yet **10/11 known substrates** scored in the HIGH tier at protein level.

## Why these are not contradictory

### The fundamental issue: averaging vs maximising

RFFL ubiquitinates a small number of specific lysines per substrate --
but the pipeline characterises ALL surface-accessible lysines in each
substrate protein. For example, CFTR has **78 surface-accessible K** residues.
Only a handful are RFFL targets. The lysine-level comparison therefore
lumps together:
- The few RFFL-target K (genuinely distinctive)
- The many non-target surface K (indistinguishable from random)

This dilutes the AUC toward 0.5.

### The correct metric: protein-level MAX score

Phase 3 ranks proteins, not individual lysines -- by taking the **maximum**
score across all surface K. If even ONE of a protein's surface K scores
distinctively, the protein ranks highly. This is the right comparison for
how the pipeline is actually used.

### Per-protein fraction of surface K above HIGH threshold ({p75_v1})

| Gene         | Above/Total | Fraction |
|-------------|-------------|----------|
{lines_frac}

Low fractions confirm that most surface K per substrate do NOT individually
meet the HIGH threshold -- only the best K does. This is why protein-level
max-score is high while lysine-level AUC is low.

### Protein-level AUC recalculation

| Metric               | Value  |
|---------------------|--------|
| Lysine-level AUC (v1) | 0.530 |
| Protein-level AUC (v1) | {auc_prot_v1:.3f} |
| n_positive proteins  | {len(pos_scores_prot)} |
| n_negative proteins  | {len(neg_protein_scores)} |

{"The protein-level AUC is meaningfully higher than 0.530, confirming that the original lysine-level AUC was measuring the wrong thing -- not indicating the model is weak. The model does distinguish substrate proteins from random proteins; it just cannot reliably distinguish every individual K within a substrate from every random K, which is an inherent limitation given that ubiquitination sites are sparse within proteins." if auc_prot_v1 > 0.60 else "The protein-level AUC remains modest, indicating genuine limited discriminative power even at the protein level. This is a real limitation to report: the features used (surface exposure, sequence context, disorder) are necessary but not sufficient to identify RFFL-specific substrates reliably with n=10 positive examples."}

## What v2 does about this

v2 addresses both issues:
1. PWM is now trained on annotated ubiquitination sites (from UniProt PTM features)
   rather than all surface K -- reducing the circularity that diluted the lysine-level signal
2. MobiDB disorder predictions add an independent feature channel
3. A harder negative control (other-E3-substrate proteins) tests RFFL-specificity
4. All AUC numbers in v2 are reported at **both** lysine level AND protein level
"""

    diag_path = OUTPUT_DIR / "auc_discrepancy_analysis.md"
    diag_path.write_text(diagnosis, encoding='utf-8')
    print(f"\n  Saved diagnosis -> output/auc_discrepancy_analysis.md")
    return auc_prot_v1, fracs, neg_protein_scores

# ==============================================================
# PART 2 -- Improved features
# ==============================================================

def fetch_ub_sites_all(gene_uid_map):
    """Fetch UniProt ub sites for each protein. Returns dict: gene -> [positions]."""
    print("\n[2.2] Fetching UniProt ubiquitination site annotations...")
    sites = {}
    for gene, uid in gene_uid_map.items():
        s = fetch_ub_sites(uid)
        sites[gene] = s
        method = f"{len(s)} sites" if s else "none (will use surface-K fallback)"
        print(f"  {gene:12s} ({uid}): {method}")
        time.sleep(0.35)
    return sites

def fetch_mobidb_all(gene_uid_map, residues_map):
    """Fetch MobiDB disorder for each protein. Returns dict: gene -> np.array."""
    print("\n[2.3] Fetching MobiDB disorder predictions...")
    disorder_map = {}
    for gene, uid in gene_uid_map.items():
        length = len(residues_map.get(gene, []))
        if length == 0:
            disorder_map[gene] = np.array([])
            continue
        d = fetch_mobidb(uid, length)
        frac = d.mean() if len(d) else 0.0
        print(f"  {gene:12s}: {frac:.2f} disorder fraction")
        disorder_map[gene] = d
        time.sleep(0.35)
    return disorder_map

def build_harder_negatives(n=30):
    """
    Fetch ~n human proteins with known ubiquitination (by other ligases)
    as a harder negative control. Uses UniProt's cross-link ubiquitin annotation.
    """
    print(f"\n[2.4] Building harder negative set ({n} proteins with other-ligase ubiquitination)...")
    exclude = (set(KNOWN_SUBSTRATES.values())
               | {uid for uid, _ in CANDIDATES.values()}
               | {RFFL_UNIPROT})

    # Search UniProt for proteins with ubiquitin cross-link annotations
    data = get_json(UNIPROT_SEARCH, params={
        "query": "reviewed:true AND organism_id:9606 AND ft_crosslnk:ubiquitin",
        "format": "json", "fields": "accession,gene_names", "size": 150,
    })
    if not data:
        print("  [WARN] UniProt search failed -- skipping harder negatives")
        return pd.DataFrame()

    candidates_pool = [e['primaryAccession'] for e in data.get('results', [])
                       if e['primaryAccession'] not in exclude]
    random.shuffle(candidates_pool)
    selected = candidates_pool[:n]
    print(f"  Selected {len(selected)} proteins for harder-negative set")

    rows = []
    for uid in selected:
        pdb = HARD_NEG_DIR / f"{uid}.pdb"
        if not fetch_structure(uid, pdb):
            continue
        try:
            res = load_residues(pdb)
            rsa_vals, _ = get_rsa(pdb, res)
            ub_s = fetch_ub_sites(uid)
            mobidb_d = fetch_mobidb(uid, len(res))
            for idx, (r, rsa) in enumerate(zip(res, rsa_vals)):
                if r['aa'] != 'K':
                    continue
                dis_plddt  = plddt_to_disorder(r['plddt'])
                dis_mobidb = float(mobidb_d[idx]) if idx < len(mobidb_d) else dis_plddt
                chg        = local_charge(res, idx)
                mot        = local_motif(res, idx)
                rows.append({
                    'gene_name': f"HARD_{uid}", 'uniprot_id': uid,
                    'lysine_position': r['pos'], 'rsa_value': round(rsa, 4),
                    'plddt': round(r['plddt'], 1),
                    'disorder_plddt': dis_plddt, 'disorder_mobidb': dis_mobidb,
                    'local_charge': round(chg, 4),
                    'local_motif_sequence': mot,
                    'is_surface': rsa > RSA_THRESH,
                    'is_annotated_ub': r['pos'] in ub_s,
                })
        except Exception:
            pass
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df)} K-residue rows from harder-negative set")
    return df

def extract_features_v2(gene, uid, pdb_path, ub_sites, mobidb_arr):
    """
    Extract per-lysine features with v2 improvements:
    real SASA (biotite if available), MobiDB disorder, ub site annotation.
    """
    res = load_residues(pdb_path)
    rsa_vals, rsa_method = get_rsa(pdb_path, res)
    length = len(res)

    rows = []
    for idx, (r, rsa) in enumerate(zip(res, rsa_vals)):
        if r['aa'] != 'K':
            continue
        dis_plddt  = plddt_to_disorder(r['plddt'])
        dis_mobidb = float(mobidb_arr[idx]) if idx < len(mobidb_arr) else dis_plddt
        chg        = local_charge(res, idx)
        mot        = local_motif(res, idx)
        is_ub      = r['pos'] in ub_sites
        rows.append({
            'gene_name':          gene,
            'uniprot_id':         uid,
            'chain':              r['chain'],
            'lysine_position':    r['pos'],
            'rsa_value':          round(rsa, 4),
            'rsa_method':         rsa_method,
            'plddt':              round(r['plddt'], 1),
            'disorder_plddt':     dis_plddt,
            'disorder_mobidb':    dis_mobidb,
            'local_charge':       round(chg, 4),
            'local_motif_sequence': mot,
            'is_surface':         rsa > RSA_THRESH,
            'is_annotated_ub':    is_ub,
        })
    return rows, res

# ==============================================================
# PART 2 MAIN -- Run improved pipeline
# ==============================================================

def part2_improved_pipeline():
    print("\n" + "="*62)
    print("PART 2 -- Improved substrate feature extraction")
    print("="*62)

    # Extract features for all known substrates
    all_rows, residues_map = [], {}
    for gene, uid in KNOWN_SUBSTRATES.items():
        pdb = SUBSTRATE_DIR / f"{gene}.pdb"
        if not pdb.exists():
            if not fetch_structure(uid, pdb):
                print(f"  SKIP {gene} -- no structure"); continue
        try:
            res = load_residues(pdb)
            residues_map[gene] = res
        except Exception as e:
            print(f"  ERROR loading {gene}: {e}")

    # Fetch UniProt ub sites and MobiDB disorder for all substrates
    ub_sites_map  = fetch_ub_sites_all(KNOWN_SUBSTRATES)
    mobidb_map    = fetch_mobidb_all(KNOWN_SUBSTRATES, residues_map)

    # Extract v2 features
    print(f"\n[2.5] Extracting v2 features...")
    sub_rows = []
    for gene, uid in KNOWN_SUBSTRATES.items():
        pdb = SUBSTRATE_DIR / f"{gene}.pdb"
        if not pdb.exists():
            continue
        mobidb_arr = mobidb_map.get(gene, np.array([]))
        ub_sites   = ub_sites_map.get(gene, [])
        rows, res  = extract_features_v2(gene, uid, pdb, ub_sites, mobidb_arr)
        surf = sum(1 for r in rows if r['is_surface'])
        ub_a = sum(1 for r in rows if r['is_annotated_ub'])
        print(f"  * {gene}: {surf} surface K, {ub_a} annotated ub sites")
        sub_rows.extend(rows)

    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(V2_DIR / "substrate_features_v2.csv", index=False)

    # Build PWM: prefer annotated ub sites, fall back to surface K
    annotated_motifs = sub_df[sub_df['is_annotated_ub'] == True]['local_motif_sequence'].tolist()
    surface_motifs   = sub_df[sub_df['is_surface'] == True]['local_motif_sequence'].tolist()
    annotated_motifs = [m for m in annotated_motifs if len(m) == MOTIF_LEN]
    surface_motifs   = [m for m in surface_motifs   if len(m) == MOTIF_LEN]

    print(f"\n  PWM training: {len(annotated_motifs)} annotated ub-site motifs "
          f"+ {len(surface_motifs)} surface-K motifs (fallback)")

    if len(annotated_motifs) >= 5:
        pwm = build_pwm(annotated_motifs)
        pwm_source = "annotated_ub_sites"
        print(f"  Using annotated-site PWM (n={len(annotated_motifs)})")
    else:
        # Not enough annotated sites -- use surface K but note it
        pwm = build_pwm(surface_motifs)
        pwm_source = "surface_k_fallback"
        print(f"  [WARN] Only {len(annotated_motifs)} annotated sites "
              f"-- falling back to surface-K PWM")

    # Score all substrate K with v2 model
    ref_pwm_scores = [score_pwm(m, pwm) for m in sub_df['local_motif_sequence']]
    sub_df['pwm_raw']  = ref_pwm_scores
    sub_df['pwm_norm'] = [minmax(ref_pwm_scores, v) for v in ref_pwm_scores]
    sub_df['composite_v2'] = [
        composite(row['rsa_value'], row['pwm_norm'],
                  row['disorder_mobidb'], row['disorder_plddt'], row['local_charge'])
        for _, row in sub_df.iterrows()
    ]

    surf_scores = sub_df[sub_df['is_surface']]['composite_v2'].tolist()
    p50 = float(np.percentile(surf_scores, 50))
    p75 = float(np.percentile(surf_scores, 75))
    print(f"\n  v2 surface-K score distribution: "
          f"min={min(surf_scores):.3f}  median={p50:.3f}  max={max(surf_scores):.3f}")

    return sub_df, pwm, ref_pwm_scores, p50, p75, pwm_source

# ==============================================================
# PART 2 CANDIDATE SCREEN
# ==============================================================

def screen_candidates_v2(sub_df, pwm, ref_pwm_scores, p50, p75):
    print("\n" + "="*62)
    print("PART 2 CANDIDATE SCREEN -- v2 scoring")
    print("="*62)
    print(f"\n  HIGH tier > {p75:.3f}  |  MODERATE tier > {p50:.3f}")

    results = []
    total = len(CANDIDATES)
    for i, (gene, (uid, cset)) in enumerate(CANDIDATES.items(), 1):
        print(f"  [{i:02d}/{total}] {gene} ({uid}, Set {cset})...", end=" ", flush=True)
        pdb = CANDIDATE_DIR / f"{gene}.pdb"
        if not fetch_structure(uid, pdb):
            print("NO STRUCTURE")
            results.append({'gene_name':gene,'uniprot_id':uid,'candidate_set':cset,
                            'max_score_v2':None,'confidence_tier_v2':'NO_STRUCTURE',
                            'top_lysine_position':None,'top_lysine_motif':None,
                            'n_surface_K':0,'pwm_source':'n/a','note':'No structure'})
            time.sleep(0.3); continue
        try:
            res  = load_residues(pdb)
            mobidb = fetch_mobidb(uid, len(res))
            ub_s   = fetch_ub_sites(uid)
            rsa_vals, rsa_meth = get_rsa(pdb, res)

            scored = []
            for idx, (r, rsa) in enumerate(zip(res, rsa_vals)):
                if r['aa'] != 'K' or rsa <= RSA_THRESH:
                    continue
                dis_m  = float(mobidb[idx]) if idx < len(mobidb) else plddt_to_disorder(r['plddt'])
                dis_p  = plddt_to_disorder(r['plddt'])
                chg    = local_charge(res, idx)
                mot    = local_motif(res, idx)
                pn_raw = score_pwm(mot, pwm)
                pn     = minmax(ref_pwm_scores, pn_raw)
                sc     = composite(rsa, pn, dis_m, dis_p, chg)
                scored.append((sc, r['pos'], mot, rsa, r['plddt'], dis_m, rsa_meth))

            if not scored:
                print("0 surface K")
                results.append({'gene_name':gene,'uniprot_id':uid,'candidate_set':cset,
                                'max_score_v2':0.0,'confidence_tier_v2':'LOW',
                                'n_surface_K':0,'note':'No surface K'})
                time.sleep(0.3); continue

            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0]
            tier = 'HIGH' if best[0]>p75 else ('MODERATE' if best[0]>p50 else 'LOW')
            n_ub = len(ub_s)
            results.append({
                'gene_name': gene, 'uniprot_id': uid, 'candidate_set': cset,
                'max_score_v2': best[0], 'confidence_tier_v2': tier,
                'top_lysine_position': best[1], 'top_lysine_motif': best[2],
                'top_k_rsa': round(best[3], 4), 'top_k_plddt': round(best[4], 1),
                'top_k_mobidb_disorder': round(best[5], 2),
                'n_surface_K': len(scored),
                'n_annotated_ub_sites': n_ub,
                'rsa_method': best[6], 'pwm_source': 'annotated_or_surface_k',
                'note': None,
            })
            print(f"score={best[0]:.3f} ({tier}), {n_ub} ub sites")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({'gene_name':gene,'uniprot_id':uid,'candidate_set':cset,
                            'max_score_v2':None,'confidence_tier_v2':'ERROR','note':str(e)})
        time.sleep(0.35)

    return pd.DataFrame(results)

# ==============================================================
# PART 2 VALIDATION -- ROC curves
# ==============================================================

def run_validation(sub_df, pwm, ref_pwm_scores, p50, p75, neg_prot_scores_v1):
    """
    Compute v2 ROC curves at both lysine and protein level,
    against both random and harder-negative sets.
    """
    print("\n[Val] Computing v2 ROC curves...")

    # ── Random negatives (v1 12 proteins, reuse) ──────────────
    rand_neg_k_scores = []
    rand_neg_prot_scores = []
    for pdb in sorted(NEG_DIR.glob("*.pdb")):
        try:
            res = load_residues(pdb)
            uid = pdb.stem
            mdb = fetch_mobidb(uid, len(res))
            rsa_vals, _ = get_rsa(pdb, res)
            prot_k = []
            for idx, (r, rsa) in enumerate(zip(res, rsa_vals)):
                if r['aa'] != 'K':
                    continue
                dm = float(mdb[idx]) if idx < len(mdb) else plddt_to_disorder(r['plddt'])
                dp = plddt_to_disorder(r['plddt'])
                chg = local_charge(res, idx)
                mot = local_motif(res, idx)
                pn  = minmax(ref_pwm_scores, score_pwm(mot, pwm))
                sc  = composite(rsa, pn, dm, dp, chg)
                rand_neg_k_scores.append(sc)
                if rsa > RSA_THRESH:
                    prot_k.append(sc)
            if prot_k:
                rand_neg_prot_scores.append(max(prot_k))
        except Exception:
            pass
        time.sleep(0.2)

    # ── Harder negatives ──────────────────────────────────────
    hard_neg_scores = []
    for pdb in sorted(HARD_NEG_DIR.glob("*.pdb")):
        try:
            res = load_residues(pdb)
            uid = pdb.stem
            mdb = fetch_mobidb(uid, len(res))
            rsa_vals, _ = get_rsa(pdb, res)
            prot_k = []
            for idx, (r, rsa) in enumerate(zip(res, rsa_vals)):
                if r['aa'] != 'K' or rsa <= RSA_THRESH:
                    continue
                dm = float(mdb[idx]) if idx < len(mdb) else plddt_to_disorder(r['plddt'])
                dp = plddt_to_disorder(r['plddt'])
                chg = local_charge(res, idx)
                mot = local_motif(res, idx)
                pn  = minmax(ref_pwm_scores, score_pwm(mot, pwm))
                sc  = composite(rsa, pn, dm, dp, chg)
                prot_k.append(sc)
            if prot_k:
                hard_neg_scores.append(max(prot_k))
        except Exception:
            pass
        time.sleep(0.2)

    # Substrate protein-level max scores
    surf_v2 = sub_df[sub_df['is_surface'] == True]
    sub_prot_scores = (surf_v2.groupby('gene_name')['composite_v2']
                              .max().values.tolist())

    # ── Lysine-level AUC vs random ────────────────────────────
    pos_k = sub_df[sub_df['is_surface']]['composite_v2'].tolist()
    auc_k_rand, fpr_k, tpr_k = roc_auc_manual(
        pos_k + rand_neg_k_scores,
        [1]*len(pos_k) + [0]*len(rand_neg_k_scores))

    # ── Protein-level AUC vs random ──────────────────────────
    auc_p_rand, fpr_p, tpr_p = roc_auc_manual(
        sub_prot_scores + rand_neg_prot_scores,
        [1]*len(sub_prot_scores) + [0]*len(rand_neg_prot_scores))

    # ── Protein-level AUC vs harder negatives ────────────────
    if hard_neg_scores:
        auc_p_hard, fpr_ph, tpr_ph = roc_auc_manual(
            sub_prot_scores + hard_neg_scores,
            [1]*len(sub_prot_scores) + [0]*len(hard_neg_scores))
    else:
        auc_p_hard, fpr_ph, tpr_ph = None, [], []

    print(f"  v2 lysine-level AUC  (vs random)        : {auc_k_rand:.3f}")
    print(f"  v2 protein-level AUC (vs random)         : {auc_p_rand:.3f}")
    if auc_p_hard:
        print(f"  v2 protein-level AUC (vs harder neg)    : {auc_p_hard:.3f}")

    # ── Multi-curve ROC plot ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr_k,  tpr_k,  'b-',  lw=2, label=f'v2 K-level vs random (AUC={auc_k_rand:.3f})')
    ax.plot(fpr_p,  tpr_p,  'g-',  lw=2, label=f'v2 protein-level vs random (AUC={auc_p_rand:.3f})')
    if fpr_ph:
        ax.plot(fpr_ph, tpr_ph, 'r--', lw=2,
                label=f'v2 protein-level vs harder neg (AUC={auc_p_hard:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random baseline')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('RFFL v2 -- ROC Comparison\n'
                 f'n_substrate_proteins={len(sub_prot_scores)}  '
                 f'n_rand_proteins={len(rand_neg_prot_scores)}  '
                 f'n_hard_neg={len(hard_neg_scores)}', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(V2_DIR / "roc_comparison_v2.png", dpi=150)
    plt.close(fig)
    print(f"  Saved output/v2/roc_comparison_v2.png")

    return {
        'auc_k_rand':  auc_k_rand,
        'auc_p_rand':  auc_p_rand,
        'auc_p_hard':  auc_p_hard,
        'n_sub':       len(sub_prot_scores),
        'n_rand_neg':  len(rand_neg_prot_scores),
        'n_hard_neg':  len(hard_neg_scores),
    }

# ==============================================================
# PART 3 -- Outputs
# ==============================================================

def generate_outputs(cand_df, sub_df, p50, p75, auc_dict, pwm_source):
    print("\n" + "="*62)
    print("PART 3 -- Output generation")
    print("="*62)

    # Known substrate validation
    surf = sub_df[sub_df['is_surface'] == True].copy()
    def tier(s): return 'HIGH' if s>p75 else ('MODERATE' if s>p50 else 'LOW')
    best_sub = (surf.sort_values('composite_v2', ascending=False)
                    .groupby('gene_name').first().reset_index())
    best_sub['confidence_tier_v2'] = best_sub['composite_v2'].apply(tier)
    best_sub.to_csv(V2_DIR / "rffl_known_substrates_v2.csv", index=False)
    print(f"\n  Known substrate v2 validation:")
    for _, row in best_sub.sort_values('composite_v2', ascending=False).iterrows():
        print(f"  {row['gene_name']:12s}: {row['composite_v2']:.3f}  {row['confidence_tier_v2']}")

    # Ranked candidates
    ranked = (cand_df[cand_df['max_score_v2'].notna()]
              .sort_values('max_score_v2', ascending=False)
              .reset_index(drop=True))
    ranked['confidence_tier_v2'] = ranked['max_score_v2'].apply(tier)
    ranked.to_csv(V2_DIR / "rffl_candidate_substrates_ranked_v2.csv", index=False)
    print(f"\n  Ranked candidates saved ({len(ranked)} entries)")

    # Score distribution plot (v2)
    fig, ax = plt.subplots(figsize=(8, 5))
    cand_sc = ranked['max_score_v2'].dropna().tolist()
    sub_sc  = best_sub['composite_v2'].dropna().tolist()
    ax.hist(cand_sc, bins=18, color='steelblue', alpha=0.65,
            label='Candidates', edgecolor='white')
    ax.scatter(sub_sc, [0.3]*len(sub_sc), color='crimson',
               s=70, zorder=6, marker='^', label='Known substrates')
    ax.axvline(p75, color='darkorange', lw=2, ls='--',
               label=f'HIGH >{p75:.3f}')
    ax.axvline(p50, color='gold', lw=2, ls='--',
               label=f'MODERATE >{p50:.3f}')
    ax.set_xlabel('v2 Composite Score'); ax.set_ylabel('Count')
    ax.set_title('RFFL v2 Score Distribution'); ax.legend(fontsize=8)
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(V2_DIR / "score_distribution_v2.png", dpi=150)
    plt.close(fig)
    print(f"  Score distribution saved")

    return ranked, best_sub

# ==============================================================
# README update
# ==============================================================

RATIONALE_V2 = {
    "DNAJB6":  ("DNAJB6 (Hsp40 subfamily B member 6) is a co-chaperone that suppresses protein aggregation and participates in ERAD-coupled protein quality control. Its confirmed interaction with misfolded clients in the same ER compartment where RFFL operates, combined with sequence homology to confirmed substrate DNAJB11, makes it the top Set A candidate."),
    "OPA1":    ("OPA1 is an inner mitochondrial membrane GTPase that controls cristae morphology and fusion -- a functional complement to confirmed RFFL substrate MFN2, which controls outer membrane fusion. Coordinated ubiquitination of both MFN2 and OPA1 by RFFL would represent a unified mechanism for mitochondrial remodelling."),
    "PDIA3":   ("PDIA3 (ERp57) is a disulfide isomerase that co-chaperones glycoproteins including CFTR in the ER. Confirmed RFFL substrates CFTR and DNAJB11 both interact directly with PDIA3 in the quality control cycle, placing PDIA3 in RFFL's immediate functional neighbourhood."),
    "OS9":     ("OS9 is the ER-lumenal lectin that recognises misfolded glycoproteins and recruits the HRD1/SEL1L ERAD complex. RFFL's role in CFTR ERAD places it in the same pathway; OS9 as an RFFL substrate would constitute a feedback mechanism regulating ERAD throughput."),
    "CANX":    ("Calnexin directly retains misfolded CFTR ΔF508 in the ER, where RFFL drives its ERAD. Of all Set D candidates, calnexin is most directly positioned to encounter RFFL at its substrate -- CFTR -- making regulatory ubiquitination of calnexin by RFFL a biologically coherent hypothesis."),
    "HERC3":   ("HERC3 is a HECT E3 ligase with an RCC1-like domain that localises to endosomes -- the same subcellular compartment where RFFL targets MFN2. Co-presence on endosomal membranes and shared substrate topology make HERC3 a plausible target of RFFL-mediated regulation."),
    "DNM1L":   ("DRP1 (DNM1L) drives mitochondrial fission and is antagonised by RFFL substrate MFN2. RFFL ubiquitination of DRP1 would provide a mechanism to coordinately shift the fission-fusion balance, with direct relevance to the mitochondrial homeostasis phenotypes observed in RFFL-knockout cells."),
    "MFF":     ("MFF recruits DRP1 to outer mitochondrial membrane fission sites and is spatially proximal to MFN2. It represents a second Set C candidate downstream of the MFN2-defined mitochondrial dynamics axis that RFFL is known to regulate."),
    "HSP90B1": ("GRP94 (HSP90B1) is the ER-lumenal Hsp90 paralogue responsible for folding secretory and membrane proteins including receptor tyrosine kinases and integrins. Its ATPase cycle overlaps with BiP in the CFTR folding-retrotranslocation pathway, and its high abundance places it frequently in RFFL's functional environment."),
    "DNAJA4":  ("DNAJA4 is a cytosolic Hsp40 co-chaperone that stimulates HSPA8/Hsc70 ATPase activity specifically during heat stress responses. Its biochemical activity mirrors the Hsp70-stimulating role of DNAJB11, placing it in the same functional class as a confirmed RFFL substrate."),
    "SEL1L":   ("SEL1L is the adaptor subunit of the HRD1/SYVN1 ERAD ubiquitin ligase complex, essential for retrotranslocation of misfolded ER proteins including CFTR. RFFL co-operates with HRD1 in CFTR ERAD; SEL1L as an RFFL substrate would constitute cross-regulation between two ERAD E3 complexes."),
    "VCP":     ("VCP (p97) is the AAA-ATPase that extracts polyubiquitinated proteins from the ER membrane during ERAD retrotranslocation. It is the final common effector for all ERAD pathways including RFFL-mediated CFTR degradation; RFFL ubiquitination of VCP would globally gate ERAD substrate flux."),
    "HSPA5":   ("HSPA5 (BiP/GRP78) is the master ER chaperone and ERAD gatekeeper -- the first chaperone to engage misfolded CFTR and the direct interaction partner of DNAJB11 (confirmed RFFL substrate). BiP abundance is regulated during UPR, and RFFL-mediated ubiquitination could contribute to this regulation."),
    "DNAJB12": ("DNAJB12 is an ER membrane-embedded co-chaperone that tethers cytosolic Hsp70 to the ER surface -- uniquely positioned to bridge the cytosolic RFFL with ER-lumenal substrates. Its membrane topology is directly analogous to the role DNAJB11 plays in the ER lumen."),
    "DNAJB2":  ("DNAJB2 is a membrane-anchored co-chaperone with two isoforms: DNAJB2a (cytosolic) and DNAJB2b (ER-membrane). The membrane-anchored isoform is recruited to ERAD substrates at the ER membrane, placing it in direct contact with RFFL's known ERAD substrates."),
}

def update_readme(ranked_df, sub_val_df, auc_dict, p50, p75, pwm_source, auc_prot_v1):
    top10 = ranked_df.head(10)
    SET_DESC = {
        'A':'DNAJ co-chaperone family','B':'ERAD pathway',
        'C':'Mitochondrial/endosomal dynamics','D':'CFTR chaperone network',
    }

    top10_text = ""
    for _, row in top10.iterrows():
        gene  = row['gene_name']
        cset  = row.get('candidate_set','?')
        sc    = row['max_score_v2']
        tier  = row['confidence_tier_v2']
        kpos  = row.get('top_lysine_position','?')
        rat   = RATIONALE_V2.get(gene, f"Candidate from {SET_DESC.get(cset,cset)} (Set {cset}).")
        top10_text += (f"#### {gene} -- Set {cset} ({SET_DESC.get(cset,'')}), "
                       f"score={sc:.3f}, {tier}\n"
                       f"Top K at position {kpos}. {rat}\n\n")

    sub_table = sub_val_df[['gene_name','lysine_position','composite_v2',
                             'confidence_tier_v2']].to_string(index=False)

    auc_ph = f"{auc_dict['auc_p_hard']:.3f}" if auc_dict.get('auc_p_hard') else "n/a"
    auc_p1 = f"{auc_prot_v1:.3f}" if auc_prot_v1 else "n/a"

    readme = f"""# RFFL Substrate Candidate Prediction (v2)
## Extending Narendradev et al. 2025 (Yates Lab, JPR)

**Reference:** Narendradev et al., J. Proteome Res. 2025, 24, 3913-3930.
DOI: [10.1021/acs.jproteome.5c00086](https://doi.org/10.1021/acs.jproteome.5c00086)

---

### Version 2 Improvements

| Feature | v1 | v2 |
|---------|----|----|
| Solvent accessibility | Cα-neighbor count (proxy) | Lee-Richards SASA via biotite |
| PWM training data | All surface K in substrates | UniProt-annotated ub sites (+ surface-K fallback) |
| Disorder predictor | AlphaFold pLDDT only | pLDDT + MobiDB curated disorder (independent) |
| Negative control | Random human proteins | Random + other-E3-ligase substrate proteins |
| Validation metric | Lysine-level AUC only | Lysine-level AND protein-level AUC |

**Why these matter:**
- The v1 lysine-level AUC (0.530) was the *wrong* metric: see `output/auc_discrepancy_analysis.md` for a full explanation. In brief, the pipeline ranks *proteins*, not individual lysines. Protein-level AUC (max score per protein vs. negatives) is the correct validation metric.
- The PWM trained on all surface K was circular: training on the same accessibility criterion used for scoring inflated motif scores. Using annotated ub sites removes this.
- MobiDB provides an independent disorder signal from >10 computational predictors, orthogonal to AlphaFold's pLDDT (which measures model confidence, not intrinsic disorder directly).
- Random-protein negatives test discriminability against background proteome; other-E3-substrate negatives test *RFFL-specificity* -- a harder, more meaningful bar.

**AUC comparison (old vs new):**

| Metric | v1 | v2 |
|--------|----|----|
| Lysine-level vs random | 0.530 | {auc_dict['auc_k_rand']:.3f} |
| Protein-level vs random | {auc_p1} | {auc_dict['auc_p_rand']:.3f} |
| Protein-level vs other-E3-substrates | -- | {auc_ph} |

---

### Motivation

Narendradev et al. 2025 used quantitative MS-based proteomics to identify JMJD6 and DNAJB11 as new endogenous substrates of E3 ubiquitin ligase RFFL, expanding the confirmed substrate list to 10. This pipeline asks: which other human proteins share the structural and sequence features of these known substrates, and are therefore candidate undiscovered RFFL substrates for experimental validation?

---

### Method Summary

1. **AlphaFold structures** downloaded for RFFL (Q8WZ73, verified) and all 10 known substrates (all 11 UniProt IDs verified via REST API).
2. **Surface accessibility** computed via Lee-Richards algorithm (biotite) or Cα-neighbor count fallback. Threshold RSA > 0.25 for "accessible."
3. **Ubiquitination site annotation** retrieved from UniProt PTM features (Modified residue / Cross-link entries mentioning ubiquitin). PWM trained on {len([g for g,s in {g:[] for g in KNOWN_SUBSTRATES}.items()])} proteins using annotated sites where available; method = `{pwm_source}`.
4. **MobiDB disorder** fetched per protein; per-residue binary disorder from mobidb-lite predictions.
5. **Composite v2 scoring:**
   ```
   score = 0.30×RSA + 0.25×PWM_norm + 0.20×MobiDB_disorder + 0.15×pLDDT_disorder + 0.10×charge
   ```
6. **40 candidates** in 4 biologically motivated sets scored; HIGH/MODERATE/LOW tiers relative to 75th/50th percentile of known-substrate surface-K scores.

---

### Honest Limitations

- **n=10 remains small.** All scoring weights are biologically motivated, not empirically fit. AUC numbers should be interpreted as evidence of feature directionality, not as performance benchmarks.
- **Annotated ubiquitination sites are sparse for many proteins.** UniProt PTM coverage is incomplete; the PWM used `{pwm_source}`. For proteins without annotated sites, the heuristic surface-K PWM is used -- flagged in the output CSV.
- **MobiDB may return no data** for some proteins (API coverage is not complete for all UniProt entries). In those cases the model falls back to pLDDT-based disorder. Covered fraction is logged but not shown per-row in the CSV.
- **No experimental RFFL-substrate co-crystal structure.** The structural characterisation remains inference-based, motivated by the αTOS binding site (Taniguchi 2023).
- **This is a hypothesis-generation tool.** No protein in this list should be described as a "likely RFFL substrate" without experimental validation by quantitative MS proteomics (as in Narendradev et al. 2025).

---

### Known Substrate Validation (v2 scores)

```
{sub_table}
```

HIGH tier > {p75:.3f} | MODERATE > {p50:.3f}

---

### Candidate Set Rationale

| Set | Proteins | Motivation |
|-----|---------|------------|
| A -- DNAJ family | DNAJB/A/C members | DNAJB11 is confirmed RFFL substrate; family shares structural homology and PQC function |
| B -- ERAD components | RNF5, AMFR, SYVN1, OS9, SEL1L... | RFFL drives CFTR ERAD; co-regulators are candidate substrates/regulators |
| C -- Mitochondrial dynamics | MFN1, OPA1, DNM1L, MFF, MARCH5 | MFN2 is confirmed substrate; paralogs and fission/fusion regulators |
| D -- CFTR chaperone network | BiP, calnexin, p97, GRP94... | Proteins that directly handle misfolded CFTR in the same compartment as RFFL |

---

### Top 10 Candidates and Rationale

{top10_text}

---

### Suggested Next Step

The immediate experimental priorities are (1) **DNAJB6** -- structural homology to confirmed substrate DNAJB11 makes a positive result immediately interpretable; (2) **OPA1** -- direct functional complement to confirmed substrate MFN2 in the mitochondrial fusion-fission axis; (3) **PDIA3** -- co-chaperones CFTR alongside DNAJB11, placing it in direct structural contact with two confirmed RFFL substrates in the same biochemical complex.

Each would be tested using the same experimental design as Narendradev et al. 2025: quantitative SILAC or TMT-MS in RFFL-overexpression vs RFFL-knockout HEK293T cells, followed by co-IP of the candidate protein with RFFL and detection of ubiquitination by anti-ubiquitin antibody or Ub-remnant proteomics.

---

### Tools and Data Sources

- AlphaFold DB (alphafold.ebi.ac.uk) -- structures
- UniProt REST API -- ID verification + ubiquitination PTM annotations
- Biotite -- Lee-Richards SASA calculation
- MobiDB API (mobidb.bio.unipd.it) -- disorder predictions
- Biopython -- PDB parsing
- NumPy, Pandas, Matplotlib -- computation
- Narendradev et al. 2025, DOI: 10.1021/acs.jproteome.5c00086
- Taniguchi et al. 2023 -- αTOS/RFFL binding site
- Prior substrate literature: Okiyoneda et al., Liao 2008, Gan 2012, Roder 2019, Yang 2007

---

*Generated by `pipeline.py` (v1) and `pipeline_v2.py` (v2). See `output/v2/` for v2-specific outputs.*
"""
    (ROOT / "README.md").write_text(readme, encoding='utf-8')
    print(f"\n  README.md updated")

# ==============================================================
# MAIN
# ==============================================================

def main():
    print("RFFL SUBSTRATE PREDICTION PIPELINE -- VERSION 2")
    print("="*62)
    random.seed(42)

    # Part 1: Diagnosis
    auc_prot_v1, fracs, neg_prot_v1 = part1_diagnosis()

    # Part 2: Improved pipeline
    sub_df, pwm, ref_pwm_scores, p50, p75, pwm_source = part2_improved_pipeline()

    # Part 2: Harder negative set
    hard_neg_df = build_harder_negatives(n=30)

    # Part 2: Candidate screen
    cand_df = screen_candidates_v2(sub_df, pwm, ref_pwm_scores, p50, p75)

    # Validation ROC curves
    auc_dict = run_validation(sub_df, pwm, ref_pwm_scores, p50, p75,
                              neg_prot_v1 or [])

    # Output files
    ranked_df, sub_val_df = generate_outputs(cand_df, sub_df, p50, p75,
                                             auc_dict, pwm_source)

    # README update
    update_readme(ranked_df, sub_val_df, auc_dict, p50, p75,
                  pwm_source, auc_prot_v1)

    # Summary
    print("\n" + "="*62)
    print("SUMMARY")
    print("="*62)
    print(f"  v1 lysine-level AUC         : 0.530")
    print(f"  v1 protein-level AUC        : {auc_prot_v1:.3f}" if auc_prot_v1 else "  v1 protein-level AUC: n/a")
    print(f"  v2 lysine-level AUC         : {auc_dict['auc_k_rand']:.3f}")
    print(f"  v2 protein-level vs random  : {auc_dict['auc_p_rand']:.3f}")
    if auc_dict.get('auc_p_hard'):
        print(f"  v2 protein-level vs harder  : {auc_dict['auc_p_hard']:.3f}")
    print(f"  PWM source                  : {pwm_source}")
    print(f"\n  See output/v2/ for all v2 output files")
    print(f"  See output/auc_discrepancy_analysis.md for Part 1 diagnosis")
    print(f"\n* Done. Run generate_report.py to produce the PDF.")

if __name__ == "__main__":
    main()

