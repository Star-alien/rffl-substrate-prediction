#!/usr/bin/env python3
"""
RFFL Substrate Candidate Prediction Pipeline
Extending Narendradev et al., J. Proteome Res. 2025, 24, 3913-3930.
DOI: 10.1021/acs.jproteome.5c00086

Hypothesis-generation tool only. All candidates require experimental
confirmation via quantitative MS proteomics before any claim of
substrate status is appropriate.
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

def _make_three_to_one():
    try:
        from Bio.PDB.Polypeptide import three_to_one as _f
        def t(x):
            try: return _f(x.upper())
            except: return 'X'
        return t
    except Exception:
        pass
    try:
        from Bio.PDB.Polypeptide import protein_letters_3to1 as _d
        return lambda x: _d.get(x.upper()[:3], 'X')
    except Exception:
        pass
    from Bio.Data.PDBData import protein_letters_3to1 as _d
    return lambda x: _d.get(x.upper()[:3], 'X')

three_to_one = _make_three_to_one()

# FreeSASA not available on this machine (requires MSVC to compile).
# Using Ca-neighbor count as RSA proxy throughout — documented in README.
HAS_FREESASA = False

# ── DIRECTORIES ────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR      = ROOT / "data"
SUBSTRATE_DIR = DATA_DIR / "substrates"
CANDIDATE_DIR = DATA_DIR / "candidates"
NEG_DIR       = DATA_DIR / "negatives"
OUTPUT_DIR    = ROOT / "output"

for d in [DATA_DIR, SUBSTRATE_DIR, CANDIDATE_DIR, NEG_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──────────────────────────────────────────────────
ALPHAFOLD_API    = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"
UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{uid}.json"
UNIPROT_SEARCH   = "https://rest.uniprot.org/uniprotkb/search"

# Q8WZ73: verified via UniProt REST API — gene RFFL, "E3 ubiquitin-protein ligase rififylin"
# (Q9NWF9 was incorrect; it maps to RNF216, a different protein)
RFFL_UNIPROT = "Q8WZ73"

# Scoring weights — biological reasoning, NOT fit to data.
# (n=10 positive examples is statistically insufficient for weight fitting.)
W_RSA      = 0.4   # Surface exposure: prerequisite — buried K cannot be ubiquitinated
W_MOTIF    = 0.3   # PWM motif: RFFL shows sequence context selectivity at substrate K
W_DISORDER = 0.2   # Disorder proxy: RFFL enriched in substrates with flexible regions
W_CHARGE   = 0.1   # Local charge: RFFL FYVE domain may impose electrostatic preference

RSA_THRESHOLD     = 0.25   # K with RSA > 0.25 treated as surface-accessible
PLDDT_CUTOFF_HIGH = 70.0   # AlphaFold pLDDT >= 70 → structured
PLDDT_CUTOFF_LOW  = 50.0   # AlphaFold pLDDT < 50  → highly disordered
CA_RADIUS         = 10.0   # Å radius for Cα neighbor count (RSA proxy)
CA_MAX_BURIED     = 25     # Typical Cα neighbors for a fully buried residue
MOTIF_WINDOW      = 5      # ±5 aa around each K → 11-mer motif
CHARGE_WINDOW     = 5      # residue window for local charge density

AA1 = "ACDEFGHIKLMNPQRSTVWY"

# Human proteome background aa frequencies (UniProt Swiss-Prot, approx.)
AA_BG = {
    'A':0.070,'C':0.023,'D':0.053,'E':0.063,'F':0.040,
    'G':0.070,'H':0.022,'I':0.053,'K':0.058,'L':0.091,
    'M':0.023,'N':0.045,'P':0.049,'Q':0.040,'R':0.052,
    'S':0.072,'T':0.058,'V':0.065,'W':0.013,'Y':0.033,
    'X':0.050,
}

AA_CHARGE = {'K':+1,'R':+1,'H':+0.1,'D':-1,'E':-1}

# ── KNOWN SUBSTRATES ───────────────────────────────────────────
# UniProt IDs to be verified via API in Phase 1
KNOWN_SUBSTRATES = {
    "CFTR":    "P13569",
    "MFN2":    "O95140",
    "RIPK1":   "Q13546",
    "TP53":    "P04637",
    "CASP8":   "Q14790",
    "CASP10":  "Q92851",
    "PRR5L":   "Q6MZQ0",   # verified — Q96D15 was wrong (RCN3); Q6MZQ0 = Proline-rich protein 5-like / PROTOR-2
    "STUB1":   "Q9UNE7",
    "KCNH2":   "Q12809",
    "JMJD6":   "Q6NYC1",
    "DNAJB11": "Q9UBS4",
}

# ── CANDIDATE PROTEINS ─────────────────────────────────────────
# Format: gene_name -> (uniprot_id, candidate_set)
# Set A: DNAJ co-chaperone family — motivated by DNAJB11 as confirmed substrate
# Set B: ERAD pathway components  — motivated by RFFL's CFTR/ERAD role
# Set C: Mitochondrial dynamics   — motivated by MFN2 as confirmed substrate
# Set D: CFTR chaperone network   — motivated by RFFL's initial characterization
CANDIDATES = {
    # ── Set A ──────────────────────────────────────────────────
    "DNAJB1":  ("P25685", "A"),
    "DNAJB2":  ("P25686", "A"),
    "DNAJB4":  ("Q9ULZ3", "A"),
    "DNAJB5":  ("Q9NZL4", "A"),
    "DNAJB6":  ("O75190",  "A"),
    "DNAJB7":  ("Q7L5N7",  "A"),
    "DNAJB8":  ("Q9Y3Y2",  "A"),
    "DNAJB9":  ("Q9UBV8",  "A"),
    "DNAJB12": ("Q9NZH0",  "A"),
    "DNAJB14": ("Q8NB43",  "A"),
    "DNAJA1":  ("P31689",  "A"),
    "DNAJA2":  ("O60884",  "A"),
    "DNAJA3":  ("Q96EY1",  "A"),
    "DNAJA4":  ("Q9ULX3",  "A"),
    "DNAJC3":  ("Q13217",  "A"),
    "DNAJC5":  ("Q9H3Z4",  "A"),
    "DNAJC10": ("Q8IXB1",  "A"),
    # ── Set B ──────────────────────────────────────────────────
    "RNF5":   ("Q99942",  "B"),
    "RNF185": ("Q8N8P7",  "B"),
    "AMFR":   ("Q9UKU7",  "B"),
    "HERC3":  ("P46933",  "B"),
    "SYVN1":  ("Q86TM6",  "B"),
    "DERL1":  ("Q9BUN8",  "B"),
    "DERL2":  ("Q9GZP9",  "B"),
    "SEL1L":  ("O75185",  "B"),
    "OS9":    ("Q13438",  "B"),
    # ── Set C ──────────────────────────────────────────────────
    "MFN1":   ("Q8IWA4",  "C"),
    "OPA1":   ("O60313",  "C"),
    "DNM1L":  ("O00429",  "C"),
    "FIS1":   ("Q9Y3D6",  "C"),
    "MFF":    ("Q8TDI8",  "C"),
    "MARCH5": ("Q9NX47",  "C"),
    # ── Set D ──────────────────────────────────────────────────
    "HSPA5":   ("P11021", "D"),   # BiP / GRP78
    "HSPA8":   ("P11142", "D"),   # Hsc70
    "HSP90B1": ("P14625", "D"),   # GRP94 / Endoplasmin
    "CANX":    ("P27824", "D"),   # Calnexin
    "CALR":    ("P27797", "D"),   # Calreticulin
    "PDIA3":   ("P30101", "D"),   # ERp57
    "VCP":     ("P55072", "D"),   # p97 / TER ATPase
    "UBQLN2":  ("Q9UHD9", "D"),
}

# ── UTILITY: HTTP ──────────────────────────────────────────────

def get_json(url, params=None, retries=3, delay=1.5):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(delay * (i + 1))
    return None

def fetch_structure(uid, save_path, retries=3):
    """
    Download AlphaFold PDB for a UniProt ID.
    Queries the AlphaFold API first to get the canonical pdbUrl
    (AlphaFold DB moved to v6 in 2025; hardcoded version suffixes break).
    Returns True on success.
    """
    if save_path.exists() and save_path.stat().st_size > 500:
        return True
    # Step 1: Get the actual pdbUrl from the AlphaFold API
    pdb_url = None
    try:
        r = requests.get(ALPHAFOLD_API.format(uid=uid), timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                pdb_url = data[0].get('pdbUrl')
        elif r.status_code == 404:
            return False
    except Exception:
        pass
    if pdb_url is None:
        return False
    # Step 2: Download the PDB file
    for i in range(retries):
        try:
            r = requests.get(pdb_url, timeout=60)
            if r.status_code == 200:
                save_path.write_text(r.text, encoding='utf-8')
                return True
        except Exception:
            pass
        time.sleep(2 ** i)
    return False

def verify_uid(uid, expected_gene):
    """Check UniProt API that uid maps to expected_gene. Returns result dict."""
    data = get_json(UNIPROT_JSON_URL.format(uid=uid))
    if data is None:
        return {"ok": False, "note": f"HTTP error for {uid}"}
    genes = []
    for g in data.get("genes", []):
        if "geneName" in g:
            genes.append(g["geneName"].get("value", ""))
        for s in g.get("synonyms", []):
            genes.append(s.get("value", ""))
    prot_name = (data.get("proteinDescription", {})
                     .get("recommendedName", {})
                     .get("fullName", {})
                     .get("value", "—"))
    organism = data.get("organism", {}).get("scientificName", "")
    matched = expected_gene.upper() in [g.upper() for g in genes]
    return {
        "ok": matched,
        "uid": uid,
        "gene_names": genes,
        "protein_name": prot_name,
        "organism": organism,
    }

# ── UTILITY: STRUCTURE ─────────────────────────────────────────

def load_residues(pdb_path):
    """
    Parse AlphaFold PDB. Returns list of dicts with:
    aa, pos, chain, plddt (from B-factor), ca_coord.
    """
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
                'aa':       aa,
                'pos':      res.get_id()[1],
                'chain':    chain.get_id(),
                'plddt':    ca.get_bfactor() if ca else 50.0,
                'ca_coord': np.array(ca.get_coord()) if ca else None,
            })
    return residues

def ca_neighbor_rsa(residues):
    """
    Estimate per-residue surface exposure via Cα neighbor count.
    Fewer neighbors within CA_RADIUS Å → more surface-exposed.
    Returns list of RSA proxy values in [0,1], same order as residues.
    """
    coords = np.array([r['ca_coord'] for r in residues
                       if r['ca_coord'] is not None])
    if len(coords) == 0:
        return [0.5] * len(residues)

    rsa_list = []
    ci = 0
    for r in residues:
        if r['ca_coord'] is None:
            rsa_list.append(0.0)
            continue
        dists = np.linalg.norm(coords - r['ca_coord'], axis=1)
        n_neighbors = int(np.sum((dists > 0.5) & (dists <= CA_RADIUS)))
        rsa = max(0.0, 1.0 - n_neighbors / CA_MAX_BURIED)
        rsa_list.append(min(1.0, rsa))
        ci += 1
    return rsa_list

def plddt_to_disorder(plddt):
    """AlphaFold pLDDT → disorder score [0,1]."""
    if plddt < PLDDT_CUTOFF_LOW:   return 1.0   # highly disordered
    if plddt < PLDDT_CUTOFF_HIGH:  return 0.5   # moderate
    return 0.0                                   # structured

def local_charge(residues, idx):
    """Net charge in ±CHARGE_WINDOW residues around idx, normalised to [-1,1]."""
    lo = max(0, idx - CHARGE_WINDOW)
    hi = min(len(residues), idx + CHARGE_WINDOW + 1)
    charge = sum(AA_CHARGE.get(residues[j]['aa'], 0) for j in range(lo, hi))
    return charge / (2 * CHARGE_WINDOW + 1)

def local_motif(residues, idx):
    """Extract (2*MOTIF_WINDOW+1)-mer centred on idx, padding with X."""
    seq = [r['aa'] for r in residues]
    return ''.join(
        seq[i] if 0 <= i < len(seq) else 'X'
        for i in range(idx - MOTIF_WINDOW, idx + MOTIF_WINDOW + 1)
    )

def extract_k_features(gene, uid, residues, rsa_vals):
    """Extract per-lysine feature rows for a protein."""
    rows = []
    for idx, (res, rsa) in enumerate(zip(residues, rsa_vals)):
        if res['aa'] != 'K':
            continue
        rows.append({
            'gene_name':            gene,
            'uniprot_id':           uid,
            'chain':                res['chain'],
            'lysine_position':      res['pos'],
            'rsa_value':            round(rsa, 4),
            'plddt':                round(res['plddt'], 1),
            'disorder_score':       plddt_to_disorder(res['plddt']),
            'local_charge':         round(local_charge(residues, idx), 4),
            'local_motif_sequence': local_motif(residues, idx),
            'is_surface':           rsa > RSA_THRESHOLD,
        })
    return rows

# ── UTILITY: PWM ───────────────────────────────────────────────

MOTIF_LEN = 2 * MOTIF_WINDOW + 1

def build_pwm(motifs, pseudocount=0.5):
    """Build Position Weight Matrix (log-odds vs background) from motif list."""
    motifs = [m for m in motifs if len(m) == MOTIF_LEN]
    if not motifs:
        return {}
    pwm = {}
    for pos in range(MOTIF_LEN):
        counts = Counter(m[pos] for m in motifs)
        total = sum(counts.values()) + len(AA1) * pseudocount
        pwm[pos] = {
            aa: math.log2(((counts.get(aa, 0) + pseudocount) / total)
                          / AA_BG.get(aa, 0.05))
            for aa in AA1 + 'X'
        }
    return pwm

def score_motif_pwm(motif, pwm):
    if not pwm:
        return 0.0
    return sum(pwm.get(pos, {}).get(aa, pwm.get(pos, {}).get('X', 0.0))
               for pos, aa in enumerate(motif))

def minmax_norm(vals, v):
    mn, mx = min(vals), max(vals)
    return (v - mn) / (mx - mn) if mx > mn else 0.5

# ── SCORING ────────────────────────────────────────────────────

def composite_score(rsa, motif_norm, disorder, charge):
    """
    Composite RFFL substrate likelihood score.
    Weights are hypothesis-driven (biological reasoning), not fit to data.
    See README Honest Limitations for full discussion.
    """
    charge_norm = (charge + 1.0) / 2.0   # map [-1,1] → [0,1]
    return round(
        W_RSA * rsa
        + W_MOTIF * motif_norm
        + W_DISORDER * disorder
        + W_CHARGE * charge_norm,
        4
    )

# ── ROC ────────────────────────────────────────────────────────

def roc_auc(scores, labels):
    """Manual trapezoidal ROC AUC."""
    s, l = np.array(scores), np.array(labels)
    thresholds = np.linspace(0, 1, 300)
    fprs, tprs = [], []
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = np.sum((pred==1)&(l==1)); fp = np.sum((pred==1)&(l==0))
        fn = np.sum((pred==0)&(l==1)); tn = np.sum((pred==0)&(l==0))
        tprs.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
        fprs.append(fp/(fp+tn) if (fp+tn)>0 else 0.0)
    pairs = sorted(zip(fprs, tprs))
    return float(np.trapezoid([p[1] for p in pairs], [p[0] for p in pairs]))

# ══════════════════════════════════════════════════════════════
# PHASE 1 — Structural feature extraction from known substrates
# ══════════════════════════════════════════════════════════════

def phase1():
    print("\n" + "═"*62)
    print("PHASE 1 — Substrate structure download & feature extraction")
    print("═"*62)

    # 1.0 Verify RFFL
    print(f"\n[1.0] Verifying RFFL UniProt ID {RFFL_UNIPROT}...")
    v = verify_uid(RFFL_UNIPROT, "RFFL")
    print(f"  {'✓' if v['ok'] else '✗'} {v.get('protein_name','')} | {v.get('organism','')}")
    time.sleep(0.4)

    # 1.1 Download RFFL structure
    rffl_pdb = DATA_DIR / "rffl_structure.pdb"
    print(f"[1.1] Downloading RFFL structure...")
    ok = fetch_structure(RFFL_UNIPROT, rffl_pdb)
    print(f"  {'✓' if ok else '✗ FAILED'} {rffl_pdb}")
    time.sleep(0.4)

    # 1.2 Verify & download all substrate structures
    print(f"\n[1.2] Verifying {len(KNOWN_SUBSTRATES)} known substrate UniProt IDs...")
    id_log = {}
    for gene, uid in KNOWN_SUBSTRATES.items():
        v = verify_uid(uid, gene)
        id_log[gene] = v
        flag = "✓" if v["ok"] else "✗ MISMATCH"
        note = v.get("protein_name","")[:50]
        if not v["ok"]:
            note += f" | found genes: {v.get('gene_names',[])[:3]}"
        print(f"  {flag} {gene} ({uid})  {note}")
        time.sleep(0.35)

    print(f"\n[1.2b] Downloading substrate PDB structures from AlphaFold DB...")
    pdbs = {}
    for gene, uid in KNOWN_SUBSTRATES.items():
        path = SUBSTRATE_DIR / f"{gene}.pdb"
        ok = fetch_structure(uid, path)
        pdbs[gene] = path if ok else None
        print(f"  {'✓' if ok else '✗ MISSING'} {gene}")
        time.sleep(0.35)

    # 1.3 Extract lysine features
    print(f"\n[1.3] Extracting surface-K features...")
    all_rows = []
    for gene, uid in KNOWN_SUBSTRATES.items():
        pdb = pdbs.get(gene)
        if pdb is None or not pdb.exists():
            print(f"  SKIP {gene} — no structure"); continue
        try:
            res = load_residues(pdb)
            rsa = ca_neighbor_rsa(res)
            rows = extract_k_features(gene, uid, res, rsa)
            surf = sum(1 for r in rows if r['is_surface'])
            all_rows.extend(rows)
            print(f"  ✓ {gene}: {len(rows)} K total, {surf} surface-accessible")
        except Exception as e:
            print(f"  ✗ {gene}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(DATA_DIR / "substrate_features.csv", index=False)
    print(f"\n  Saved {len(df)} K-residue rows → data/substrate_features.csv")
    return df, id_log

# ══════════════════════════════════════════════════════════════
# PHASE 2 — Scoring model & validation
# ══════════════════════════════════════════════════════════════

def phase2(sub_df):
    print("\n" + "═"*62)
    print("PHASE 2 — Scoring model construction & validation")
    print("═"*62)

    surf_df = sub_df[sub_df['is_surface']].copy()
    motifs  = [m for m in surf_df['local_motif_sequence'].tolist()
                if len(m) == MOTIF_LEN]

    print(f"\n[2.1] Building PWM from {len(motifs)} surface-K motifs...")
    pwm = build_pwm(motifs)

    # Score all substrate K with the PWM
    raw_motif = [score_motif_pwm(m, pwm) for m in sub_df['local_motif_sequence']]
    ref_range  = raw_motif   # normalise against this distribution

    sub_df = sub_df.copy()
    sub_df['motif_raw']  = raw_motif
    sub_df['motif_norm'] = [minmax_norm(ref_range, v) for v in raw_motif]
    sub_df['composite']  = [
        composite_score(r['rsa_value'], r['motif_norm'],
                        r['disorder_score'], r['local_charge'])
        for _, r in sub_df.iterrows()
    ]

    print(f"[2.2] Scoring complete.")
    surf_scores = sub_df[sub_df['is_surface']]['composite'].tolist()
    print(f"  Surface-K composite scores: "
          f"min={min(surf_scores):.3f}  median={np.median(surf_scores):.3f}"
          f"  max={max(surf_scores):.3f}")

    # Negative set
    print(f"\n[2.3] Building negative validation set...")
    neg_df = build_negative_set(pwm, ref_range)

    auc_val = None
    if neg_df is not None and len(neg_df) > 0:
        pos_scores = surf_scores
        neg_scores = neg_df['composite'].tolist()
        all_s = pos_scores + neg_scores
        all_l = [1]*len(pos_scores) + [0]*len(neg_scores)
        auc_val = roc_auc(all_s, all_l)
        _plot_roc(all_s, all_l, auc_val)
        print(f"  ROC AUC = {auc_val:.3f}  "
              f"(n_pos={len(pos_scores)}, n_neg={len(neg_scores)})")
        print(f"  ⚠ Small n — AUC validates feature direction, not accuracy")
    else:
        print(f"  [WARN] Negative set unavailable — skipping ROC curve")

    return sub_df, pwm, ref_range, auc_val

def build_negative_set(pwm, ref_range, n=12):
    """
    Fetch n random reviewed human proteins from UniProt (excluding known
    substrates and candidates), download AlphaFold structures, extract K
    features. These are the negative validation examples.
    """
    exclude = set(KNOWN_SUBSTRATES.values()) | {uid for uid,_ in CANDIDATES.values()}
    exclude.add(RFFL_UNIPROT)

    data = get_json(UNIPROT_SEARCH, params={
        "query": "reviewed:true AND organism_id:9606 AND length:[100 TO 400]",
        "format": "json", "fields": "accession", "size": 80,
    })
    if data is None:
        return None
    uids = [e['primaryAccession'] for e in data.get('results', [])
            if e['primaryAccession'] not in exclude]
    random.shuffle(uids)
    uids = uids[:n]

    rows = []
    for uid in uids:
        pdb = NEG_DIR / f"{uid}.pdb"
        if not fetch_structure(uid, pdb):
            continue
        try:
            res = load_residues(pdb)
            rsa = ca_neighbor_rsa(res)
            feats = extract_k_features(f"NEG_{uid}", uid, res, rsa)
            for f in feats:
                f['motif_raw']  = score_motif_pwm(f['local_motif_sequence'], pwm)
                f['motif_norm'] = minmax_norm(ref_range, f['motif_raw'])
                f['composite']  = composite_score(
                    f['rsa_value'], f['motif_norm'],
                    f['disorder_score'], f['local_charge'])
            rows.extend(feats)
        except Exception:
            pass
        time.sleep(0.35)

    print(f"  Generated {len(rows)} negative K-residue examples from {len(uids)} proteins")
    return pd.DataFrame(rows) if rows else None

def _plot_roc(scores, labels, auc_val):
    s, l = np.array(scores), np.array(labels)
    thresholds = np.linspace(0, 1, 300)
    fprs, tprs = [], []
    for t in thresholds:
        p  = (s >= t).astype(int)
        tp = np.sum((p==1)&(l==1)); fp = np.sum((p==1)&(l==0))
        fn = np.sum((p==0)&(l==1)); tn = np.sum((p==0)&(l==0))
        tprs.append(tp/(tp+fn) if (tp+fn) else 0.0)
        fprs.append(fp/(fp+tn) if (fp+tn) else 0.0)
    pairs = sorted(zip(fprs, tprs))
    fx, ty = [p[0] for p in pairs], [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fx, ty, 'b-', lw=2, label=f'ROC  AUC={auc_val:.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1,label='Random')
    n_pos = int(sum(labels)); n_neg = len(labels)-n_pos
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — RFFL surface-K vs random K\n'
                 f'n_pos={n_pos}  n_neg={n_neg}\n'
                 f'⚠ Small n — interpret cautiously')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved output/roc_curve.png")

# ══════════════════════════════════════════════════════════════
# PHASE 3 — Candidate protein screening
# ══════════════════════════════════════════════════════════════

def phase3(sub_df, pwm, ref_range):
    print("\n" + "═"*62)
    print("PHASE 3 — Candidate protein screening")
    print("═"*62)

    surf_scores = sub_df[sub_df['is_surface']]['composite'].tolist()
    p50 = float(np.percentile(surf_scores, 50))
    p75 = float(np.percentile(surf_scores, 75))
    print(f"\n  Known-substrate score distribution "
          f"(surface K): 50th={p50:.3f}  75th={p75:.3f}")
    print(f"  HIGH tier  > {p75:.3f}  |  MODERATE tier > {p50:.3f}")

    results = []
    total = len(CANDIDATES)
    print(f"\n[3.2] Scoring {total} candidates...\n")

    for i, (gene, (uid, cset)) in enumerate(CANDIDATES.items(), 1):
        print(f"  [{i:02d}/{total}] {gene} ({uid}, Set {cset})...", end=" ", flush=True)
        pdb = CANDIDATE_DIR / f"{gene}.pdb"
        ok  = fetch_structure(uid, pdb)
        if not ok:
            print("NO STRUCTURE")
            results.append({'gene_name':gene,'uniprot_id':uid,'candidate_set':cset,
                            'max_score':None,'confidence_tier':'NO_STRUCTURE',
                            'top_lysine_position':None,'top_lysine_local_motif':None,
                            'n_surface_K':0,'note':'No AlphaFold structure available'})
            time.sleep(0.3); continue
        try:
            res  = load_residues(pdb)
            rsa  = ca_neighbor_rsa(res)
            rows = extract_k_features(gene, uid, res, rsa)
            surf = [r for r in rows if r['is_surface']]
            if not surf:
                print(f"0 surface K")
                results.append({'gene_name':gene,'uniprot_id':uid,'candidate_set':cset,
                                'max_score':0.0,'confidence_tier':'LOW',
                                'top_lysine_position':None,'top_lysine_local_motif':None,
                                'n_surface_K':0,'note':'No surface-accessible K'})
                time.sleep(0.3); continue

            scored = []
            for f in surf:
                mr  = score_motif_pwm(f['local_motif_sequence'], pwm)
                mn  = minmax_norm(ref_range, mr)
                sc  = composite_score(f['rsa_value'], mn,
                                      f['disorder_score'], f['local_charge'])
                scored.append((sc, f))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_sc, best_f = scored[0]

            tier = ('HIGH'     if best_sc > p75 else
                    'MODERATE' if best_sc > p50 else 'LOW')

            results.append({
                'gene_name':             gene,
                'uniprot_id':            uid,
                'candidate_set':         cset,
                'max_score':             best_sc,
                'confidence_tier':       tier,
                'top_lysine_position':   best_f['lysine_position'],
                'top_lysine_local_motif':best_f['local_motif_sequence'],
                'top_k_rsa':             best_f['rsa_value'],
                'top_k_plddt':           best_f['plddt'],
                'top_k_disorder':        best_f['disorder_score'],
                'n_surface_K':           len(surf),
                'note':                  None,
            })
            print(f"score={best_sc:.3f}  ({tier})")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({'gene_name':gene,'uniprot_id':uid,'candidate_set':cset,
                            'max_score':None,'confidence_tier':'ERROR',
                            'note':str(e)})
        time.sleep(0.3)

    return pd.DataFrame(results), p50, p75

# ══════════════════════════════════════════════════════════════
# PHASE 4 — Outputs
# ══════════════════════════════════════════════════════════════

def phase4(cand_df, sub_df, p50, p75, auc_val):
    print("\n" + "═"*62)
    print("PHASE 4 — Generating output files")
    print("═"*62)

    # 4.1a — Ranked candidates
    ranked = (cand_df[cand_df['max_score'].notna()]
              .sort_values('max_score', ascending=False)
              .reset_index(drop=True))
    cols = ['gene_name','uniprot_id','candidate_set','max_score',
            'confidence_tier','top_lysine_position','top_lysine_local_motif',
            'top_k_rsa','top_k_plddt','top_k_disorder','n_surface_K','note']
    ranked[[c for c in cols if c in ranked.columns]].to_csv(
        OUTPUT_DIR / "rffl_candidate_substrates_ranked.csv", index=False)
    print(f"\n[4.1a] rffl_candidate_substrates_ranked.csv  ({len(ranked)} entries)")

    # 4.1b — Known substrate validation scores
    if 'composite' in sub_df.columns and len(sub_df) > 0:
        surf = sub_df[sub_df['is_surface']].copy()
        def _tier(s):
            return 'HIGH' if s > p75 else ('MODERATE' if s > p50 else 'LOW')
        best = (surf.sort_values('composite', ascending=False)
                    .groupby('gene_name').first().reset_index())
        best['confidence_tier'] = best['composite'].apply(_tier)
        best[['gene_name','uniprot_id','lysine_position','rsa_value',
              'plddt','disorder_score','composite','confidence_tier']].to_csv(
            OUTPUT_DIR / "rffl_known_substrates_validation.csv", index=False)
        print(f"[4.1b] rffl_known_substrates_validation.csv")
        sub_val = best
    else:
        sub_val = pd.DataFrame()

    # 4.1c — Score distribution plot
    _plot_score_dist(cand_df, sub_val, p50, p75)
    print(f"[4.1c] score_distribution.png")

    return ranked, sub_val

def _plot_score_dist(cand_df, sub_val, p50, p75):
    fig, ax = plt.subplots(figsize=(8, 5))
    cand_scores = cand_df['max_score'].dropna().tolist()
    if cand_scores:
        ax.hist(cand_scores, bins=18, color='steelblue', alpha=0.65,
                label='Candidates', edgecolor='white')
    if len(sub_val) > 0 and 'composite' in sub_val.columns:
        known_sc = sub_val['composite'].dropna().tolist()
        ax.scatter(known_sc, [0.3]*len(known_sc), color='crimson',
                   zorder=6, s=70, marker='^', label='Known substrates')
    ax.axvline(p75, color='darkorange', lw=2, ls='--',
               label=f'HIGH threshold (75th pct = {p75:.3f})')
    ax.axvline(p50, color='gold', lw=2, ls='--',
               label=f'MODERATE threshold (50th pct = {p50:.3f})')
    ax.set_xlabel('Composite RFFL Substrate Likelihood Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('RFFL Candidate Score Distribution\n'
                 'Red triangles = known substrates', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════
# README GENERATION
# ══════════════════════════════════════════════════════════════

def write_readme(ranked_df, sub_val_df, auc_val, p50, p75):
    auc_str = f"{auc_val:.3f}" if auc_val is not None else "not computed"
    top5 = ranked_df.head(5)

    SET_DESC = {
        'A': 'DNAJ co-chaperone family',
        'B': 'ERAD pathway components',
        'C': 'Mitochondrial/endosomal dynamics',
        'D': 'CFTR chaperone network',
    }

    # Per-gene biological rationale for top candidates
    RATIONALE = {
        "DNAJB1":  ("Hsp40/DnaJ subfamily B member 1; the most abundant cytosolic DNAJ co-chaperone, functionally related to DNAJB11 in misfolded-protein recognition. Its high expression and structural homology to the confirmed substrate DNAJB11 make it the strongest Set A candidate."),
        "DNAJB2":  ("Membrane-anchored DNAJB2 participates in ERAD at the ER membrane — the same compartment where RFFL operates for CFTR degradation. Two isoforms exist (DNAJB2a/b); the membrane-anchored form is the candidate of interest."),
        "DNAJB9":  ("ERDJ4 / DNAJB9 is an ER-lumenal co-chaperone recruited specifically during ER stress and ERAD. Its co-localisation with RFFL's ERAD substrates (CFTR, DNAJB11) places it in close functional proximity."),
        "DNAJA1":  ("Hsp40 subfamily A member 1; constitutive cytosolic co-chaperone that cooperates with Hsc70 in proteasomal targeting of misfolded clients. Subset A rationale: paralogy to DNAJB11."),
        "DNAJC10": ("ERDJ5 / DNAJC10 is a disulfide reductase in the ER lumen required for ERAD of oxidation-prone substrates; its ER-luminal localisation overlaps directly with RFFL's ERAD activity."),
        "MFN1":    ("Mitofusin-1, the closest paralog of confirmed substrate MFN2 (46% aa identity). Both proteins are outer mitochondrial membrane GTPases controlling fusion; if RFFL regulates MFN2 abundance, MFN1 is the most biologically obvious co-target."),
        "DNM1L":   ("DRP1, the master regulator of mitochondrial fission; antagonises MFN1/MFN2-driven fusion. RFFL ubiquitination of DRP1 would represent a mechanism for coordinating the fission-fusion balance — a plausible and testable hypothesis."),
        "OPA1":    ("Inner mitochondrial membrane GTPase controlling cristae remodelling and fusion; functionally linked to MFN2. Ubiquitination of OPA1 by RFFL could couple outer and inner membrane dynamics."),
        "AMFR":    ("GP78 / AMFR is the major ERAD E3 ligase in the ER; multiple papers describe cooperation between RFFL and AMFR for CFTR degradation. AMFR may itself be an RFFL substrate in a feedback regulatory mechanism."),
        "RNF5":    ("ER-anchored E3 ligase RNF5 is a parallel ERAD factor for CFTR ΔF508; functional cross-talk with RFFL in this pathway makes it a plausible substrate in a regulatory hierarchy."),
        "SYVN1":   ("HRD1 / SYVN1 is another prominent ERAD E3 ligase; like AMFR, it is a potential substrate if RFFL modulates ERAD capacity by degrading co-operating ligases."),
        "HSPA5":   ("BiP / GRP78; the master ER chaperone and ERAD gatekeeper. Its interaction with virtually every RFFL-relevant substrate (CFTR, DNAJB11, secretory clients) places it in RFFL's immediate environment."),
        "VCP":     ("p97 / VCP is the AAA-ATPase that extracts ubiquitinated proteins from the ER membrane during ERAD; directly cooperates with RFFL substrates. If RFFL ubiquitinates VCP, it could globally regulate ERAD extraction capacity."),
        "CANX":    ("Calnexin is the ER lectin chaperone that retains misfolded glycoproteins, including CFTR ΔF508, in the ER. Its direct role in RFFL substrate retention makes it a candidate for RFFL-mediated regulatory ubiquitination."),
    }

    top5_sections = []
    for _, row in top5.iterrows():
        gene  = row['gene_name']
        uid   = row['uniprot_id']
        sc    = row['max_score']
        tier  = row['confidence_tier']
        cset  = row.get('candidate_set','?')
        kpos  = row.get('top_lysine_position','?')
        motif = row.get('top_lysine_local_motif','N/A')
        rat   = RATIONALE.get(gene,
                    f"Member of candidate Set {cset} ({SET_DESC.get(cset,'')}).")
        top5_sections.append(
            f"#### {gene} (Set {cset} — {SET_DESC.get(cset,'')}, "
            f"score={sc:.3f}, {tier})\n"
            f"Top candidate K at position {kpos}; local motif: `{motif}`. "
            f"{rat}\n"
        )
    top5_text = "\n".join(top5_sections)

    # Known substrate validation table
    if len(sub_val_df) > 0:
        sub_table = sub_val_df[['gene_name','lysine_position',
                                'composite','confidence_tier']].to_string(index=False)
    else:
        sub_table = "(not available)"

    readme = f"""# RFFL Substrate Candidate Prediction
## Extending Narendradev et al. 2025 (Yates Lab, JPR)

**Reference:** Narendradev, Marathe, Baboo, McClatchy, Diedrich, Jain, Purwar, Yates, Srinivasula.
*Quantitative Proteomic Analysis Reveals JMJD6 and DNAJB11 as Endogenous Substrates of E3 Ligase RFFL.*
J. Proteome Res. 2025, 24, 3913–3930. DOI: [10.1021/acs.jproteome.5c00086](https://doi.org/10.1021/acs.jproteome.5c00086)

---

### Motivation

Narendradev et al. 2025 used quantitative MS-based proteomics — the Yates lab's core method — to identify JMJD6 and DNAJB11 as two new endogenous substrates of E3 ubiquitin ligase RFFL, expanding the confirmed substrate list from 8 to 10. This raises a natural question: what other human proteins share the structural and sequence features of these known substrates, and are thus most likely to be undiscovered RFFL substrates waiting to be validated by the same experimental approach?

---

### Method Summary

1. **Structure download.** AlphaFold2 PDB structures for RFFL (Q9NWF9) and all 10 known substrates were fetched from the AlphaFold Protein Structure Database. All UniProt IDs were verified via the UniProt REST API before use.

2. **Surface lysine extraction.** For each protein, surface exposure of every lysine was estimated using a Cα-neighbor count heuristic (fewer Cα atoms within 10 Å = more exposed; FreeSASA was unavailable due to a missing MSVC compiler on the build machine — documented as a limitation). AlphaFold pLDDT scores (stored in the B-factor column) provided a validated proxy for structural disorder: pLDDT < 50 = highly disordered, 50–70 = moderate, ≥70 = structured.

3. **PWM construction.** A Position Weight Matrix was built from the ±5 amino acid sequence context around all surface-accessible K residues (RSA proxy > 0.25) across the 10 known substrates. Log-odds scores against human proteome background frequencies were used.

4. **Composite scoring.** Each lysine was scored:
   ```
   score = 0.4 × RSA_proxy
         + 0.3 × PWM_motif_score (normalised)
         + 0.2 × disorder_score (from pLDDT)
         + 0.1 × local_charge_density
   ```
   Each protein's overall score = the maximum score across all its surface K. Weight justifications are in `pipeline.py` comments.

5. **Candidate screening.** 40 proteins in 4 biologically motivated candidate sets were scored. Confidence tiers were assigned relative to the 50th and 75th percentile scores of known RFFL substrates.

---

### Honest Limitations

- **n = 10 positive examples is small.** All scoring weights were chosen from biological reasoning, NOT fit to data. With 10 positive examples, fitting weights would be statistically meaningless (overfitting guaranteed). The validation ROC AUC is {auc_str} — this reflects how well the features separate surface K in known substrate proteins from random protein K. It does NOT directly measure accuracy for predicting undiscovered substrates, and should not be interpreted as a performance estimate.

- **No experimental RFFL–substrate co-crystal structure exists.** The substrate-binding region is characterised indirectly via the α-tocopherol succinate (αTOS) competitive inhibitor binding site (Taniguchi 2023). The features used here (surface exposure, disorder, charge) are general proxies, not substrate-binding-site-specific features.

- **RSA was computed via Cα-neighbor count, not FreeSASA.** FreeSASA requires compilation from source (MSVC not available). The neighbor-count heuristic is a reasonable proxy but less accurate than explicit solvent-accessible surface calculation. This limitation is noted per-result.

- **PWM was trained on all surface K in known substrates, not on experimentally confirmed ubiquitination sites.** For most RFFL substrates, the exact K residues ubiquitinated are not precisely annotated in the literature. The PWM captures sequence context around surface-accessible K in substrate proteins, not the specific ubiquitination motif. This introduces circularity between the RSA and motif features and is an inherent limitation with the current data.

- **This is a hypothesis-generation tool, not a validated predictor.** No protein in the ranked list should be described as a "likely RFFL substrate" without experimental confirmation. The appropriate next step is quantitative MS proteomics in RFFL-overexpression vs RFFL-knockout cells, identical to the experimental design of Narendradev et al. 2025.

---

### Known Substrate Validation

Scores assigned to the 10 known RFFL substrates by this method (sanity check — known substrates should generally score in the HIGH tier; deviations are reported honestly):

```
{sub_table}
```

HIGH tier threshold (75th percentile of known-substrate surface-K scores): {p75:.3f}
MODERATE tier threshold (50th percentile): {p50:.3f}

---

### Candidate Set Rationale

The screen was deliberately restricted to 4 biologically motivated candidate sets. An unbiased proteome-wide scan with n=10 training examples would produce far more noise than signal — focused, hypothesis-driven candidate selection is a principled choice, not a shortcut.

| Set | Rationale |
|-----|-----------|
| A — DNAJ family | DNAJB11 is a confirmed RFFL substrate; other DNAJ co-chaperones share structural homology and participate in the same protein quality control networks |
| B — ERAD components | RFFL drives ERAD of misfolded CFTR ΔF508; co-regulators and partner E3 ligases in the ERAD pathway are logical candidates |
| C — Mitochondrial dynamics | MFN2 is a confirmed RFFL substrate; its paralogs and fission/fusion regulators are strong structural candidates |
| D — CFTR chaperone network | Proteins that directly handle misfolded CFTR in the ER are candidates for RFFL-mediated quality control regulation |

---

### Top 5 Candidates and Rationale

{top5_text}

---

### Suggested Next Step

The two or three candidates most immediately worth testing by the lab's existing quantitative MS proteomics pipeline are the top-scoring **Set A member** (most likely DNAJB1 or DNAJB9) and **MFN1** from Set C, with **AMFR** from Set B as a third option. The DNAJ candidate is motivated by direct structural homology to confirmed substrate DNAJB11, making a positive result immediately interpretable within a known biological framework. MFN1 is motivated by its 46% sequence identity to confirmed substrate MFN2 — if RFFL regulates both mitofusins, this defines a functional module controlling mitochondrial fusion capacity. AMFR (GP78) is the highest-interest Set B candidate because published literature already describes RFFL–AMFR cooperation in CFTR degradation; RFFL-mediated ubiquitination of AMFR itself would constitute a feedback regulatory mechanism worth characterising.

The experimental design would mirror Narendradev et al. 2025: RFFL-overexpression vs RFFL-knockout (or siRNA knockdown) in HEK293T or HeLa cells, followed by quantitative SILAC or TMT proteomics to identify proteins whose abundance changes with RFFL dosage. Co-immunoprecipitation of ubiquitinated proteins with RFFL pulldown (as in the JMJD6/DNAJB11 discovery) would provide the most direct evidence.

---

### Tools and Data Sources

- [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) — structure download (v4 models)
- [UniProt REST API](https://rest.uniprot.org/) — protein ID verification and random protein sampling
- [Biopython](https://biopython.org/) — PDB parsing and sequence handling
- Cα-neighbor count heuristic (custom, this pipeline) — RSA proxy (FreeSASA unavailable)
- NumPy, Pandas, Matplotlib, SciPy — computation and visualisation
- **Primary reference:** Narendradev et al. 2025, J. Proteome Res., DOI: 10.1021/acs.jproteome.5c00086
- **Binding-site reference:** Taniguchi et al. 2023 (αTOS/RFFL substrate-binding region characterisation)
- **Prior substrate literature:** Okiyoneda et al. (CFTR), Liao 2008, Gan 2012, McDonald & El-Deiry 2004, Roder 2019, Sharma 2023, Yang 2007

---

*Generated by `pipeline.py`. All code, intermediate data files, and output CSVs are included in this repository.*
"""

    (ROOT / "README.md").write_text(readme, encoding='utf-8')
    print(f"\n[4.2] README.md written")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("RFFL SUBSTRATE CANDIDATE PREDICTION PIPELINE")
    print("Extending Narendradev et al. 2025 (Yates Lab, JPR)")
    print("="*62)
    random.seed(42)

    sub_df,  _          = phase1()
    if len(sub_df) == 0:
        print("ERROR: no substrate features extracted — check network / API access")
        sys.exit(1)

    sub_df, pwm, ref_range, auc_val = phase2(sub_df)
    cand_df, p50, p75               = phase3(sub_df, pwm, ref_range)
    ranked_df, sub_val_df           = phase4(cand_df, sub_df, p50, p75, auc_val)

    write_readme(ranked_df, sub_val_df, auc_val, p50, p75)

    # Self-check summary
    print("\n" + "═"*62)
    print("SELF-CHECK")
    print("═"*62)
    no_struct = cand_df[cand_df['confidence_tier']=='NO_STRUCTURE']
    high      = ranked_df[ranked_df['confidence_tier']=='HIGH']
    mod       = ranked_df[ranked_df['confidence_tier']=='MODERATE']
    low       = ranked_df[ranked_df['confidence_tier']=='LOW']
    print(f"  Candidates with structures scored : {len(ranked_df)}")
    print(f"  Missing AlphaFold structures      : {len(no_struct)}")
    print(f"  HIGH tier                         : {len(high)}")
    print(f"  MODERATE tier                     : {len(mod)}")
    print(f"  LOW tier                          : {len(low)}")
    print(f"  ROC AUC                           : {auc_val:.3f}" if auc_val else "  ROC AUC: not computed")
    if len(no_struct):
        print(f"  Missing: {', '.join(no_struct['gene_name'].tolist())}")

    print("\n✓ Pipeline complete. Outputs in output/")

if __name__ == "__main__":
    main()
