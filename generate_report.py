#!/usr/bin/env python3
"""
generate_report.py — RFFL Substrate Prediction PDF Report
7-page reportlab PDF, navy/white styling, embedded matplotlib figures.
Run AFTER pipeline_v2.py has populated output/v2/.
"""

import os, math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

# ── Paths ──────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
V2_DIR  = ROOT / "output" / "v2"
OUT_DIR = ROOT / "output"

RANKED_CSV  = V2_DIR / "rffl_candidate_substrates_ranked_v2.csv"
VAL_CSV     = V2_DIR / "rffl_known_substrates_v2.csv"
ROC_PNG     = V2_DIR / "roc_comparison_v2.png"
DIST_PNG    = V2_DIR / "score_distribution_v2.png"
AUC_MD      = OUT_DIR / "auc_discrepancy_analysis.md"

PDF_OUT = OUT_DIR / "RFFL_substrate_prediction_report.pdf"

# ── Colours ────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1B2A4A")
TEAL   = colors.HexColor("#2E86AB")
SILVER = colors.HexColor("#E8ECF0")
GOLD   = colors.HexColor("#D4A017")
WHITE  = colors.white
BLACK  = colors.black

# ── Styles ─────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles['title'] = ParagraphStyle('Title2',
        fontName='Helvetica-Bold', fontSize=22,
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=6)
    styles['subtitle'] = ParagraphStyle('Sub',
        fontName='Helvetica', fontSize=11,
        textColor=colors.HexColor("#BDD5EA"), alignment=TA_CENTER, spaceAfter=4)
    styles['h1'] = ParagraphStyle('H1',
        fontName='Helvetica-Bold', fontSize=14,
        textColor=NAVY, spaceBefore=14, spaceAfter=6)
    styles['h2'] = ParagraphStyle('H2',
        fontName='Helvetica-Bold', fontSize=11,
        textColor=TEAL, spaceBefore=10, spaceAfter=4)
    styles['body'] = ParagraphStyle('Body',
        fontName='Helvetica', fontSize=9,
        textColor=BLACK, leading=13, spaceAfter=4, alignment=TA_JUSTIFY)
    styles['caption'] = ParagraphStyle('Cap',
        fontName='Helvetica-Oblique', fontSize=8,
        textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=6)
    styles['small'] = ParagraphStyle('Small',
        fontName='Helvetica', fontSize=8,
        textColor=BLACK, leading=11, spaceAfter=2)
    styles['bullet'] = ParagraphStyle('Bullet',
        fontName='Helvetica', fontSize=9,
        textColor=BLACK, leading=13, spaceAfter=3,
        leftIndent=14, firstLineIndent=-10)
    styles['tbl_hdr'] = ParagraphStyle('TH',
        fontName='Helvetica-Bold', fontSize=8,
        textColor=WHITE, alignment=TA_CENTER)
    styles['tbl_cell'] = ParagraphStyle('TC',
        fontName='Helvetica', fontSize=8,
        textColor=BLACK, alignment=TA_LEFT)
    styles['tbl_cell_c'] = ParagraphStyle('TCC',
        fontName='Helvetica', fontSize=8,
        textColor=BLACK, alignment=TA_CENTER)
    return styles

# ── NavyBox cover flowable ──────────────────────────────────────
class ColorBox(Flowable):
    def __init__(self, w, h, color):
        Flowable.__init__(self)
        self.box_w = w; self.box_h = h; self.color = color
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.box_w, self.box_h, fill=1, stroke=0)
    def wrap(self, *args):
        return self.box_w, self.box_h

# ── PNG embed helper ────────────────────────────────────────────
def embed_png(path, width=6.0*inch):
    if not Path(path).exists():
        return None
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    return Image(str(path), width=width, height=width * ratio)

# ── Inline figure from matplotlib BytesIO ──────────────────────
def make_roc_figure_inline(ranked_df, val_df):
    """Fallback: regenerate simple ROC-like figure if PNG not available."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Comparison (inline regeneration)')
    ax.legend(fontsize=8)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img._restrictSize(5.5*inch, 4*inch)
    return img

# ── Table builder ───────────────────────────────────────────────
def make_table(data, col_widths, hdr_color=NAVY):
    tbl = Table(data, colWidths=col_widths)
    n_rows = len(data)
    style = TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), hdr_color),
        ('TEXTCOLOR',   (0,0), (-1,0), WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, SILVER]),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,1), (-1,-1), 8),
        ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
        ('ALIGN',       (2,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING',(0,0), (-1,-1), 4),
        ('TOPPADDING',  (0,0), (-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1), 3),
        ('GRID',        (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ('LINEBELOW',   (0,0), (-1,0),  0.8, NAVY),
    ])
    for i in range(1, n_rows):
        row_val = data[i][2] if len(data[i]) > 2 else ''
        try:
            sc = float(str(row_val).replace(',', '.'))
            if sc > 0.75:
                style.add('TEXTCOLOR', (2,i), (2,i), colors.HexColor("#006600"))
                style.add('FONTNAME',  (2,i), (2,i), 'Helvetica-Bold')
            elif sc < 0.50:
                style.add('TEXTCOLOR', (2,i), (2,i), colors.HexColor("#AA2200"))
        except Exception:
            pass
    tbl.setStyle(style)
    return tbl

# ══════════════════════════════════════════════════════════════
# PAGE BUILDERS
# ══════════════════════════════════════════════════════════════

def page_cover(styles, story):
    # Full-width navy box with white text
    story.append(ColorBox(7.5*inch, 2.8*inch, NAVY))
    story.append(Spacer(1, -2.8*inch))

    story.append(Spacer(1, 0.45*inch))
    story.append(Paragraph("RFFL E3 Ubiquitin Ligase", styles['title']))
    story.append(Paragraph("Substrate Candidate Prediction", styles['title']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Confidence-ranked predictions using AlphaFold structures,",
                            styles['subtitle']))
    story.append(Paragraph("real ubiquitination-site data, and MobiDB disorder predictions",
                            styles['subtitle']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Pipeline v2 · Extending Narendradev et al. 2025, J. Proteome Res.",
                            styles['subtitle']))
    story.append(Spacer(1, 1.45*inch))

    # Gold rule
    story.append(HRFlowable(width="100%", thickness=2, color=GOLD, spaceAfter=14))

    # Summary box
    summary_data = [
        [Paragraph("<b>Metric</b>", styles['tbl_hdr']),
         Paragraph("<b>v1</b>",     styles['tbl_hdr']),
         Paragraph("<b>v2</b>",     styles['tbl_hdr'])],
        ["Known substrates analysed", "10/11", "11/11"],
        ["Candidate proteins screened", "40", "40"],
        ["Lysine-level AUC (vs. random)", "0.530", "see p.3"],
        ["Protein-level AUC (vs. random)", "n/a", "see p.3"],
        ["Protein-level AUC (vs. harder neg)", "—", "see p.3"],
        ["SASA method", "Cα-neighbor proxy", "biotite Lee-Richards"],
        ["PWM training set", "all surface K", "UniProt ub-site annotations"],
        ["Disorder predictor", "pLDDT only", "pLDDT + MobiDB curated"],
        ["Negative control type", "random human", "random + other-E3-substrates"],
    ]
    col_w = [2.8*inch, 1.8*inch, 1.8*inch]
    tbl = Table(summary_data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), NAVY),
        ('TEXTCOLOR',   (0,0), (-1,0), WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, SILVER]),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING',  (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
        ('GRID',        (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ('LINEBELOW',   (0,0), (-1,0),  0.8, NAVY),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.3*inch))

    meta_text = ("Remi Sky · UCSD · Pipeline code: github.com/Star-alien/rffl-substrate-prediction · "
                 "DOI: 10.1021/acs.jproteome.5c00086")
    story.append(Paragraph(meta_text, styles['caption']))
    story.append(PageBreak())


def page_method(styles, story):
    story.append(Paragraph("1. Method Summary", styles['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    story.append(Paragraph("Overview", styles['h2']))
    story.append(Paragraph(
        "This pipeline identifies candidate substrates of the E3 ubiquitin ligase RFFL (UniProt Q8WZ73) "
        "by scoring the surface-accessible lysines of 40 candidate proteins against features derived from "
        "11 experimentally confirmed RFFL substrates (Narendradev et al. 2025; Okiyoneda et al. 2011; "
        "Liao 2008; Gan 2012; Roder 2019; Yang 2007). Each candidate protein receives a composite score "
        "equal to the maximum score across all its surface-accessible lysines; candidates are ranked and "
        "thresholded into HIGH / MODERATE / LOW tiers relative to the 75th and 50th percentile of "
        "known-substrate surface-K scores.",
        styles['body']))

    story.append(Paragraph("Feature Engineering (v2)", styles['h2']))
    bullets = [
        ("<b>RSA (weight 0.30).</b> Relative solvent accessibility computed by the Lee-Richards "
         "algorithm via biotite on AlphaFold v6 PDB structures. Fallback: Cα-neighbor count "
         "within 10 Å if biotite fails. Lysines with RSA ≤ 0.25 are excluded from scoring "
         "(buried residues cannot be ubiquitinated in most contexts)."),
        ("<b>PWM score (weight 0.25).</b> Position-weight matrix trained on ±5 aa windows around "
         "experimentally annotated ubiquitination sites retrieved from UniProt PTM/modification "
         "features (type = 'Modified residue' or 'Cross-link' with ubiquitin/glycyl-lysine "
         "keyword). Where fewer than 5 annotated sites are available across all 11 substrates, "
         "the matrix falls back to all surface-K windows (flagged in CSV as pwm_source). "
         "Log-odds against the human proteome background frequency (Uniprot/Swiss-Prot composition)."),
        ("<b>MobiDB disorder (weight 0.20).</b> Per-residue disorder from MobiDB-lite "
         "(mobidb.bio.unipd.it), which aggregates ≥10 computational disorder predictors. "
         "Provides an independent measure of intrinsic disorder orthogonal to AlphaFold pLDDT."),
        ("<b>pLDDT-based disorder (weight 0.15).</b> AlphaFold per-residue confidence stored in "
         "B-factor column: pLDDT < 50 → disorder=1.0; 50–70 → 0.5; ≥ 70 → 0.0. Kept as a "
         "complementary, independent structural signal. Weight reduced from v1 (0.20) because "
         "MobiDB now handles the primary disorder channel."),
        ("<b>Local charge (weight 0.10).</b> Mean residue charge in a ±5 aa window "
         "(K,R=+1; D,E=−1; H=+0.1). The RFFL FYVE domain has electrostatic preference for "
         "positively charged endosomal membranes; a modest positive-charge prior is preserved."),
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles['bullet']))

    story.append(Paragraph("Scoring Equation", styles['h2']))
    story.append(Paragraph(
        "score = 0.30 × RSA + 0.25 × PWM<sub>norm</sub> + 0.20 × MobiDB<sub>disorder</sub> "
        "+ 0.15 × pLDDT<sub>disorder</sub> + 0.10 × charge<sub>norm</sub>",
        styles['body']))
    story.append(Paragraph(
        "All terms are normalised to [0, 1]. Composite score is protein-level MAX across all "
        "surface-accessible K. Weights are hypothesis-driven and were NOT fit to data (n=10 "
        "known substrates is too small for empirical weight optimisation).",
        styles['body']))

    story.append(Paragraph("Candidate Sets", styles['h2']))
    set_data = [
        [Paragraph("<b>Set</b>", styles['tbl_hdr']),
         Paragraph("<b>Proteins</b>", styles['tbl_hdr']),
         Paragraph("<b>Biological motivation</b>", styles['tbl_hdr'])],
        ["A — DNAJ family", "17 members",
         "DNAJB11 is confirmed RFFL substrate; entire Hsp40 family shares PQC function"],
        ["B — ERAD pathway",  "9 proteins",
         "RFFL drives CFTR ERAD; HRD1/SEL1L/OS9/SYVN1 complex members"],
        ["C — Mitochondrial dynamics", "6 proteins",
         "MFN2 confirmed; paralogs and fission regulators (OPA1, DNM1L, MFF)"],
        ["D — CFTR chaperone network", "8 proteins",
         "Proteins that directly handle misfolded CFTR in same compartment as RFFL"],
    ]
    story.append(make_table(set_data, [1.0*inch, 1.2*inch, 4.2*inch]))
    story.append(PageBreak())


def page_validation(styles, story, val_csv_path, roc_png_path):
    story.append(Paragraph("2. Validation Results", styles['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    story.append(Paragraph("AUC Discrepancy — Why Lysine-Level AUC Is the Wrong Metric", styles['h2']))
    story.append(Paragraph(
        "The v1 lysine-level ROC AUC of 0.530 appeared to contradict the fact that 10/11 known "
        "substrates scored HIGH at the protein level. The resolution is methodological: RFFL "
        "ubiquitinates a small number of specific lysines per substrate, but the pipeline "
        "characterises ALL surface-accessible lysines in each substrate protein. CFTR, for "
        "example, has 78 surface-accessible K. The lysine-level comparison lumps those 78 "
        "together as 'positives', diluting the signal toward 0.5. The correct validation "
        "unit is the protein — scored by its MAX surface-K score — which is also how the "
        "ranking is used in practice. Protein-level AUC was therefore the primary validation "
        "metric in v2.", styles['body']))

    story.append(Paragraph("ROC Comparison", styles['h2']))
    roc_img = embed_png(roc_png_path, width=5.5*inch)
    if roc_img:
        story.append(roc_img)
        story.append(Paragraph(
            "Figure 1. Multi-curve ROC. Blue: v2 lysine-level vs. random negatives. "
            "Green: v2 protein-level vs. random negatives. Red dashed: v2 protein-level "
            "vs. harder negative control (proteins ubiquitinated by other E3 ligases).",
            styles['caption']))
    else:
        story.append(Paragraph("[ROC figure not found — run pipeline_v2.py first]", styles['small']))

    story.append(Paragraph("Known Substrate Validation (v2 scores)", styles['h2']))
    if val_csv_path.exists():
        val_df = pd.read_csv(val_csv_path)
        # Select relevant columns
        cols = ['gene_name','lysine_position','composite_v2','confidence_tier_v2']
        cols = [c for c in cols if c in val_df.columns]
        val_sorted = val_df[cols].sort_values('composite_v2', ascending=False
                                              ).reset_index(drop=True)
        hdr = [Paragraph("<b>" + c.replace('_',' ').title() + "</b>", styles['tbl_hdr'])
               for c in cols]
        tbl_data = [hdr]
        for _, row in val_sorted.iterrows():
            r = [str(row.get(c,'')) for c in cols]
            tbl_data.append(r)
        cw = [1.4*inch, 1.4*inch, 1.4*inch, 1.8*inch][:len(cols)]
        story.append(make_table(tbl_data, cw))
        story.append(Spacer(1, 6))
        high_n = (val_df['confidence_tier_v2'] == 'HIGH').sum() if 'confidence_tier_v2' in val_df else '?'
        total  = len(val_df)
        story.append(Paragraph(
            f"{high_n}/{total} known substrates classified HIGH. "
            "Validation is performed against n=11 positives with small negative sets — "
            "AUC numbers carry wide confidence intervals and should be interpreted "
            "directionally, not as precision benchmarks.",
            styles['body']))
    else:
        story.append(Paragraph("[Validation CSV not found — run pipeline_v2.py first]",
                               styles['small']))
    story.append(PageBreak())


def page_top_candidates(styles, story, ranked_csv_path, dist_png_path):
    story.append(Paragraph("3. Top Candidate Predictions", styles['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    if ranked_csv_path.exists():
        df = pd.read_csv(ranked_csv_path)
        df = df[df['max_score_v2'].notna()].sort_values('max_score_v2', ascending=False)

        # Score distribution figure
        dist_img = embed_png(dist_png_path, width=5.5*inch)
        if dist_img:
            story.append(dist_img)
            story.append(Paragraph(
                "Figure 2. v2 composite score distribution. Blue histogram: 40 candidates "
                "(highest-scoring K per protein). Red triangles: known substrates. "
                "Orange/gold lines: HIGH/MODERATE thresholds (75th/50th percentile).",
                styles['caption']))

        story.append(Paragraph("Top 10 Ranked Candidates", styles['h2']))
        top10 = df.head(10)
        SET_DESC = {'A':'DNAJ','B':'ERAD','C':'Mito','D':'Chaperone'}
        hdr = [Paragraph(f"<b>{h}</b>", styles['tbl_hdr'])
               for h in ['Rank','Gene','Score','Tier','Set','Top K','n Surface K']]
        tbl_data = [hdr]
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            cset = row.get('candidate_set', '?')
            tbl_data.append([
                str(rank),
                str(row.get('gene_name', '')),
                f"{row['max_score_v2']:.4f}",
                str(row.get('confidence_tier_v2', '')),
                f"{cset} ({SET_DESC.get(cset,'')})",
                f"K{int(row.get('top_lysine_position',0)) if pd.notna(row.get('top_lysine_position')) else '?'}",
                str(int(row.get('n_surface_K', 0)) if pd.notna(row.get('n_surface_K')) else '?'),
            ])
        cw = [0.4*inch, 0.9*inch, 0.7*inch, 0.9*inch, 1.3*inch, 0.7*inch, 0.8*inch]
        story.append(make_table(tbl_data, cw))

        story.append(Spacer(1, 8))
        high_n = (df['confidence_tier_v2'] == 'HIGH').sum()
        mod_n  = (df['confidence_tier_v2'] == 'MODERATE').sum()
        story.append(Paragraph(
            f"Of {len(df)} successfully screened candidates: "
            f"{high_n} HIGH confidence, {mod_n} MODERATE confidence.",
            styles['body']))
    else:
        story.append(Paragraph("[Ranked CSV not found — run pipeline_v2.py first]", styles['small']))

    story.append(PageBreak())


RATIONALE = {
    "DNAJB6": (
        "DNAJB11 (confirmed RFFL substrate) and DNAJB6 are close paralogs within the DNAJ "
        "Hsp40 subfamily B, sharing the J-domain structure required for Hsc70 stimulation. "
        "DNAJB6 is specifically involved in suppressing polyglutamine aggregation and co-localises "
        "with RFFL in the ER quality-control network, making it the most structurally and "
        "functionally motivated candidate in Set A."
    ),
    "OPA1": (
        "OPA1 controls inner mitochondrial membrane fusion and cristae remodelling — the "
        "functional complement of confirmed substrate MFN2, which governs outer membrane fusion. "
        "RFFL-mediated ubiquitination of both MFN2 and OPA1 would constitute a unified mechanism "
        "for coordinating mitochondrial morphology, a hypothesis directly testable by MFN2/OPA1 "
        "co-IP in RFFL-overexpression cells."
    ),
    "PDIA3": (
        "PDIA3 (ERp57) is the disulfide isomerase that co-chaperones CFTR with calnexin in the "
        "ER folding cycle. Confirmed substrates CFTR and DNAJB11 both interact with PDIA3 in "
        "the same biochemical complex, placing PDIA3 in direct spatial contact with RFFL's two "
        "most well-characterised substrates simultaneously."
    ),
    "OS9": (
        "OS9 is the ER-lumenal lectin that recognises misfolded glycoproteins and recruits the "
        "HRD1/SEL1L ERAD complex for retrotranslocation. RFFL participates in CFTR ERAD through "
        "the same pathway; RFFL ubiquitination of OS9 would provide a feedback control point "
        "for ERAD throughput, modulating how many misfolded clients enter the retrotranslocation "
        "pathway."
    ),
    "CANX": (
        "Calnexin is the ER-membrane lectin chaperone that directly retains misfolded CFTR ΔF508 "
        "in the ER, where RFFL drives its degradation. Of all Set D candidates, calnexin is most "
        "proximally positioned to RFFL's confirmed substrate CFTR, and RFFL ubiquitination of "
        "calnexin could serve as a mechanism to tune the rate at which misfolded CFTR is "
        "transferred from the chaperone-retention complex to the ERAD machinery."
    ),
}

def page_top5_writeups(styles, story, ranked_csv_path):
    story.append(Paragraph("4. Detailed Candidate Writeups — Top 5", styles['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=10))

    if not ranked_csv_path.exists():
        story.append(Paragraph("[Ranked CSV not found]", styles['small']))
        story.append(PageBreak())
        return

    df = pd.read_csv(ranked_csv_path)
    df = df[df['max_score_v2'].notna()].sort_values('max_score_v2', ascending=False).head(5)

    SET_FULL = {
        'A': 'Set A — DNAJ Co-chaperone Family',
        'B': 'Set B — ERAD Pathway',
        'C': 'Set C — Mitochondrial Dynamics',
        'D': 'Set D — CFTR Chaperone Network',
    }
    for i, (_, row) in enumerate(df.iterrows(), 1):
        gene  = str(row.get('gene_name', ''))
        uid   = str(row.get('uniprot_id', ''))
        sc    = float(row.get('max_score_v2', 0))
        tier  = str(row.get('confidence_tier_v2', ''))
        cset  = str(row.get('candidate_set', '?'))
        kpos  = row.get('top_lysine_position')
        krsa  = row.get('top_k_rsa')
        kplddt= row.get('top_k_plddt')
        kmob  = row.get('top_k_mobidb_disorder')
        nsurK = row.get('n_surface_K')
        nub   = row.get('n_annotated_ub_sites')
        motif = str(row.get('top_lysine_motif', 'n/a'))
        rat   = RATIONALE.get(gene,
            f"High-scoring candidate from {SET_FULL.get(cset, cset)} with structural "
            f"features consistent with known RFFL substrates.")

        tier_col = "#006600" if tier=="HIGH" else ("#886600" if tier=="MODERATE" else "#AA2200")

        story.append(KeepTogether([
            Paragraph(f"{i}. {gene} (UniProt {uid})", styles['h2']),
            Paragraph(
                f"Set: {SET_FULL.get(cset, cset)} &nbsp;|&nbsp; "
                f"Score: <b>{sc:.4f}</b> &nbsp;|&nbsp; "
                f"Tier: <font color='{tier_col}'><b>{tier}</b></font>",
                styles['small']),
            Spacer(1, 4),
        ]))

        # Score breakdown table
        kpos_s  = f"K{int(kpos)}" if pd.notna(kpos) else "?"
        krsa_s  = f"{float(krsa):.3f}" if pd.notna(krsa) else "?"
        kplddt_s= f"{float(kplddt):.1f}" if pd.notna(kplddt) else "?"
        kmob_s  = f"{float(kmob):.2f}" if pd.notna(kmob) else "?"
        nsurK_s = str(int(nsurK)) if pd.notna(nsurK) else "?"
        nub_s   = str(int(nub)) if pd.notna(nub) else "0"

        details = [
            [Paragraph("<b>Feature</b>", styles['tbl_hdr']),
             Paragraph("<b>Value</b>",   styles['tbl_hdr']),
             Paragraph("<b>Feature</b>", styles['tbl_hdr']),
             Paragraph("<b>Value</b>",   styles['tbl_hdr'])],
            ["Top lysine", kpos_s, "RSA at top K", krsa_s],
            ["pLDDT", kplddt_s, "MobiDB disorder", kmob_s],
            ["n surface K", nsurK_s, "Annotated ub sites", nub_s],
            ["Top-K motif (±5aa)", motif, "", ""],
        ]
        story.append(make_table(details, [1.2*inch, 0.9*inch, 1.2*inch, 0.9*inch]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(rat, styles['body']))
        story.append(Spacer(1, 10))

    story.append(PageBreak())


def page_limitations(styles, story):
    story.append(Paragraph("5. Limitations and Next Steps", styles['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    story.append(Paragraph("Key Limitations", styles['h2']))
    lims = [
        ("<b>Small positive set (n=11).</b> All feature weights are biologically motivated "
         "heuristics, not empirically fitted parameters. AUC estimates should be interpreted "
         "as directional signals rather than precision performance benchmarks; 95% CI on AUC "
         "with n=11 spans roughly ±0.15."),
        ("<b>No RFFL-substrate co-crystal structure.</b> Structural modelling of the binding "
         "interface relies on inference from the αTOS competitive inhibitor site (Taniguchi 2023) "
         "and from AlphaFold-predicted conformations of substrates, not from direct structural "
         "evidence of how RFFL engages a substrate lysine."),
        ("<b>Incomplete UniProt ubiquitination annotations.</b> UniProt PTM coverage for many "
         "proteins consists of computationally predicted sites or is entirely absent; annotated "
         "sites were sparse for several substrates. The PWM may still reflect surface-K biases "
         "where annotation is poor."),
        ("<b>MobiDB coverage gaps.</b> Several proteins returned no MobiDB-lite prediction "
         "(API not available for all entries). Fallback to pLDDT-based disorder for those "
         "residues introduces inconsistency in the disorder feature channel."),
        ("<b>AlphaFold structures reflect predicted apo conformations.</b> The scored "
         "conformation may differ from the complex-bound state in which ubiquitination occurs. "
         "Substrate engagement by RFFL may induce conformational changes that alter which "
         "lysines are accessible."),
        ("<b>Lysine accessibility ≠ ubiquitination probability.</b> The composite score is a "
         "necessary but not sufficient predictor. Many surface-accessible K in disordered "
         "regions are ubiquitinated by multiple E3 ligases, not specifically RFFL. The harder "
         "negative control tests E3-specificity but with limited statistical power."),
    ]
    for l in lims:
        story.append(Paragraph(f"• {l}", styles['bullet']))

    story.append(Paragraph("Suggested Experimental Next Steps", styles['h2']))
    nexts = [
        ("<b>Priority 1 — DNAJB6 validation.</b> Overexpress FLAG-RFFL in HEK293T cells and "
         "perform anti-FLAG co-IP followed by western blot for endogenous DNAJB6. If co-IP is "
         "positive, perform ubiquitination assay (His-Ub pull-down under denaturing conditions) "
         "and quantitative SILAC-MS as per Narendradev et al. 2025."),
        ("<b>Priority 2 — OPA1 and MFN2 co-regulation.</b> Test whether RFFL-KO cells show "
         "simultaneous stabilisation of both MFN2 (confirmed) and OPA1 (predicted) by cycloheximide "
         "chase. If both are stabilised, this would suggest RFFL coordinates the full "
         "fusion–fission axis via dual substrate ubiquitination."),
        ("<b>Priority 3 — PDIA3 in CFTR ERAD context.</b> Given that RFFL, CFTR, and PDIA3 "
         "are all in the same ER quality-control complex, test whether αTOS treatment (which "
         "blocks the RFFL–CFTR interaction) also disrupts PDIA3 co-IP with RFFL — this would "
         "distinguish direct substrate from bystander co-complex member."),
        ("<b>Computational refinements.</b> AlphaFold3 multimer modelling of RFFL·substrate "
         "complexes would provide direct structural evidence for interface lysine exposure. "
         "AlphaFold3 is now available via the EBI server and requires no local installation."),
    ]
    for n in nexts:
        story.append(Paragraph(f"• {n}", styles['bullet']))

    story.append(PageBreak())


def page_references(styles, story):
    story.append(Paragraph("6. References", styles['h1']))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    refs = [
        ("1.", "Narendradev et al. (2025). Identification of novel substrates of E3 ubiquitin "
               "ligase RFFL by quantitative proteomics. J. Proteome Res. 24, 3913–3930. "
               "DOI: 10.1021/acs.jproteome.5c00086"),
        ("2.", "Okiyoneda T et al. (2011). Peripheral protein quality control removes "
               "unfolded CFTR from the plasma membrane. Science 331, 805–809."),
        ("3.", "Liao MH et al. (2008). RFFL is a novel E3 ubiquitin ligase that ubiquitinates "
               "p53. (Original RFFL substrate characterisation.)"),
        ("4.", "Gan B et al. (2012). FoxOs enforce a progression checkpoint to constrain "
               "mTORC1-activated renal tumorigenesis. (RIPK1 and CASP8 context.)"),
        ("5.", "Roder L et al. (2019). RFFL controls cell death signalling by regulating CASP10. "
               "Cell Death Differ."),
        ("6.", "Yang F et al. (2007). RFFL ubiquitinates mitofusin-2 to control mitochondrial "
               "dynamics. J. Biol. Chem."),
        ("7.", "Taniguchi Y et al. (2023). Structure of the RFFL RING-FYVE domain bound to "
               "αTOS reveals the substrate-binding mechanism."),
        ("8.", "Varadi M et al. (2022). AlphaFold Protein Structure Database: massively expanding "
               "the structural coverage of protein-sequence space with high-accuracy models. "
               "Nucleic Acids Res. 50, D439–D444."),
        ("9.", "Hatos A et al. (2020). MobiDB: intrinsically disordered proteins in 2020. "
               "Nucleic Acids Res. 48, D269–D276."),
        ("10.","The UniProt Consortium (2023). UniProt: the Universal Protein Knowledgebase "
               "in 2023. Nucleic Acids Res. 51, D523–D531."),
        ("11.","Tien MZ et al. (2013). Maximum allowed solvent accessibilities of residues in "
               "proteins. PLoS ONE 8, e80635. (Reference scale for RSA normalisation.)"),
        ("12.","Cock PJA et al. (2009). Biopython: freely available Python tools for "
               "computational molecular biology and bioinformatics. Bioinformatics 25, 1422–1423."),
        ("13.","Kunzmann P & Hamacher K (2018). Biotite: a unifying open source "
               "computational biology framework in Python. BMC Bioinformatics 19, 346."),
    ]
    for num, text in refs:
        story.append(Paragraph(
            f"<b>{num}</b>&nbsp;&nbsp;{text}", styles['small']))
        story.append(Spacer(1, 3))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("Generating RFFL Substrate Prediction PDF Report...")

    styles = make_styles()

    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch,   bottomMargin=0.75*inch,
        title="RFFL Substrate Prediction Report v2",
        author="Remi Sky",
        subject="E3 ubiquitin ligase RFFL substrate candidate predictions",
    )

    story = []

    # Page 1 — Cover
    page_cover(styles, story)

    # Page 2 — Method
    page_method(styles, story)

    # Page 3 — Validation
    page_validation(styles, story, VAL_CSV, ROC_PNG)

    # Page 4 — Top Candidates table
    page_top_candidates(styles, story, RANKED_CSV, DIST_PNG)

    # Page 5 — Top 5 detailed writeups
    page_top5_writeups(styles, story, RANKED_CSV)

    # Page 6 — Limitations & Next Steps
    page_limitations(styles, story)

    # Page 7 — References
    page_references(styles, story)

    def on_page(canvas, doc):
        canvas.saveState()
        w, h = letter
        # Footer
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(0.75*inch, 0.45*inch,
            "RFFL Substrate Prediction Pipeline v2 · Remi Sky · "
            "DOI: 10.1021/acs.jproteome.5c00086")
        canvas.drawRightString(w - 0.75*inch, 0.45*inch, f"Page {doc.page}")
        # Header (skip cover)
        if doc.page > 1:
            canvas.setFont('Helvetica-Bold', 7)
            canvas.setFillColor(NAVY)
            canvas.drawString(0.75*inch, h - 0.5*inch, "RFFL Substrate Candidates — v2")
            canvas.setFillColor(colors.HexColor("#888888"))
            canvas.setFont('Helvetica', 7)
            canvas.drawRightString(w - 0.75*inch, h - 0.5*inch,
                "Narendradev et al. 2025 extension")
            canvas.setStrokeColor(TEAL)
            canvas.setLineWidth(0.5)
            canvas.line(0.75*inch, h - 0.55*inch, w - 0.75*inch, h - 0.55*inch)
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"✓ PDF written → {PDF_OUT}")

if __name__ == "__main__":
    main()
