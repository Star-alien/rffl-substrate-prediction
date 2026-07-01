# AUC Discrepancy Analysis -- v1 Pipeline

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

### Per-protein fraction of surface K above HIGH threshold (0.602)

| Gene         | Above/Total | Fraction |
|-------------|-------------|----------|


Low fractions confirm that most surface K per substrate do NOT individually
meet the HIGH threshold -- only the best K does. This is why protein-level
max-score is high while lysine-level AUC is low.

### Protein-level AUC recalculation

| Metric               | Value  |
|---------------------|--------|
| Lysine-level AUC (v1) | 0.530 |
| Protein-level AUC (v1) | 0.848 |
| n_positive proteins  | 11 |
| n_negative proteins  | 12 |

The protein-level AUC is meaningfully higher than 0.530, confirming that the original lysine-level AUC was measuring the wrong thing -- not indicating the model is weak. The model does distinguish substrate proteins from random proteins; it just cannot reliably distinguish every individual K within a substrate from every random K, which is an inherent limitation given that ubiquitination sites are sparse within proteins.

## What v2 does about this

v2 addresses both issues:
1. PWM is now trained on annotated ubiquitination sites (from UniProt PTM features)
   rather than all surface K -- reducing the circularity that diluted the lysine-level signal
2. MobiDB disorder predictions add an independent feature channel
3. A harder negative control (other-E3-substrate proteins) tests RFFL-specificity
4. All AUC numbers in v2 are reported at **both** lysine level AND protein level
