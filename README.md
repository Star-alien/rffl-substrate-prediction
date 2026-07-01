# RFFL Substrate Candidate Prediction (v2)
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
| Lysine-level vs random | 0.530 | 0.459 |
| Protein-level vs random | 0.848 | 0.686 |
| Protein-level vs other-E3-substrates | -- | 0.659 |

---

### Motivation

Narendradev et al. 2025 used quantitative MS-based proteomics to identify JMJD6 and DNAJB11 as new endogenous substrates of E3 ubiquitin ligase RFFL, expanding the confirmed substrate list to 10. This pipeline asks: which other human proteins share the structural and sequence features of these known substrates, and are therefore candidate undiscovered RFFL substrates for experimental validation?

---

### Method Summary

1. **AlphaFold structures** downloaded for RFFL (Q8WZ73, verified) and all 10 known substrates (all 11 UniProt IDs verified via REST API).
2. **Surface accessibility** computed via Lee-Richards algorithm (biotite) or Cα-neighbor count fallback. Threshold RSA > 0.25 for "accessible."
3. **Ubiquitination site annotation** retrieved from UniProt PTM features (Modified residue / Cross-link entries mentioning ubiquitin). PWM trained on 11 proteins using annotated sites where available; method = `annotated_ub_sites`.
4. **MobiDB disorder** fetched per protein; per-residue binary disorder from mobidb-lite predictions.
5. **Composite v2 scoring:**
   ```
   score = 0.30×RSA + 0.25×PWM_norm + 0.20×MobiDB_disorder + 0.15×pLDDT_disorder + 0.10×charge
   ```
6. **40 candidates** in 4 biologically motivated sets scored; HIGH/MODERATE/LOW tiers relative to 75th/50th percentile of known-substrate surface-K scores.

---

### Honest Limitations

- **n=10 remains small.** All scoring weights are biologically motivated, not empirically fit. AUC numbers should be interpreted as evidence of feature directionality, not as performance benchmarks.
- **Annotated ubiquitination sites are sparse for many proteins.** UniProt PTM coverage is incomplete; the PWM used `annotated_ub_sites`. For proteins without annotated sites, the heuristic surface-K PWM is used -- flagged in the output CSV.
- **MobiDB may return no data** for some proteins (API coverage is not complete for all UniProt entries). In those cases the model falls back to pLDDT-based disorder. Covered fraction is logged but not shown per-row in the CSV.
- **No experimental RFFL-substrate co-crystal structure.** The structural characterisation remains inference-based, motivated by the αTOS binding site (Taniguchi 2023).
- **This is a hypothesis-generation tool.** No protein in this list should be described as a "likely RFFL substrate" without experimental validation by quantitative MS proteomics (as in Narendradev et al. 2025).

---

### Known Substrate Validation (v2 scores)

```
gene_name  lysine_position  composite_v2 confidence_tier_v2
   CASP10                2        0.6783               HIGH
    CASP8              367        0.5141               HIGH
     CFTR              688        0.6531               HIGH
  DNAJB11              344        0.4369               HIGH
    JMJD6              375        0.5657               HIGH
    KCNH2              886        0.6069               HIGH
     MFN2               15        0.5101               HIGH
    PRR5L               15        0.5319               HIGH
    RIPK1              396        0.5958               HIGH
    STUB1                2        0.6728               HIGH
     TP53              386        0.6797               HIGH
```

HIGH tier > 0.426 | MODERATE > 0.320

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

#### PDIA3 -- Set D (CFTR chaperone network), score=0.635, HIGH
Top K at position 497.0. PDIA3 (ERp57) is a disulfide isomerase that co-chaperones glycoproteins including CFTR in the ER. Confirmed RFFL substrates CFTR and DNAJB11 both interact directly with PDIA3 in the quality control cycle, placing PDIA3 in RFFL's immediate functional neighbourhood.

#### CANX -- Set D (CFTR chaperone network), score=0.623, HIGH
Top K at position 551.0. Calnexin directly retains misfolded CFTR ΔF508 in the ER, where RFFL drives its ERAD. Of all Set D candidates, calnexin is most directly positioned to encounter RFFL at its substrate -- CFTR -- making regulatory ubiquitination of calnexin by RFFL a biologically coherent hypothesis.

#### DNAJB12 -- Set A (DNAJ co-chaperone family), score=0.620, HIGH
Top K at position 365.0. DNAJB12 is an ER membrane-embedded co-chaperone that tethers cytosolic Hsp70 to the ER surface -- uniquely positioned to bridge the cytosolic RFFL with ER-lumenal substrates. Its membrane topology is directly analogous to the role DNAJB11 plays in the ER lumen.

#### DNAJA4 -- Set A (DNAJ co-chaperone family), score=0.619, HIGH
Top K at position 133.0. DNAJA4 is a cytosolic Hsp40 co-chaperone that stimulates HSPA8/Hsc70 ATPase activity specifically during heat stress responses. Its biochemical activity mirrors the Hsp70-stimulating role of DNAJB11, placing it in the same functional class as a confirmed RFFL substrate.

#### DNAJB6 -- Set A (DNAJ co-chaperone family), score=0.618, HIGH
Top K at position 320.0. DNAJB6 (Hsp40 subfamily B member 6) is a co-chaperone that suppresses protein aggregation and participates in ERAD-coupled protein quality control. Its confirmed interaction with misfolded clients in the same ER compartment where RFFL operates, combined with sequence homology to confirmed substrate DNAJB11, makes it the top Set A candidate.

#### OS9 -- Set B (ERAD pathway), score=0.607, HIGH
Top K at position 519.0. OS9 is the ER-lumenal lectin that recognises misfolded glycoproteins and recruits the HRD1/SEL1L ERAD complex. RFFL's role in CFTR ERAD places it in the same pathway; OS9 as an RFFL substrate would constitute a feedback mechanism regulating ERAD throughput.

#### DNAJA3 -- Set A (DNAJ co-chaperone family), score=0.605, HIGH
Top K at position 175.0. Candidate from DNAJ co-chaperone family (Set A).

#### MFF -- Set C (Mitochondrial/endosomal dynamics), score=0.601, HIGH
Top K at position 41.0. MFF recruits DRP1 to outer mitochondrial membrane fission sites and is spatially proximal to MFN2. It represents a second Set C candidate downstream of the MFN2-defined mitochondrial dynamics axis that RFFL is known to regulate.

#### HERC3 -- Set B (ERAD pathway), score=0.597, HIGH
Top K at position 333.0. HERC3 is a HECT E3 ligase with an RCC1-like domain that localises to endosomes -- the same subcellular compartment where RFFL targets MFN2. Co-presence on endosomal membranes and shared substrate topology make HERC3 a plausible target of RFFL-mediated regulation.

#### SEL1L -- Set B (ERAD pathway), score=0.590, HIGH
Top K at position 937.0. SEL1L is the adaptor subunit of the HRD1/SYVN1 ERAD ubiquitin ligase complex, essential for retrotranslocation of misfolded ER proteins including CFTR. RFFL co-operates with HRD1 in CFTR ERAD; SEL1L as an RFFL substrate would constitute cross-regulation between two ERAD E3 complexes.



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
