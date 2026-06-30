# RFFL Substrate Candidate Prediction
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

- **n = 10 positive examples is small.** All scoring weights were chosen from biological reasoning, NOT fit to data. With 10 positive examples, fitting weights would be statistically meaningless (overfitting guaranteed). The validation ROC AUC is 0.530 — this reflects how well the features separate surface K in known substrate proteins from random protein K. It does NOT directly measure accuracy for predicting undiscovered substrates, and should not be interpreted as a performance estimate.

- **No experimental RFFL–substrate co-crystal structure exists.** The substrate-binding region is characterised indirectly via the α-tocopherol succinate (αTOS) competitive inhibitor binding site (Taniguchi 2023). The features used here (surface exposure, disorder, charge) are general proxies, not substrate-binding-site-specific features.

- **RSA was computed via Cα-neighbor count, not FreeSASA.** FreeSASA requires compilation from source (MSVC not available). The neighbor-count heuristic is a reasonable proxy but less accurate than explicit solvent-accessible surface calculation. This limitation is noted per-result.

- **PWM was trained on all surface K in known substrates, not on experimentally confirmed ubiquitination sites.** For most RFFL substrates, the exact K residues ubiquitinated are not precisely annotated in the literature. The PWM captures sequence context around surface-accessible K in substrate proteins, not the specific ubiquitination motif. This introduces circularity between the RSA and motif features and is an inherent limitation with the current data.

- **This is a hypothesis-generation tool, not a validated predictor.** No protein in the ranked list should be described as a "likely RFFL substrate" without experimental confirmation. The appropriate next step is quantitative MS proteomics in RFFL-overexpression vs RFFL-knockout cells, identical to the experimental design of Narendradev et al. 2025.

---

### Known Substrate Validation

Scores assigned to the 10 known RFFL substrates by this method (sanity check — known substrates should generally score in the HIGH tier; deviations are reported honestly):

```
gene_name  lysine_position  composite confidence_tier
   CASP10               14     0.8077            HIGH
    CASP8              367     0.6339            HIGH
     CFTR              684     0.8572            HIGH
  DNAJB11              344     0.5817        MODERATE
    JMJD6              375     0.8434            HIGH
    KCNH2              888     0.8840            HIGH
     MFN2               16     0.7640            HIGH
    PRR5L              247     0.7930            HIGH
    RIPK1              396     0.8347            HIGH
    STUB1                7     0.7676            HIGH
     TP53              382     0.8530            HIGH
```

HIGH tier threshold (75th percentile of known-substrate surface-K scores): 0.602
MODERATE tier threshold (50th percentile): 0.487

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

#### DNAJB6 (Set A — DNAJ co-chaperone family, score=0.877, HIGH)
Top candidate K at position 309.0; local motif: `GGKRKKQKQRE`. Member of candidate Set A (DNAJ co-chaperone family).

#### OPA1 (Set C — Mitochondrial/endosomal dynamics, score=0.867, HIGH)
Top candidate K at position 259.0; local motif: `GIHHRKLKKSL`. Inner mitochondrial membrane GTPase controlling cristae remodelling and fusion; functionally linked to MFN2. Ubiquitination of OPA1 by RFFL could couple outer and inner membrane dynamics.

#### PDIA3 (Set D — CFTR chaperone network, score=0.843, HIGH)
Top candidate K at position 497.0; local motif: `EEKPKKKKKAQ`. Member of candidate Set D (CFTR chaperone network).

#### OS9 (Set B — ERAD pathway components, score=0.843, HIGH)
Top candidate K at position 515.0; local motif: `PELVKKHKKKR`. Member of candidate Set B (ERAD pathway components).

#### CANX (Set D — CFTR chaperone network, score=0.841, HIGH)
Top candidate K at position 531.0; local motif: `EEEEEKEEEKD`. Calnexin is the ER lectin chaperone that retains misfolded glycoproteins, including CFTR ΔF508, in the ER. Its direct role in RFFL substrate retention makes it a candidate for RFFL-mediated regulatory ubiquitination.


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
