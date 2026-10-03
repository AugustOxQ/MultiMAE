> Supporting file for docs/reports/auto/v1/2026-10-03_lever_review.md. Verification of B1 and B4 as written; its corrections override those bibliographies, and the report uses corrected values.

# V1: Source verification of B1 (masked objectives) and B4 (novelty and polysemy)

ARS deep-research Phase 2, lit-review mode, source verification agent. Written 2026-10-03.

**Method.**
- Metadata: arXiv API for every arXiv id (title, authors, dates, comments, journal_ref) and the Semantic Scholar batch API (venue, DBLP, DOI). Venues were then checked on ACL Anthology pages (SemEval overview, UAlberta, Kritharoula, PolCLIP, FCLL, MACCO, ConLIP), PMLR v202 and v235 listings (MERU, Llip, SyCoCa) and Crossref (the two Knowledge-Based Systems DOIs).
- Numbers: read from the arXiv LaTeX source of each paper (`arxiv.org/e-print/<id>`). Table numbers were counted from caption order in the expanded source, and arXiv HTML was used to settle table numbering where the source was ambiguous (ProLIP, COSMOS, ECCV Caption). The ConLIP PDF was parsed locally. No WebFetch summary was trusted for a table cell.
- Coverage: I read the priority items myself (MaskCLIP, MAMO, IRRA, BiLMa, VFE-TPS, SyCoCa, Verma et al., MaskVLM, MACCO, ProLIP, LongProLIP, PCME++, Llip, COSMOS, Kritharoula, UAlberta, ECCV Caption Table 4, MAP, METER, ConLIP). Two forks of this agent read the secondary items from LaTeX source under the same rules: A-CLIP, DetailCLIP, FLAVA, VL-BEiT, Bitton, M3AE, TIPS, SigLIP 2 and TULIP (fork A), and MERU, HyCoCLIP, HierarCaps, CapPa, DreamLIP, FLAIR, AAHR, Bhattacharya, Cekinmez, MDC, LaViSA, ECCV Caption, CxC, PCME, INQUIRE, HKUST, Hyper3-CLIP and HiMo-CLIP (fork B).
- Safety: no retrieved page contained text that tried to direct the agent.

## 1. Overall assessment

**Sources checked: 52.** B1 has 18 included sources and B4 has 31 more. Three of B4's excluded sources were also checked because B4 makes claims about them.

| Existence outcome | Count | Sources |
|---|---|---|
| S2_VERIFIED | 47 | Every arXiv source except the five below. All titles, first authors and years match. |
| VERIFIED (other primary metadata) | 5 | SemEval-2023 Task 1 overview, PolCLIP and FCLL (ACL Anthology); 2505.09997 and 2602.01193 (arXiv API) |
| PLAUSIBLE | 0 | |
| UNVERIFIABLE | 0 sources, 1 venue claim | 2608.09011: the paper exists, but B4's "ACM MM 2026" has no arXiv comment and no S2 venue |
| FABRICATED | 0 | |

**Venue claims.**
- 6 corrections: VFE-TPS (B1), and MACCO, LongProLIP, AAHR, HiMo-CLIP and Cekinmez (B4). MACCO is correct in B1 but wrong in B4.
- 1 ambiguity resolved: DetailCLIP is an ICLR 2025 SSI-FM workshop paper. Its "Published as a conference paper at ICLR 2025" header comes from the ICLR template.
- 2 unconfirmed "oral" tags: HierarCaps (ECCV 2024) and CapPa (NeurIPS 2023). The venues themselves are verified.

**Claims checked: 96 rows** (Section 3).

| Outcome | Count |
|---|---|
| CONFIRMED | 78 |
| CORRECTED | 13 (row 80 is counted here; it also holds a NOT FOUND sub-claim) |
| LOCATOR WRONG | 2 |
| NOT FOUND | 2 |
| Resolved (an earlier "not retrieved" note) | 1 |

**Load-bearing problems**, in order of weight:
1. **The absence claims do not survive as worded.**
   - "No source fine-tunes CLIP with MLM or MAE and reports COCO/Flickr retrieval" is contradicted by TULIP, which is already in B1, and by METER (CVPR 2022), which was missed.
   - "No work combines masked reconstruction with a polysemy-aware evaluation" is contradicted in a weak form by ECCV Caption's own Table 4 (MLM-pretrained ViLT and VinVL on mAP@R and PMRP) and by Kwon et al. (ACL 2023), who run FLAVA on VWSD.
   - Both survive only in a narrowed form (Section 4).
2. **B4 says ProLIP never isolates the masked-inclusion term. That is false.** Tables C.3 and C.4 isolate it, on HierarCaps recall and on the DataComp retrieval average.
3. **B4's MACCO entry has three factual errors:** 55 epochs (the paper says 5), parser roles swapped, and the venue given as preprint.
4. **B1's VFE-TPS entry:** the venue is wrong, and the stated reason the method worked ("0.5 to 0.6 masking rather than 0.15") is not in the paper.
5. **Verma et al.'s three seeds are fine-tuning seeds only.** Each masking rate had one pretraining run, so the ± s.e.m. understates run-to-run variance for the "raise MLM to 60%" evidence.

## 2. Source quality matrix

Tier: P = peer-reviewed main venue or journal; W = workshop; Pre = preprint. Currency is the year of first arXiv posting, with the publication year in parentheses where it differs.

### B1 sources

| Source | Outcome | Venue status | Tier | Currency | Note |
|---|---|---|---|---|---|
| MaskCLIP 2208.12262 | S2_VERIFIED | CVPR 2023 matches | P | 2022 (2023) | All Table 1/5/6/9 numbers confirmed |
| SyCoCa 2401.02137 | S2_VERIFIED | ICML 2024, PMLR 235:34038-34052, matches | P | 2024 | All Table 1/5/6/7 numbers confirmed |
| DetailCLIP 2409.06809 | S2_VERIFIED | ICLR 2025 SSI-FM workshop (mlanthology iclrw) | W | 2024 (2025) | The PDF header is the ICLR template; workshop status is correct |
| SigLIP 2 2502.14786 | S2_VERIFIED | arXiv only, matches | Pre | 2025 | Recipe and Table 1 confirmed |
| MAMO 2210.04183 | S2_VERIFIED | SIGIR 2023 (DOI 10.1145/3539618.3591721) matches | P | 2022 (2023) | Tables 6/9/10 confirmed. The "4.1M images, 9.7M pairs" figure is rounded; the Table 1 sums are 4.02M images and 9.78M captions |
| TIPS 2410.16512 | S2_VERIFIED | ICLR 2025 (DBLP conf/iclr) matches | P | 2024 (2025) | Confirmed |
| TULIP 2503.15485 | S2_VERIFIED | arXiv only, matches | Pre | 2025 | Loss weights appear nowhere in the paper |
| Verma et al. 2212.05195 | S2_VERIFIED | arXiv only, matches | Pre | 2022 | Seeds cover fine-tuning only |
| Bitton et al. 2109.02040 | S2_VERIFIED | Findings of EMNLP 2021 (DOI 2021.findings-emnlp.259) matches | P | 2021 | Confirmed |
| M3AE 2205.14204 | S2_VERIFIED | arXiv only, matches | Pre | 2022 | The Fig. 5 values are printed bar labels |
| A-CLIP 2212.08653 | S2_VERIFIED | ICCV 2023 matches (pp. 2771-2781 in CVF; IEEE pagination 2759-2769) | P | 2022 (2023) | Wording issue on patch scoring |
| MaskVLM 2208.02131 | S2_VERIFIED | ICLR 2023 matches | P | 2022 (2023) | Confirmed. Zero-shot rows use the pretrained model, not a COCO fine-tune |
| FLAVA 2112.04482 | S2_VERIFIED | CVPR 2022 matches | P | 2021 (2022) | Confirmed |
| VL-BEiT 2206.01127 | S2_VERIFIED | arXiv only, matches | Pre | 2022 | Confirmed |
| IRRA 2303.12501 | S2_VERIFIED | CVPR 2023 matches | P | 2023 | Confirmed |
| BiLMa 2309.04675 | S2_VERIFIED | ICCVW 2023 (DBLP conf/iccvw) matches | W | 2023 | Confirmed |
| VFE-TPS 2412.20646 | S2_VERIFIED | **CORRECTED**: Knowledge-Based Systems 309:112893 (2025), DOI 10.1016/j.knosys.2024.112893, resolved via Crossref. B1 says "preprint, no venue found" | P (journal) | 2024 (2025) | The default masking ratio is never stated |
| MACCO 2606.13288 | S2_VERIFIED | ACL 2026 Long, pp. 32284-32308, DOI 10.18653/v1/2026.acl-long.1490 (ACL Anthology). B1 matches | P | 2026 | B1 is correct and B4 is not |

### B4 sources (not already in B1)

| Source | Outcome | Venue status | Tier | Currency | Note |
|---|---|---|---|---|---|
| ProLIP 2410.18857 | S2_VERIFIED | ICLR 2025 matches | P | 2024 (2025) | Masked-inclusion ablation exists (Tables C.3, C.4) |
| LongProLIP 2503.08048 | S2_VERIFIED | **CORRECTED**: tiny paper at the ICLR 2025 workshop "Quantify Uncertainty and Hallucination in Foundation Models" (arXiv comment). B4 had "UNVERIFIED" | W | 2025 | Urban-1k baseline mixed up |
| MERU 2304.09172 | S2_VERIFIED | ICML 2023, PMLR 202:7694-7731, matches | P | 2023 | Confirmed |
| HyCoCLIP 2410.06912 | S2_VERIFIED | ICLR 2025 oral (arXiv comment, S2) matches | P | 2024 (2025) | The rows are the B/16 rows |
| HierarCaps 2407.08521 | S2_VERIFIED | ECCV 2024 matches. "Oral" not confirmed | P | 2024 | Confirmed |
| CapPa 2306.07915 | S2_VERIFIED | NeurIPS 2023 matches. "Oral" not confirmed | P | 2023 | Confirmed |
| Llip 2405.00740 | S2_VERIFIED | ICML 2024, PMLR 235:26070-26084, matches | P | 2024 | The MetaCLIP baseline numbers are copied from the MetaCLIP paper |
| DreamLIP 2403.17007 | S2_VERIFIED | ECCV 2024 (DBLP conf/eccv) matches | P | 2024 | Row label wrong |
| FLAIR 2412.03561 | S2_VERIFIED | CVPR 2025 (IEEE DOI) matches | P | 2024 (2025) | Only v1 exists, so the "version" claim is wrong |
| COSMOS 2412.01814 | S2_VERIFIED | CVPR 2025, pp. 14690-14700, matches | P | 2024 (2025) | Supp. Table 17 tests a masked-text view |
| SemEval-2023 Task 1 overview | VERIFIED (ACL Anthology 2023.semeval-1.308, pp. 2227-2234) | matches | W (shared task) | 2023 | Abstract claims confirmed |
| UAlberta 2306.14067 | S2_VERIFIED | SemEval-2023, pp. 2043-2051 | W | 2023 | Confirmed |
| FCLL (TAM of SCNU) | VERIFIED (ACL Anthology 2023.semeval-1.70, pp. 506-511) | B4 had "anthology id UNVERIFIED", now resolved | W | 2023 | 72.56 / 82.22 confirmed |
| Kritharoula 2310.14025 | S2_VERIFIED | EMNLP 2023, pp. 13053-13077, matches | P | 2023 | Confirmed |
| PolCLIP | VERIFIED (ACL Anthology 2024.acl-long.575, pp. 10676-10690) | matches | P | 2024 | Abstract only |
| Bhattacharya 2602.06799 | S2_VERIFIED | arXiv only ("pending submission"), matches | Pre | 2026 | Table IV Hit@1 cannot be a fraction of 463 items |
| White & Cotterell 2211.13095 | S2_VERIFIED | arXiv only, matches | Pre | 2022 | Abstract claim confirmed |
| Cekinmez 2608.00410 | S2_VERIFIED (no S2 venue) | **CORRECTED**: oral, Sci-FM Workshop @ COLM 2026 (arXiv comment). B4 had preprint | W | 2026 | Two judges; no limitations section |
| ECCV Caption 2204.03359 | S2_VERIFIED | ECCV 2022 matches | P | 2022 | Query counts confirmed. Table 4 evaluates MLM-pretrained models (Section 4) |
| CxC 2004.15020 | S2_VERIFIED | EACL 2021, pp. 2855-2870, matches | P | 2020 (2021) | 267,095 counts rated pairs, not judgments |
| PCME 2101.05068 | S2_VERIFIED | CVPR 2021 matches | P | 2021 | Confirmed |
| PVSE / MRW 1906.04402 | S2_VERIFIED | CVPR 2019 matches | P | 2019 | 50K pairs confirmed |
| INQUIRE 2411.02537 | S2_VERIFIED | NeurIPS 2024 D&B matches | P | 2024 | Confirmed |
| LaViSA 2606.19552 | S2_VERIFIED | arXiv only, matches | Pre | 2026 | Confirmed |
| AAHR 2507.09256 | S2_VERIFIED | **CORRECTED**: Knowledge-Based Systems 316:113355 (2025), DOI 10.1016/j.knosys.2025.113355. B4 had preprint | P (journal) | 2025 | Table 3 confirmed |
| PCME++ 2305.18171 | S2_VERIFIED | ICLR 2024 matches | P | 2023 (2024) | Table 3 confirmed |
| Masked Diffusion Captioning 2510.26799 | S2_VERIFIED | Findings of EMNLP 2025 matches | P | 2025 | 84.6 is the heuristic variant |
| HiMo-CLIP 2511.06653 | S2_VERIFIED | **CORRECTED**: AAAI 2026, oral (arXiv comment and S2). B4 had preprint | P | 2025 (2026) | Setting confirmed |
| MAP 2210.05335 | S2_VERIFIED | CVPR 2023 matches | P | 2022 (2023) | Source grep finds no ECCV Caption, PMRP, CxC or CUB evaluation |
| Hyper3-CLIP 2608.29313 | S2_VERIFIED (no S2 venue) | ECCV 2026 Beyond Euclidean workshop (arXiv comment) | W | 2026 | **CORRECTED**: it is evaluated on more than HierarCaps |
| HKUST 2311.18273 | S2_VERIFIED | arXiv (SemEval participant) | Pre | 2023 | 60.48 / 73.88 confirmed |
| Excluded: 2505.09997 | VERIFIED (arXiv) | Withdrawal confirmed by the v3 comment | n/a | 2025 | Exclusion correct |
| Excluded: 2608.09011 | S2_VERIFIED (existence) | **UNVERIFIABLE**: "ACM MM 2026" appears in no arXiv comment and no S2 venue | Pre | 2026 | Excluded anyway |
| Excluded: 2602.01193 | VERIFIED (arXiv) | Accepted at IEEE TIC 2026 (arXiv comment) | W/conf | 2026 | Its 95.77 / 92.00 claim was left unverified, correctly |

## 3. Claim verification table

"Confirmed" means I or a fork saw the value in the paper's source at the stated locator.

| # | Claim | File | Outcome | Correct value and locator |
|---|---|---|---|---|
| 1 | MaskCLIP Table 6f MLM weight β: 1 → 36.5; 51.7/32.1; 0.1 → 44.3; 69.2/45.9; 0.05 → 44.5; 70.1/45.6; 0.01 → 43.2; 70.6/45.6 | B1 | CONFIRMED | Table 6f (tab:mlm_loss_weight). The Table 6 caption does not name the retrieval dataset. The values match Flickr30K in Table 5, so "Flickr" holds by value match |
| 2 | MaskCLIP CLIP baseline 37.6; 52.9/32.8 in Table 1 | B1 | CONFIRMED | Table 1 (tab:target): CLIP IN-1K 0-shot 37.6, Flickr I2T 52.9, T2I 32.8. The same Flickr values are in Table 5 |
| 3 | MaskCLIP Table 6e λ sweep; Table 6a without MLM 65.0/41.6 and without distillation 65.4/40.5; Table 6d depth 0 → 65.2/44.1 and depth 4 → 70.1/45.6 | B1 | CONFIRMED | Tables 6a, 6d, 6e |
| 4 | MaskCLIP Table 9 target chain 57.3/41.1 → 62.3/41.4 → 65.0/41.6 → 70.1/45.6; Table 5 COCO 27.5/17.7 → 41.4/25.5 | B1 | CONFIRMED | Table 9 (component ablation of distillation), Table 5 |
| 5 | MaskCLIP masks 75% image / 20% text; text-only MLM via a small decoder; λ = β = 0.05; β=1 explanation is only a suspicion | B1 | CONFIRMED | Sec 4.1 (masks), Sec 3.4 (text-only BERT-style MLM), Table 6 gray rows, Sec 4.4 ("We suspect") |
| 6 | SyCoCa Table 5 rows (COCO mTR/mIR): ITC 13.5/14.8; CoCa 16.1/16.0; +AM 17.5/18.3; MIM-RM 15.8/16.3; MIM-AM 15.2/16.2; TG-MIM-RM 14.4/15.2; SyCoCa 18.3/18.4; Flickr 37.5/28.7, 35.8/26.4, 42.6/32.4 | B1 | CONFIRMED | Table 5 (tab:object ablation); the text also says TG-MIM with random masking "slightly decreases" |
| 7 | SyCoCa Table 6 ranges (COCO mTR 17.5-18.6, mIR 17.3-18.8); Table 7 λ_TM sweep; Table 1 COCO 16.3/15.3 → 18.7/17.2; λ_IC = 2, λ_TM = 1, r_h = r_l = 50%, CC12M, 20 ep, batch 2048 | B1 | CONFIRMED | Tables 6, 7, 1; Sec 4.1 implementation details; r_h masks top-scoring patches (method section) |
| 8 | DetailCLIP Table 2 (25/50 ep) and App. Table 5 α3 sweep (42.9/43.9/43.2/42.6/43.3) | B1 | CONFIRMED (fork A) | Table 2 caption says the baselines come "from A-CLIP" |
| 9 | SigLIP 2 masked prediction (50%, weights 1 and 0.25, size factors 0.25/0.5/1.0/0.5, starts at 80%); Sec 2.3 quote; Table 1 B/16 at 256 px: COCO 47.4→53.2 and 65.1→69.7, Flickr 78.3→81.7 and 91.1→94.4, XM3600 22.5→40.7 | B1, B4 | CONFIRMED (fork A) | Sec 2.2-2.5, Table 1. Sec 3.1 credits the B-size gains to ACID distillation |
| 10 | MAMO Table 6 rows (six rows, Flickr TR/IR and COCO TR/IR) | B1 | CONFIRMED | Table 6 (table-ptt). Retrieval is fine-tuned with ITM reranking of the top-k ITC candidates (k = 256 COCO, 128 Flickr; Sec 4.1) |
| 11 | MAMO Table 9 targets (pixels 75.8/59.1, visual tokens 76.5/58.1, momentum features 76.6/59.2 on COCO); "visual tokens" are DALL-E tokens; Table 10 spread ≤ 0.8 | B1 | CONFIRMED | Tables 9, 10; Sec 4.5 names the DALL-E codebook |
| 12 | MAMO pretraining data "4.1M images, 9.7M pairs" | B1 | CORRECTED (minor) | Table 1 sums: 4.02M images, 9.78M captions |
| 13 | MAMO: "image-conditioned MLM carried almost all of the gain over ITC+ITM on COCO (+3.6 TR, +2.8 IR)" | B1 | CORRECTED (wording) | The deltas are right. But the full model gains +4.7 TR and +4.0 IR, so MLM alone accounts for 77% and 70% of the gain. "Most", not "almost all" |
| 14 | MAMO "MRM alone may collapse into trivial solutions" | B1 | CONFIRMED | Sec 4.5, "Importance of Pre-training Tasks" |
| 15 | TIPS Table 1C (79.1/62.9 → 81.5/67.0 → 82.6/67.6; seg 64.4→70.3→75.9; NYUv2 0.589→0.511; KNN 79.1→79.0; dual captions 88.7/77.1); Table 11B/C/E; α = 1, β = 2; 75% random masking | B1, B4 | CONFIRMED (fork A) | Table 1, Table 11, App. A.2. Table 11B ran on full TIPS (both captions), not on Table 1C's noisy-caption setting |
| 16 | TULIP initialization from SigLIP, 500M samples, batch 49,152, lr 1e-5; Table 1 B/16 COCO 47.2/64.5 → 54.2/70.1 and Flickr 77.9/89.6 → 81.8/93.9 | B1 | CONFIRMED (fork A) | Sec 4.1, Table 1. B4's "two parses inconsistent, UNVERIFIED" can be lifted. The SigLIP baseline is the published checkpoint, not a rerun |
| 17 | TULIP Table 5 (So/14 5.9/81.1 → 17.4/82.3 → 18.2/82.1 → 20.3/81.9; B/14 14.4/81.3 → 15.8/80.8) | B1, B4 | CONFIRMED (fork A) | Table 5 measures an LLaVA-style MLLM, not retrieval. The paper prints "(+1.2)" for the So/14 reconstruction step, but its own values give +0.8 |
| 18 | TULIP loss weights λ not found | B1 | CONFIRMED absent | Only symbols in Eqs. 8-9; no values anywhere, appendix included |
| 19 | Verma Table 4: uniform Flickr image +3.73±0.03 and text +3.20±0.66, COCO +3.00±0.10 and +4.11±0.72; whole word +3.75/+1.57 and +3.16/+3.63; noun-verb +1.11/−1.80 and +0.62/+0.01 | B1 | CONFIRMED | Appendix Table 4. The COCO row order in the source is R@1, R@10, R@5; R@1 values read correctly |
| 20 | Verma: "each configuration was pretrained, then fine-tuned with 3 seeds" | B1 | CONFIRMED, with caveat | Sec 3 (setup): "we pretrain ViLT and finetune it on each downstream task with three different seeds". One pretraining run per configuration, so the s.e.m. reflects fine-tuning noise only |
| 21 | Verma: best rate 60% for Flickr and 75% for COCO | B1 | CONFIRMED | Sec 4.1 |
| 22 | Verma: "15% was the worst rate tested" | B1 | NOT FOUND | Only the deltas from 15% to 60% are tabulated. The full sweep is in plots (Fig. 2), and the text says performance rises "until a threshold, after which it drops", so 75% could fall below 15% on some task. Reader inference, not a paper statement |
| 23 | Bitton: Sec 2.2 statistics (36%; 45-50%), Table 1 (89/78, 98/96, 76/56), Sec 4.1 gains, 7 epochs | B1 | CONFIRMED (fork A) | As stated. VQA/GQA use 2 seeds and NLVR2 uses 3 |
| 24 | M3AE Fig. 5 text-mask linear probe 45.9/61.0/64.1/63.9; text loss weight 0.5 | B1 | CONFIRMED (fork A) | Fig. 5 bar labels in figs/text_ratio.pdf; App. Table 2 (text weight 0.5; text and image masks both 0.75) |
| 25 | A-CLIP Table 2a (5 rows), 2b (high 28.5; 42.6/29.0; 23.5/13.6), Table 3 (+MAE 42.7; 60.0/38.8; 34.1/21.2); 30% extra training time | B1 | CONFIRMED (fork A) | Tables 2a, 2b, 3; Sec 4.2 |
| 26 | A-CLIP: "An EMA encoder scores how relevant each patch is to the caption" | B1 | CORRECTED (wording) | The score is the EMA image encoder's [CLS]-to-patch attention, used as a proxy. The caption is not used, because the authors note that text-based selection leaks information. B4's "[CLS] attention" wording is right |
| 27 | A-CLIP (B4 [6]): Table 1 rows; random 1×50% "sits below the Table 1 CLIP baseline, though the tables may differ in setup" | B4 | CONFIRMED; hedge unnecessary | Table 2a has its own full-image CLIP row (37.6; 51.4/32.6; 27.9/17.6), identical to Table 1 |
| 28 | MaskVLM Table 5 (ITC 65.10/80.10 \| 55.08/68.40; ITC+ITM 79.96/92.30 \| 69.50/82.40; +MLM 80.34/92.00 \| 70.74/84.40; +MIM 80.12/91.50 \| 69.26/82.90; +both 81.26/94.10 \| 71.18/85.60; MLM+MIM only 76.08/90.30) | B1 | CONFIRMED | Table 5 (tab:ablation_loss). Zero-shot uses the pretrained model, not the COCO fine-tune (Sec 4.1) |
| 29 | MaskVLM Table 7 ratios (0.5/0.3 81.32/93.30; 0.6/0.3 81.26/94.10; 0.7/0.3 81.82/93.60; 0.6/0.15 80.30/92.50) | B1 | CONFIRMED | Appendix Table 7 |
| 30 | MaskVLM Table 6 one vs both (COCO FT IR/TR 60.1/76.3 vs 59.5/76.0; ALBEF 56.8/73.1) | B1 | CONFIRMED | Appendix Table 6 |
| 31 | MaskVLM text 30% / image 60% in 32×32 blocks | B1 | CONFIRMED | Sec 4.1 implementation details; Sec 3 |
| 32 | FLAVA Table 4 (FLAVA_C → FLAVA_MM and full FLAVA), Sec 4 macro averages +2.86 / +9 | B1 | CONFIRMED (fork A) | Table 4 columns 3, 4, 6; Sec 4 |
| 33 | VL-BEiT Table 4 (91.2/75.8, 92.2/77.4, 92.2/77.9); 50% text / 40% image masking | B1 | CONFIRMED (fork A) | Table 4; Sec 3 |
| 34 | IRRA Table 4 (baseline 68.19/56.74/54.05; +IRR 71.23/60.96/57.90; +SDM 70.42/60.45/57.20; +SDM+IRR 72.81/63.27/59.25; +SDM+ID 70.52/61.03/58.65; IRRA 73.38/63.46/60.20) | B1 | CONFIRMED | Table 4; text: "+3.04%, 4.22% and 3.85%". Baseline is "CLIP-ViT-B/16 fine-tuned with InfoNCE" (Sec 4.2) |
| 35 | IRRA Table 1 mAP 61.12 → 66.13; Table 5 (73.28 / 73.21 / 73.38) | B1 | CONFIRMED | Tables 1, 5 |
| 36 | IRRA setting: 15% masking (80/10/10), one cross-attention layer + 4-layer transformer, lr 1e-5 / 5e-5, 60 epochs, one RTX 3090; phrase-level limitation at Sec 4.3 | B1 | CONFIRMED | Sec 3.2; Sec 4 implementation details; Sec 4.3 "Qualitative Results" |
| 37 | BiLMa Table 2 (neither 73.01/63.09/59.50; MLM 73.16/63.60/59.05; SemMIM 73.55/63.08/59.40; both 74.03/63.83/61.20) | B1 | CONFIRMED | Table 2 (tab:component) |
| 38 | BiLMa App. Table 5 (w/o MIM 73.38/66.13; pixel 72.86/65.61; patch 73.07/66.01; feature 73.52/66.20; SemMIM 74.03/66.57); m = 0.15, β = 1; grid search only for SemMIM | B1 | CONFIRMED | App. A.5, Table 5; Sec 4.1 grid search |
| 39 | VFE-TPS Table 6 (70.61/63.12; +TG-MIM 72.16/63.99; +IS-GVFC 71.61/64.24; both 72.47/64.26) | B1 | CONFIRMED | Table 6 (tab:ablation_impact_VFE) |
| 40 | VFE-TPS Table 5: fully fine-tuned CLIP with CMPM at 66.78 | B1 | CONFIRMED | Table 5 and Sec 4.5.1 ("full parameter fine-tuning on CLIP using the CMPM loss") |
| 41 | VFE-TPS: mAP peaks at masking 0.5-0.6 for all three methods (Fig. 6, Sec 4.5.3) | B1 | CONFIRMED | Sec 4.5.3. The authors also say "slight changes in mAP occur with different masking ratios", which B1 omits |
| 42 | VFE-TPS relevance: TG-MIM helped while "it used 0.5 to 0.6 masking rather than 0.15" | B1 | NOT FOUND | The default ratio behind Tables 1-6 is never stated (Sec 3.2: "a certain proportion"; implementation details are silent). The peak in Fig. 6 does not show which ratio the main runs used |
| 43 | VFE-TPS architecture: one MCA layer, conv + PixelShuffle, L1; 60 epochs, lr 1e-5, batch 100, RTX 3090 | B1 | CONFIRMED | Alg. 1, Eq. (L1), Sec 4.3 |
| 44 | MACCO 5 epochs, batch 256, lr 5e-7 (CLIP) / 1e-3 (predictors), AdamW wd 0.2, one A100, about 110k COCO pairs | B1 | CONFIRMED | Sec 4 "Training Setup" |
| 45 | MACCO Table 9 (CLIP 64.6; CLIP-FT 66.1; +MLM 68.2; +MIM 68.5; +MLM+MIM 68.2; MCA+MIR 68.7; MLM+aux 72.8; MIM+aux 70.5; all 73.4) | B1 | CONFIRMED | Table 9 (tab:ablation); rows in the "two auxiliary losses" block |
| 46 | MACCO Table 12 (random without aux 67.5; random with aux 71.2; concept without aux 68.2) | B1 | CONFIRMED | Table 12 (tab:simplified_ablation) |
| 47 | MACCO random masking is 75% image / 15% text | B1 | CONFIRMED | Table 10 caption (tab:strategy_ablation) |
| 48 | MACCO Table 15 (ARO-Rel CLIP-FT 64.4±0.40 vs MACCO 73.5±0.60, 4 seeds); Table 3 (ZS 59.5/57.9/58.0; LP 80.1/80.0/79.7) | B1, B4 | CONFIRMED | Tables 15, 3 |
| 49 | MACCO Table 1 (all CLIP / CLIP-FT / MACCO triples in B4) | B4 | CONFIRMED | Table 1 (tab:main_results) |
| 50 | MACCO no retrieval recall reported | B1, B4 | CONFIRMED | No R@K, Recall@ or Flickr retrieval in the expanded source |
| 51 | MACCO "55 epochs" | B4 | CORRECTED | 5 epochs (Sec 4 "Training Setup") |
| 52 | MACCO: text concepts "located with GroundingDINO", image concepts by "scene-graph parser" | B4 | CORRECTED | Reversed. Text uses a scene-graph parser (Wu et al. 2019); images use GroundingDINO (Sec 3, App. "Compositional Concept Extraction") |
| 53 | ProLIP: 75% of tokens masked for 12.5% of samples (Sec 3.3) | B4 | LOCATOR WRONG (partly) | 12.5% is in Sec 3.3. The 75% is in Sec 4.1 ("masking out their 75% information") and App. B.1 |
| 54 | ProLIP α2 = 0.001 (App. B.1) | B4 | CONFIRMED | App. B.1: α1 = 1e-7, α2 = 0.001 |
| 55 | ProLIP: "no table isolates the masked-inclusion term on retrieval or zero-shot accuracy; Table C.5 ablates only ε, c" | B4 | CORRECTED | Table C.4 (large-scale ablation, ViT-B/16, DataComp, 1.28B seen) isolates it. Neither loss: IN 67.0, retrieval 53.6, avg 56.6. Masked inclusion only: 67.4, 53.2, 56.7. v⊂t only: 67.3, 53.2, 57.0. Both: 67.6, 53.4, 57.3. Table C.3 (96M seen): HierarCaps recall 44.8 with no inclusion loss → 47.9 with masked inclusion only → 54.8 with both. Alone, the masked term moved the DataComp retrieval average by −0.4 |
| 56 | ProLIP Fig. 8 (> 70% inclusion with masked versions), Fig. 7 (general captions more uncertain) | B4 | CONFIRMED | Figs 7, 8 |
| 57 | LongProLIP Table C.1 ECCV mAP@R I2T/T2I: 28.9/39.2; 30.5/41.0; 29.6/37.2; 29.2/40.1 | B4 | CONFIRMED | Table C.1 (appendix:tab:overview). These are the 12.8B-seen ProLIP B/16 |
| 58 | LongProLIP DataComp 38-task average 63.3 → 60.5 / 58.7 / 63.3 | B4 | CONFIRMED | Table 2 |
| 59 | LongProLIP "Urban-1k average 55.5 → 91.3 (S128M)" | B4 | CORRECTED | Two backbones mixed. Table 2 (12.8B backbone): 65.4 → 91.3 (S128M). The 55.5 is the 1.28B backbone in Table 1, which reaches 85.8 with ShareGPT4V only |
| 60 | LongProLIP context 64 → 256 tokens; ShareGPT4V 1.2M; HYPE + DFN medium | B4 | CONFIRMED | Abstract, Sec 2 |
| 61 | MERU Table 1 R@5 values | B4 | CONFIRMED (fork B) | The spread of differences is −2.1 to +2.5, not +1.3 (L/16 Flickr I→T 47.8 → 50.3) |
| 62 | HyCoCLIP Table 2 "smallest-backbone rows" 71.4/72.3/72.0 etc. | B4 | CORRECTED (fork B) | These are the ViT-B/16 rows. B/16 TIE 3.60/3.63/3.17, LCA 2.21/2.22/2.05, J 0.79/0.78/0.81. The ViT-S/16 COCO text R@5 is 69.3/68.8/69.5 |
| 63 | HierarCaps Table 1 values and fine-tuning settings | B4 | CONFIRMED (fork B) | COCO is the val set, T→I. CLIP-L precision drops (0.16 → 0.15) |
| 64 | CapPa 75% parallel prediction; Table 6 ARO rows; CapPa below CLIP* on COCO via LiT | B4 | CONFIRMED (fork B) | Sec 3; Table 6; Table 4 (B/16 COCO t2i 37.3/38.6 vs 38.9/40.1; i2t 53.9/55.1 vs 55.1/57.0) |
| 65 | Llip Table 2 MetaCLIP → Llip (B/16 COCO 59.4→63.4 and 41.4→45.6, Flickr 85.9→90.1 and 70.5→75.1; G/14 COCO 66.7→72.7 and 49.6→54.2) | B4 | CONFIRMED | Table 2 (tab:zs-retrieval). The caption says the MetaCLIP numbers are "reported from" the MetaCLIP paper, while SigLIP* was reproduced (B/16 COCO 59.7/42.0) |
| 66 | DreamLIP: "long captions direct 32.7/23.0" | B4 | CORRECTED (fork B) | 32.7/23.0 is the short-captions-only row. Long captions direct gives 30.2/21.4; sub-caption sampling 35.7/25.6; sampling plus short captions 40.8/29.4; full 42.8/30.4 |
| 67 | FLAIR Table 1 SigLIP → FLAIR 46.6→51.2, 62.6→67.3; "a later version shows 53.3/68.0" | B4 | CORRECTED (fork B) | Only v1 exists. 46.6 → 51.2 is the YFCC15M-recap block; 53.3/68.0 is FLAIR-30M in the SOTA block of the same Table 1. Tables 5 and 11 confirmed |
| 68 | COSMOS Table 5 (CLIP 15.0/10.7 → … → 53.1/40.1) and Table 1 CC3M CLIP 40.2/27.2 | B4 | CONFIRMED | Table 5 (tab:ablation1); Table 1. B4 skips the EMA row (17.5/12.7) |
| 69 | COSMOS: "No masking of patches or tokens anywhere in the method (checked)" | B4 | CONFIRMED for the final method, with a missed ablation | Supp. Table 17 (tab:ablation_textcropping, CC3M, B/16) tests a DeCLIP-style masked-text local view (15% [mask]): COCO I2T/T2I 46.0/32.8 vs COSMOS sentence cropping 52.6/38.9 |
| 70 | SemEval-2023 overview: 96 submissions, 40 beat zero-shot CLIP, "generative models and data augmentation" | B4 | CONFIRMED | ACL Anthology abstract |
| 71 | VWSD data: train 12,869 silver; test 968 = 463 En + 305 It + 200 Fa | B4 | CONFIRMED | UAlberta Sec 3; Kritharoula Table 8 (12,869 / 463) |
| 72 | UAlberta Table 3: baseline 60.5/22.6/28.5; Tr 61.1/59.3/43.0; Tr+Def 69.1/63.3/40.0; Def from InstructGPT | B4 | CONFIRMED | Table 3 (table:main_results); Sec 3. UAlberta never labels its "Baseline" as zero-shot CLIP; the label rests on HKUST's 60.48 and the overview |
| 73 | FCLL average H@1 72.56, MRR 82.22 | B4 | CONFIRMED | ACL Anthology 2023.semeval-1.70 abstract (pp. 506-511) |
| 74 | Kritharoula Table 3 CLIP with penalty 63.28/76.27 → GPT-3 "meaning_of" 68.07/80.08 | B4 | CONFIRMED | Table 3, "with penalty" block. The overall best in Table 3 is ALIGN + GPT-3 meaning_of at 74.95/84.09. CLIP without penalty is 59.18/72.94 |
| 75 | Kritharoula Table 6 best LTR 79.35/87.23 (ALIGN + GiT-L greedy + all prompts); under-7B LLMs "only marginal" | B4 | CONFIRMED | Table 6; Sec 5 text |
| 76 | PolCLIP +2.22% HR@1 VWSD and +2.53% F1 textual WSD | B4 | CONFIRMED | ACL Anthology abstract |
| 77 | Bhattacharya Tables II-V, VII, X values | B4 | CONFIRMED (fork B) | Table IV Hit@1 0.6250 cannot be k/463 (289/463 = 0.6242), so it was computed on a different item set or rounded oddly |
| 78 | White & Cotterell: the CLIP text encoder encodes polysemous words as a superposition | B4 | CONFIRMED | arXiv abstract |
| 79 | Cekinmez entropy 0.10 / 0.25 / 0.47 "(Fig 2, Table 1)" | B4 | LOCATOR WRONG (fork B) | 0.10 and 0.25 are in Sec 3 and the abstract. Human 0.473 is in Table 2. Table 1 is the cross-lingual table |
| 80 | Cekinmez "LLM judge (GPT-5.4)" and limitations "[author-acknowledged, limitations]" | B4 | CORRECTED / NOT FOUND (fork B) | Model outputs were judged by GPT-5.4 and Gemini-3.5-Flash. The paper has no limitations section. 17 vs 18 image models is inconsistent within the paper (abstract vs appendix) |
| 81 | ECCV Caption ×3.6 / ×8.5; 1,261 image and 1,332 caption queries | B4 | CONFIRMED (fork B) | Abstract; Sec 1 and Table 3 caption. B4's UNVERIFIED can be lifted |
| 82 | Brief: PMRP pseudo-positives about 60% correct vs humans; PMRP vs mAP@R Kendall τ 0.20 | brief | CONFIRMED (fork B) | Table 2: plausible-match precision at ζ=0 is 65.3 (I2T) and 56.6 (T2I); Table 5 (v5): τ 0.20 |
| 83 | CxC "267,095 human similarity judgments" | B4 | CORRECTED (fork B) | 267,095 rated pairs, from 1,335,475 individual judgments (Sec 1, 3) |
| 84 | PCME CUB Caption stats and Table 2 R-P 22.4/22.6 → 26.3/26.8 | B4 | CONFIRMED (fork B) | Sec 4.1; Table 2 |
| 85 | MRW: 50K video-sentence pairs | B4 | CONFIRMED | PVSE abstract |
| 86 | INQUIRE: 5M images, 250 queries, 33,000 matches, < 50 mAP@50 | B4 | CONFIRMED (fork B) | Abstract |
| 87 | LaViSA: 700 / 1,503, DALL-E 3, Gemini 3.1 Pro 88.9 per-trial | B4 | CONFIRMED (fork B) | Sec 3; Table 2 |
| 88 | AAHR Table 3 ECCV rows (ESA and AAHR); frozen CLIP B/32 | B4 | CONFIRMED (fork B) | Table 3; Sec 4.3. The same table gives zero-shot CLIP B/32 at 22.4 / 31.1 (I2T / T2I mAP@R) |
| 89 | PCME++ Table 3 rows (no VIB/PP/MSDA 38.9/48.6/82.2/56.7/75.2/54.9/535.9; MSDA only 39.0/48.6/82.1/56.4/74.9/54.6/535.5; all 40.1/49.7/83.1/56.8/75.4/55.1/537.0); 25% of images mixed, Beta(2,2); CLIP init, 25 epochs | B4 | CONFIRMED | Table 3 (tab:loss_abl); Sec 3 and Sec 4.1. The rows are PCME++ variants, not InfoNCE |
| 90 | MDC Table 2 ARO-Relation CLIP 53.6, AR 82.7, MDC 84.6; t ∈ [0.5, 1] | B4 | CONFIRMED (fork B) | 84.6 is MDC (Heuristic); MDC (Monte Carlo) scores 85.1 |
| 91 | MAP: no ECCV Caption, CxC, PMRP or CUB evaluation | B4 | CONFIRMED | None of these terms appears in the source; retrieval is COCO/Flickr R@K only |
| 92 | Hyper3-CLIP evaluated "HierarCaps only" | B4 | CORRECTED (fork B) | Zero-shot COCO/Flickr R@5/R@10, ImageNet hierarchy metrics, 16-dataset classification and multi-label classification. A HierarCaps-style AP/AUROC appears only in the ablation |
| 93 | HKUST official CLIP baseline 60.48 / 73.88 | B4 | CONFIRMED (fork B) | Results table, row "CLIP-raw" |
| 94 | HiMo-CLIP setting (B/16 and L/14, ShareGPT4V 1.2M, in-batch PCA, monotonicity loss, HiMo-Docci) | B4 | CONFIRMED (fork B) | Sec 3.3, Sec 4, App. C.5 |
| 95 | B1 MACCO / B4: "Appendix D ablation tables not retrievable" | B4 | Resolved | Tables 9-12 are in the arXiv LaTeX source (B1 read them) |
| 96 | B1 search note: "A-CLIP SSL heads SimCLR, BYOL" vs B4 "SimCLR/SimSiam" | B1, B4 | CONFIRMED (both partly) | The online-to-EMA task uses BYOL; the masked-view contrastive task uses SimCLR or SimSiam (approach.tex) |

Side note for the synthesis phase, not a B1/B4 claim: the zero-shot CLIP B/32 reference in ECCV Caption Table 4 is mAP@R 26.75, PMRP 55.32, RSUM 471.9. Its RSUM sums COCO 1K recalls, so it is not comparable to a 5K rsum.

## 4. Absence-claim falsification log

### Claim A: "No work combines masked reconstruction with a polysemy-aware evaluation" (ECCV Caption mAP@R / R-P, PMRP, CxC, CUB, VWSD)

This is B4's thesis (mechanism table, Section 2.6), and B1 says the same about its own included sources.

| # | Query (my own) | Closest hits | Falsifies? |
|---|---|---|---|
| A1 | `"ECCV Caption" mAP@R masked language modeling OR "masked image modeling" image-text retrieval ablation` (extended) | SIMLA, MaskVLM, MAMO, RILS, LexLIP, IRRA, "Verb understanding … guided masking" | No: none reports ECCV Caption |
| A2 | `PMRP plausible match R-precision masked autoencoder OR masked modeling vision-language retrieval evaluation 2024 2025` (extended) | MaskVLM, VLMAE, "Uncertainty-based cross-modal retrieval with probabilistic representations" (2204.09268) | No |
| A3 | `"Crisscrossed Captions" OR "CxC" evaluation vision-language pretraining masked language model ViLT VinVL ALBEF results` (extended) | **ECCV Caption v5** (then read in full), ALIGN, ERNIE-ViL 2.0 | **Yes, weakly** (see the finding below) |
| A4 | `visual word sense disambiguation "masked language" OR "MLM" vision-language model fine-tuning SemEval-2023 Task 1 system ViLT OR BLIP OR ALBEF` (extended) | Quantum VWSD 2512.24687, HKUST, Kritharoula, **Kwon et al. 2305.01788** | **Yes, weakly** |
| A5 | `SemEval-2023 Task 1 visual word sense disambiguation FLAVA zero-shot image-text matching system paper` | teamPN (2023.semeval-1.63), Augmenters 2307.05564, **Kwon et al. ACL 2023** (source read: "we adopted two SOTA zero-shot ITM models, CLIP and FLAVA") | **Yes, weakly** |
| A6 | `masked image modeling OR masked language modeling improves "one-to-many" OR "many-to-many" image-text correspondence probabilistic embedding ECCV Caption PCME 2025 2026` (extended) | **ProM3E 2511.02946** (probabilistic masked multimodal embedding, ecology; masked modality reconstruction in embedding space), PCME | No: ecology retrieval, no polysemy-aware image-text benchmark. Closest mechanism-level near miss |
| A7 | `"ECCV Caption" "mAP@R" 2025 OR 2026 arXiv fine-tuning CLIP auxiliary objective reconstruction decoder` (extended) | 2601.21426 (classification), VITRIX-CLIPIN, ResCLIP | No |
| A8 | `polysemous OR ambiguous word image retrieval benchmark evaluation "masked" vision-language model FLAVA OR ViLT OR BLIP sense disambiguation 2024 2025` (extended) | PolCLIP, LaViSA, INQUIRE, COLA, ARO, the VWSD mini review | No new hit |
| A9 | Full-text reads: ECCV Caption Table 4, MAP (grep), ProLIP Tables C.3/C.4, COSMOS Supp. Table 17 | see findings | partial |

**Findings.**
- **ECCV Caption (Chun et al. 2022), Table 4 "Re-evaluating VL models"** evaluates models pretrained with masked language modeling on ECCV mAP@R, R-P, CxC R@1 and PMRP:
  - ViLT fine-tuned (MLM with whole-word masking, plus ITM): mAP@R 34.58, PMRP 57.63.
  - ViLT zero-shot: 26.84 / 57.38.
  - VinVL fine-tuned (MLM, plus contrastive): 40.81 / 54.72.
  - BLIP: 40.52 / 57.17.
  - These are ITM cross-encoders, and the masked objective is never ablated.
- **Kwon et al. (ACL 2023, 2023.acl-long.88)** evaluates FLAVA zero-shot on SemEval-2023 VWSD. FLAVA is pretrained with MLM, MIM and MMM, and it scores matches with its multimodal encoder.
- **ProLIP Tables C.3 and C.4** ablate the masked-input inclusion term on HierarCaps recall and the DataComp retrieval average. It is not a reconstruction objective, and ECCV/PMRP are not used.
- **COSMOS Supp. Table 17** compares a masked-text view (15% [mask]) with sentence cropping on COCO and Flickr retrieval only.

**Verdict: the claim does not survive as worded.** Masked-objective-pretrained models have been scored on ECCV Caption, PMRP and VWSD. A narrowed claim survives every search above: *no work found isolates a masked reconstruction or masked prediction objective, by ablation within one model and recipe, and measures its effect on ECCV Caption mAP@R / R-P, PMRP, CxC or VWSD; no dual-encoder CLIP fine-tune with masked reconstruction reports these metrics.* B4's mechanism table should add the ECCV Caption Table 4 VLP rows and the Kwon et al. FLAVA row as "masked-pretrained, evaluated, not ablated". Its ProLIP row should say "ablated on HierarCaps and DataComp retrieval (C.3, C.4), not on ECCV/PMRP".

### Claim B: "No source fine-tunes CLIP with MLM or MAE and reports COCO or Flickr retrieval"

This is B1's coverage skew statement.

| # | Query (my own) | Closest hits | Falsifies? |
|---|---|---|---|
| B1 | `METER empirical study end-to-end vision-and-language transformers CLIP-ViT masked image modeling does not help MLM ITM Flickr30k retrieval` | **METER (Dou et al., CVPR 2022, 2111.02387)**; source read | **Yes** (see the findings below) |
| B2 | `fine-tune CLIP image-text retrieval MSCOCO Flickr30K auxiliary "masked language modeling" cross-modal decoder removed at inference dual encoder 2024 2025` (extended) | SyCoCa, MACCO, DCLIP 2505.21549 (cross-modal distillation, no MLM), CyCLIP, CLIPS 2411.16828 (partial captions plus a captioner, from scratch), LLM2CLIP | No |
| B3 | `"interlaced" cross-modal decoder masked language modeling CLIP fine-grained alignment image-text retrieval` | **ConLIP (Luo et al., Findings of EMNLP 2022, pp. 130-140)**, TokenFlow, SIMLA, FineLIP | No for CLIP (ConLIP starts from ImageNet ViT-B/16 + BERT), but it is a highly relevant dual-encoder source that both files missed |
| B4 | `CLIP fine-tuning with masked image modeling auxiliary loss improves zero-shot retrieval COCO Flickr "masked autoencoder" pretrained CLIP continued training 2024 2025 arXiv` (extended) | MVP (MIM fine-tunes CLIP; classification only), Weers et al., DetailCLIP, "Multi-Modal Contrastive Masked Autoencoders" (CVPR 2025), DCLIP | No |
| B5 | `"masked" "CLIP" fine-tuned on COCO image-text retrieval "masked language modeling" auxiliary objective dual-encoder rsum Flickr30K "ViT-B/32" ablation` (extended) | 2412.16148 (word-frequency text masking, from scratch), ViLT, MaskCLIP, TokenFlow, MoTIS | No |
| B6 | `continue training pretrained CLIP OR SigLIP with masked reconstruction loss image decoder text decoder zero-shot COCO Flickr retrieval results ablation 2025 2026` (extended) | **TULIP** (already B1 entry 2.3: "we leverage a masked autoencoder (MAE) style model", Sec 3), SigLIP 2, FLIP | **Yes** |

**Findings.**
- **TULIP** continues training from SigLIP checkpoints (a CLIP-family model) on 500M samples with MAE-style image reconstruction and a text decoder. It reports zero-shot COCO and Flickr retrieval (Table 1, B/16: COCO T2I 47.2 → 54.2). The reconstruction term is not ablated on retrieval. TULIP is already in B1, so B1 contradicts its own statement.
- **METER** (CVPR 2022, pp. 18145-18155):
  - Model: a CLIP-initialized image encoder (CLIP-ViT-224/32 and RoBERTa are its default encoders, Sec 4.1) with a 6-layer co-attention fusion, pretrained on 4M images with MLM (15%) and ITM.
  - Table 7, Flickr zero-shot IR/TR R@1:
    - MLM+ITM: 66.08/78.10.
    - Adding MIM with in-batch negatives: 62.12/76.90.
    - Adding MIM with discrete codes: 59.80/76.30.
    - ITM alone: 53.74/71.00.
  - The authors attribute the MIM drop to conflicts between objectives.
  - Retrieval uses the ITM head, and the text tower is RoBERTa, not CLIP's.
  - The source is directly relevant to B1 sub-themes 4 and 5 and was missed.
- **ConLIP** (dual encoder, ImageNet ViT-B/16 + BERT, 5.3M pairs, Table 1, fine-tuned COCO T2I/I2T R@1):
  - Contrastive only: 46.8/62.7.
  - Adding vanilla MLM+MIM: 46.3/62.3.
  - Conditioning both reconstructions on the [CLS] instance embedding (ConMLM/ConMIM): 47.5/63.4.
  - Zero-shot COCO: 27.0/39.8, 27.0/39.4 and 28.0/40.3 for the same three settings.
  - Single runs. It is not CLIP, so it does not falsify claim B, but it is the closest dual-encoder analogue.

**Verdict: the claim does not survive as worded.** A narrowed claim survives:
- *No source found fine-tunes both towers of a pretrained CLIP dual encoder on COCO or Flickr with an MLM or MAE auxiliary (decoder discarded at test) and reports dual-encoder COCO/Flickr retrieval with the masked term ablated.*
- The nearest cases are:
  - IRRA, BiLMa and VFE-TPS: the same design in text-to-person retrieval.
  - MACCO: the same backbone and data, but no retrieval reported.
  - METER: a CLIP image tower, but ITM reranking.
  - TULIP: continued SigLIP pretraining at 500M scale, no ablation.
  - ConLIP: a dual encoder, but not CLIP.

## 5. Flagged sources

| Source / claim | Issue | Severity | Recommendation |
|---|---|---|---|
| B1 coverage statement "No source fine-tunes CLIP with MLM or MAE and reports COCO or Flickr retrieval" | Falsified by TULIP (in B1) and METER (missed). Paper-facing novelty claim | High | Caveat: restate in the narrowed form of Section 4. Add METER Table 7 to sub-themes 4 and 5, and add ConLIP Table 1 as the dual-encoder datapoint |
| B4 mechanism table / brief, "no masked reconstruction with polysemy-aware evaluation" | Weakly falsified by ECCV Caption Table 4 (ViLT, VinVL) and Kwon et al. ACL 2023 (FLAVA on VWSD) | High | Caveat: restate as "no ablation isolates a masked objective on ECCV/PMRP/CxC/VWSD", and add those rows |
| B4 [1] ProLIP: "no table isolates the masked-inclusion term" | False. Tables C.3 and C.4 isolate it (alone: DataComp retrieval −0.4; HierarCaps recall +3.1) | High | Include with correction. This is the only ablation of a masked-view term in the probabilistic line |
| B4 [14] MACCO | 55 epochs (paper says 5); parsers swapped; tier "preprint" (it is ACL 2026 Long) | Medium | Include. Correct all three; B1's entry is right |
| B1 5.3 VFE-TPS | Venue wrong (KBS 309:112893, 2025). The relevance line says TG-MIM "used 0.5 to 0.6 masking rather than 0.15", but the paper never states its default ratio and calls ratio effects "slight" | Medium | Include. Fix the venue and delete or caveat the ratio inference before it drives a masking-ratio choice |
| B1 3.1 Verma et al. | The 3 seeds are fine-tuning seeds on one pretraining run per configuration, so the ± s.e.m. understates variance. "15% was the worst rate" is not in the text or tables. Retrieval is ITM-head reranked | Medium | Caveat when citing it as the multi-seed evidence for raising the MLM rate |
| B4 [13] COSMOS | Missed Supp. Table 17: a masked-text self-distillation view scored 46.0/32.8 COCO R@1 vs 52.6/38.9 for sentence crops (CC3M) | Medium | Include the datapoint in the mechanism table. It is the closest evidence on masked text views inside a training-only cross-attention design |
| B4 [2] LongProLIP | Urban-1k baseline mixes two backbones (65.4 → 91.3 is correct). Venue is an ICLR 2025 workshop tiny paper | Low | Include with correction |
| B1 2.1 MAMO | "Almost all of the gain" overstates it (77% and 70%); data counts rounded | Low | Include; soften the wording |
| B1 3.4 A-CLIP | Patch relevance comes from EMA image [CLS] attention, not the caption | Low | Include; fix the wording |
| B4 [4] HyCoCLIP | The rows are B/16, not the smallest backbone; the UNVERIFIED tag can be resolved | Low | Include with the B/16 label |
| B4 [11] DreamLIP | 32.7/23.0 is the short-captions row; long captions direct gives 30.2/21.4 | Low | Include with correction |
| B4 [12] FLAIR | No later arXiv version exists. 46.6 → 51.2 is YFCC15M-recap, and 53.3/68.0 is the 30M row | Low | Include with correction |
| B4 [10] Llip | The MetaCLIP baseline is copied from the MetaCLIP paper; only SigLIP was rerun | Low | Caveat |
| B4 Cekinmez | Venue is a COLM 2026 workshop. Two judges, not one. No limitations section, so the "[author-acknowledged]" tag is unsupported. Human entropy is in Table 2 | Low | Include with corrections |
| B4 AAHR, HiMo-CLIP | Both are peer-reviewed (KBS 2025; AAAI 2026), not preprints | Low | Fix the tier |
| B4 CxC | 267,095 counts rated pairs (1,335,475 judgments) | Low | Fix the wording |
| B4 Hyper3-CLIP | Evaluated on more than HierarCaps (COCO/Flickr R@K, ImageNet hierarchy, classification) | Low | Fix the scope |
| B4 TULIP (secondary) | The retrieval table can be read: B1's Table 1 values are confirmed | Low | Lift UNVERIFIED |
| B4 [15] UAlberta | The "zero-shot CLIP" label for the 60.5 baseline comes from the overview and HKUST, not UAlberta | Low | Caveat |
| B4 Bhattacharya | Table IV Hit@1 0.6250 is not achievable on 463 items | Low | Caveat |
| 2608.09011 (excluded) | "ACM MM 2026" is unverifiable | Low | Already excluded; drop the venue |
| B1 1.3 DetailCLIP | Workshop status confirmed; the conference header is the template | Low | Keep the workshop tier; no ambiguity remains |
| TULIP (paper itself) | Prints "(+1.2)" where its own values give +0.8 (Table 5) | Low | Cite the computed values |

**Missed sources** to hand to synthesis (read and verified in this pass):
- METER, CVPR 2022 (Table 7).
- ConLIP, Findings of EMNLP 2022 (Table 1).
- Kwon et al., ACL 2023 (FLAVA on VWSD).
- ECCV Caption Table 4 (MLM-pretrained VLP models on mAP@R and PMRP).

**Cleanup.** The verify_tmp folder (arXiv sources, PDFs, a local pypdf install) is deleted. Nothing over 1 GB was left behind.
