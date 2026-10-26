> Supporting file for docs/reports/literature/2026-10-06_masked_decoder_readings_review.md. Verification of N as written; its corrections override N, and the report uses corrected values. Its summary counts (section 1) differ from its claim-row table (section 3); the report uses the rows (26 confirmed, 6 corrected, 1 not found). Row 17's locator 'Table 1' for Cekinmez et al. is wrong: Table 1 is the cross-lingual table; 0.47 and 0.73 are in Fig 2 and Table 2 (0.473, 0.730).
> The final review (tests/20261003_ml_improve/review_2026-10-06_masked_decoder_readings_review.md) superseded V3 problem 1 (rows 18 to 20, Panagopoulou et al.) and row 24 (UniDisc); the report uses the final review's values.

# V3: Source verification of N (novelty sweep for H-b)

Verification agent, ARS deep-research Phase 2, lit-review mode. Written 2026-10-06. Same method as V2; corrections here override N downstream.

## Method

- Metadata: arXiv API (14 ids, https), web search for venues the arXiv record does not give (Lan et al., UniDisc, AmbiBench).
- Numbers: raw arXiv HTML downloaded by me to `dl_V2/nh/<id>.html` (10 papers: 2609.13254, 2606.08031, 2606.31407, 2410.02773, 2608.00410, 2405.19423, 2302.01328, 2503.20853, 2306.05240, 2507.21794), converted to text with my own script (`scratchpad/hb_review/h2t_v2.py`) and read with grep/sed. MixGen re-read from the LaTeX source from V2. I did not use `dl_N/`.
- Absence claim: 3 extended web searches plus N's own query list.
- Not re-read in full: Yao et al. 2508.04945 (N6), Wang and Nalisnick 2505.11804, AMER 2511.02770, CLIP Under the Microscope, FOCUS. Their identity and venue were checked from arXiv records only.

## 1. Overall assessment

| Item | Count |
|---|---|
| Sources checked (N1 to N10, 6 secondary entries) | 16 |
| VERIFIED (arXiv record plus venue statement) | 14 |
| PLAUSIBLE | 1 (UniDisc: exists, no venue found; AmbiBench dataset exists, see below) |
| UNVERIFIABLE | 1 (AmbiBench paper: OpenReview behind a challenge page; title, authors and decision cannot be confirmed) |
| FABRICATED | 0 |
| Claim rows checked | 33 |
| CONFIRMED | 21 |
| CORRECTED | 9 |
| NOT FOUND | 1 (the CLIP "opposite preferences to generative models" statement) |
| NOT RE-CHECKED | 2 |

All numbers named in the priority list reproduce. No fabrication. The load-bearing problems, in order of weight:

1. **N1 (Panagopoulou) misattributes "almost opposite preferences".** The paper says that about BLIP-2 OPT versus the Flan-T5 variants of BLIP-2 (same training data), not CLIP versus generative models. It also does not say CLIP is "the most sensitive to brightness, tint": brightness and tint had "minimal effects" on every model; CLIP varied most under rotation and prompt changes. What the paper does say about CLIP: the three CLIP variants had identical probability distributions with high variance across images of one category, and the abstract-level result is that most models prefer one reading. The S3 line "CLIP vs likelihood-scored generative VLMs differ in which reading and in sensitivity" is therefore weaker than written.
2. **N2 (Takatsuki) omits its own counter-case.** The "exclusive reports" conclusion holds for duck-rabbit and Visual Anagrams, but on the Raven-Bear figure-ground stimulus three of five LLaVA models put 43 to 51% of beam mass on continuations naming both readings at the original orientation (Appendix B.4.1). Also, the 10% vs at least 98% contrast is LLaVA-1.5-7B, beam search with 8 beams, continuations classified by an LLM judge (GPT-5-nano), not next-token probabilities.
3. **N5 (Lan et al.) venue is wrong in N: it is AAAI 2025**, the same paper G2 lists as "[8] Lan et al. 2025". N says "preprint 2024, no venue stated". arXiv v1 (2410.02773) is dated 2024-09-17 and carries no venue; the AAAI page is ojs.aaai.org/index.php/AAAI/article/view/32468. Use AAAI 2025 everywhere.
4. **N8 (UniDisc) headline overstates.** The 0.64 vs 0.17 gap is joint retrieval only (DataComp validation, one correct pair among 16, random distractors). On the same dataset UniDisc ties the AR baseline in single-direction retrieval: text retrieval 0.85 vs 0.85, image retrieval 0.95 vs 0.96. No contrastive baseline (confirmed). Venue: none found.
5. **N4 (VSE) improvement is quoted against SE, not against the strongest baseline**, and "sampled-answer entropy under-reads ambiguity" is the authors' hypothesis, tested on incorrectly answered samples of the VILP adversarial split with a LogitLens visual-entropy proxy for "confidence". No human ambiguity labels (as N says).
6. **N3 (Agate) tier**: workshop paper (ICML 2026 Workshop on Philosophy of Machine Learning, arXiv comment), not a bare preprint. The 50/69 figure is SAE feature pools on CLIP layer-22 patch activations, thresholded at the median activation on opposite-aspect controls; per-group counts are uneven (duck_rabbit 12/12 but face_vase 7/16, schroeder_stairs 1/4).
7. Smaller: N10 MixGen reconciliation is fine (different settings, section 3); IC3 "up to 84.46%" text claim is not reproducible from Table 6 and Table 6's R@1 and R@5 columns are identical in the arXiv HTML, which suggests a table error; Cekinmez et al. is a workshop paper (COLM 2026 Sci-FM, oral), and its same table shows models predict human-like diversity (about 0.73) when asked, which N omits.

## 2. Per-source metadata table

| # | Source | arXiv | Authors (arXiv) | Date | Venue verified | Outcome |
|---|---|---|---|---|---|---|
| N1 | Evaluating VLMs on Bistable Images | 2405.19423 | Panagopoulou, Melkin, Callison-Burch | 2024-05 | CMCL 2024 workshop (S2 string, N cites ACL Anthology 2024.cmcl-1.2) | VERIFIED |
| N2 | (How) Do MLLMs Report Bistable Images Like Humans? | 2609.13254 | Takatsuki, Doi, Watahiki, Seth, Yanaka | 2026-09-06 | EMNLP 2026 Main, author-stated in the arXiv comment and the paper's thanks note; no proceedings page checked | VERIFIED (author-stated) |
| N3 | Vision-Language Asymmetry in Bistable Image Captioning | 2606.08031 | Agate | 2026-06-06 | ICML 2026 Workshop on Philosophy of ML (arXiv comment) | VERIFIED |
| N4 | Visual Semantic Entropy | 2606.31407 | Huy, Nguyen, Chowdhury, Yadav, To, Liao, Verjans, Phan | 2026-06-30 | ECCV 2026, author-stated (arXiv comment "Accepted at ECCV2026") | VERIFIED (author-stated) |
| N5 | Mind the Uncertainty in Human Disagreement | 2410.02773 | Lan, Frassinelli, Plank | 2024-09-17 (v1) | AAAI 2025 (ojs article 32468; mlanthology lan2025aaai) | VERIFIED; N's venue wrong |
| N6 | Visual activity recognition, verb sense clustering | 2508.04945 | Yao, Jarvis, Jiang | 2025-08 | Findings of EACL 2026 (arXiv comment) | VERIFIED |
| N7 | IC3 | 2302.01328 | Chan, Myers, Vijayanarasimhan, Ross, Canny | 2023-02 | EMNLP 2023 (arXiv comment "To Appear") | VERIFIED |
| N8 | UniDisc | 2503.20853 | Swerdlow, Prabhudesai, Gandhi, Pathak, Fragkiadaki | 2025-03 | none found (project page only) | VERIFIED as arXiv preprint |
| N9 | Pezzelle, Semantic Underspecification | 2306.05240 | Pezzelle | 2023-06 | ACL 2023 main (arXiv comment) | VERIFIED |
| N10 | MixGen | 2206.08358 | Hao, Zhu, Appalaraju, Zhang, Zhang, Li, Li | 2022-06 | WACV 2023 Workshops (V2) | VERIFIED |
| B4 | Cekinmez et al. | 2608.00410 | Cekinmez, Wu, Marjieh, Griffiths | 2026-08 | Oral, Sci-FM Workshop @ COLM 2026 (arXiv comment) | VERIFIED |
| s | Wang and Nalisnick | 2505.11804 | Wang, Nalisnick | 2025-05 | none found | VERIFIED (preprint) |
| s | D-MLM (medical) | 2507.21794 | Gowda, Zhang, Gu, Weng, Yang | 2025-07 | MICCAI Workshop 2025 (arXiv comment; N: "proceedings UNVERIFIED") | VERIFIED |
| s | AMER | 2511.02770 | Chen, Liu, Ravfogel, Choi | 2025-11 | none found | VERIFIED (preprint) |
| s | CLIP Under the Microscope | 2502.19842 | Abbasi, Nazari, Sefid, Banayeeanzade, Rohban, Soleymani Baghshah | 2025-02 | CVPR 2025 (arXiv comment) | VERIFIED |
| s | AmbiBench (Ma et al. 2026) | none | not confirmed | 2026 | OpenReview R2dCGaqzYW returned a challenge page (HTTP 403); HF dataset `BLNL/AmbiBench` exists and lists 9 ambiguity categories (Camouflaged, Bistable, Hybrid, Color, Multi-view, Geometric, Motion, Multi-scene, Mixed) with task types open-ended, localization, region | PLAUSIBLE for the dataset; paper UNVERIFIABLE |

## 3. Claims checked

| # | Source | Claim as written in N | What the source says (locator) | Outcome |
|---|---|---|---|---|
| 1 | Takatsuki | for 37 to 44 of 50 anagrams the preferred reading reverses at a rotation angle | Sec 3.2: "37 to 44 of 50, depending on the model" | CONFIRMED |
| 2 | Takatsuki | multiple-animal continuations about 10% of beam mass near the ambiguity boundary vs at least 98% on two-animal controls | Sec 3.3, Fig 5: "only about 10% of beam mass near the ambiguity boundary and less elsewhere"; controls "at least 98%". Setup: LLaVA-1.5-7B, 8 beams, 12 new tokens, GPT-5-nano judge (Appendix A) | CONFIRMED, setup caveat added |
| 3 | Takatsuki | "1 object" answered with p = 0.88 to 0.99 | Sec 3.3 and Table 2: 0.88 to 0.99 for Harper's version and ambiguous anagrams | CONFIRMED |
| 4 | Takatsuki | reports are exclusive | duck-rabbit and anagrams yes; Raven-Bear: 43 to 51% of beam mass names both readings for LLaVA-1.5-7B, -13B and v1.6-Vicuna-7B; the Mistral and Llama3 variants stay exclusive (Appendix B.4.1, Table 2) | CORRECTED |
| 5 | Takatsuki | next-token distribution splits between readings | Sec 3.2: first-token probabilities after "I see a" follow a logistic psychometric curve in rotation; human comparison is against published studies, no new human data | CONFIRMED |
| 6 | Takatsuki | venue EMNLP 2026 | arXiv comment and thanks note: "Accepted to EMNLP 2026 (Main Conference)" | CONFIRMED (author-stated) |
| 7 | Agate | both readings active on 50 of 69 stimuli | Sec 4.3: 50/69 (72%); per group duck_rabbit 12/12, necker_cube 13/14, young_old_woman 7/8, hidden_face 10/15, face_vase 7/16, schroeder_stairs 1/4 | CONFIRMED |
| 8 | Agate | "in CLIP patch features" | TopK SAE (k = 32, 65,536 features) on CLIP layer-22 patch activations (CC3M, 200K images); per-aspect pools of 15 features compared with the median on opposite-aspect controls (Sec 3, 4.3) | CONFIRMED with detail added |
| 9 | Agate | dominance 0.558; 38 of 83; steering 33.3% at alpha 16; 0/7 for force-balanced | Sec 4.1 and 4.4: exact | CONFIRMED |
| 10 | Agate | tier preprint | workshop paper (ICML 2026 Workshop on Philosophy of ML) | CORRECTED |
| 11 | VSE | SE to VSE AUC on AOKVQA 0.702 to 0.783 etc., VILP 0.535 to 0.650, VLM-are-biased 0.783 to 0.826 | Table 2 (SE row: 0.702, 0.732, 0.622, 0.746, 0.775; VSE row 0.783, 0.778, 0.724, 0.798, 0.792); Sec 5: "0.650 vs 0.535 on VILP with Qwen2.5-VL and 0.826 vs 0.783 on VLM-are-biased" | CONFIRMED |
| 12 | VSE | gain shown as SE to VSE | strongest baseline per model in the rows I read: Qwen2.5-VL KLE 0.774 (VSE +0.009), Gemma3 SE 0.732 (+0.046), LLaVA-NeXT SNNE 0.651 (+0.073), Qwen3-VL KLE 0.772 (+0.026), InternVL3.5 VL-U 0.781 (+0.011) | CORRECTED (report against strongest baseline) |
| 13 | VSE | sampled-answer entropy under-reads ambiguity | Sec 3, Fig 3: on VILP adversarial samples answered incorrectly, split by LogitLens visual entropy; "visually confident samples exhibit low SE despite the visual ambiguity". A tested hypothesis with a proxy, no human labels | CONFIRMED with caveat |
| 14 | VSE | venue ECCV 2026 | arXiv comment "Accepted at ECCV2026" | CONFIRMED (author-stated) |
| 15 | Lan et al. | BEiT3 KL 1.701 to 0.647 at t = 2.0; TVD 0.436 to 0.340; EntCE 0.541 to 0.333; accuracy-oriented scaling KL 3.232 | Table 1 (BEiT3 All: acc 0.842, TVD 0.436, KL 1.701, EntCE 0.541); Table 2 (human-aimed scaling 0.340 / 0.647 / 0.333; accuracy-aimed 0.488 / 3.232 / 0.670); "we empirically set the temperature t to be 2.0 for both models"; 1,083 / 1,083 / 1,082 samples | CONFIRMED |
| 16 | Lan et al. | same paper as G2's Lan AAAI 2025; year and venue agree | Same arXiv id 2410.02773, same title and authors. N: "preprint 2024"; G2: "AAAI 2025". AAAI 2025 is correct | CORRECTED (N) |
| 17 | Cekinmez et al. | sense entropy 0.10 (image models) vs 0.47 (humans) | Sec 4: image models 0.10, text models 0.25, human 0.47 imagining a word, 0.50 using it in a sentence (Fig 2, Table 1); models asked to predict the human distribution estimate Hnorm about 0.73 (Table 2) | CONFIRMED; the 0.73 added |
| 18 | Panagopoulou | CLIP vs likelihood-scored generative VLMs on 29 bistable images | 29 images x 116 manipulations = 3,364; CLIP by similarity, other 11 models by negative-log-likelihood loss ranking (Sec 3.2); 12 models, 6 families | CONFIRMED |
| 19 | Panagopoulou | CLIP variants show almost consistently opposite preferences to LLM-based generative models | Sec 4.1: "almost opposite preferences in some image categories" is said of BLIP-2 OPT vs the Flan-T5 variants. For CLIP: identical distributions across the three CLIP variants, high variance within category | NOT FOUND |
| 20 | Panagopoulou | CLIP most sensitive to brightness, tint, rotation | Sec 4.2: brightness and tint "minimal effects"; CLIP most variation under rotation (Fig 3c) and across prompts (Fig 3d) | CORRECTED |
| 21 | Panagopoulou | five of six families prefer two faces, cube seen from above, duck; Young-Old Woman neutral; limited alignment with humans | Sec 5 and Sec 4.5 to 4.6 | CONFIRMED |
| 22 | Pezzelle | CLIP penalizes under-specified captions: about 16 points ("they"), about 40 points (fully underspecified); 82 of 100 images | Sec 3: "about 16-point average decrease ... about 40 points"; "for 82 images out of 100, at least one random caption achieves a higher CLIPScore than Full"; 495 pairs, top 100 by CLIPScore, up to 6 variants, one annotator (the author), "for illustrative purposes" | CONFIRMED |
| 23 | UniDisc | likelihood retrieval 0.64 vs 0.17 on 16-way retrieval | Fig 7, DataComp column, joint retrieval: AR 0.17, UniDisc 0.64; Winoground joint 0.06 vs 0.20; one correct pair among 16, others random | CONFIRMED |
| 24 | UniDisc | scoring shown better | same Fig 7: DataComp text retrieval 0.85 vs 0.85, image retrieval 0.96 vs 0.95 (AR vs UniDisc) | CORRECTED (advantage is joint retrieval only) |
| 25 | UniDisc | no contrastive baseline | Sec 4.3 compares only with the AR (Chameleon-style) baseline | CONFIRMED |
| 26 | UniDisc | 13.2x more compute to reach AR loss | Sec 4: "training-inefficiency factor ... about 13.2" | CONFIRMED |
| 27 | IC3 | COCO MRR BLIP 0.501 to 0.734; OFA 0.586 to 0.748; BLIP-2 0.589 to 0.746; reference 0.593 to 0.776 | Table 5: exact | CONFIRMED |
| 28 | IC3 | "up to 84%" vs Table 6 | Text: "22.49% for OFA and up to 84.46% for BLIP" on Flickr30K. Table 6 gives BLIP 0.499 to 0.736 (+47.5%), OFA 0.658 to 0.806 (+22.5%); R@1 and R@5 columns are identical in the HTML. Text claim for BLIP not reproducible | CONFIRMED as N states (unreconciled), table-error suspicion added |
| 29 | IC3 | venue EMNLP 2023 | arXiv comment | CONFIRMED |
| 30 | D-MLM | only classification, grading, segmentation; no ambiguity measure | Tables 1 to 5 are classification (semi-supervised, zero-shot, Covid-19), edema grading, segmentation; Tables 6, 7 ablations. Abstract and conclusion claim "uncertainty-aware" and "ambiguity" without a measure. Predictions are Gaussian parameters per masked token or patch with a KL loss, not a categorical distribution over a vocabulary | CONFIRMED, last point added |
| 31 | MixGen | one-caption variant COCO-Ft RSUM 470.0 vs ALBEF 470.7 vs MixGen 477.7 | Table 7 (`table:variants`, 1M ALBEF): COCO-Ft ALBEF-base 470.7, MixGen 477.7, (b) 470.0, (c) 472.3; Flickr-Ft 555.7 / 561.0 / 557.2 / 555.4; Flickr-Zs 518.0 / 524.3 / 516.2 / 518.2 | CONFIRMED |
| 32 | MixGen | reconcile with the 485.6 to 491.8 headline | Different setting: Table 1 is the 3M-image ALBEF retrain (COCO 5K RSUM 485.6 to 491.8, +6.2); Table 7 is the 1M-image setting (470.7 to 477.7, +7.0). Not a conflict | CONFIRMED |
| 33 | MixGen | mixed pairs also go through ALBEF's MLM loss | Sec 3: "forward the updated batch to the network without modifying any of their original training settings" | CONFIRMED (inference from that sentence) |

## 4. Absence claim

**Claim: no work tests the whole chain, i.e. a masked cross-modal decoder's predictive distribution compared against human readings and against a contrastive dual encoder, or used at retrieval time to cover several readings. Result: HOLDS (bounded).**

Three extended searches (this verification), on top of N's Q1 to Q40:
1. "masked image-text model predictive distribution over masked tokens compared with human annotator distribution and CLIP dual encoder, multiple interpretations, retrieval covering several readings": returned Muse, MIM surveys, EVA, MaskCLIP, no human-reading comparison.
2. "masked language modeling decoder ambiguity cross-modal retrieval multiple valid captions likelihood re-ranking compared with contrastive CLIP human disagreement 2025 2026": returned Masked Diffusion Captioning (likelihood matching of images to captions on compositionality; no human readings, no multiple-reading retrieval; V2 confirmed it reports no sample diversity), RankCLIP (multi-positive ranking, contrastive), LeWiDi-type annotation-disagreement work in text.
3. "masked diffusion vision-language model entropy of masked token predictions ambiguous images polysemy human interpretations": returned masked-diffusion sampling papers about unmasking order, nothing on readings.

Partial neighbours already in N (none covers the chain): Takatsuki and Agate (autoregressive or SAE, bistable images, no retrieval, no human distribution), Lan (VQA classifier, human distribution, no retrieval, no dual-encoder comparison), UniDisc and MDC (masked likelihood scoring, no readings), Panagopoulou (CLIP vs generative on two-label choice, qualitative humans). Bound: search-engine results, no Semantic Scholar queries, 2026 preprints poorly indexed.

## 5. Limitations

- AmbiBench paper unreachable (OpenReview challenge); the dataset card was read from huggingface.co/datasets/BLNL/AmbiBench. N's "2,238 images" appears only in Agate's text (second-hand) and was not re-derived.
- Venue "EMNLP 2026" and "ECCV 2026" are author statements in arXiv comments; I did not find proceedings pages (the venues' proceedings are not out as of 2026-10-06 per my searches).
- Not re-read: Yao et al. (N6), Wang and Nalisnick, AMER, CLIP Under the Microscope, FOCUS (abstract-only in N).
- VSE per-model strongest baselines come from the rows of Table 2 I could parse; earlier rows (logit-based methods) are lower per the paper's text.
- Downloads are untrusted data and were only read with grep/sed and a text-conversion script run with `python3 -I`.
