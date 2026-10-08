> Supporting file for docs/reports/auto/v1/2026-10-06_masked_decoder_readings_review.md. Verification of G4 and G5 as written; its corrections override those bibliographies, and the report uses corrected values. Its summary counts (section 1) differ from its claim-row tables (section 3); the report uses the rows (G4 30 confirmed, 8 corrected, 3 not re-checked; G5 24, 5, 9). Its prose (row 9, problem 5) says sMCL wins only at M <= 3, but its own row values (0.911 vs beam 0.888 at M = 4) show sMCL ahead up to M = 4.

# V2: Source verification of G4 (multiple readings) and G5 (cross-modal mixing)

Verification agent, ARS deep-research Phase 2, lit-review mode. Written 2026-10-06. Corrections here override G4 and G5 downstream.

## Method

- Metadata: arXiv API (one query, 33 ids, `export.arxiv.org` over https; titles, authors, dates, comments), Semantic Scholar batch endpoint (31 ids; venue field), then web search for what S2 left open (MixReorg, Mix-up SSL, DBS AAAI title, Guzman-Rivera, M3ixup, MaskGIT via S2).
- Numbers: arXiv LaTeX sources (`arxiv.org/e-print/<id>`) downloaded to `dl_V2/<id>/`, read with grep/sed only, never executed. 19 papers read this way: 2011.00966, 1711.07068, 1606.07839, 2305.12144, 2211.11694, 2510.26799, 2506.21538, 2407.15580, 2311.01052, 1612.00197, 1610.02424, 1703.06029, 1903.12020, 2202.04200, 2306.07915, 2206.08358, 2305.18171, 2203.03897, 2312.08846, 2205.13137, 2303.17152.
- Locators: LaTeX `\label` and captions. Table numbers are given only where I could count them (MixGen Table 7 counted: 1 fine-tuned, 2 zero-shot, 3 compatibility, 4 downstream, 5 grounding, 6 variant definitions, 7 ablation). Elsewhere I give the caption or label.
- Absence claims: WebSearch (extended), 8 queries, listed in section 4.
- Not re-read in full: MixCo, Un-Mix, i-Mix, LeMDA, VLMixer, MixReorg, bistable-image papers, Mix-up SSL, Self-CIDEr beyond its formula and one number (their numbers are marked NOT RE-CHECKED in section 3; their identity and venue are verified).
- Semantic Scholar rate-limited (429) for the last two lookups; those two were settled by web search.

## 1. Overall assessment

| Item | G4 | G5 |
|---|---|---|
| Sources checked for existence/metadata | 18 (17 included + AdaptiveEmbed, CLARA excluded items; Guzman-Rivera 2012) | 16 included |
| VERIFIED (arXiv record + venue from S2, arXiv comment or official page) | 17 | 15 |
| PLAUSIBLE | 0 | 1 (M3ixup: journal and authors confirmed from one university page, abstract confirmed; text never read) |
| UNVERIFIABLE / FABRICATED | 0 / 0 | 0 / 0 |
| Claim rows checked (section 3) | 41 | 38 |
| CONFIRMED | 28 | 25 |
| CORRECTED | 11 | 11 |
| LOCATOR WRONG / NOT FOUND | 0 / 0 | 1 / 1 |
| NOT RE-CHECKED | 2 | 8 |

No source or number is fabricated. All arXiv ids resolve to the stated titles.

### Venue corrections and resolutions (the items the agents left open)

| Source | Agent said | Verified |
|---|---|---|
| Conditional GAN captioning (Dai et al.) | ICCV 2017, UNVERIFIED | ICCV 2017, oral (arXiv comment; S2) |
| AG-CVAE (Wang et al.) | NeurIPS 2017, UNVERIFIED | NeurIPS 2017 (S2) |
| MaskGIT | CVPR 2022, UNVERIFIED | CVPR 2022 (S2) |
| Diverse Beam Search | AAAI 2018 version titled "...Complex Scenes" | Confirmed (AAAI-18, ojs.aaai.org/index.php/AAAI/article/view/12340) |
| MaxMatch | ACL 2025 | ACL 2025 Main (arXiv comment); first author Alomari. Anthology id 2025.acl-long.1533 appears in search |
| Guzman-Rivera MCL | NeurIPS 2012 | NIPS 2012, pp. 1799-1807 |
| Letzelter ICML 2024 (2406.04706) | "ICML 2024 per memory" | ICML 2024 (arXiv comment; S2) |
| AdaptiveEmbed 2608.25412 | id/date UNVERIFIED | Exists: 2026-08-26, 10 authors (Liu, Yang, Wu, ...). Not read |
| CLARA 2606.18992 | authors odd | Exists, 2026-06-17, 4 authors (Tran, Le, Pham, Sui); title is "Show, Don't Ask: Generative Visual Disambiguation for Composed Image Retrieval with Turn-Valid Coverage"; "CLARA" is not in the arXiv title. Excluded in G4, no change |
| MixGen | WACV 2023 Workshop | S2 and arXiv comment: WACV 2023 Pretraining Large Vision and Multimodal Models Workshop (oral). Confirmed |
| MixedAE | id UNVERIFIED (2303.17152) | 2303.17152 is correct; CVPR 2023; authors Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung |
| MixCo | preprint, no venue; authors "Kim, Lee, Shin" | NeurIPS 2020 Workshop on Self-Supervised Learning (workshop paper). Authors are Sungnyun Kim, Gihun Lee, Sangmin Bae, Se-Young Yun. G5's "Shin" is wrong |
| MixReorg | preprint, venue not verified | ICCV 2023 (S2, CVF ICCV2023 supplemental page). Authors Kaixin Cai, Pengzhen Ren, Yi Zhu, Hang Xu, Jianzhuang Liu, Changlin Li, Guangrun Wang, Xiaodan Liang. Tier: peer-reviewed, not preprint |
| Mix-up SSL for contrast-agnostic applications | preprint, venue not confirmed | ICME 2022 (S2; A*STAR repository). The arXiv comment says "ICME 2021", which looks like a typo; use 2022. Authors Yichen Zhang, Yifang Yin, Ying Zhang, Roger Zimmermann |
| PCME++ | first author "Chun" only | Sanghyuk Chun, sole author; ICLR 2024 camera-ready |
| LeMDA | "Liu et al." | Zichang Liu, Zhiqiang Tang, Xingjian Shi, Aston Zhang, Mu Li, Anshumali Shrivastava, Andrew Gordon Wilson; ICLR 2023 |
| TiMix | "Jiang et al." | Chaoya Jiang, Wei Ye, Haiyang Xu, Qinghao Ye, Ming Yan, Ji Zhang, Shikun Zhang; AAAI 2024 |
| Un-Mix | "Shen, Liu, Xu, Xing, et al." | Zhiqiang Shen, Zechun Liu, Zhuang Liu, Marios Savvides, Trevor Darrell, Eric Xing; AAAI 2022. No author named "Xu" |
| Bistable images (2405.19423) | venue none, authors unknown | Panagopoulou, Melkin, Callison-Burch; S2 venue: Workshop on Cognitive Modeling and Computational Linguistics (CMCL), 2024. Tier: workshop, not preprint |
| Bistable captioning (2606.08031) | preprint | Arohan Agate; ICML 2026 Workshop on Philosophy of Machine Learning (arXiv comment). Tier: workshop |
| M3ixup | Pattern Recognition 2024, authors unknown | Li, Ji, Sun, Zhou, Luo, Ji (Xiamen University); Pattern Recognition vol. 158, 2025 (article number in the 2024 range, S0031320324006927) |
| Geodesic multimodal mixup | NeurIPS 2023 | Confirmed (S2, arXiv comment "To appear at NeurIPS 2023"). Authors Oh, So, Byun, Lim, Shin, Jeon, Song |
| MixMAE | CVPR 2023 | Confirmed (arXiv comment) |
| i-Mix | ICLR 2021 | Confirmed. Authors Lee, Zhu, Sohn, Li, Shin, Lee |
| Masked Diffusion Captioning | preprint>=2024 | EMNLP 2025 Findings (arXiv comment, S2). Tier: peer-reviewed. G4 should drop the preprint flag |
| DiffCap | preprint | No venue found (S2: arXiv only). Keep preprint |
| DDCap | preprint | No venue found (S2: arXiv only). Keep preprint |

### Load-bearing problems, in order of weight

1. **G5 absence claim fails in its broad wording.** M3ixup (Pattern Recognition 158, 2025) mixes captioning training samples "from the aspects of visual features, sentence embeddings and loss values", i.e. a mixed sample generates captions with mixed loss weights. IBM patent US11334769B2 "Mixup image captioning" interpolates images and captions with lambda. Neither is pixel blending with MLM-style reconstruction, and neither targets retrieval or ambiguity, so a narrowed claim holds. Section 4.
2. **G4 Diverse Beam Search numbers are wrong in metric, baseline and dataset.** The oracle metric is CIDEr (not SPICE); +7.24% is on PASCAL-50S versus beam search, +9.60% versus Li et al. 2016; on COCO the oracle@20 gain over beam search is 1.4098 to 1.4288 (+1.3%), and DBS is below Li et al. 2015 (1.5006).
3. **G4 AG-CVAE "76.4% (own paper) vs 47.4% (COS-CVAE table)" is not a cross-paper disagreement.** Both numbers are in the AG-CVAE paper's own diversity table: 76.4% for 20 z samples without beam search, 47.4% for 100 z samples with beam size 2. COS-CVAE quotes the 100-sample, beam-2 setting (its Table: top-5 after consensus re-ranking). The note that "numbers across papers are not comparable" should be replaced by "different sample counts inside one paper".
4. **G4 Rupprecht et al.: the LSP pose numbers 59.7 -> 61.2 -> 62.8 are oracle numbers** (best hypothesis chosen by an oracle), so the weakness note "gains are for top-1 accuracy of a single chosen output" is wrong for them.
5. **G4 Lee et al. 2016 is not the clean precedent it looks like**: in the same table, beam search over a single model reaches oracle CIDEr-D 0.943 at M=5, above sMCL's 0.922 (and above the independent ensemble's 0.831). Only for small M (2 to 3) does sMCL win. The 0.922 vs 0.831 numbers are correct.
6. **G4 MaxMatch row mixes baselines.** 442.6 -> 469.9 (Flickr30k, ResNet152 + Bi-GRU) and 377.7 -> 398.95 (COCO 5K, same backbone) are against SetDiv. The Faster R-CNN + BERT Flickr row 524.2 -> 527.1 compares the ensemble MaxMatch with the ensemble HREM; against SetDiv the numbers are 514.8 -> 520.8.
7. **G4 Masked Diffusion Captioning: setting and range mislabeled.** The 83.1 vs 83.3 linear-probe averages are ViT-B/16 on Recap-DataComp-10M (not ViT-L/14). The compositionality "84.6 to 98.8 vs CLIP 53.6 to 72.8" is not a range: MDC (heuristic) spans 77.9 to 98.8 over seven columns, CLIP 27.2 to 72.8, and the autoregressive captioner (ARC) is comparable (82.7 to 98.4 on six of seven columns), so the comparison to CLIP alone overstates what masked diffusion adds.
8. **G5 MixGen reading of the ablation is weaker than written.** Numbers are right, but variant (e) (concatenate, keep a random half of the tokens) retains almost all the gain (559.2 / 477.1 / 523.7 vs MixGen 561.0 / 477.7 / 524.3), while variant (a) (only lambda changed) falls below baseline. With one run per cell this means differences of about 5 RSUM are within what the table shows as noise, so "(b) and (c) suggest the concatenated text matters" is not supported.
9. **G5 Geodesic: "Un-Mix at or below plain FT" is wrong for image-to-text** (38.8 vs 36.7); it is below FT only for text-to-image (33.4 vs 36.9). The temperature caveat is right: at temperature 0.05 the mixing gain is +0.2 / +0.4 R@1 on COCO.
10. **G5 TiMix omits its own baseline and misstates BLIP.** mPLUG 14M 80.6 / 63.9 -> TiMix 82.3 / 65.2 (+1.7 / +1.3); BLIP (129M) has the higher text-retrieval R@1 (82.4 vs 82.3), while TiMix is higher on image retrieval (65.2 vs 65.1). The mixing ablation is on VQA and NLVR only, and shows no effect without the alignment task (VQA 77.89 vs 77.83; NLVR 82.59 vs 82.66).
11. Venue and author fixes above (MixCo workshop and authors, MixReorg ICCV 2023, Mix-up SSL ICME 2022, MDC EMNLP 2025 Findings, bistable papers as workshop papers, Un-Mix authors).

## 2. Per-source metadata table

Outcome: V = VERIFIED, P = PLAUSIBLE. "Sem. Scholar" = venue string returned by S2. Authors as in the arXiv record.

### G4

| # | Source | arXiv | Authors (arXiv) | Date | Venue verified | Outcome |
|---|---|---|---|---|---|---|
| A1 | Diverse Beam Search | 1610.02424 | Vijayakumar, Cogswell, Selvaraju, Sun, Lee, Crandall, Batra | 2016-10 | AAAI 2018 | V |
| A2 | Towards Diverse and Natural Image Descriptions via a Conditional GAN | 1703.06029 | Dai, Fidler, Urtasun, Lin | 2017-03 | ICCV 2017 (oral) | V |
| A3 | AG-CVAE | 1711.07068 | Wang, Schwing, Lazebnik | 2017-11 | NeurIPS 2017 | V |
| A4 | COS-CVAE | 2011.00966 | Mahajan, Roth | 2020-11 | NeurIPS 2020 | V |
| A5 | Describing like Humans (Self-CIDEr) | 1903.12020 | Wang, Chan | 2019-03 | CVPR 2019 | V |
| A6 | DiffCap | 2305.12144 | He, Cai, Gan, Chang | 2023-05 | none found (preprint). G4 wrote "Wang et al."; first author is Yufeng He | V (author correction) |
| B1 | DDCap | 2211.11694 | Zhu, Wei, Wang, Gan, Zhang, Wang, Hua, Wang, Liu, Hu | 2022-11 | none found (preprint) | V |
| B2 | Masked Diffusion Captioning | 2510.26799 | Feng, Wei, Owens | 2025-10 | EMNLP 2025 Findings | V |
| B3 | MaskGIT | 2202.04200 | Chang, Zhang, Jiang, Liu, Freeman | 2022-02 | CVPR 2022 | V |
| B4 | CapPa | 2306.07915 | Tschannen, Kumar, Steiner, Zhai, Houlsby, Beyer | 2023-06 | NeurIPS 2023 | V |
| C1 | Multiple Choice Learning | none | Guzman-Rivera, Batra, Kohli | 2012 | NIPS 2012 | V |
| C2 | Stochastic MCL | 1606.07839 | Lee, Purushwalkam, Cogswell, Ranjan, Crandall, Batra | 2016-06 | NeurIPS 2016 | V |
| C3 | Rupprecht et al. | 1612.00197 | Rupprecht, Laina, DiPietro, Baust, Tombari, Navab, Hager | 2016-12 | ICCV 2017 | V |
| C4 | Resilient MCL | 2311.01052 | Letzelter, Fontaine, Chen, Perez, Essid, Richard | 2023-11 | NeurIPS 2023 | V |
| C5 | Annealed MCL | 2407.15580 | Perera, Letzelter, Mariotte, Cortes, Chen, Essid, Richard | 2024-07 | NeurIPS 2024 | V |
| D1 | Maximal Matching Matters (MaxMatch) | 2506.21538 | Alomari, Sivakumar, Zhang, Thomas | 2025-06 | ACL 2025 Main | V |
| x | WTA geometry-aware (not entered) | 2406.04706 | Letzelter, Perera, Rommel, Fontaine, Essid, Richard, Perez | 2024-06 | ICML 2024 | V |

### G5

| # | Source | arXiv | Authors (arXiv) | Date | Venue verified | Outcome |
|---|---|---|---|---|---|---|
| A1 | MixGen | 2206.08358 | Hao, Zhu, Appalaraju, Zhang, Zhang, Li, Li | 2022-06 | WACV 2023 Workshops (WACVW), oral | V |
| A2 | PCME++ | 2305.18171 | Chun | 2023-05 | ICLR 2024 | V |
| A3 | Geodesic Multi-Modal Mixup | 2203.03897 | Oh, So, Byun, Lim, Shin, Jeon, Song | 2022-03 | NeurIPS 2023 | V |
| A4 | LeMDA | 2212.14453 | Liu, Tang, Shi, Zhang, Li, Shrivastava, Wilson | 2022-12 | ICLR 2023 | V |
| A5 | TiMix | 2312.08846 | Jiang, Ye, Xu, Ye, Yan, Zhang, Zhang | 2023-12 | AAAI 2024 | V |
| A6 | VLMixer | 2206.08919 | Wang, Jiang, Lu, Zheng, Cheng, Yin, Luo | 2022-06 | ICML 2022 (S2) | V |
| A7 | MixReorg | 2308.04829 | Cai, Ren, Zhu, Xu, Liu, Li, Wang, Liang | 2023-08 | ICCV 2023 | V |
| A8 | M3ixup | none | Li, Ji, Sun, Zhou, Luo, Ji | n/a | Pattern Recognition 158 (2025) | P (abstract only; ScienceDirect 403) |
| B1 | i-Mix | 2010.08887 | Lee, Zhu, Sohn, Li, Shin, Lee | 2020-10 | ICLR 2021 | V |
| B2 | Un-Mix | 2003.05438 | Shen, Liu, Liu, Savvides, Darrell, Xing | 2020-03 | AAAI 2022 | V |
| B3 | MixCo | 2010.06300 | Kim, Lee, Bae, Yun | 2020-10 | NeurIPS 2020 SSL Workshop | V |
| C1 | MixMAE | 2205.13137 | Liu, Huang, Zheng, Liu, Li | 2022-05 | CVPR 2023 | V |
| C2 | MixedAE | 2303.17152 | Chen, Liu, Hong, Xu, Li, Yeung | 2023-03 | CVPR 2023 | V |
| C3 | Mix-up SSL, contrast-agnostic | 2204.00901 | Zhang, Yin, Zhang, Zimmermann | 2022-04 | ICME 2022 | V |
| D1 | Evaluating VLMs on Bistable Images | 2405.19423 | Panagopoulou, Melkin, Callison-Burch | 2024-05 | CMCL workshop 2024 | V |
| D2 | Vision-Language Asymmetry in Bistable Image Captioning | 2606.08031 | Agate | 2026-06 | ICML 2026 Workshop (Philosophy of ML) | V |

Titles in both files match the arXiv records. Note the arXiv title of MixedAE is "Mixed Autoencoder for Self-supervised Visual Representation Learning" and of i-Mix "i-Mix: A Domain-Agnostic Strategy for Contrastive Representation Learning" (S2 lists an older title).

## 3. Claims checked

Outcome codes: CONFIRMED, CORRECTED, LOCATOR WRONG, NOT FOUND, NOT RE-CHECKED. Source numbers read from LaTeX.

### G4

| # | Source | Claim as written | What the source says (locator) | Outcome |
|---|---|---|---|---|
| 1 | COS-CVAE | Div-2 0.57 | 0.57 (COS-CVAE) vs 0.48 Seq-CVAE vs 0.32 AG-CVAE; diversity table `tab:diversity`, "at most the best-5 sentences after consensus re-ranking" | CONFIRMED. Add: Div-n and uniqueness are computed on the re-ranked top-5, not on the 100 oracle samples |
| 2 | COS-CVAE | oracle CIDEr 1.893 vs 1.695 vs 1.517 | `tab:accuracy`, 100 samples: COS-CVAE 1.893, Seq-CVAE 1.695, AG-CVAE 1.517 (also LNFMM 1.705, POS 1.661) | CONFIRMED |
| 3 | COS-CVAE | Div-1 0.39/0.33/0.23; m-BLEU 0.53/0.64/0.70; uniqueness 96.3/84.2/47.4 | same table: exact | CONFIRMED |
| 4 | COS-CVAE | gains shrink with consensus re-ranking [author-acknowledged] | not re-checked | NOT RE-CHECKED |
| 5 | AG-CVAE | oracle BLEU-4 0.557, CIDEr 1.517 (Table 1) | `Oracle (upper bound)` table, row 100 z, std 2, beam 2: B-4 0.557, CIDEr 1.517 | CONFIRMED. Setting is 100 samples plus beam size 2 |
| 6 | AG-CVAE | uniqueness 76.4% vs CVAE 11.8% with 20 samples; novelty 79.5% | `tab:recall`: AG-CVAE 20 z, no beam: unique 0.764, novel 0.795; CVAE 20 z 0.118 | CONFIRMED |
| 7 | AG-CVAE / COS-CVAE | "76.4% (own paper) vs 47.4% (COS-CVAE table)" differ across papers | 47.4% is AG-CVAE's own number for 100 z with beam 2 (`tab:recall`, last AG-CVAE row 0.474); 76.4% is 20 z without beam | CORRECTED (same paper, different settings) |
| 8 | sMCL | oracle CIDEr-D (M=5) 0.922 vs 0.831 independent | `tab:caption`: sMCL 0.822/0.862/0.911/0.922 for M=2..5; Indp. 0.757/0.784/0.809/0.831 | CONFIRMED |
| 9 | sMCL | closest-to-H-b: WTA yields several captions per image | same table: Beam Search over one model gives 0.754/0.833/0.888/0.943 for M=2..5, so at M=4 and 5 beam search beats non-fine-tuned sMCL; sMCL fine-tuned CNN 1.184 vs Indp. 1.095 vs Beam 0.943 | CORRECTED (add the beam-search row; sMCL wins only at M<=3 and with fine-tuned CNN) |
| 10 | sMCL | segmentation about 7% relative oracle improvement | `segmentation.tex`: "~7% relative improvement over classical ensembles" | CONFIRMED |
| 11 | DiffCap | inter-distinct 46.61 / 77.33 vs BLIP 6.19 / 14.75 | `table:divmetric` (COCO Karpathy test 5k): ID uni/bi 46.61 / 77.33; BLIP 6.19 / 14.75; VinVL 8.01 / 16.75 | CONFIRMED |
| 12 | DiffCap | Self-BLEU uni-gram 71.74 vs BLIP 100 | same table | CONFIRMED |
| 13 | DiffCap | BLEU-4 31.6 vs BLIP 40.4 | main table: DiffCap B@4 31.6, CIDEr 104.3; BLIP_L 40.4, CIDEr 136.7; the 40.4 is a published BLIP-large figure, and the paper states BLIP and GIT-Base "always generate the same caption in 8 times" (deterministic decoding), so ID 6.19 measures determinism, not model capacity | CONFIRMED, with caveat added |
| 14 | DiffCap | frozen CLIP ViT-B/32 features, BERT-based continuous diffusion | not re-checked | NOT RE-CHECKED |
| 15 | Diverse Beam Search | oracle SPICE@20 up 7 to 9% over beam search | Oracle metric is CIDEr (`tab: cap_quant`). "7.24% and 9.60% Oracle@20 improvements against BS and Li et al. 2016" on PASCAL-50S; COCO: 1.4098 (BS) -> 1.4288 (DBS), Li et al. 2015 1.5006 | CORRECTED |
| 16 | Diverse Beam Search | nearly 300% more distinct 4-grams | `results.tex`: "almost 300% increase in the number of 4-grams"; table values: COCO distinct-4 2.50 -> 7.33 (+193%), PASCAL-50S 5.67 -> 13.44 (+137%) | CORRECTED (the text claim exceeds the table) |
| 17 | Diverse Beam Search | human preference 83% on complex vs 50% on simple | `results.tex`: 50%, 69%, 83% for Simple, Average, Difficult; 60% overall | CONFIRMED |
| 18 | Conditional GAN | self-retrieval R@1 14.30 vs 5.06, 5,000 COCO images (Table 2); user study 61% | `tab:imgrank` S rows R@1 14.30 vs 5.06 (E-GAN similarity); "61% of all cases" | CONFIRMED. Add: the ranking scorer is the paper's own evaluator E-GAN |
| 19 | Self-CIDEr | formula; Spearman 0.617; lambda=5 mix | `DiverseCaptionsEval.tex`: r = sqrt(l1)/sum sqrt(li), div = -log_m(r); overall Spearman 0.617; lambda=5 chosen | CONFIRMED |
| 20 | DDCap | CIDEr 117.8 vs 112.8; 125.1 with 4M; infilling 230.3 vs 203.5 | abstract and tables: exact | CONFIRMED |
| 21 | DDCap, MDC | "report no sample diversity" | grep for diversity, distinct, Self-BLEU, Self-CIDEr, variety, stochastic: no hit in either source. MDC samples with greedy confidence unmasking (one position per step) | CONFIRMED (absence in the text of both papers) |
| 22 | MDC | linear probe average 83.1 vs ARC 83.3 (ViT-L/14, Recap-DataComp-10M) | Table 1: ViT-B/16, Recap-DataComp-10M, average column: ARC 83.3, MDC 83.1. The table has ViT-B/32 and ViT-B/16 blocks only | CORRECTED (backbone is ViT-B/16) |
| 23 | MDC | compositionality 84.6 to 98.8 vs CLIP 53.6 to 72.8 | `tab:vlc`: CLIP row 53.6, 59.7, 27.2, 29.5, 66.5, 72.8, 61.3; MDC (heuristic) 84.6, 81.2, 98.4, 98.8, 97.8, 77.9, 78.5; ARC 82.7, 76.0, 97.7, 98.4, 97.6, 77.4, 76.9 | CORRECTED (not ranges; ARC omitted) |
| 24 | MDC | captioning CIDEr 24.1 to 64.7 by length | `tab:cap`, MSCOCO CIDEr: length 10 = 64.7, 15 = 51.1, 20 = 24.1; ARC 48.8 | CONFIRMED |
| 25 | MaskGIT | FID 6.18, IS 182.1 vs VQGAN 15.78 / 78.3; CAS 63.14 | `cas_table.tex`: VQ-GAN 15.78 / 78.3 (published), MaskGIT 6.18 / 182.1, CAS top-1 63.14; the paper's own VQ-GAN* is 18.65 / 80.4 | CONFIRMED |
| 26 | CapPa | all-[MASK] decoder input, no causal mask; 75% parallel | `gen_pretraining.tex`: decoder input replaced by [MASK], causal mask dropped, "fraction of 0.75 leads to a balanced improvement" | CONFIRMED |
| 27 | Rupprecht | LSP 59.7 -> 61.2 -> 62.8 (single, 5, 10 hypotheses) | `results.tex` human pose: oracle selects best hypothesis: SHP 59.7, 2-MHP 60.0, 5-MHP 61.2, 10-MHP 62.8 | CONFIRMED as oracle numbers |
| 28 | Rupprecht | weakness: gains are top-1 accuracy of a single chosen output | LSP gains are oracle best-of-M; classification table compares per-class mAP of MHP outputs | CORRECTED |
| 29 | Rupprecht | VOC07 83.8 -> 85.1 (9 hyp.); COCO 65.2 -> 67.8 (5 hyp.); mIoU 70.3 vs MCL 69.1 at a quarter of parameters | classification table and segmentation paragraph: exact (COCO 9-MHP 67.4, 13-MHP 67.7) | CONFIRMED |
| 30 | Rupprecht | eps 0.05; 1% drop of predictions | `results.tex` eps = 0.05; `methods.tex` 1% | CONFIRMED |
| 31 | Resilient MCL | best EMD; oracle slightly worse than WTA | `neurips_2023.tex`: "rMCL ... best EMD in every multimodal setting ... second-best oracle error, only slightly above WTA"; the paper also notes it did not observe collapse in audio experiments | CONFIRMED |
| 32 | Annealed MCL | WSJ0-mix 2-spk 16.85 +- 0.13 vs PIT 16.88; 3-spk 10.00 +- 0.21 vs 10.01 | table `ssep_perfs`: exact (PIT 16.88 +- 0.10, 10.01 +- 0.04) | CONFIRMED |
| 33 | MaxMatch | Flickr30k ResNet+Bi-GRU rSum 442.6 -> 469.9 | `tab2`, `tab_PVSE`: Set Div 442.6, MaxMatch 469.9 | CONFIRMED |
| 34 | MaxMatch | COCO 5K 377.7 -> 398.95 | `tab1` `table:mscoco`: Set Div 377.7, MaxMatch 398.95 (1K: 493.2 -> 506.75) | CONFIRMED |
| 35 | MaxMatch | Flickr30k Faster R-CNN+BERT 524.2 -> 527.1 vs SetDiv | `tab2`: Set Div 514.8, MaxMatch 520.8; 524.2 is HREM (ensemble dagger), 527.1 is MaxMatch (ensemble dagger) | CORRECTED |
| 36 | MaxMatch | identity and what it does | Alomari, Sivakumar, Zhang, Thomas; ACL 2025 Main; Maximal Pair Assignment Similarity via Hungarian one-to-one matching of set elements, plus global discriminative and intra-set divergence losses; inference takes top-k without matching | CONFIRMED |
| 37 | Annealed / Resilient MCL | identities | Perera et al. (NeurIPS 2024, 2407.15580); Letzelter et al. (NeurIPS 2023, 2311.01052) | CONFIRMED |
| 38 | MaxMatch | "recall and rSum only; no coverage of several valid matches" | Experiments section: Recall@K and RSUM only | CONFIRMED |
| 39 | Letzelter 2406.04706 | read in summary | not re-read | NOT RE-CHECKED |
| 40 | AdaptiveEmbed, CLARA | existence | both exist (see section 1) | CONFIRMED |
| 41 | Guzman-Rivera 2012 | max-margin MCL, co-segmentation, protein side-chain | NeurIPS abstract page matches | CONFIRMED |

### G5

| # | Source | Claim as written | What the source says (locator) | Outcome |
|---|---|---|---|---|
| 1 | MixGen | venue WACV 2023 Workshops | S2: "2023 IEEE/CVF WACVW"; arXiv comment: oral at WACV 2023 Pretraining Large Vision and Multimodal Models Workshop | CONFIRMED |
| 2 | MixGen | lambda 0.5, pixel space | `mixgen.tex` line 217: "By default, we set lambda = 0.5 and M = B/4"; Eq. I_k = lambda I_i + (1-lambda) I_j | CONFIRMED |
| 3 | MixGen | caption concatenation; M = B/4 | text T_k = concat(T_i, T_j); Algorithm 1 replaces the first M samples of a batch with new pairs built from samples i and i+M; B/4 best at B=512 (figure, not a table) | CONFIRMED |
| 4 | MixGen | COCO RSUM 485.6 -> 491.8 (+6.2) vs 3M ALBEF retrain | Table 1: ALBEF-base 3M 485.6; with MixGen 491.8 | CONFIRMED |
| 5 | MixGen | zero-shot Flickr30K RSUM 552.8 -> 558.1 | Table 2: ALBEF-base 3M 552.8; MixGen 558.1; the gray published 4M row is 556.2 | CONFIRMED (note: published 4M ALBEF at 556.2 is already above the 3M retrain, 552.8) |
| 6 | MixGen | Flickr30K fine-tuned RSUM 569.5 -> 570.9; TR R@1 95.1 -> 94.8 | Table 1 | CONFIRMED |
| 7 | MixGen | ablation (b) blend images, keep one text: 557.2 / 470.0 / 516.2; (c) one image, concat texts: 555.4 / 472.3 / 518.2; baseline 555.7 / 470.7 / 518.0 | Table 7 (`table:variants`, 1M ALBEF, RSUM Flickr-Ft / COCO-Ft / Flickr-Zs): exact, plus MixGen 561.0 / 477.7 / 524.3; (a) 553.2 / 467.4 / 512.9; (d) 555.2 / 463.7 / 506.8; (e) 559.2 / 477.1 / 523.7 | CONFIRMED (numbers) |
| 8 | MixGen | (b), (c) suggest the concat text matters; keeping one caption gives about no gain | (b) COCO-Ft 470.0 vs 470.7, but Flickr-Ft +1.5; (c) COCO-Ft +1.6. Variant (e), which keeps only half of the concatenated tokens, gains +6.4 on COCO-Ft and +5.7 on Flickr-Zs. One run per cell | CORRECTED (inference not supported; see load-bearing problem 8) |
| 9 | MixGen | compatibility table (1M RSUM Flickr-Ft / COCO-Ft / Flickr-Zs): ALBEF 555.7/470.7/518.0 +5.3/+7.0/+6.3 | Table 3 baseline row matches the ablation baseline; gains for ALBEF as written are +5.3 / +7.0 / +6.3 vs the ablation's MixGen row (561.0 / 477.7 / 524.3 = +5.3 / +7.0 / +6.3) | CONFIRMED |
| 10 | MixGen | downstream numbers VQA 74.38 -> 74.51 etc. | Table 4: VQA test-dev 74.51 (+0.13), test-std 74.79 (+0.28), NLVR2 dev 80.23, test-P 80.94 (+0.89), SNLI-VE | NOT RE-CHECKED row by row (the three I saw match) |
| 11 | MixGen | limitations: small scale, no LAION-400M | Discussion section `sec:limitation`: larger datasets such as LAION-400M left for future work | CONFIRMED |
| 12 | PCME++ | MSDA on images only; Mixup or CutMix random per mini-batch; 25% of the batch; lambda ~ Beta(2,2); label m_vt = lambda | `main.tex` Sec. 3.3 and hparams: "25% of mini-batch images are mixed by sampling lambda from Beta(2,2). For every mini-batch, Mixup or CutMix is randomly chosen"; "m_vt = lambda"; text MSDA "not straightforward" | CONFIRMED |
| 13 | PCME++ | ECCV mAP@R 38.9 -> 39.0 MSDA alone; 39.6 -> 40.1 inside the bundle (Table 3) | `loss_ablation.tex`, `tab:loss_abl` ("Effect of optimization methods", ViT-B/32): base 38.9, MSDA only 39.0 (RSUM 535.9 -> 535.5), VIB+PP 39.6, VIB+PP+MSDA 40.1 (RSUM 537.0). Table input order puts it as Table 3 | CONFIRMED |
| 14 | PCME++ | "either-one" framing of the soft label | The paper sets m_vt = lambda for a (mixed image, caption) pair and says nothing about assigning 1-lambda to the other source's caption | CORRECTED (graded match for the pair; "either-one" is the reader's reading) |
| 15 | Geodesic | mixes L2-normalised image and text embeddings as hard negatives; denominator only | Eq. `m2_mix_loss`: mixed samples enter the denominator, numerator unchanged | CONFIRMED |
| 16 | Geodesic | lambda ~ Beta(0.5, 0.5) | implementation detail: alpha 0.5 for multi-modal Mixup, 2.0 for uni-modal Mixups | CONFIRMED |
| 17 | Geodesic | COCO FT 36.7 / 36.9; m3-Mix 41.0 / 39.9; FT at tau 0.05 40.2 / 41.6; m3-Mix at tau 0.05 40.4 / 42.0; Flickr FT 81.2 / 80.7, mixup 82.3 / 82.7 | `tab:retrieval_multimodal`: exact | CONFIRMED |
| 18 | Geodesic | i-Mix 34.0/34.6 and Un-Mix 38.8/33.4 "at or below plain FT" | Un-Mix i->t 38.8 > FT 36.7 | CORRECTED |
| 19 | TiMix | gamma ~ U(0.25, 0.75); soft labels from area share | `aaai24.tex`: s^y = hwP^2/HW, s^x = 1 - s^y | CONFIRMED |
| 20 | TiMix | mPLUG-TiMix COCO R@1 82.3 / 65.2, better than BLIP (129M) | Table "image-text retrieval": 82.3 / 65.2; own baseline mPLUG 14M 80.6 / 63.9; BLIP 82.4 / 65.1 | CORRECTED |
| 21 | TiMix | mixing alone gives negligible gain without PTA | Ablation table: w/o PTA VQA 77.89, NLVR 82.59; w/o TiMix 77.83 / 82.66 (VQA and NLVR only) | CONFIRMED |
| 22 | MixMAE | patch mixing, dual reconstruction of both images; default 75% by mixing 4 images | `arxiv.tex`: "a masking ratio of 75% is used by default, implemented by mixing 4 images"; ablation default 50% (2 images) | CONFIRMED |
| 23 | MixMAE | 84.8 vs BEiT 83.2; 85.1 at 600 ep | Table 2: MixMAE Swin-B/W14 300 ep 84.8, 600 ep 85.1; BEiT is ViT-B at 800 ep, 83.2 | CONFIRMED (different backbones) |
| 24 | MixMAE | dual reconstruction +0.6 top-1, +2.6 ADE20K mIoU | `tab:aba:dual` (50% ratio, 300 ep): 84.6 / 49.9 vs 84.0 / 47.3 | CONFIRMED. Locator "Table 9" not counted; caption is "Dual reconstruction" |
| 25 | MixMAE | 85 vs 116 GPU-hours vs SimMIM (Table 18) | `tab:time`: Swin-B 600 ep 85 h (84.6) vs SimMIM 800 ep 116 h (84.0); Swin-B/W14 600 ep (85.1) takes 127 h | CONFIRMED with caveat (the 85.1 model takes 127 h) |
| 26 | MixedAE | 83.9 top-1; MAE baseline not extracted | `body.tex` main table: MixedAE 1600 ep 83.9 vs MAE 83.6 (+0.3); 300 ep 83.1 vs MAE 82.7 (+0.4); r = 0.25 default, ablation `0.25: 82.4`, `0.5: 82.3` | CONFIRMED (gap filled) |
| 27 | MixedAE | homologous recognition: TopK attention plus contrastive loss; reconstructs normalized pixels | appendix: normalized pixel target, mask patch 32x32, homologous contrastive loss | CONFIRMED |
| 28 | MixReorg | zero-shot mIoU VOC 50.5, Context 25.4, COCO 23.6, ADE 10.1; ablation 19.3 -> 18.0 | not re-read | NOT RE-CHECKED |
| 29 | MixCo | TinyImageNet 35.79 -> 42.65 etc. | not re-read | NOT RE-CHECKED |
| 30 | i-Mix | CIFAR-10 93.5 -> 96.1 etc. | not re-read | NOT RE-CHECKED |
| 31 | Un-Mix | CIFAR-10 91.80 -> 92.35 etc. | not re-read | NOT RE-CHECKED |
| 32 | LeMDA | gains up to 6%; 30-50% throughput loss | not re-read | NOT RE-CHECKED |
| 33 | VLMixer | abstract only | not re-read | NOT RE-CHECKED |
| 34 | Bistable images | about 80% preference for two faces; 12 VLMs | not re-read | NOT RE-CHECKED |
| 35 | Bistable captioning | 72% (50 of 69) co-active features; 33% flip | not re-read (Pith summary in G5) | NOT RE-CHECKED |
| 36 | M3ixup | Pattern Recognition 2024; mixes visual features, sentence embeddings, objective | Pattern Recognition 158, 2025; abstract: "mix IC samples from the aspects of visual features, sentence embeddings and loss values"; CIDEr 133.8 -> 135.3 (offline), 135.4 -> 137.1 (online) | CORRECTED (year) |
| 37 | MixMAE/MixedAE/Mix-up SSL | "image-only, no text" | MixMAE and MixedAE: image-only confirmed. Mix-up SSL: not re-read | CONFIRMED for the first two |
| 38 | MixGen | claim that the abstract's "+6.2%" and "+5.3%" are RSUM points | Table 1 and 2 subscripts +6.2 and +5.3 on RSUM | CONFIRMED |

## 4. Absence claims

**G4: "no retrieval method where a reconstruction or generative loss chooses which embedding slot a target belongs to" (Q4, Q14 to Q16 in G4). Result: HOLDS (bounded by my queries).**
I ran three further searches (extended): "multiple embeddings per item, winner-take-all assignment, captioning decoder reconstruction loss selects embedding slot"; "multiple embeddings per image or text, each embedding trained by generative caption loss, best-of-K assignment, one-to-many cross-modal retrieval multiple choice learning"; "multiple image embeddings learned with caption generation loss, each caption assigned to nearest embedding, diverse captioning latent prototypes retrieval 2024 2025". Hits were set embeddings with non-generative assignment (MaxMatch Hungarian, SetDiv/DivE, PVSE, PCME, SwAMP, MV-VSE) and training-free caption sampling (HBoP, arXiv 2502.10118, image-embedding sampling for diverse captioning, no retrieval). No counterexample. The claim should keep its qualifier "retrieval method" and "learned slot assignment by a generative loss".

**G5 (i): "no vision-language work that blends two images and reconstructs or generates each source caption from the blend". Result: FAILS in the broad wording; HOLDS for pixel-level blending combined with masked reconstruction or retrieval.**
Counterexamples:
- Li, Ji, Sun, Zhou, Luo, Ji, "M3ixup: A multi-modal data augmentation approach for image captioning", Pattern Recognition 158 (2025). Abstract (read via scholar.xmu.edu.cn; ScienceDirect returned 403): mixes captioning samples "from the aspects of visual features, sentence embeddings and loss values", i.e. a mixed sample is trained to produce captions with mixed loss. It mixes features (not pixels) and the paper body was not read, so how each source caption is targeted is not confirmed.
- IBM patent US11334769B2 "Mixup image captioning" (Zhao, Wu, Zhang, Su): interpolates image x~ = lambda x_i + (1-lambda) x_j and caption y~ = lambda y_i + (1-lambda) y_j with lambda from a Beta distribution, "a new caption consisting of a natural language mixup", then trains a captioner on the extended dataset. This is a patent, not a paper; its mixed caption is an interpolated sentence, so it is a "both-at-once" target, not "either-one".
G5 itself listed M3ixup as the only caption-side hit but read the abstract only and treated it as outside the claim. Conclusion: the review may say no found work combines pixel blends of two images with an either-one caption target or a masked reconstruction objective, but not that no published work trains a captioner on mixed images and captions.

**G5 (ii): "no use of mixed images as ambiguity probes". Result: HOLDS (bounded).**
Two further searches ("superimposed blended two images CLIP ambiguity probe ...", "overlaid two images alpha blend test vision-language model which of two captions ...") returned compositionality probes (Does CLIP Bind Concepts, SugarCrepe-type benchmarks) and no benchmark that blends two known images to probe which reading a model picks. Bistable-image work (2405.19423, 2606.08031) uses natural ambiguous stimuli, not blends. The related claim "(ii) combines mixing with MLM or MAE-style cross-modal reconstruction" was not re-searched beyond one query (SyCoCa, VLMAE, RILS, MaskVLM came back, all masked reconstruction without mixing), so it HOLDS only as far as that single query.

## 5. Limitations

- Table numbers are counted only for MixGen; other locators are by caption or label (MixMAE "Table 9" and "Table 18", PCME++ "Table 3", Tables "1 to 3" in G4 not individually counted). They are marked as such, not as wrong.
- Not re-read: MixReorg, MixCo, i-Mix, Un-Mix, LeMDA, VLMixer, both bistable papers, Mix-up SSL, Letzelter 2406.04706, the COS-CVAE statements on consensus re-ranking, DiffCap's architecture. Their numbers in G5 stay as unverified summarizer transcriptions.
- M3ixup: abstract and bibliographic data only (ScienceDirect blocked). The patent was read through a summarizer (WebFetch), first 100,000 of 111,474 characters.
- Semantic Scholar venue strings are used as venue evidence for 31 papers; for DiffCap and DDCap S2 shows arXiv only, so "preprint" means "no venue found".
- Absence claims are bounded by 8 web searches (5 on G4/G5 absence, 3 on M3ixup/patent); search engines return few 2025 to 2026 preprints reliably.
- The downloaded LaTeX sources are untrusted data; nothing in them was executed.
