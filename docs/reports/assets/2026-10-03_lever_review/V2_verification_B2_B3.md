> Supporting file for docs/reports/auto/v1/2026-10-03_lever_review.md. Verification of B2 and B3 as written; its corrections override those bibliographies, and the report uses corrected values.

# V2: Source verification of B2 (retrieval representation) and B3 (recipes and leaderboard)

Verification agent, ARS deep-research Phase 2, lit-review mode. Written 2026-10-03. Adversarial pass: every claim was
re-read at its locator in the primary source by me; nothing below is marked CONFIRMED on the strength of B2's or B3's own
text.

**How the sources were read.** I downloaded the arXiv LaTeX source (`arxiv.org/e-print/<id>`) of 31 papers myself and
read the tables at their `\label`s, with table numbers derived from `\input` order and `\numberwithin` settings. I did not
use the B3 agent's downloads in `scratchpad/b3/` for any claim. Metadata came from the arXiv API (all 44 ids in one
query, plus 6 more ids), the Semantic Scholar batch API (38 ids), Crossref (title search), the ICML 2026 virtual site
(two poster pages), and CVF/AAAI pages through web search. I fetched the ITRA documentation page and the PLOS ONE paper
(PMC) as raw HTML and parsed them locally. WebFetch's summarizer was used only for page metadata (ICML poster page, TMLR
anthology page, one HF model card, two arXiv abstracts), never for a number.

---

## 1. Overall assessment

| Item | Count |
|---|---|
| Sources checked for existence and metadata | 48 (all sources named in B2 and B3, including excluded and screened-out items; plus the claimed TMLR companion of Sun) |
| S2_VERIFIED | 26 |
| VERIFIED (arXiv record, venue page, Crossref or ICML/CVF page; no matching S2 venue) | 21 |
| PLAUSIBLE | 1 (the TMLR 2024 paper as the "published version" of arXiv 2209.02127) |
| UNVERIFIABLE | 0 |
| FABRICATED | 0 |
| Withdrawn by authors (exists, but withdrawn) | 1 (arXiv 2505.09997, DITM; excluded by both files anyway) |
| Claim rows re-checked (§3; many rows bundle several numbers) | 121 |
| CONFIRMED | 109 (2 of them confirmed against ECCV Caption Table 4 while the paper's own appendix contradicts them, see §5 issue 2) |
| CORRECTED | 6 (FILIP gain; PCME++ "1K" selection; Multiplicity venue; Sun TMLR authors; CLIP4Clip venue; source of the listwise venue) |
| LOCATOR WRONG | 1 (ProLIP 75% masking) |
| NOT FOUND | 3 (ALBEF k = 256 for COCO; PCME++ no-GPO pooling; CUSA baseline recipe) |
| NOT CHECKED (excluded-source numbers only; stated as such) | 1 full row plus half of 1 row |
| Absence claim | survives only in its narrow form (cross-encoder / ITM re-ranking); its broad wording is falsified by PCME++ Table C.7 |

Bottom line: both bibliographies are accurate on numbers. All 40 arXiv ids resolve to the stated titles and first
authors, and no source or number was fabricated. The corrections that matter for choosing an experiment are:
(1) B2 misread FILIP's ablation (the late-interaction gain is +5.5/+3.8 R@1, not "about +1");
(2) ECCV Caption's own appendix (Table D.3 per-direction PMRP) contradicts its Table 4 PMRP for 11 of 25 models, including
the negative-mining rows on which B3's "what moved PMRP" rests;
(3) PCME++ does not say what pooling the no-GPO model used, so the "+2.6 from GPO" lever has an unknown baseline;
(4) B3's +23.3 temperature row compares a VACSR run against a row copied from PCME++.
There are also venue fixes: Multiplicity is ICML 2026 (Position track), CLIP4Clip is in Neurocomputing 2022, and the TMLR
paper linked to Sun has two authors.

---

## 2. Source quality matrix

Tier: P = peer-reviewed, W = workshop, PP = preprint, G = grey literature. Currency: year of the version read, and
`>=2024` when it falls in the brief's recent window.

| # | Source (file) | Outcome | Venue status | Tier | Currency | Note |
|---|---|---|---|---|---|---|
| 1 | ALBEF, Li et al., 2107.07651 (B2) | S2_VERIFIED | NeurIPS 2021 (S2) | P | 2021 | |
| 2 | BLIP, Li et al., 2201.12086 (B2) | S2_VERIFIED | ICML 2022 (S2) | P | 2022 | |
| 3 | BLIP-2, Li et al., 2301.12597 (B2) | S2_VERIFIED | ICML 2023 (S2) | P | 2023 | |
| 4 | X-VLM, Zeng et al., 2111.08276 (B2) | S2_VERIFIED | ICML 2022 (S2 + arXiv comment) | P | 2022 | |
| 5 | MaskVLM, Kwon et al., 2208.02131 (B2) | S2_VERIFIED | ICLR 2023 | P | 2023 | |
| 6 | Geigle et al., Retrieve Fast, Rerank Smart, 2103.11920 (B2) | S2_VERIFIED | TACL 10 (2022), DOI 10.1162/tacl_a_00473 | P | 2022 | |
| 7 | LoopITR, Lei et al., 2203.05465 (B2) | VERIFIED | none found (S2: arXiv; two web searches found no venue) | PP | 2022 | B2 correctly treats it as preprint |
| 8 | FILIP, Yao et al., 2111.07783 (B2) | S2_VERIFIED | ICLR 2022 | P | 2022 | ablation misread, see §3 |
| 9 | ELIP, Zhan et al., 2502.15682 (B2) | S2_VERIFIED | CBMI 2025, DOI 10.1109/CBMI66578.2025.11339290 | P | 2025, >=2024 | |
| 10 | ECCV Caption, Chun et al., 2204.03359 (B2, B3) | S2_VERIFIED | ECCV 2022 | P | 2022 (v5, 2024-01) | internal PMRP inconsistency, see §5 |
| 11 | PVSE, Song and Soleymani, 1906.04402 (B2, B3) | S2_VERIFIED | CVPR 2019, DOI 10.1109/CVPR.2019.00208 | P | 2019 | |
| 12 | PCME, Chun et al., 2101.05068 (B2) | S2_VERIFIED | CVPR 2021 | P | 2021 | |
| 13 | PCME++, Chun, 2305.18171 (B2, B3) | S2_VERIFIED | ICLR 2024 (S2; arXiv "ICLR 2024 camera-ready") | P | 2024, >=2024 | |
| 14 | DivE, Kim et al., 2211.16761 (B2, B3) | S2_VERIFIED | CVPR 2023 (Highlight per arXiv) | P | 2023 | |
| 15 | ProbVLM, Upadhyay et al., 2307.00398 (B2) | VERIFIED | ICCV 2023 (CVF open-access page; S2 venue field empty) | P | 2023 | CVF title spells "Vison" |
| 16 | GroVE, Venkataramanan et al., 2505.05163 (B2) | S2_VERIFIED | UAI 2025 (S2 + arXiv comment) | P | 2025, >=2024 | |
| 17 | ProLIP, Chun et al., 2410.18857 (B2) | S2_VERIFIED | ICLR 2025 (S2) | P | 2025, >=2024 | |
| 18 | Llip, Lavoie et al., 2405.00740 (B2) | S2_VERIFIED | ICML 2024 | P | 2024, >=2024 | |
| 19 | CUSA, Huang et al., 2403.05261 (B2, B3) | S2_VERIFIED | AAAI 2024, DOI 10.1609/aaai.v38i16.29789 | P | 2024, >=2024 | |
| 20 | SoftCLIP, Gao et al., 2303.17561 (B2) | S2_VERIFIED | AAAI 2024, 38(3) 1860-1868, DOI 10.1609/aaai.v38i3.27955 (OJS) | P | 2024, >=2024 | |
| 21 | NegCLIP, Yuksekgonul et al., 2210.01936 (B3) | S2_VERIFIED | ICLR 2023 (oral per arXiv) | P | 2023 | |
| 22 | ITRA docs, "Fine-tuning CLIP for MS-COCO Retrieval" (B3) | VERIFIED | documentation page, fetched (HTTP 200), no authors named | G | n.d. | all numbers re-read from the page |
| 23 | CLIP4Clip, Luo et al., 2104.08860 (B3) | S2_VERIFIED | Neurocomputing 508 (2022), DOI 10.1016/j.neucom.2022.07.028 | P | 2022 | B3 marked the journal version UNVERIFIED; it exists |
| 24 | NeighborRetr, Lin et al., 2503.10526 (B3) | S2_VERIFIED | CVPR 2025, DOI 10.1109/CVPR52734.2025.00865 | P | 2025, >=2024 | |
| 25 | VACSR, Wei et al., 2605.30968 (B3) | VERIFIED | ICML 2026: poster on icml.cc/virtual/2026/poster/65621 (same 5 authors, Hall A #1605); LaTeX uses `icml2026[accepted]`. S2 still lists arXiv | P | 2026, >=2024 | proceedings PDF not checked |
| 26 | Sun, Design of the topology..., 2209.02127 (B3) | VERIFIED | arXiv preprint, sole author Zhun Sun, v2 2023-10 | PP | 2023 | |
| 27 | Sun and Li, TMLR 2024, "Analyzing the Impact of Learnable SoftMax Temperature..." (B3, as possible published version) | PLAUSIBLE | TMLR 2024 (mlanthology; OpenReview rx1QNhsNsK) | P | 2024 | two authors (Zhun Sun, Chao Li), not "the same sole author"; same idea (multiple class tokens), different title; link to 2209.02127 not proven |
| 28 | LongProLIP, Chun and Yun, 2503.08048 (B3) | VERIFIED | ICLR 2025 workshop tiny paper (arXiv comment); S2: arXiv | W | 2025, >=2024 | |
| 29 | AAHR, Chen et al., 2507.09256 (B3) | S2_VERIFIED | Knowledge-Based Systems 316, 113355 (2025), DOI 10.1016/j.knosys.2025.113355 | P | 2025, >=2024 | |
| 30 | Listwise ranking, Li et al., 2305.16566 (B3) | VERIFIED | Knowledge-Based Systems (2024), DOI 10.1016/j.knosys.2024.111431 (Crossref) | P | 2024, >=2024 | B3 says "per Semantic Scholar", but the S2 record lists arXiv only; Crossref is the source that confirms it |
| 31 | BCLS, Li et al., 2210.11319 (B2, B3) | VERIFIED | arXiv preprint (S2: arXiv; no Crossref match) | PP | 2022 | |
| 32 | Pishdad et al., 2204.09268 (B3) | VERIFIED | arXiv preprint (S2: arXiv) | PP | 2022 | |
| 33 | Multiplicity, Chun and Russakovsky, 2505.19614 (B2, B3) | VERIFIED | ICML 2026 Position Track (arXiv v2 comment; icml.cc poster 67073 titled "Position: Multiplicity is...") | P | 2026, >=2024 | B3 lists it as preprint: wrong |
| 34 | CxC, Parekh et al., 2004.15020 (brief) | S2_VERIFIED | EACL 2021, DOI 10.18653/v1/2021.eacl-main.249 | P | 2021 | |
| 35 | BayesVLM, Baumann et al., 2412.06014 (B2, excluded) | VERIFIED | "Published at ICLR 2026" (arXiv v5 comment); S2: arXiv; OpenReview not checked | P (per authors) | 2026, >=2024 | exclusion reason (no retrieval table) not re-checked |
| 36 | MaxMatch, Alomari et al., 2506.21538 (B2, excluded) | S2_VERIFIED | ACL 2025 | P | 2025, >=2024 | |
| 37 | beta-CLIP, Zohra et al., 2512.12678 (B2, B3, excluded) | VERIFIED | arXiv preprint | PP | 2026 (v2), >=2024 | |
| 38 | FNE, Li et al., 2308.04380 (B2, excluded) | S2_VERIFIED | ACM MM 2023 | P | 2023 | |
| 39 | Deep Boosting Learning, Diao et al., 2404.18114 (B2, excluded) | S2_VERIFIED | IEEE TIP 33 (2024) | P | 2024, >=2024 | |
| 40 | DITM, Jang et al., 2505.09997 (B2, B3, excluded) | VERIFIED (withdrawn) | arXiv v3 comment: authors withdraw the paper as incomplete | PP | 2025 | should not be cited for content |
| 41 | LightningDOT, Sun et al. (B2, excluded) | VERIFIED | arXiv 2103.08784, NAACL 2021 (arXiv comment) | P | 2021 | B2 gave no id |
| 42 | Thinking Fast and Slow, Miech et al. (B2, excluded) | VERIFIED | arXiv 2103.16553, CVPR 2021 (arXiv comment) | P | 2021 | B2 gave no id |
| 43 | FLYP, Goyal et al., 2212.00638 (B3, excluded) | VERIFIED | arXiv record (venue not checked) | PP/P | 2022 | |
| 44 | WiSE-FT, Wortsman et al., 2109.01903 (B3, excluded) | VERIFIED | CVPR 2022 (arXiv comment) | P | 2022 | |
| 45 | MAFA, Byun et al., 2312.06112 (B3, excluded) | VERIFIED | CVPR 2024 (arXiv comment) | P | 2024, >=2024 | "mentions ECCV Caption only in text" not re-checked |
| 46 | MAP, Ji et al., 2210.05335 (B3, excluded) | VERIFIED | CVPR 2023 (arXiv comment) | P | 2023 | |
| 47 | Kim et al., 2304.03391 (B3, excluded) | VERIFIED | CVPR 2023 MULA workshop (arXiv comment) | W | 2023 | |
| 48 | AOQ, Chen et al., 2003.03669 (B3, seen as snippet) | VERIFIED | ECCV 2020 (arXiv comment) | P | 2020 | |

---

## 3. Claim verification table

Locators are mine and are given only when they differ from the file or add precision. "Avg" = mean of i2t and t2i.

### 3.1 PCME++ (2305.18171), both files

| Claim | File | Outcome | Correct value + locator |
|---|---|---|---|
| Table 1 B/32, InfoNCE vs PCME++: mAP@R 39.0 -> 40.1, R-P 48.7 -> 49.7, ECCV R@1 81.7 -> 83.1, CxC 54.9 -> 56.8, 5K R@1 53.0 -> 55.1, 1K RSUM 532.6 -> 537.0 | B2, B3 | CONFIRMED | Table 1 (`tab:main`); caption: "average of three different runs" |
| Table 1 B/32 rows for CLIP ZS, PCME, VSE∞, PCME++ (μ only), PCME++ SWA | B2, B3 | CONFIRMED | ZS 26.8/36.9/67.1/42.0/40.3/471.9; PCME 39.1/48.9/81.4/54.7/53.0/532.0; VSE∞ 40.0/49.5/83.1/57.1/55.2/536.5; μ only 39.5/49.1/82.7/57.0/55.2/536.2; SWA 40.2/49.8/82.9/56.8/55.2/537.3 |
| B/16 InfoNCE 41.1 / 59.3, VSE∞ 41.7 / 60.7, PCME++ 42.1 / 61.1 (mAP@R / 5K R@1); L/14 InfoNCE 35.6 / 45.9, VSE∞ 20.2 / 22.7, PCME 41.2 / 61.9, PCME++ 42.1 / 64.3 | B2, B3 | CONFIRMED | Table 1 |
| Run-to-run spread, B/32: InfoNCE i2t mAP@R 31.2 ± 0.1, t2i 46.8 ± 0.5; PCME++ 32.3 ± 0.2, 47.8 ± 0.2 | B2, B3 | CONFIRMED | Tables C.10, C.11 (`tab:appendix_i2t_full`, `tab:appendix_t2i_full`). Table 1's caption calls them "standard errors"; C.10/C.11 do not say std vs s.e. |
| 5K RSUM computed from the appendix: B/32 InfoNCE 444.1, VSE∞ 447.1, PCME 443.7, PCME++ 452.1; B/16 InfoNCE 470.3, VSE∞ 474.6, PCME++ 475.8 | B3 | CONFIRMED (recomputed) | InfoNCE B/32 = 60.1+85.9+92.3 + 46.0+75.1+84.7 = 444.1 (R@1 from C.10/C.11, R@5/R@10 from C.8/C.9); the other six sums also reproduce exactly |
| Recipe: 25 epochs, batch 128, AdamP, initial lr 5e-4, ×0.1 for the last 10 epochs, wd 1e-4, visual ×0.01 (= 5e-6), text ×0.1 (= 5e-5), LLRD 0.7, visual tower frozen 2 epochs then 1 epoch linear warmup, SizeAugment, GPO, 1024-d, B/32 1 V100 38 h | B2, B3 | CONFIRMED | App. B.2 text and Table B.1 (`tab:hparam`) |
| InfoNCE baseline initial temperature 1.0 | B3 | CONFIRMED | App. B.2: "the initial softmax temperature for InfoNCE ... is set to 1.0" |
| Model selection "on validation 1K RSUM" | B3 | CORRECTED | Paper says only "the best model based on the validation rsum" (Sec 3.1, App. B.3). The 1K/5K protocol of the validation rsum is not stated; the reported RSUM column is 1K |
| All baselines share hyperparameters tuned on VSE∞ B/32 validation rsum | B2 | CONFIRMED | Sec 3.1: "optimization hyperparameters for all experiments are fixed ... based on VSE∞ ViT-B/32 validation rsum score" |
| PP and MSDA applied to PCME++ only; "non-trivial" for triplet loss | B2 | CONFIRMED | Sec 3.2 and App. A.5 |
| Table 3 ablation: none 38.9; VIB 39.3; PP 39.0; MSDA 39.0; VIB+PP 39.6; all 40.1 (RSUM 535.9 / 534.5 / 536.0 / 535.5 / 534.8 / 537.0) | B2, B3 | CONFIRMED | Table 3 (`tab:loss_abl`) |
| PP α sweep: mAP@R 40.0 to 40.3, 5K R@1 54.8 -> 52.6 | B2 | CONFIRMED | Table C.3 (`tab:appendix_alpha_abl`) |
| GPO ablation 37.4 -> 40.0 mAP@R, 5K R@1 49.2 -> 55.3, RSUM 521.8 -> 537.1 | B3 | CONFIRMED (numbers) | Table C.6 (`tab:arch_abl`): rows "1 layer σ head, no GPO" vs "1 layer σ head, GPO", both under the PCME++ loss |
| What the no-GPO baseline pooled with | B3 (implied "GPO vs no GPO") | NOT FOUND | The paper never states the replacement pooling (no "CLS", "EOS" or "average pooling" anywhere in main.tex). It is also unstated whether SizeAugment, which exists "for the generalizability of GPO", was kept. Treat the +2.6 as GPO vs an unspecified pooling, under the probabilistic loss, not InfoNCE |
| Table C.7: mean-only 40.2 vs CSD 40.2 mAP@R (5K R@1 55.2 vs 55.5; RSUM 536.3 vs 537.3) | B2, B3 | CONFIRMED | Table C.7 (`tab:inference_distance_comparisons`), SWA B/32, 3-run average; 2-Wasserstein also 40.2 |
| Definition of the Table 1 "μ only" row | B2 (flags it), B3 ("pairwise sigmoid-style loss") | CONFIRMED that it is undefined | Not defined anywhere in the text; Table 1 marks it non-probabilistic. B3's label "pairwise sigmoid-style loss" is an inference, not the paper's wording |
| Noise 50%: InfoNCE 33.6 vs PCME++ 35.7 mAP@R | B2 | CONFIRMED | Table 2 (`tab:noise_ratio`) |
| Table C.1: BLIP 40.5 / 73.1, VinVL 40.8 / 66.4, PCME++ B/16 42.2 / 61.3 | B2 | CONFIRMED | Table C.1 (`tab:sota`). The row is labelled "PCME++ (B/16)" without "SWA"; its values equal Table 1's B/16 SWA row |
| "Gap between VSE∞ and PCME++ (SWA) not significant" at B/32 | B3 | CONFIRMED | App. D "Limitations and Discussions" |

### 3.2 ECCV Caption (2204.03359), both files

| Claim | File | Outcome | Correct value + locator |
|---|---|---|---|
| Table 4: CLIP B/32 ZS 26.75 / 36.91 / CxC 41.97 / 5K 40.28 / PMRP 55.32 | B2, B3 | CONFIRMED | Table 4 (`tab:main_results`) |
| Table 4: B/16 ZS 29.25/38.99/56.58/44.26/42.69; L/14 ZS 27.98/37.80/57.70/48.14/46.44 | B3 | CONFIRMED | Table 4 |
| Table 4: BLIP 40.52 / 48.43 / ECCV R@1 90.99 / CxC 74.30 / 1K 86.12 / 5K 73.11 / PMRP 57.17 | B2, B3 | CONFIRMED | Table 4 |
| Table 4: VSE∞ BUTD region 40.46/49.97/56.64/52.40/50.38; grid 40.40/50.09/56.87/53.47/51.60; WSL 42.41/51.43/57.65/60.79/59.01 | B2, B3 | CONFIRMED | Table 4 |
| Table 4: PVSE K=1 33.98/44.49/53.56; K=2 40.26/49.92/55.52 (Δ +6.28 mAP@R, +1.96 PMRP, 5K R@1 36.20 -> 38.13) | B2, B3 | CONFIRMED | Table 4 |
| Table 4: PVSE K=1 re-implementations, no NM / semi-hard / hardest: mAP@R 33.34 / 36.63 / 35.76; PMRP 56.67 / 55.15 / 54.37; 5K R@1 30.65 / 36.00 / 36.88 | B3 | CONFIRMED (Table 4) but see §5 | Table 4 bottom block. The same paper's Tables D.3 give per-direction PMRP whose means are 44.84 / 46.41 / 47.17, the opposite trend |
| Table 4: PCME 37.11 -> CutMix re-impl. 41.74 mAP@R, PMRP 56.71 -> 57.65 | B2, B3 | CONFIRMED | Table 4 |
| Table 4: VSE++ 35.01/45.50/54.26/37.95/35.79; VSRN 42.28/51.84/55.44/48.85/46.74; ViLT ZS/FT and VinVL ZS/FT rows | B3 | CONFIRMED | Table 4 |
| PMRP range over 25 models 46.95 to 57.70 | B2 | CONFIRMED | Table 4 (VSE0 46.95; CLIP L/14 57.70) |
| Kendall τ: 5K R@1 vs mAP@R 0.39; mAP@R vs PMRP 0.20; 1K R@1 vs PMRP 0.45; mAP@R vs 1K R@1 0.47, vs CxC 0.39, vs RSUM 0.52, vs R-P 0.90; PMRP vs 5K R@1 0.45, vs RSUM 0.43, vs ECCV R@1 0.29, vs R-P 0.17; CxC vs 5K R@1 1.00; 1K R@1 vs 5K R@5 0.97 | B2, B3 | CONFIRMED | Table 5 (`tab:metric_wise_kendall_full`) |
| Table 2 PM precision/recall: 65.3 / 56.5 (i2t), 56.6 / 61.8 (t2i); COCO 96.9/21.1, 96.4/13.8; CxC 95.5/23.0, 93.2/16.1 | B2, B3 | CONFIRMED | Table 2 (`tab:cxc_pm_pr`); "PM precision 57 to 65%" holds |
| Table 3: ×8.47 positive images, ×3.58 positive captions | B3 | CONFIRMED | Table 3 |
| Query counts 1,333 caption / 1,261 image; 5.3% / 25.2% | B3 | CONFIRMED | Sec 3.2 and Sec 5 say 1,333 subsampled; Sec 1 and Table 3 say 1,332 caption queries after post-processing |
| Human study: 3,200 preferences, BT scores 70.85 / 13.15 / 10.66 / 4.89 / 0.44 | B3 | CONFIRMED | Sec 4.1 (all five) and App. D.1 (first four) |
| "VinVL takes 25 hours ... VSE++ only takes 1 minute" | B2 | CONFIRMED | Sec 4.1 |
| "contrastive models without a negative mining strategy are specialized to PMRP"; best COCO/ECCV models "show inferior PMRP scores"; PMRP "only captures the existence or absence of the objects" | B2, B3 | CONFIRMED | Sec 4.2 |
| "PMRP is a noisy metric compared to others" | B3 | CONFIRMED | Sec 2 (related work) |
| "only three machine annotators (PVSE, PCME and VSRN) achieve better rankings on ECCV mAP@R compared to the COCO R@1 ranking" | B2, B3 | CONFIRMED | App. E.3 |
| Machine annotators PVSE, VSRN, PCME, ViLT, CLIP B/32 | B2, B3 | CONFIRMED | Table 1 (`tab:model_overview`) |
| ECCV Caption does not state how BLIP was run (k, ITM) | B2 | CONFIRMED | No BLIP inference details in Sec 4 or App. D.2 |
| CLIP B/32 per-direction (App D.3): i2t 50.14/75.00/83.42, t2i 30.42/55.96/66.89; 5K RSUM 361.83; mAP@R 22.39/31.11 | B3 | CONFIRMED | Tables D.3 (`tab:main_results_it`, `_ti`); sum recomputed 361.83 |

### 3.3 CUSA (2403.05261)

| Claim | File | Outcome | Correct value + locator |
|---|---|---|---|
| COCO 5K CLIP B/32: i2t R@1 56.3 -> 57.3, t2i 42.8 -> 44.2, RSUM 422.6 -> 429.7 (422.5 recomputed) | B2, B3 | CONFIRMED | Table 1 (`tab:itr_mscoco_flickr`); 56.3+81.7+89.4+42.8+71.2+81.1 = 422.5 |
| ECCV, CLIP B/32: i2t mAP@R 28.5 -> 29.6, R-P 39.4 -> 40.7, R@1 72.5 -> 72.0; t2i 41.7 -> 45.2, 50.8 -> 53.6, 83.0 -> 85.7; avg mAP@R 35.1 -> 37.4; avg R-P 45.1 -> 47.15 | B2, B3 | CONFIRMED | Table 2 (`tab:itr_eccv`); arithmetic correct |
| L/14@336: avg mAP@R 39.15 -> 40.6, avg R-P 48.8 -> 49.95, 5K R@1 avg 59.35 -> 60.15, 5K RSUM 469.6 -> 473.1 | B2, B3 | CONFIRMED | Tables 1, 2 |
| Ablation: ECCV "Avg" 52.6 / 54.5 (CSA) / 53.6 (USA) / 54.5 (both); 5K RSUM 422.6 / 427.7 / 425.7 / 429.7 | B2, B3 | CONFIRMED | Table 5 (`tab:ablation_clip`). "Avg" = mean of the six ECCV metrics (recomputed 52.65 and 54.47) |
| X2VLM-base (author checkpoint, re-ranked) ECCV mAP@R 36.6 / 43.8 (avg 40.2) -> 37.6 / 48.4; 5K R@1 83.5 / 66.2 | B2 | CONFIRMED | Tables 1, 2 |
| Teachers: Unicom (ViT-B/32) and all-mpnet-base-v2 | B2, B3 | CONFIRMED | Sec 3.1 |
| "All CLIP model reports are based on fine-tuned results using InfoNCE" | B3 | CONFIRMED | Sec 4.1.2 |
| CUSA's baseline training recipe | task item | NOT FOUND in the paper | No epochs, lr or batch in main.tex; the paper defers details to a GitHub appendix (not read by me or by B3). No seeds or variance anywhere |

### 3.4 VACSR (2605.30968)

| Claim | File | Outcome | Correct value + locator |
|---|---|---|---|
| Sigmoid baseline 39.3/48.7, CxC 57.3, 5K R@1 55.6 -> VACSR 40.7/50.1/59.7/58.1 (+1.4 mAP@R) | B3 | CONFIRMED | `tab:ablation`; baseline = "GPO and sigmoid loss to fine-tune CLIP directly" (Sec 4.3) |
| InfoNCE learnable τ 39.0; τ fixed at 1 15.7 (R-P 25.5, CxC 16.4, 5K 14.8); sigmoid fixed a,b 0.2 | B3 | CONFIRMED | `tab:loss` (the table carries two labels, `tab:ablation_study` and `tab:loss`) |
| The InfoNCE row equals PCME++'s Table 1 InfoNCE row; PCME++ comparison rows copied | B3 | CONFIRMED | 39.0/48.7/81.7/54.9/74.0/53.0/532.6 in both. Consequence: the "+23.3 from learnable temperature" row compares a VACSR run (fixed τ) with a PCME++ run (learnable τ) |
| Removing L_KL or L_σ raises COCO and CxC R@1 but lowers mAP@R | B3 | CONFIRMED | `tab:ablation`: w/o L_KL 40.1 / CxC 60.4 / 5K 58.8; w/o L_σ 39.8 / 60.7 / 59.1; Sec 4 "Metric Sensitivity under Uncertainty Modeling" |
| Setting: PCME++ framework, B/32 and B/16, GPO, AdamP, 25 epochs, lr 5e-4 decayed to 10% at epoch 15, two-layer MLP adapter | B3 | CONFIRMED | Appendix implementation details |
| VACSR B/32 5K RSUM 462.5 (computed) | B3 | CONFIRMED | 66.5+88.3+93.9+49.8+77.7+86.3 (`tab:coco_comparison`) |

### 3.5 Re-ranking sources (B2 §2.1)

| Claim | File | Outcome | Correct value + locator |
|---|---|---|---|
| ALBEF Flickr k ablation: TR 97.30 (s_itc) -> 98.60 / 98.57 / 98.57; IR 90.95 -> 93.64 / 93.99 / 93.95; "not sensitive to changes in k" | B2 | CONFIRMED | Table 6 (`tbl:retrieval_ablation`), Sec 6.5 |
| ALBEF COCO 5K 4M 73.1/56.8, 14M 77.6/60.7 | B2 | CONFIRMED | Table 2 |
| ALBEF COCO "re-ranked with k = 256" | B2 | NOT FOUND | ALBEF's paper never gives k for COCO (Sec 5 says only "top-k"). k = 256 for COCO is stated by BLIP (Sec 5.1) and X-VLM ("Following ALBEF") |
| ALBEF multi-positive ITC, 1/#positives | B2 | CONFIRMED | Sec 5 |
| BLIP k = 256 COCO, 128 Flickr; COCO 5K 81.9/64.3 (129M B), 82.4/65.1 (L) | B2 | CONFIRMED | Sec 5.1, Table 5 |
| BLIP-2: k = 128; COCO 5K 83.5/66.3 (L), 85.4/68.3 (g); ITC+ITM 84.5/67.2 vs +ITG 85.4/68.3 | B2 | CONFIRMED | Sec 4.3, Tables 5, 6 |
| X-VLM: k 256/128 "Following ALBEF"; 1/#positives; 81.2/63.4 (16M) | B2 | CONFIRMED | Sec 4 and appendix retrieval paragraph; Table 2 |
| MaskVLM: ITC 65.10/80.10; ITC+ITM 79.96/92.30; MLM+MIM 76.08/90.30; all four 81.26/94.10 (IR/TR R@1, fine-tuned Flickr) | B2 | CONFIRMED | `tab:ablation_loss` (Table 5 by input order) |
| Geigle COCO 5K: Joint+BE 52.5/66.7 (rsum 472.2) -> Joint+Coop 54.7/70.8 (481.9); BE 52.2/66.9 (472.4) -> Sep+Coop 52.8/70.2 (478.6); CE (own OSCAR) 52.6/69.3 (476.0) | B2 | CONFIRMED | Table 1; all rsums recomputed |
| Geigle k (COCO 5K Joint+Coop): k = 10/20/50 IR 54.8/54.7/54.6, TR 70.9/70.8/70.7 | B2 | CONFIRMED | Table 8 (`tab:res:rr-k`) |
| Geigle timing: 5K eval 30 s / 25 min / 50 h; latency 50k images 16 ms / 74 ms / 2 min | B2 | CONFIRMED | Tables 4, 3 |
| Geigle score fusion (Flickr): Joint+Coop IR 76.4; add λ 0.1 76.7; λ 0.9 74.6 | B2 | CONFIRMED | Table 9 (`tab:res:rr-sum`) |
| LoopITR: dual 67.6/51.7 (rsum 471.9) -> cross re-ranked 75.1/58.0 (494.7) | B2 | CONFIRMED | Table 9 and Table 7; re-ranking of dual top-k confirmed by the appendix footnote ("we report 'cross encoder' results in this setting if not otherwise stated") |
| LoopITR distillation (COCO 5K val) dual 62.64/46.78 -> 65.00/50.53 | B2 | CONFIRMED | Table 4; Sec 4.1 says ablations report the 5K val split |
| LoopITR Flickr timing: 89.6/77.2 in 11 s; re-rank top 16 94.5/83.4 in 76 s; full cross 94.4/83.1 in 2342 s | B2 | CONFIRMED | Table 13 (`tab:efficiency`) |
| LoopITR CxC R@1 dual only, I->T 69.2, T->I 53.5 | B2 | CONFIRMED | Table 12 (`tab:cxc`); no CxC row for the cross encoder |
| FILIP late interaction: I2T 29.2 -> 30.5, T2I 17.9 -> 18.5 ("w/ back translation" to "w/ late interaction"); relevance line "about +1 R@1" | B2 | CORRECTED | Table 4 rows are single additions to the baseline, not cumulative. Correct baseline is "Baseline (ViT-B/32)" 25.0 / 14.7 -> late interaction 30.5 / 18.5: **+5.5 I2T, +3.8 T2I R@1** (Sec 4.4: "absolute R@1 gain of 5.5% (resp. 3.8%)"). The gain is of the same size as the cross-encoder re-ranking gains in this theme, not smaller |
| FILIP cost: 1.31 -> 2.85 s/iter, 14.3 -> 26.0 GB (fp32, all tokens); final 1.39 s/iter, 16.1 GB | B2 | CONFIRMED | Table 5 (`tab:efficiency-late`) |
| FILIP fine-tuned COCO 5K 78.9/61.2 vs ALBEF 77.6/60.7 | B2 | CONFIRMED | Table 3 |
| ELIP: CLIP T2I R@1 40.16 -> 45.61; SigLIP 54.21 -> 61.03; SigLIP-2 56.87 -> 62.91; BLIP-2 68.25 -> 68.41; CLIP R@100 97.67; Flickr late fusion 69.58 vs 72.30 vs 67.56; k = 100; 144 A40 GPU-hours | B2 | CONFIRMED | Sec 4 table, Sec 5.1, supplement (Top-k recall table; late-fusion table; compute table) |

### 3.6 Probabilistic and set embeddings (B2 §2.2, B3)

| Claim | File | Outcome | Correct value + locator |
|---|---|---|---|
| PVSE COCO 5K R@1 i2t 41.7 -> 45.2, t2i 30.6 -> 32.4 | B2 | CONFIRMED | `tab:results_coco` (Table 2) |
| PCME-paper PMRP for PVSE (5K): 29.3/30.1 -> 31.8/32.0 | B2 | CONFIRMED | PCME Table 3 (`tab:coco`), marked "*" (published models) |
| PVSE on CUB (HNM): i2t R-P 22.34 (K=1) -> 19.67 (K=2) -> 18.38 (K=4) | B2 | CONFIRMED | PCME `tab:supp-cub-comparison-full` |
| PCME COCO 5K PMRP 34.0 -> 34.1 (i2t), 34.3 -> 34.4 (t2i); R@1 43.5 -> 44.2, 31.7 -> 31.9; 1K PMRP 45.0 -> 45.0, 45.9 -> 46.0; R@1 68.0 -> 68.8 | B2 | CONFIRMED | PCME Table 3 |
| PCME CUB R-P 24.7 -> 26.3, 25.6 -> 26.8; DoF 0/1/512: 24.7/25.7/26.3 | B2 | CONFIRMED | `tab:cub-comparison`, `tab:sigma-isotropic` |
| DivE ECCV/CxC: i2t mAP@R 34.8 -> 36.0, t2i 50.0 -> 51.0 (avg 42.4 -> 43.5); R-P avg 51.45 -> 52.45; CxC i2t 67.9 -> 72.3, t2i 53.7 -> 55.5 (avg 60.8 -> 63.9) | B2, B3 | CONFIRMED | Table A1 (`tab:supp_eccv`), App. B.2; "our best model" wording confirmed; single vs ensemble not stated |
| DivE COCO 5K RSUM: VSE∞ 468.9 (ensemble 474.8), DivE 474.9 (ensemble 482.0) | B2 | CONFIRMED | Table 1 |
| DivE K sweep, Flickr RSUM 492.6 / 495.5 / 497.4 / 500.8 / 498.4 / 499.3 | B2 | CONFIRMED, clarified | Table 4. All ablations use "ROI visual features on Flickr30K" (Sec 4.4), not the ResNeXt+BERT model |
| DivE: enlarging VSE∞ to equal FLOPs lowers Flickr RSUM by 10.8 | B2 | CONFIRMED | App. B.1 |
| DivE: VSRN is an annotator, "inflated evaluations" | B2, B3 | CONFIRMED | App. B.2 |
| ProbVLM: "all these models use the same underlying embeddings and achieve the same performance on the retrieval task"; BLIP COCO −SR²: 0.80 vs PFE* 0.58, PCME* 0.62, TTDA 0.29; 100 epochs, lr 1e-4 | B2 | CONFIRMED | Sec 4.1 text and implementation paragraph |
| GroVE R@1 (CLIP B/32, frozen): COCO i2t deterministic 0.715, GroVE 0.512, PCME++ 0.397, ProbVLM 0.303; t2i 0.515, 0.288, 0.125, 0.156 | B2 | CONFIRMED | App. C (`tab:retrieval`, "Retrieval performance using CLIP"); Q = 5 for COCO confirmed. COCO protocol (1K vs 5K) is not stated, as B2 says |
| ProLIP Table C.4 (DataComp 1.28B): retrieval 53.6 (no inclusion), 53.2 (masked only), 53.2 (image-text only), 53.4 (both); average 56.6 / 56.7 / 57.0 / 57.3 | B2 | CONFIRMED | Table C.4 (`tab:zero_shot_abl`) |
| ProLIP Table C.3 (96M): masked inclusion HierarCaps 44.8 -> 47.9, IN 37.4 -> 37.5; both 54.8 / 37.0 | B2 | CONFIRMED | Table C.3 (`tab:abl`) |
| ProLIP Table C.8 occlusion: σ_v 0.0148 -> 0.0153, IN 74.6 -> 73.2 | B2 | CONFIRMED | Table C.8 |
| ProLIP Table 1 retrieval avg CLIP 53.4, SigLIP 53.4, ProLIP 53.0; C.1 Flickr/COCO/WinoGAViL 71.13/45.73/42.12 (mean 53.0) | B2 | CONFIRMED | Table 1, Table C.1 |
| ProLIP masking "75% of input tokens", masked copies for 12.5% of the batch, "(Sec 3.3)" | B2 | LOCATOR WRONG | 12.5% is in Sec 3.3; "75% information" is in Sec 4.1 (implementation details) |
| ProLIP "often counterintuitive ... under noisy image-text correspondences" | B2 | CONFIRMED | Sec 3.3 |
| Llip zero-shot COCO: B/16 I2T 59.7 -> 63.4, T2I 42.0 -> 45.6; L/14 65.4 -> 68.1, 48.1 -> 50.6; "slightly higher than CLIP" inference time | B2 | CONFIRMED | Table 2 (`tab:zs-retrieval`); inference-time sentence is in the appendix "Inference time" paragraph (section number not checked) |
| SoftCLIP zero-shot COCO 5K: RN50 29.4 -> 36.0 / 18.9 -> 22.2; B/16 30.7 -> 30.9 / 19.1 -> 19.2 (YFCC15M-V2, own CLIP implementation) | B2 | CONFIRMED | `sotaretrieval` table |

### 3.7 Recipes and other B3 sources

| Claim | File | Outcome | Correct value + locator |
|---|---|---|---|
| NegCLIP recipe: open_clip, 5 epochs, lr sweep {1e-5, 5e-6, 1e-6} chosen on COCO val retrieval, 50 warmup steps, AdamW, cosine, batch 1024, one RTX 2080 Ti; ViT-B/32 | B3 | CONFIRMED | App. C "Fine-tuning Details"; Sec 4 and App. "models" for B/32 |
| NegCLIP results: COCO Image R@1 0.30 / 0.42 / 0.41; Text 0.50 / 0.59 / 0.56; Flickr Image 0.59 / 0.67 / 0.67, Text 0.78 / 0.83 / 0.79; VG-Relation 0.59 / 0.63 / 0.81; COCO-PRC 0.46 / 0.36 / 0.86 | B3 | CONFIRMED | `table:negative-clip-perf` |
| NegCLIP limitation: single GPU, N = 1024, "expect to gain further improvements from larger batch sizes" | B3 | CONFIRMED | App. C "Limitations" |
| ITRA: RN50, 10 epochs, batch 256, lr 1e-5, AdamW, wd 0.5, warmup 100, 8×2080 Ti; zero-shot 58.39 -> 73.98 | B3 | CONFIRMED | fetched page, "Getting Started" |
| ITRA sweeps: lr 72.91 / 73.98 / 73.97 / 73.32 / 72.46 / 69.34; epochs (lr 1e-5) 72.66 / 73.98 / 74.43 / 74.45 / 73.96; (lr 2e-5) 72.86 / 73.97 / 74.28 / 74.02 / 74.03; batch 800 / 512 / 256 / 128 / 64 / 32 = 74.89 / 74.85 / 73.98 / 72.14 / 69.24 / 65.04 (lr scaled); wd range ±0.43 | B3 | CONFIRMED | fetched page, sections 1 to 4 |
| ITRA freezing: improved baseline 75.04; lock image 71.98; lock text 71.68; MLP heads 66.71; linear heads 62.34; batch 1792, lr 7e-5: 76.02 | B3 | CONFIRMED | fetched page |
| ITRA in 5K-RSUM units: 443.85 at bs 256; 128 -> 256 +11.0; 128 -> 512 +16.3 | B3 | CONFIRMED (recomputed) | 64.84+86.62+92.30+44.99+72.76+82.34 = 443.85; (73.98−72.14)×6 = 11.04; (74.85−72.14)×6 = 16.26 |
| CLIP4Clip: Adam, cosine, lr 1e-7 towers / 1e-4 new modules, batch 128, 5 epochs; "comparable result for batch size 128 and 256" | B3 | CONFIRMED | Sec 4.2, Sec 4.4 |
| NeighborRetr: Sec 5 says CLIP B/32; supplement says B/16 for COCO/Flickr; Adam, batch 128, lr 1e-4, 10 epochs | B3 | CONFIRMED | sec/5_experiments.tex l.18-22; supplement implementation details |
| NeighborRetr ECCV: t2i 49.8/57.7/92.1, i2t 34.4/45.6/82.1 (avg mAP@R 42.1, R-P 51.65); CUSA row = CUSA L/14@336 | B3 | CONFIRMED | `tab:comp_eccv_caption` |
| NeighborRetr COCO: 69.5/90.5/95.3, 53.2/79.7/88.2, RSUM 476.4, first block labelled "Text-to-Image" | B3 | CONFIRMED | supplement `tab:itr_combined` |
| Sun: i2t mAP@R 30.5 -> 30.9, t2i 40.5 -> 41.7 (avg 35.5 -> 36.3); R-P avg 45.45 -> 46.1; 5K R@1 avg 53.65 -> 53.35; CxC avg 55.1 -> 55.05; OpenAI B/16 23.7 / 34.8 (avg 29.25) | B3 | CONFIRMED | `tab:clip_eccv` |
| Sun: batch 32,768, 32 epochs, data mix includes COCO captions; oblique model max temperature 3.95 vs 100 | B3 | CONFIRMED | appendix hyperparameter table and text; Sec 4 data list cites COCO captions. That this confounds geometry with temperature is an inference: the table states the setting, it does not discuss the confound |
| LongProLIP ECCV mAP@R avg 34.1 (i2t 28.9, t2i 39.2) -> 35.7 (S24M) -> 33.4 (S128M) -> 34.6 (S+HYPE+DFN 128M); ViT-B/16, 12.8B seen | B3 | CONFIRMED | `tab:longprolip_main`, `appendix:tab:overview` |
| AAHR: w/o PGA 33.7 / 49.3 (avg 41.5) -> 34.2 / 49.7 (41.95); R-P avg 50.8 -> 51.1; CLIP B/32 ZS row 22.4 / 31.1; frozen CLIP B/32 global features | B3 | CONFIRMED | `tab:comparisonECCV`; method section ("without fine-tuning") |
| Listwise: triplet 26.80 / 46.29 (avg 36.55) -> + S-NDCG 27.75 / 47.35 (37.55); S-NDCG alone 21.40 / 41.72 (31.56) | B3 | CONFIRMED | `ablation_eccv` |
| BCLS: VSE++ 20.8 / 38.3 (29.55) -> BCLS 21.8 / 39.2 (30.5); R-P avg 40.45 -> 41.4 | B2, B3 | CONFIRMED | ECCV tables in 2210.11319 (main and ablation) |
| Pishdad: 5K i2t PMRP VSE++† 29.1 -> 31.3; VSRN† 27.7 -> 34.2; PCME 34.1; PVSE 31.8; PMRP averaged over ζ ∈ {0,1,2} | B3 | CONFIRMED | `tab:results_coco`, metrics paragraph |
| Multiplicity re-quotes τ 0.47 and PCME++ 40.0 / 20.2 vs 40.1 / 42.1 | B2, B3 | CONFIRMED | 2505.19614 Sec 3 (the 0.47 is ECCV Caption's 1K R@1 vs mAP@R) |
| Multiplicity venue | B3 ("preprint") | CORRECTED | ICML 2026 Position Track (arXiv v2 comment; icml.cc poster 67073) |
| Sun TMLR 2024 companion "by the same sole author" | B3 | CORRECTED | TMLR 2024 paper authors: Zhun Sun and Chao Li (mlanthology record); link to 2209.02127 remains unproven |
| CLIP4Clip "journal version UNVERIFIED" | B3 | CORRECTED | Neurocomputing 508 (2022), DOI 10.1016/j.neucom.2022.07.028 (S2) |
| Listwise "Knowledge-Based Systems 2024 per Semantic Scholar" | B3 | CORRECTED (source of the venue) | Venue true per Crossref (DOI 10.1016/j.knosys.2024.111431, 2024-03); the S2 record lists arXiv only |
| B3 §4 step counts: PCME++ ~111k steps; NegCLIP ~2.8k; brief ~44k; ITRA 10 epochs 0.45 mean recall (~2.7 5K-RSUM) below 15 epochs | B3 | CONFIRMED (recomputed) | 25×567k/128 ≈ 110.7k; 5×567k/1024 ≈ 2.77k; 74.43−73.98 = 0.45 |
| "What moved PMRP": no NM vs hardest +2.3; B/32 -> L/14 ZS +2.4; CutMix +0.94; ViLT FT +0.25; VinVL FT +7.46 | B3 | CONFIRMED against Table 4; contradicted by App. D.3 for four of these five | see §5, issue 2 |

### 3.8 Numbers from excluded sources

| Claim | File | Outcome | Note |
|---|---|---|---|
| MaxMatch Table 5: VSE∞ RSUM 468.9, Set Div 474.9 | B2 | NOT CHECKED | Source excluded by B2; the values equal DivE Table 1 (468.9, 474.9), so they are probably re-quoted |
| LightningDOT 60.1/45.8, re-ranked 74.2/57.4 | B2 | 60.1/45.8 CONFIRMED (as reproduced in LoopITR Table 9); 74.2/57.4 NOT CHECKED | excluded source |

---

## 4. Absence-claim falsification log

Claim tested (task wording): *no study compares one model's dual-encoder ranking with its own re-ranked ranking on ECCV
mAP@R / PMRP / CxC.* B2's own wording (§3) is narrower: "its ITM re-ranked ranking".

| # | Query or route | Closest hits | Falsifies? |
|---|---|---|---|
| 1 | WebSearch (extended): `"ECCV Caption" mAP@R re-ranking ITM cross-encoder rerank dual encoder same model image-text retrieval` | ECCV Caption itself; ELIP (re-ranker, no polysemy metric); TempRet and SCOUT (video, person retrieval) | No |
| 2 | WebSearch: `BLIP ITC only versus ITM rerank "ECCV Caption" OR "CxC" OR "PMRP" evaluation` | ECCV Caption, BLIP; HF card Abdrah/scout-eccv-itm-v3 (person retrieval; "eccv" = an ECCV 2026 workshop; reports Val mAP@10 only; says "pure ITM re-ranking alone is worse than ITC alone", no ECCV Caption numbers) | No |
| 3 | WebSearch: `multimodal LLM reranker image-text retrieval "ECCV Caption" evaluation 2025` | MM-Embed, MLLM-reranker papers (2407.21439 and others); none reported ECCV Caption in the search results (full texts not opened) | No |
| 4 | WebSearch: `"CxC" Crisscrossed Captions R@1 dual encoder and cross encoder rerank same model ALBEF X-VLM BLIP results` | CxC, X-VLM, surveys | No |
| 5 | WebSearch: `"mAP@R" "ECCV Caption" BLIP-2 OR "X-VLM" OR ALBEF "without re-ranking" OR "w/o ITM" OR "ITC only"` | ECCV Caption only | No |
| 6 | WebSearch: `"plausible match" PMRP cross-encoder re-ranking improves or hurts image-text retrieval false negatives analysis 2024 2025 arXiv` | Pishdad, PCME, FNE, PMPGuard (remote sensing) | No |
| 7 | Semantic Scholar citations, fetched by me: ECCV Caption (58 citing records), CxC (74), screened by title keywords (rerank, cross-encoder, two-stage, fusion, ITM, distill, uncertainty, false negative, ambiguity, many-to-many) | (a) CPRD, "How to Make Cross Encoder a Good Teacher..." (CVPR 2024, 2407.07479), full LaTeX read; (b) DSSLP, "Dual-stage framework with soft-label distillation..." (PLOS ONE 2025), full HTML read | Near miss (a), no (b); see below |
| 8 | WebSearch for 2407.15239 (Benchmark Granularity, SIGIR 2025) and LoopITR venue | no ECCV/CxC re-ranking content | No |
| 9 | READMEs of naver-ai/eccv-caption and naver-ai/pcmepp | no ITC vs ITM rows | No |
| 10 | Re-read of PCME++ App. C.7 (`tab:inference_distance_comparisons`) | **FAISS mean-only ranking vs "FAISS + σ re-ranking" of the same PCME++ B/32 model: ECCV mAP@R 40.1 vs 40.1, R-P 49.7 vs 49.7, ECCV R@1 83.5 vs 83.2, CxC R@1 56.4 vs 56.6, 5K R@1 54.6 vs 54.8** | **Yes, for the broad wording** |

Near misses in detail:
- **PCME++ Table C.7** reports one model's first-stage ranking (ANN on μ) and its own re-ranked list (top-K re-scored with
  μ distance plus σ) on ECCV mAP@R and CxC R@1. The re-ranker is the model's own uncertainty, not a cross-attention or ITM
  head. Re-ranking changed mAP@R by 0.0 and CxC R@1 by +0.2.
- **CPRD (CVPR 2024)**, CxC table (`tab:image_text_ranking`): Spearman correlation with CxC human image-text similarity
  (SITS) for its dual encoder (61.8; 65.1 after COCO fine-tuning) and for its cross-encoder teacher (67.3; 69.4). The DE
  is a separately trained student (BERT + ViT-B/16) and the CE a separate ALBEF teacher. The metric is a graded
  correlation, not CxC R@1, and there is no top-K re-ranking. Its Sec 3 also compares BLIP's dual encoder with BLIP's
  re-ranked cross encoder, but on COCO R@K only.
- **LoopITR** (CxC R@1 for the dual encoder only) and **CUSA** (X2VLM ECCV mAP@R for the re-ranked model only) each report
  one side of the comparison.
- **DSSLP (PLOS ONE 2025)** reports ECCV mAP@R for CLIP B/32 and L/14@336 fine-tunes, with no re-ranking. Its CLIP
  baseline rows are identical to CUSA's (28.5/39.4/72.5/41.7/50.8/83.0), so they were copied. DSSLP B/32 reaches
  i2t 29.8 and t2i 45.1 (avg 37.45). B2 and B3 both missed it; it does not change any conclusion.

**Verdict.** The narrow claim survives: no study found reports the same model's dual-encoder ranking and its own
cross-attention or ITM re-ranked ranking on ECCV mAP@R, PMRP or CxC R@1. The task's broad wording ("its own re-ranked
ranking") does not survive, because PCME++ Table C.7 is such a comparison, with an uncertainty-based re-ranker that changed
nothing. The synthesis should keep the qualifier "cross-encoder / ITM" and cite C.7 and CPRD as the nearest evidence.
Coverage caveats: US web index; MLLM-reranker papers were screened from search results, not full text; S2 citation lists
are incomplete.

---

## 5. Flagged sources and claims

| # | Source / claim (file) | Issue | Severity | Recommendation |
|---|---|---|---|---|
| 1 | FILIP [R6] (B2) | Ablation misread. Table 4 rows are single additions to the vanilla baseline, so late interaction gives +5.5 I2T / +3.8 T2I R@1 (25.0/14.7 -> 30.5/18.5; Sec 4.4), not +1.3/+0.6. B2's relevance line ("about +1 R@1, smaller than cross-encoder re-ranking") is wrong: the gain is about the size of the re-ranking gains, and it comes with an offline-indexable representation (zero-shot, YFCC subset, B/32) | Medium | Caveat; replace the numbers and the relevance line |
| 2 | ECCV Caption Table 4 PMRP (B2 E1 rows; B3 leaderboard, the "What moved PMRP" paragraph and §4 answers) | The paper contradicts itself. Its Tables D.3 give per-direction PMRP whose means match Table 4 for the 14 ResNet/region models but not for the 11 VLP and re-implemented rows. CLIP B/32: 54.40/50.69 (mean 52.55) vs 55.32; BLIP: 82.32/53.15 (67.73) vs 57.17; PVSE K=1 no NM / semi-hard / hardest: 44.84 / 46.41 / 47.17 vs Table 4's 56.67 / 55.15 / 54.37, a **reversed** trend; ViLT ZS -> FT +3.30 vs +0.25; VinVL ZS -> FT +24.09 vs +7.46. The D.3 i2t "PMRP" values for BLIP and the NM rows sit close to their own 5K i2t R@1, so the appendix column may be mislabelled. v5's changelog says only Table 4's 1K R@1 was fixed. The project reproduces CLIP B/32 zero-shot PMRP at 55.31, which supports Table 4 for that one row only. mAP@R values agree between the two tables | Medium | Include with a caveat. Cite Table 4, but say the paper's appendix disagrees for these rows. Do not build a hypothesis ("PMRP rises when hard-negative pressure falls") on the NM rows alone without re-running them |
| 3 | PCME++ GPO ablation, +2.6 mAP@R (B3 Δ table, §4 "pooling") | The baseline pooling of the "no GPO" row is never stated. The ablation is under the PCME++ probabilistic loss with SizeAugment, not InfoNCE. It is one row pair, and the "1 layer + GPO" row (40.0 / 55.3 / 537.1) differs from the Table 1 PCME++ row (40.1 / 55.1 / 537.0), so run noise is of order 0.1 to 0.2 | Medium | Caveat: "GPO vs an unspecified pooling, PCME++ loss". It does not show that GPO helps a CLIP-pooled InfoNCE fine-tune |
| 4 | VACSR, "learnable vs fixed temperature +23.3" (B3 Δ table, first row) | The learnable-τ row (39.0) is PCME++'s InfoNCE row copied into VACSR's table; the fixed-τ row (15.7) is VACSR's own run. τ fixed at 1 means a logit scale of 1 on cosine similarity, an extreme setting that says little about realistic temperatures. B3 notes the copying for the sigmoid row but not for this one | Low to Medium | Caveat: a cross-source contrast; use it only as "a fixed scale of 1 collapses training" |
| 5 | ALBEF "k = 256 (Table 2)" (B2 R1) | k for COCO is not in the ALBEF paper; BLIP and X-VLM state 256 "following ALBEF" | Low | Re-attribute to BLIP Sec 5.1 / X-VLM |
| 6 | Multiplicity position paper (B3 C3) | Listed as preprint; it is an ICML 2026 Position-track paper | Low | Correct the tier |
| 7 | Sun 2209.02127 and its TMLR companion (B3 B4) | The TMLR 2024 paper has two authors (Zhun Sun, Chao Li) and a different title; the link stays unproven | Low | Cite the arXiv version as read; keep TMLR as "related, unverified" |
| 8 | CLIP4Clip venue (B3 A4) | Journal version exists: Neurocomputing 508 (2022) | Low | Upgrade the tier |
| 9 | Listwise ranking venue (B3 B8) | Venue correct (KBS 2024, 111431) but attributed to Semantic Scholar, whose record shows arXiv only; Crossref confirms | Low | Re-attribute to Crossref |
| 10 | PCME++ "validation 1K RSUM" (B3 A1) | The paper says "validation rsum" without 1K/5K | Low | Drop "1K" |
| 11 | PCME++ "μ only" as a "pairwise sigmoid-style loss" (B3 Δ table) | The paper never defines the μ-only row | Low | Mark as inference or drop the label |
| 12 | ProLIP masking locator (B2 P7) | The 75% is in Sec 4.1, not Sec 3.3 | Low | Fix the locator |
| 13 | DivE K sweep (B2 P4) | Run on Flickr30K with ROI features, not the ResNeXt+BERT model whose COCO numbers are quoted next to it | Low | Add "(ROI features)" |
| 14 | DivE ECCV comparison (B2 P4, B3 B6) | "Our best model" vs the single official VSE∞ checkpoint; ensemble status unknown (both files already flag it) | Low | Keep the caveat |
| 15 | CUSA (B2 S1, B3 B2) | Baseline recipe not in the paper; no seeds or variance. Its CLIP baselines are re-used verbatim by DSSLP (2025) | Low | Keep the caveats |
| 16 | DITM 2505.09997 (B2, B3 exclusions) | Withdrawn by the authors (arXiv v3 comment) | Low | Exclude; do not cite its abstract as evidence |
| 17 | GroVE COCO protocol (B2 P6) | 1K vs 5K not stated in the paper (B2 already flags it) | Low | Keep the caveat |
| 18 | LoopITR (B2 R5) | No venue found; preprint | Low | Include as preprint (B2 already does) |
| 19 | NeighborRetr's ECCV table (B3 A5) | Besides the B/32 vs B/16 inconsistency B3 notes: its PCME++ row (t2i 49.5 / 57.0 / 90.8; i2t 34.4 / 45.0 / 81.3) matches no row of the PCME++ paper | Low | Caveat; B3 did not use that row |

No source was found FABRICATED. No retrieved page contained instructions aimed at the reader.

---

## 6. Limitations of this verification

- Venue claims rest on the S2 batch API (one call succeeded after one 429 retry), arXiv comments, Crossref, and two
  icml.cc poster pages. Proceedings PDFs for ICML 2026 and ICLR 2026 were not opened.
- Table numbers were derived from LaTeX `\input` order and `\numberwithin`. Where a source has floats that could
  reorder (ELIP, MaskVLM, BCLS), I confirmed the numbers at the `\label` and did not certify the printed table number.
- The DSSLP and CPRD papers were found during falsification and are not in either bibliography. They are reported only
  as near misses.
- Downloads used for this check (`scratchpad/verify_tmp2/`) were deleted after writing this file.
