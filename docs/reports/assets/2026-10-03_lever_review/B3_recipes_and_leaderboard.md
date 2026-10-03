> Supporting file for docs/reports/auto/v1/2026-10-03_lever_review.md. Phase 2 bibliographies are as written; V1/V2 list the corrections, and the report uses corrected values.

Erratum (2026-10-03): the "zeta=0" readings below describe the paper's text; the released PM files hold zeta <= 2 matches (verified 2026-10-03 against instances_val2014.json).

# B3: CLIP fine-tuning recipes on COCO, and what has moved the polysemy-aware COCO metrics

Theme agent B3, Phase 2 (investigation), written 2026-10-03. Scope: (a) what a strong CLIP fine-tuning recipe on
COCO looks like; (b) a leaderboard of ECCV Caption mAP@R, R-Precision (R-P), PMRP, CxC R@1 and COCO 5k R@1, and
which changes moved them over a same-recipe baseline; (c) how these metrics disagree, and how valid PMRP is.
No cross-theme synthesis and no project recommendations below; the only project-facing text is the per-source
"Relevance" line, marked [reader-inferred].

Conventions. Unless a row says otherwise, every metric is the mean of image-to-text (i2t) and text-to-image (t2i),
as ECCV Caption and PCME++ report it. Where a paper reports only the two directions, the mean was computed here and
is marked "avg computed". "COCO RSUM" in ECCV Caption and PCME++ is the **1K** RSUM (six recalls on the 5-fold 1K
test, max 600), whereas the MultiMAE brief's "rsum" is the **5k** RSUM; the two are not interchangeable. 5k RSUMs below
were recomputed from the papers' per-direction R@1/5/10 tables and are marked "5k RSUM computed".

## 1. Search strategy

- **Primary route to numbers**: arXiv LaTeX sources (`https://arxiv.org/e-print/<id>`), unpacked and read at the
  table source (ar5iv timed out from this container on 2026-10-03; LaTeX tables are the exact printed numbers).
  Table numbers were derived from the order of `\input{tables/...}` in the main file; where that was ambiguous the
  LaTeX label is given.
- **Discovery**:
  - Semantic Scholar Graph API, citations of ECCV Caption (arXiv 2204.03359): 58 records returned (51 from 2023 on).
  - Semantic Scholar Graph API, citations of PCME (arXiv 2101.05068): 321 records returned (259 from 2023 on),
    screened by title for image-text retrieval work that could report ECCV Caption or PMRP.
  - WebSearch queries (2026-10-03): `"ECCV Caption" mAP@R image-text matching CLIP fine-tuned COCO 2024`;
    `PMRP "ECCV Caption" noisy correspondence image-text matching mAP@R R-Precision results`;
    `"PMRP" image-text retrieval 2025 OR 2026 arXiv CLIP fine-tuning COCO "plausible match"`;
    `fine-tuning CLIP COCO retrieval batch size ablation in-batch negatives effect recall image-text matching`;
    `CLIP ViT-B/16 fine-tuned MSCOCO 5K retrieval baseline learning rate epochs ...`;
    `"Analyzing the Impact of Learnable Softmax Temperature ..." TMLR` (venue check).
  - Direct checks of named recipe papers: NegCLIP 2210.01936, FLYP 2212.00638, WiSE-FT 2109.01903, CLIP4Clip
    2104.08860, PCME++ 2305.18171, ProLIP 2410.18857.
- **Date range**: 2019 to 2026-10; last searched 2026-10-03. Active search for 2024, 2025 and 2026 work (the
  citation lists above run to arXiv 2609.x).
- **Inclusion**: (i) a dual-encoder (preferably CLIP) fine-tuned on COCO with a stated recipe and COCO retrieval
  numbers; or (ii) ECCV Caption mAP@R / R-P or PMRP reported together with a baseline inside the same paper; or
  (iii) an analysis of agreement between ECCV mAP@R, PMRP, CxC and COCO R@K, or of PMRP validity.
- **Exclusion**: no COCO retrieval numbers (FLYP: classification only, checked full LaTeX; WiSE-FT: classification
  only, checked full LaTeX; ProLIP itself reports no ECCV Caption or PMRP numbers, checked full LaTeX); no
  ECCV-Caption mAP@R or PMRP results in the LaTeX (24 papers from 2022 to 2026, including MAFA 2312.06112 and MAP
  2210.05335, which mention ECCV Caption only in text; plus 2304.03391, whose MAP@R is on its own benchmark);
  source not retrievable as LaTeX or HTML (DITM 2505.09997: abstract says it evaluates on CxC, not ECCV Caption).
- **Counts**: identified about 360 records (58 + 259 citation records, about 45 web results, overlapping);
  screened by title/abstract about 360; full LaTeX source fetched and grepped for "mAP@R"/"PMRP"/"ECCV Caption":
  44 papers (plus 1 wrong-id download discarded); full text read at the relevant sections: 16 (plus the ITRA web
  page); included: 16 documents in 15 source entries (B8 holds two companion papers; C1 re-reads B1).
- **Coverage skew**: (1) PMRP is almost absent after 2022: of the 10 screened papers after ECCV Caption that report
  ECCV Caption mAP@R (PCME++, VACSR, CUSA, NeighborRetr, Sun, LongProLIP, DivE, AAHR, listwise, BCLS), none reports
  ECCV-style PMRP. The PMRP evidence therefore rests mostly on ECCV Caption itself.
  (2) Most 2023 to 2026 papers that report mAP@R are region-feature or probabilistic ITM papers; few fine-tune
  plain CLIP. (3) The batch-size evidence for COCO fine-tuning comes from one non-peer-reviewed documentation page
  (ResNet-50 CLIP) and one video-retrieval paper; no peer-reviewed ViT-B CLIP batch-size ablation on COCO was found.

## 2. Leaderboard (the core deliverable)

All rows are COCO 5k test (Karpathy) or its ECCV Caption / CxC re-annotations. "FT" = fine-tuned on COCO train.
"ZS" = no COCO training. "-" = not reported. Source locators point to the table the number was read from.

| Method | Backbone (image / text) | FT? | ECCV mAP@R | ECCV R-P | PMRP | CxC R@1 | COCO 5k R@1 | Source locator |
|---|---|---|---|---|---|---|---|---|
| CLIP | ViT-B/32 / CLIP text | ZS | 26.75 | 36.91 | 55.32 | 41.97 | 40.28 | ECCV Caption Table 4 |
| CLIP | ViT-B/16 / CLIP text | ZS | 29.25 | 38.99 | 56.58 | 44.26 | 42.69 | ECCV Caption Table 4 |
| CLIP | ViT-L/14 / CLIP text | ZS | 27.98 | 37.80 | 57.70 | 48.14 | 46.44 | ECCV Caption Table 4 |
| ViLT | ViT-B/32 single-stream | ZS / FT | 26.84 / 34.58 | 36.81 / 44.27 | 57.38 / 57.63 | 50.35 / 53.72 | 48.63 / 52.18 | ECCV Caption Table 4 |
| VinVL | region, single-stream | ZS / FT | 22.18 / 40.81 | 32.93 / 49.55 | 47.26 / 54.72 | 33.74 / 67.76 | 32.07 / 66.39 | ECCV Caption Table 4 |
| BLIP | ViT, ITM re-ranking (not efficient retrieval) | FT | 40.52 | 48.43 | 57.17 | 74.30 | 73.11 | ECCV Caption Table 4 |
| VSE++ (re-impl.) | ResNet-152 / Bi-GRU | FT | 35.01 | 45.50 | 54.26 | 37.95 | 35.79 | ECCV Caption Table 4 |
| PVSE K=1 / K=2 | ResNet-152 / Bi-GRU | FT | 33.98 / 40.26 | 44.49 / 49.92 | 53.56 / 55.52 | 38.38 / 40.18 | 36.20 / 38.13 | ECCV Caption Table 4 |
| PVSE K=1, no NM / semi-hard NM / hardest NM (re-impl.) | ResNet-152 / Bi-GRU | FT | 33.34 / 36.63 / 35.76 | 44.44 / 47.36 / 46.50 | 56.67 / 55.15 / 54.37 | 32.69 / 38.17 / 39.02 | 30.65 / 36.00 / 36.88 | ECCV Caption Table 4 (bottom block) |
| PCME (official) / PCME CutMix-pretrained (re-impl.) | ResNet-152 / Bi-GRU | FT | 37.11 / 41.74 | 47.82 / 51.45 | 56.71 / 57.65 | 40.09 / 41.70 | 38.03 / 39.51 | ECCV Caption Table 4 |
| VSRN | BUTD region / Bi-GRU | FT | 42.28 | 51.84 | 55.44 | 48.85 | 46.74 | ECCV Caption Table 4 |
| VSE∞ BUTD region / BUTD grid / WSL grid | region or grid features (text encoder not given in Table 4) | FT | 40.46 / 40.40 / 42.41 | 49.97 / 50.09 / 51.43 | 56.64 / 56.87 / 57.65 | 52.40 / 53.47 / 60.79 | 50.38 / 51.60 / 59.01 | ECCV Caption Table 4 |
| InfoNCE (CLIP init) | ViT-B/32 / CLIP text, GPO pooling | FT | 39.0 | 48.7 | - | 54.9 | 53.0 | PCME++ Table 1 (3-run mean) |
| VSE∞ (CLIP init) | ViT-B/32 | FT | 40.0 | 49.5 | - | 57.1 | 55.2 | PCME++ Table 1 |
| PCME (CLIP init) | ViT-B/32 | FT | 39.1 | 48.9 | - | 54.7 | 53.0 | PCME++ Table 1 |
| PCME++ (μ only) / PCME++ / PCME++ SWA | ViT-B/32 | FT | 39.5 / 40.1 / 40.2 | 49.1 / 49.7 / 49.8 | - | 57.0 / 56.8 / 56.8 | 55.2 / 55.1 / 55.2 | PCME++ Table 1 |
| InfoNCE / VSE∞ / PCME++ (CLIP init) | ViT-B/16 | FT | 41.1 / 41.7 / 42.1 | 50.4 / 50.6 / 51.2 | - | 60.9 / 62.3 / 62.6 | 59.3 / 60.7 / 61.1 | PCME++ Table 1 |
| InfoNCE / PCME / PCME++ (CLIP init) | ViT-L/14 | FT | 35.6 / 41.2 / 42.1 | 45.8 / 50.3 / 50.8 | - | 48.0 / 63.4 / 65.9 | 45.9 / 61.9 / 64.3 | PCME++ Table 1 |
| Sigmoid pairwise loss, learnable a,b (VACSR's baseline) | ViT-B/32, PCME++ framework | FT | 39.3 | 48.7 | - | 57.3 | 55.6 | VACSR tab:loss and tab:ablation |
| InfoNCE with temperature fixed at 1 | ViT-B/32, PCME++ framework | FT | 15.7 | 25.5 | - | 16.4 | 14.8 | VACSR tab:loss |
| VACSR | ViT-B/32 / B/16, PCME++ framework | FT | 40.7 (B/32) | 50.1 | - | 59.7 | 58.1 | VACSR tab:loss, tab:ablation |
| CLIP InfoNCE fine-tune / + CUSA | ViT-B/32 | FT | 35.1 / 37.4 (avg computed) | 45.1 / 47.15 (avg computed) | - | - | 49.55 / 50.75 (avg computed) | CUSA tab:itr_eccv, tab:itr_mscoco_flickr |
| CLIP InfoNCE fine-tune / + CUSA | ViT-L/14@336 | FT | 39.15 / 40.6 (avg computed) | 48.8 / 49.95 (avg computed) | - | - | 59.35 / 60.15 (avg computed) | CUSA tab:itr_eccv, tab:itr_mscoco_flickr |
| NeighborRetr | CLIP ViT-B/32 (main text) or B/16 (supplement) | FT | 42.1 (avg computed) | 51.65 (avg computed) | - | - | - | NeighborRetr tab:comp_eccv_caption |
| CLIP re-implementation / multi-token oblique (from scratch) | ViT-B/16, pretrained from scratch on a public mix that includes COCO train | pretrain | 35.5 / 36.3 (avg computed) | 45.45 / 46.1 (avg computed) | - | 55.1 / 55.05 (avg computed) | 53.65 / 53.35 (avg computed) | Sun, appendix tab:clip_eccv |
| ProLIP (pre-trained) / LongProLIP (ShareGPT4V 24M) | ViT-B/16, DataComp 12.8B seen | ZS | 34.1 / 35.7 | - | - | - | - | LongProLIP tab:longprolip_main |
| VSE∞ WSL (official) / DivE | ResNeXt-101 WSL / BERT | FT | 42.4 / 43.5 (avg computed) | 51.45 / 52.45 (avg computed) | - | 60.8 / 63.9 (avg computed) | - | DivE supp. tab:supp_eccv |
| AAHR w/o PGA / AAHR | BUTD region + BERT + frozen CLIP B/32 global | FT | 41.5 / 41.95 (avg computed) | 50.8 / 51.1 (avg computed) | - | - | - | AAHR tab:comparisonECCV |
| VSE++ triplet / + S-NDCG listwise | BUTD region / GRU | FT | 36.55 / 37.55 (avg computed) | 46.85 / 47.62 (avg computed) | - | - | - | Listwise tab:ablation_eccv |
| VSE++ / + BCLS | BUTD region / GRU | FT | 29.55 / 30.5 (avg computed) | 40.45 / 41.4 (avg computed) | - | - | - | BCLS tab:ablationeccv |
| [context, from 00_brief.md, not literature] MultiMAE contrastive FT / multilearner | ViT-B/32 / CLIP text, CLIP pooled | FT | 36.94 / 36.95 | - | 56.49 / 56.88 | - | - | brief (3 seeds) |

Notes on the table.
- The MultiMAE zero-shot measurement in the brief (mAP@R 26.72, PMRP 55.32) matches ECCV Caption Table 4 (26.75,
  55.32); the 5k RSUM of zero-shot CLIP B/32 recomputed from ECCV Caption Appendix D.3 (i2t 50.14/75.00/83.42,
  t2i 30.42/55.96/66.89) is 361.83, against 361.98 in the brief.
- 5k RSUM computed from PCME++ appendix tables (3-run means): ViT-B/32 InfoNCE 444.1, VSE∞ 447.1, PCME 443.7,
  PCME++ 452.1; ViT-B/16 InfoNCE 470.3, VSE∞ 474.6, PCME++ 475.8. CUSA's CLIP B/32 fine-tune: 422.6 (as printed;
  422.5 recomputed). VACSR B/32: 462.5 (computed). ITRA ResNet-50 CLIP fine-tune (bs 256): 443.85 (computed).
- NegCLIP (no ECCV metrics) is absent from the table; its COCO R@1 are in entry A2.

### Changes with a same-recipe (or same-paper) baseline, sorted by size of the ECCV mAP@R change

| Change (kind) | Setting | Baseline -> new mAP@R (Δ) | Other metrics | Source |
|---|---|---|---|---|
| Learnable vs fixed InfoNCE temperature (τ fixed at 1) | CLIP B/32, PCME++ framework | 15.7 -> 39.0 (+23.3) | 5k R@1 14.8 -> 53.0 | VACSR tab:loss |
| PCME++ loss vs InfoNCE at ViT-L/14 (InfoNCE unstable) | CLIP L/14 FT | 35.6 -> 42.1 (+6.5) | 5k R@1 45.9 -> 64.3 | PCME++ Table 1 |
| Multiple embeddings, K=1 -> K=2 | PVSE, official checkpoints | 33.98 -> 40.26 (+6.3) | PMRP 53.56 -> 55.52; 5k R@1 36.20 -> 38.13 | ECCV Caption Table 4 |
| CutMix-pretrained backbone | PCME (re-impl. vs official) | 37.11 -> 41.74 (+4.6) | PMRP 56.71 -> 57.65 | ECCV Caption Table 4 |
| Negative mining: none -> semi-hard | PVSE K=1 re-impl. | 33.34 -> 36.63 (+3.3) | PMRP 56.67 -> 55.15 (-1.5); 5k R@1 30.65 -> 36.00 | ECCV Caption Table 4 |
| GPO pooling vs no GPO (same PCME++ loss) | CLIP B/32 FT | 37.4 -> 40.0 (+2.6) | 5k R@1 49.2 -> 55.3; 1K RSUM 521.8 -> 537.1 | PCME++ appendix tab:arch_abl |
| Negative mining: none -> hardest | PVSE K=1 re-impl. | 33.34 -> 35.76 (+2.4) | PMRP 56.67 -> 54.37 (-2.3) | ECCV Caption Table 4 |
| Soft labels from unimodal teachers (CUSA) | CLIP B/32 InfoNCE FT | 35.1 -> 37.4 (+2.3) | t2i alone 41.7 -> 45.2; 5k RSUM 422.6 -> 429.7 | CUSA tab:itr_eccv |
| CUSA | CLIP L/14@336 FT | 39.15 -> 40.6 (+1.45) | | CUSA tab:itr_eccv |
| VACSR (variational similarity adapter) vs sigmoid baseline | CLIP B/32, PCME++ framework | 39.3 -> 40.7 (+1.4) | 5k R@1 55.6 -> 58.1 | VACSR tab:ablation |
| VIB + pseudo-positives + MSDA (PCME++ optimization) | CLIP B/32, PCME++ loss | 38.9 -> 40.1 (+1.2) | 1K RSUM 535.9 -> 537.0 | PCME++ Table 3 |
| PCME++ vs InfoNCE | CLIP B/32 FT | 39.0 -> 40.1 (+1.1) | 5k RSUM 444.1 -> 452.1 (computed) | PCME++ Table 1 |
| Set of diverse embeddings (DivE) vs VSE∞ (official) | ResNeXt WSL + BERT | 42.4 -> 43.5 (+1.1) | CxC 60.8 -> 63.9 | DivE supp. |
| PCME++ vs InfoNCE | CLIP B/16 FT | 41.1 -> 42.1 (+1.0) | 5k RSUM 470.3 -> 475.8 (computed) | PCME++ Table 1 |
| Listwise S-NDCG loss on caption-similarity relevance | VSE++ region | 36.55 -> 37.55 (+1.0) | | Listwise tab:ablation_eccv |
| Kendall ranking loss on continuous pseudo labels (BCLS) | VSE++ region | 29.55 -> 30.5 (+0.95) | | BCLS tab:ablationeccv |
| Multi-token oblique embedding vs sphere | B/16 from scratch | 35.5 -> 36.3 (+0.8) | 5k R@1 53.65 -> 53.35 (down) | Sun tab:clip_eccv |
| VIB + PP only | CLIP B/32 | 38.9 -> 39.6 (+0.7) | | PCME++ Table 3 |
| Pairwise sigmoid-style loss (PCME++ μ only) vs InfoNCE | CLIP B/32 FT | 39.0 -> 39.5 (+0.5) | CxC 54.9 -> 57.0 | PCME++ Table 1 |
| VIB alone | CLIP B/32 | 38.9 -> 39.3 (+0.4) | | PCME++ Table 3 |
| Prototype module (PGA) | AAHR | 41.5 -> 41.95 (+0.45) | | AAHR tab:comparisonECCV |
| Sigmoid pairwise loss vs InfoNCE (learnable scales) | CLIP B/32 | 39.0 -> 39.3 (+0.3) | 5k R@1 53.0 -> 55.6 | VACSR tab:loss (InfoNCE row reused from PCME++) |
| SWA over last 10 epochs | PCME++ B/32 | 40.1 -> 40.2 (+0.1) | | PCME++ Table 1 |
| Probabilistic (CSD) vs mean-only inference, same trained model | PCME++ B/32 SWA | 40.2 -> 40.2 (0.0) | 1K RSUM 536.3 -> 537.3 | PCME++ appendix tab:inference_distance_comparisons |

What moved PMRP (ECCV-style PMRP, ζ=0, R capped at 50): only ECCV Caption Table 4 provides comparable,
same-family contrasts: dropping negative mining raised PMRP (PVSE K=1: hardest NM 54.37, semi-hard 55.15, none
56.67) while lowering mAP@R and R@1; larger CLIP backbones raised zero-shot PMRP (B/32 55.32, B/16 56.58, L/14
57.70) although L/14 had lower mAP@R than B/16; CutMix pretraining raised PCME's PMRP by 0.94; fine-tuning raised
ViLT's PMRP by only 0.25 and VinVL's by 7.46. No 2023 to 2026 paper found here reports ECCV-style PMRP.

## 3. Annotated bibliography

### Sub-theme A: CLIP fine-tuning recipes on COCO

**A1. Chun, S. (2024). Improved Probabilistic Image-Text Representations (PCME++).** ICLR 2024 (arXiv 2305.18171,
"ICLR 2024 camera-ready" per arXiv comments).
- Tier: peer-reviewed (ICLR 2024).
- Read scope: full LaTeX: Sec 4 (experiments), Table 1 (tab:main), Table 2 (noise), Table 3 (tab:loss_abl), Table 4
  (probability distance), appendix hyperparameters (tab:hparam and "Experimental Protocol Details"), SWA,
  ablations (tab:arch_abl, MSDA, VIB, α), per-direction full tables with standard errors.
- Setting: fine-tuning OpenAI CLIP ViT-B/32, B/16, L/14 (both towers) on COCO Karpathy train; new modules (GPO
  pooling, log σ² head, 1024-d embedding) randomly initialized. Recipe (appendix): 25 epochs, batch 128, AdamP,
  initial lr 5e-4 decayed ×0.1 for the last 10 epochs, weight decay 1e-4, visual backbone lr multiplier ×0.01
  (that is 5e-6) and text backbone ×0.1 (5e-5), layer-wise lr decay 0.7 per transformer block, visual backbone frozen
  for the first 2 epochs then 1 epoch linear warmup, SizeAugment, model selection on validation 1K RSUM. InfoNCE
  baseline: initial softmax temperature 1.0. Compute: B/32 on 1 V100 for 38 h.
- Change tested and baseline: closed-form probabilistic matching loss (PCME++) plus VIB, pseudo-positives (PP) and
  mixed-sample augmentation (MSDA), against InfoNCE, VSE∞ (triplet + hardest negative), PCME, DAA, P2RM, all
  trained by the author with the same backbone and recipe.
- Effect: see leaderboard and the Δ table. B/32 InfoNCE -> PCME++: mAP@R 39.0 -> 40.1, R-P 48.7 -> 49.7, CxC R@1
  54.9 -> 56.8, 5k R@1 53.0 -> 55.1 (Table 1). GPO vs none: mAP@R 37.4 -> 40.0 (tab:arch_abl). Run-to-run standard
  errors over 3 runs for InfoNCE B/32: i2t mAP@R ±0.1, t2i ±0.5 (tab:appendix_i2t_full, t2i_full).
- Polysemy-aware metrics: yes, ECCV mAP@R, R-P, R@1 and CxC R@1; PMRP not reported.
- Relevance to MultiMAE [reader-inferred]: its InfoNCE B/32 fine-tune reaches 5k RSUM 444.1 (computed) and ECCV mAP@R
  39.0, against the brief's contrastive fine-tune at 441.94 and 36.94, so a similar R@K level coexists with a ~2-point
  mAP@R gap; the recipes differ in pooling (GPO vs CLIP pooled), heads, lr per tower and model selection.
- Method weaknesses:
  - Model selection uses validation RSUM on COCO, whose labels miss many positives; the author proposes SWA because
    of this [author-acknowledged, appendix "SWA and model selection"]. This can bias selection toward R@K over mAP@R.
  - At B/32 the gains are small; the author states that the gap between VSE∞ and PCME++ (SWA) is not significant
    [author-acknowledged, "Limitations and Discussions"].
  - The InfoNCE baseline is not a plain CLIP fine-tune: it inherits GPO, new 1024-d heads, step schedule and τ init 1.0;
    the baseline at ViT-L/14 collapses (mAP@R 35.6) [reader-inferred]. Gains over such a baseline may overstate gains
    over a tuned CLIP fine-tune at large scale.

**A2. Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., & Zou, J. (2023). When and why vision-language
models behave like bags-of-words, and what to do about it? (NegCLIP).** ICLR 2023 (arXiv 2210.01936).
- Tier: peer-reviewed (ICLR 2023, oral per arXiv comments).
- Read scope: full LaTeX: Sec 4 (protocol), Appendix C "Negative Mining" (fine-tuning details, limitations, results
  table, label table:negative-clip-perf, Table 7 by count).
- Setting: fine-tuning OpenAI CLIP ViT-B/32 on COCO train (open_clip code), 5 epochs, lr swept over {1e-5, 5e-6,
  1e-6} and chosen on COCO validation retrieval, 50 warmup steps, AdamW, cosine schedule, batch 1024 on one RTX
  2080 Ti. At 1024 and 5 epochs this is about 2.8k optimizer steps (computed from ~567k pairs), against ~44k for a
  10-epoch run at batch 128.
- Change tested and baseline: composition-aware hard negatives (swapped-word captions, nearest-neighbour images)
  vs CLIP-FT (same recipe without hard negatives) vs zero-shot CLIP.
- Effect (Appendix C table, values as printed, two decimals of a fraction): COCO Image R@1 (t2i) 0.30 ZS -> 0.42
  CLIP-FT -> 0.41 NegCLIP; COCO Text R@1 (i2t) 0.50 -> 0.59 -> 0.56; Flickr30k Image R@1 0.59 -> 0.67 -> 0.67, Text R@1
  0.78 -> 0.83 -> 0.79; ARO VG-Relation 0.59 -> 0.63 -> 0.81, COCO-PRC 0.46 -> 0.36 -> 0.86.
- Polysemy-aware metrics: no.
- Relevance to MultiMAE [reader-inferred]: a short, large-batch CLIP B/32 fine-tune (CLIP-FT) reaches about 59/42 5k R@1,
  a reference point for how far a plain InfoNCE fine-tune gets; hard negatives traded ~3 points of i2t R@1 for
  compositional accuracy.
- Method weaknesses:
  - Batch 1024 was set by one GPU; the authors expect gains from larger batches but did not test them
    [author-acknowledged, Appendix C "Limitations"].
  - The test split for the COCO R@1 is not stated; the zero-shot row (0.50 / 0.30) matches ECCV Caption's 5k numbers
    (50.14 / 30.42), so 5k is likely [reader-inferred]. Single runs, no variance; the chosen lr is not reported
    [reader-inferred].

**A3. ITRA project documentation (n.d.). "Fine-tuning CLIP for MS-COCO Retrieval."** ITRA 0.1 documentation,
https://itra.readthedocs.io/en/latest/Contents/example-usage/clip-finetuning.html (authors not named on the page).
- Tier: non-peer-reviewed documentation (grey literature).
- Read scope: full page (downloaded HTML, all tables).
- Setting: OpenAI-style ResNet-50 CLIP (open_clip builder), fine-tuned on COCO 2017 train (118,287 images) with
  InfoNCE on 8×2080 Ti; baseline 10 epochs, batch 256, lr 1e-5, AdamW, wd 0.5, 100 warmup steps. "Mean Recall" =
  mean of i2t and t2i R@1/5/10 on "mscoco_captions" retrieval (split not named).
- Change tested and baseline: one-factor sweeps from the baseline (73.98 mean recall; zero-shot 58.39).
  - Learning rate at batch 256: 5e-6 72.91, 1e-5 73.98, 2e-5 73.97, 3e-5 73.32, 5e-5 72.46, 1e-4 69.34.
  - Epochs at lr 1e-5: 5 72.66, 10 73.98, 15 74.43, 20 74.45, 30 73.96 ("15-20 epochs seems already reached the
    saturation"). At lr 2e-5: 72.86, 73.97, 74.28, 74.02, 74.03.
  - Batch size with lr scaled linearly: 32 65.04, 64 69.24, 128 72.14 (lr 5e-6), 256 73.98, 512 74.85, 800 74.89.
  - Weight decay 0.01 to 2.5: 73.64 to 74.07 (range ±0.43).
  - Freezing at batch 800 (improved baseline 75.04): lock image, tune text 71.98; lock text, tune image 71.68;
    frozen CLIP + MLP heads 66.71; + linear heads 62.34. Freezing the low image stages and text embeddings allowed
    batch 1792 (lr 7e-5): 76.02.
  - LLRD, EMA and WiSE-FT sections give commands only, no results.
- Effect: as listed. In 5k-RSUM units (×6): batch 128 -> 256 is +11.0, 128 -> 512 is +16.3 (computed).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE [reader-inferred]: the only COCO-scale sweep found that isolates batch size, epochs and lr for a
  CLIP InfoNCE fine-tune; it suggests batch 128 sits on the steep part of the batch curve for ResNet-50 CLIP.
- Method weaknesses [all reader-inferred]: ResNet-50, not ViT; single runs without variance; the batch sweep changes lr
  with batch, so the two are confounded (the batch-128 row uses lr 5e-6, half of 1e-5); the retrieval split and its
  overlap with the COCO 2017 train set are not stated; mean recall only.

**A4. Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., & Li, T. (2021). CLIP4Clip: An Empirical Study of
CLIP for End to End Video Clip Retrieval.** arXiv 2104.08860 (journal version UNVERIFIED).
- Tier: preprint as checked (arXiv record lists no venue).
- Read scope: full LaTeX: Sec 4.2 experimental details, Sec 4.4 hyperparameter study (Figure "fig_study_sdy_fig").
- Setting: fine-tuning CLIP ViT-B/32 for text-video retrieval (MSR-VTT etc.), Adam, cosine schedule, lr 1e-7 for the
  CLIP towers and 1e-4 for new modules, batch 128, 5 epochs.
- Change tested and baseline: batch size, lr, frozen layers, frame count (MSR-VTT).
- Effect: text only, numbers are in a figure: "the performance increases and it achieves comparable result for
  batch size 128 and 256"; fine-tuning all transformer layers is better than freezing.
- Polysemy-aware metrics: no.
- Relevance to MultiMAE [reader-inferred]: origin of the "CLIP4Clip-style" recipe (tiny tower lr, 1e-4 for new modules,
  batch 128) that later image-retrieval papers such as NeighborRetr copy.
- Method weaknesses [reader-inferred]: video domain with small training sets (MSR-VTT 9k videos); batch-size
  findings read from a figure without numbers; lr 1e-7 is tied to Adam on video and may not transfer to 567k COCO
  pairs.

**A5. Lin, Z., Wang, Z., Qian, T., Mu, P., Chan, S., & Bai, C. (2025). NeighborRetr: Balancing Hub Centrality in
Cross-Modal Retrieval.** CVPR 2025 (arXiv 2503.10526).
- Tier: peer-reviewed (CVPR 2025, per Semantic Scholar venue).
- Read scope: full LaTeX: Sec 5 (settings, ECCV table tab:comp_eccv_caption), supplement (COCO table
  tab:itr_combined, implementation details).
- Setting: CLIP initialization, Adam, batch 128, initial lr 1e-4, 10 epochs on MS-COCO. Sec 5 states ViT-B/32; the
  supplement states ViT-B/16 for MS-COCO and Flickr30K.
- Change tested and baseline: hub-aware training (abstract and Sec 4: hubs found by sample centrality get more weight,
  a one-to-many contrastive term separates relevant from irrelevant hubs, and a globally uniform retrieval-probability
  term aligns distributions). The ECCV table compares with
  ViLT, PVSE, PCME, SGRAF, CUSA and PCME++ rows taken from other papers; no same-recipe CLIP baseline is shown.
- Effect: ECCV mAP@R t2i 49.8, i2t 34.4 (avg 42.1 computed); R-P 57.7 / 45.6; R@1 92.1 / 82.1. COCO 5k (as printed,
  column labels as in the source): 69.5 / 90.5 / 95.3 and 53.2 / 79.7 / 88.2, RSUM 476.4.
- Polysemy-aware metrics: ECCV mAP@R, R-P, R@1; no PMRP, no CxC.
- Relevance to MultiMAE [reader-inferred]: uses almost the brief's recipe shape (batch 128, 10 epochs) with a higher lr
  (1e-4, Adam), and reports a high ECCV mAP@R, but without a matched baseline the recipe's own contribution cannot be
  separated from the method's.
- Method weaknesses [reader-inferred]: backbone stated inconsistently (B/32 vs B/16); the CUSA row in the ECCV table
  equals CUSA's ViT-L/14@336 result, so baselines differ in backbone; whether the 1e-4 lr applies to the CLIP towers
  is not stated; the COCO table's "Text-to-Image" column carries values that look like i2t (69.5 R@1).

### Sub-theme B: leaderboard sources and changes that moved ECCV mAP@R or PMRP

**B1. Chun, S., Kim, W., Park, S., Chang, M., & Oh, S. J. (2022). ECCV Caption: Correcting False Negatives by
Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO.** ECCV 2022 (arXiv 2204.03359, LaTeX
of the latest arXiv revision, dated 2024-01).
- Tier: peer-reviewed (ECCV 2022).
- Read scope: full LaTeX: Sec 3 (Table 2 precision/recall of COCO, CxC, PM; Table 3 positive counts), Sec 4 (Table
  4 main results, Table 5 Kendall τ between metrics, human study), Sec 5 (limitations), Appendix D.3 (per-direction
  tables), appendix bias analysis and model-similarity tables.
- Setting: re-evaluation of 25 released or re-implemented models on COCO 5k test with ECCV Caption (1,333 caption
  and 1,261 image queries, subsampled; human-verified positives from the top-5 of five machine annotators: PVSE,
  VSRN, PCME, ViLT, CLIP ViT-B/32), CxC, COCO 1K/5K and PMRP (ζ=0, R capped at min(R, 50)).
- Change tested and baseline: no new method; within-paper contrasts include negative mining (none, semi-hard,
  hardest) on a re-implemented PVSE K=1, PVSE K=1 vs K=2, PCME vs CutMix-pretrained PCME, zero-shot vs fine-tuned
  ViLT and VinVL, and CLIP backbone size.
- Effect: leaderboard rows above (Table 4). Negative mining: mAP@R 33.34 (none) -> 36.63 (semi-hard) -> 35.76
  (hardest) while PMRP 56.67 -> 55.15 -> 54.37 and 5k R@1 30.65 -> 36.00 -> 36.88. The paper notes that hardest
  negative mining helps R@1 more than mAP@R, and that "contrastive models without a negative mining strategy are
  specialized to PMRP" (Sec 4.2).
- Polysemy-aware metrics: yes (defines ECCV mAP@R and R-P; reports PMRP, CxC).
- Relevance to MultiMAE [reader-inferred]: it gives the zero-shot anchor the brief reproduces (26.75 / 55.32), and its
  own data show PMRP rising when hard-negative pressure falls, a direction that matters when reading a PMRP gain that
  comes without an mAP@R gain.
- Method weaknesses:
  - Positives come only from the top-5 lists of five machine annotators, one of them CLIP ViT-B/32; models similar to
    the annotators can be favoured [author-acknowledged, Sec 5 "Potential machine biases", appendix bias analysis].
    The authors report that only PVSE, PCME and VSRN rank better on mAP@R than on COCO R@1 [author-acknowledged,
    appendix].
  - Query subsampling (5.3% of caption queries, 25.2% of image queries) [author-acknowledged, Sec 5 "Scale"]; with
    about 1.3k queries per direction, sub-point differences carry sampling noise [reader-inferred].
  - "Partially YES" answers count as positives, so some positives are only plausible [author-acknowledged, Sec 5
    "Noisy annotations"].

**B2. Huang, H., Nie, Z., Wang, Z., & Shang, Z. (2024). Cross-Modal and Uni-Modal Soft-Label Alignment for
Image-Text Retrieval (CUSA).** AAAI 2024 (arXiv 2403.05261; code repo named aaai24_itr_cusa).
- Tier: peer-reviewed (AAAI 2024).
- Read scope: full LaTeX main paper: method, tab:itr_mscoco_flickr, tab:itr_eccv, tab:ablation_clip,
  implementation details (training hyperparameters are deferred to a GitHub appendix that was not read).
- Setting: fine-tuning CLIP ViT-B/32 and ViT-L/14@336 on COCO with InfoNCE ("All CLIP model reports are based on
  fine-tuned results using InfoNCE"); also SGRAF and X2VLM.
- Change tested and baseline: soft targets from frozen unimodal teachers (Unicom for images, Sentence-BERT for
  text) for cross-modal (CSA) and unimodal (USA) alignment, against the same model fine-tuned with InfoNCE alone.
- Effect: CLIP B/32: ECCV mAP@R i2t 28.5 -> 29.6, t2i 41.7 -> 45.2 (avg 35.1 -> 37.4 computed); R-P avg 45.1 -> 47.15;
  COCO 5k RSUM 422.6 -> 429.7. L/14@336: mAP@R avg 39.15 -> 40.6; 5k RSUM 469.6 -> 473.1. Ablation (B/32): CSA alone
  raises the ECCV average of six metrics 52.6 -> 54.5, USA alone 52.6 -> 53.6 (tab:ablation_clip).
- Polysemy-aware metrics: ECCV mAP@R, R-P, R@1 per direction; no PMRP, no CxC.
- Relevance to MultiMAE [reader-inferred]: one of the few CLIP fine-tunes with a same-recipe InfoNCE baseline on ECCV
  Caption; the gain comes mainly from t2i and from external similarity structure injected as soft labels.
- Method weaknesses [reader-inferred]: external teachers add knowledge beyond COCO, so the gain is not a pure
  objective effect; the CLIP B/32 InfoNCE baseline (5k RSUM 422.6) is weaker than PCME++'s (444.1), leaving more
  headroom; single runs; recipe not in the paper.

**B3. Wei, W., Gui, Z., Peng, D., Ye, T., & Wu, H. (2026). Variational Adapter for Cross-modal Similarity
Representation (VACSR).** ICML 2026 per arXiv comments (arXiv 2605.30968).
- Tier: peer-reviewed per arXiv comment "Accepted by ... ICML 2026" (proceedings not checked); flag preprint>=2024
  for the arXiv text read.
- Read scope: full LaTeX: Sec 4 (comparison and ablation), tables tab:ablation, tab:loss (contrastive vs sigmoid),
  ECCV/CxC comparison, appendix implementation details and sensitivity analysis.
- Setting: PCME++ framework (CLIP ViT-B/32 and B/16, GPO, AdamP, 25 epochs, lr 5e-4 decayed to 10% at epoch 15),
  with the variance head replaced by a two-layer MLP adapter that models a d-dimensional similarity vector.
- Change tested and baseline: variational similarity adapter with KL and σ losses and a GMM prior, against its own
  sigmoid-loss baseline ("w/o All Components") and against PCME++ numbers copied from PCME++.
- Effect: mAP@R 39.3 (sigmoid baseline) -> 40.7; R-P 48.7 -> 50.1; 5k R@1 55.6 -> 58.1; vs PCME++ 40.1. Loss table:
  InfoNCE with learnable τ (init 1) 39.0, InfoNCE with τ fixed at 1 15.7; sigmoid with learnable a,b 39.3, with a,b
  fixed 0.2. Removing the KL or σ loss raised COCO and CxC R@1 slightly while lowering mAP@R (tab:ablation).
- Polysemy-aware metrics: ECCV mAP@R, R-P, R@1; CxC R@1; no PMRP.
- Relevance to MultiMAE [reader-inferred]: a 2026 data point that a change can raise mAP@R by 1.4 within the PCME++
  recipe; the fixed-temperature collapse shows how strongly the similarity scale controls this benchmark.
- Method weaknesses:
  - Removing the uncertainty losses raises R@1 but lowers mAP@R; the authors read this as a metric-sensitivity effect
    [author-acknowledged, Sec 4 "Metric Sensitivity under Uncertainty Modeling"].
  - The InfoNCE row (39.0 / 48.7 / 81.7 / 54.9 / 74.0 / 53.0 / 532.6) is identical to PCME++'s Table 1 InfoNCE row, and
    the comparison table reuses PCME++'s rows, so baselines were copied, not re-run; no variance is reported
    [reader-inferred].

**B4. Sun, Z. (2022, revised). Design of the topology for contrastive visual-textual alignment.** arXiv 2209.02127
(TMLR template; a TMLR 2024 paper by the same sole author, "Analyzing the Impact of Learnable SoftMax Temperature
in Contrastive Visual-Textual Alignment Systems", appears to be its published version: UNVERIFIED link).
- Tier: preprint as read (possible TMLR 2024 version, UNVERIFIED).
- Read scope: full LaTeX: appendix ECCV table (tab:clip_eccv), hyperparameter appendix, Sec 4 datasets.
- Setting: CLIP-style pretraining from scratch, ViT-B/16, batch 32,768, 32 epochs, on a public mix that includes
  COCO train (Sec 4 "Datasets"); evaluated without COCO fine-tuning.
- Change tested and baseline: multi-token oblique-manifold embedding (several class tokens) against the author's
  own spherical CLIP re-implementation with the same data and schedule.
- Effect: ECCV mAP@R i2t 30.5 -> 30.9, t2i 40.5 -> 41.7 (avg 35.5 -> 36.3 computed); R-P avg 45.45 -> 46.1; COCO 5k R@1
  avg 53.65 -> 53.35 (down); CxC R@1 avg 55.1 -> 55.05. The same table reports OpenAI CLIP B/16 at mAP@R 23.7 / 34.8
  (avg 29.25, matching ECCV Caption Table 4).
- Polysemy-aware metrics: ECCV mAP@R, R-P, R@1, CxC R@1; no PMRP.
- Relevance to MultiMAE [reader-inferred]: an embedding-geometry change that raised mAP@R by 0.8 while COCO R@1 fell,
  i.e. the two metrics moved in opposite directions under one change.
- Method weaknesses: the oblique model's maximum temperature was reduced to 3.95 while the sphere kept the default,
  so geometry and temperature are confounded [author-acknowledged, appendix hyperparameters]; COCO train is in the
  pretraining mix, so "zero-shot" COCO numbers are in-domain [reader-inferred]; single comparison, no variance
  [reader-inferred].

**B5. Chun, S., & Yun, S. (2025). LongProLIP: A Probabilistic Vision-Language Model with Long Context Text.**
ICLR 2025 workshop tiny paper (workshop "Quantify Uncertainty and Hallucination in Foundation Models", per arXiv
comments; arXiv 2503.08048).
- Tier: workshop paper; preprint>=2024.
- Read scope: full LaTeX (short paper): tab:longprolip_main, appendix tab:overview.
- Setting: ProLIP ViT-B/16 pretrained on DataComp 1B (12.8B seen samples), then fine-tuned for long captions
  (ShareGPT4V, optionally with HYPE + DFN filtered DataComp); zero-shot evaluation.
- Change tested and baseline: long-context fine-tuning data mixes vs the pretrained ProLIP.
- Effect: ECCV mAP@R avg 34.1 (pretrained; i2t 28.9, t2i 39.2) -> 35.7 (ShareGPT4V, 24M seen) -> 33.4 (ShareGPT4V,
  128M seen) -> 34.6 (ShareGPT4V + HYPE + DFN, 128M).
- Polysemy-aware metrics: ECCV mAP@R only.
- Relevance to MultiMAE [reader-inferred]: a zero-shot probabilistic B/16 model already reaches 34.1 mAP@R without
  COCO training, about 5 points above zero-shot CLIP B/16 (29.25).
- Method weaknesses [reader-inferred]: zero-shot only; the base models differ from CLIP in data (DataComp) and in the
  probabilistic objective, so the 34.1 vs 29.25 gap mixes both; no variance.

**B6. Kim, D., Kim, N., & Kwak, S. (2023). Improving Cross-Modal Retrieval with Set of Diverse Embeddings (DivE).**
CVPR 2023 (arXiv 2211.16761).
- Tier: peer-reviewed (CVPR 2023).
- Read scope: LaTeX: supplementary ECCV/CxC section and table (tab:supp_eccv); backbone description in Sec 4.
- Setting: a set of embeddings per sample from a set-prediction module, matched with a smooth-Chamfer similarity,
  on ResNeXt-101 (Instagram-pretrained) features + BERT, fine-tuned on COCO.
- Change tested and baseline: set of diverse embeddings vs the official VSE∞ (WSL) checkpoint (not a same-recipe
  re-run).
- Effect: ECCV mAP@R i2t 34.8 -> 36.0, t2i 50.0 -> 51.0 (avg 42.4 -> 43.5 computed); CxC R@1 avg 60.8 -> 63.9.
- Polysemy-aware metrics: ECCV mAP@R, R-P, R@1; CxC R@1; no PMRP.
- Relevance to MultiMAE [reader-inferred]: multi-embedding (one-to-many) representations again move mAP@R by about 1,
  as PVSE K=2 did in ECCV Caption.
- Method weaknesses: VSRN, a comparison row, is one of ECCV Caption's machine annotators, which may inflate its
  scores [author-acknowledged, supp. ECCV section]; whether the reported "best model" is a single model or an
  ensemble is not stated in that section [reader-inferred].

**B7. Chen, J., Gao, Y., Ge, M., & Li, M. (2025). Ambiguity-Aware and High-Order Relation Learning for
Multi-Grained Image-Text Matching (AAHR).** Knowledge-Based Systems 316, 113355 (arXiv 2507.09256).
- Tier: peer-reviewed (journal).
- Read scope: LaTeX: ECCV table (tab:comparisonECCV), implementation details, discussion of CLIP use.
- Setting: Faster R-CNN region features + BERT, plus global features from a frozen CLIP ViT-B/32; AdamW lr 5e-4,
  batch 256 on COCO.
- Change tested and baseline: prototype-guided alignment (PGA) and other modules; ablation AAHR w/o PGA.
- Effect: ECCV mAP@R avg 41.5 -> 41.95 (computed) with PGA; its zero-shot CLIP B/32 row (i2t 22.4, t2i 31.1 mAP@R)
  matches ECCV Caption Appendix D.3.
- Polysemy-aware metrics: ECCV mAP@R, R-P, R@1; no PMRP.
- Relevance to MultiMAE [reader-inferred]: shows that prototype/clustering-style modules on top of an embedding gave
  under 0.5 points of mAP@R here.
- Method weaknesses [reader-inferred]: comparison rows are other papers' numbers; single runs; the ablation for PGA
  on ECCV comes from the main comparison table, not a dedicated controlled run.

**B8. Li, Z., Guo, C., Wang, X., Feng, Z., & Wang, Y. (2023/2024). Integrating Listwise Ranking into
Pairwise-based Image-Text Retrieval.** Knowledge-Based Systems 2024 per Semantic Scholar (publisher record not
checked); arXiv 2305.16566. **Companion: Li, Z., Guo, C., Feng, Z., Hwang, J.-N., Jin, Y., & Zhang, Y. (2022).
Image-Text Retrieval with Binary and Continuous Label Supervision (BCLS).** arXiv 2210.11319 (venue UNVERIFIED).
- Tier: peer-reviewed (listwise, per Semantic Scholar) and preprint (BCLS).
- Read scope: LaTeX: ECCV ablation tables in both papers (ablation_eccv; ablationeccv, eccv), method summaries.
- Setting: VSE++ on Faster R-CNN region features, fine-tuned on COCO; relevance degrees (continuous pseudo labels)
  derived from caption-caption semantic similarity.
- Change tested and baseline: adding a listwise smooth-NDCG loss (or a Kendall ranking loss in BCLS) over graded
  relevance to the triplet loss, against triplet only.
- Effect: listwise: mAP@R i2t 26.80 -> 27.75, t2i 46.29 -> 47.35 (avg 36.55 -> 37.55 computed). BCLS: i2t 20.8 -> 21.8,
  t2i 38.3 -> 39.2 (avg 29.55 -> 30.5 computed). S-NDCG alone (no triplet) fell to avg 31.6.
- Polysemy-aware metrics: ECCV mAP@R, R-P, R@1; no PMRP.
- Relevance to MultiMAE [reader-inferred]: graded, text-similarity-derived relevance targets moved mAP@R by about 1
  on a weak backbone.
- Method weaknesses: BCLS notes that Kendall τ against its own pseudo labels only measures fit to pseudo labels
  [author-acknowledged, BCLS results section]; non-CLIP, region features; single runs [reader-inferred].

### Sub-theme C: metric disagreement and PMRP validity

**C1. ECCV Caption (B1), metric-analysis parts.** Read scope as in B1.
- Kendall τ-b between metric rankings over 25 models (Table 5): ECCV mAP@R vs COCO 1K R@1 0.47, vs COCO 5K R@1 0.39,
  vs CxC R@1 0.39, vs RSUM 0.52, vs ECCV R-P 0.90, vs PMRP 0.20; PMRP vs COCO 5K R@1 0.45, vs RSUM 0.43, vs ECCV R@1
  0.29, vs ECCV R-P 0.17; CxC R@1 vs COCO 5K R@1 1.00; COCO 1K R@1 vs COCO 5K R@5 0.97.
- PM pseudo-positive quality (Table 2, ζ=0): precision 65.3 (i2t) / 56.6 (t2i), recall 56.5 / 61.8; original COCO
  precision 96.9 / 96.4 but recall 21.1 / 13.8; CxC precision 95.5 / 93.2, recall 23.0 / 16.1. ECCV Caption has ×8.47
  positive images and ×3.58 positive captions vs COCO on its query subset (Table 3).
- Human study (Sec 4.1): 3,200 pairwise preferences over five synthetic ranking patterns; Bradley-Terry scores
  ordered exactly as mAP@R ranks them (70.85, 13.15, 10.66, 4.89, 0.44), unlike R@1 and R@5.
- The authors call PMRP "a noisy metric compared to others" (Sec 2) and explain its different behaviour as PMRP
  capturing only object presence or absence (Sec 4.2).
- Weaknesses: τ is computed across 25 heterogeneous models (region, grid, VLP), so it says how metrics rank families
  of models, not how they rank close variants of one recipe [reader-inferred]; the human study uses synthetic ranking
  patterns, not model outputs [reader-inferred].

**C2. Pishdad, L., Zhang, R., Derpanis, K. G., Jepson, A., & Fazly, A. (2022). Uncertainty-based Cross-Modal
Retrieval with Probabilistic Representations.** arXiv 2204.09268 (venue not stated in the source).
- Tier: preprint.
- Read scope: LaTeX results section: metric definitions and COCO table (tab:results_coco).
- Setting: probabilistic VSE++ and VSRN (region features, non-CLIP) on COCO.
- Change tested and baseline: probabilistic embeddings vs deterministic VSE++/VSRN.
- Effect: PMRP (5k, i2t) VSE++ 29.1 (higher-capacity model retrained with the official code) -> 31.3, VSRN 27.7 (same
  kind of baseline) -> 34.2; PCME 34.1, PVSE 31.8 (tab:results_coco).
- Polysemy-aware metrics: PMRP in PCME's original form (mean over ζ ∈ {0, 1, 2}, no cap on R) and a new CxC-based
  R-Precision (RPC²).
- Relevance to MultiMAE [reader-inferred]: shows that "PMRP" values in the literature use at least two definitions
  (PCME's ζ-average vs ECCV Caption's ζ=0 with R ≤ 50), so cross-paper PMRP numbers are not comparable; the brief's
  PMRP follows ECCV Caption.
- Method weaknesses: the authors note that PMRP depends on exhaustive object annotations and that images with the
  same object classes can mean different things [author-acknowledged, metrics paragraph]; non-CLIP.

**C3. Chun, S., & Russakovsky, O. (2025). Multiplicity is an Inevitable and Inherent Challenge in Multimodal
Learning.** arXiv 2505.19614 (position paper).
- Tier: preprint, preprint>=2024.
- Read scope: LaTeX, sections on multiplicity in training and evaluation.
- Setting: position paper, no new experiments on these metrics.
- Content used: restates ECCV Caption's τ = 0.47 (COCO R@1 vs mAP@R ranking) and PCME++'s result that training that
  assumes one-to-one correspondence breaks down at scale (40.0 mAP@R at B/32 vs 20.2 at L/14, the VSE∞ rows of
  PCME++ Table 1; PCME++ 40.1 vs 42.1); argues single-positive R@K is misaligned with human judgment under
  multiplicity.
- Polysemy-aware metrics: discusses ECCV mAP@R; no new numbers.
- Relevance to MultiMAE [reader-inferred]: the 2025 statement of the PCME/ECCV line's position on which metric to trust.
- Method weaknesses: not assessed beyond noting that it re-cites the authors' own results [reader-inferred].

## 4. Answers extracted for the B3 questions (within-theme, no recommendations)

- **Recipe comparison.** Brief's recipe: 10 epochs, batch 128, lr 1e-5 towers, about 44k steps. Reported recipes:
  PCME++ 25 epochs at batch 128 (about 111k steps) with much lower lower-layer lrs (visual 5e-6 with LLRD 0.7, text
  5e-5), 2-epoch visual freeze and step decay; NegCLIP 5 epochs at batch 1024 (about 2.8k steps), lr ≤ 1e-5;
  NeighborRetr 10 epochs at batch 128, lr 1e-4 (Adam); CLIP4Clip 5 epochs at batch 128, tower lr 1e-7 (video); ITRA
  (ResNet-50) lr 1e-5 to 2e-5 best at batch 256, epochs saturating at 15 to 20 (10 epochs is 0.45 mean recall, about
  2.7 5k-RSUM, below the 15-epoch value). By these sources the brief's run is not over-trained in epochs; by ITRA it
  is slightly short of the epoch plateau for ResNet-50.
- **Batch size.** Only ITRA isolates it on COCO: batch 128 -> 512 (lr scaled) is +2.71 mean recall (about +16 5k RSUM)
  for ResNet-50 CLIP; CLIP4Clip found 128 ≈ 256 on video. NegCLIP's authors expected gains from larger batches but did
  not measure them. No source measured batch-size effects on ECCV mAP@R or PMRP.
- **Changes that moved ECCV mAP@R by ≥ 0.5 over a same-recipe baseline**: the loss family (probabilistic or pairwise
  sigmoid-style vs InfoNCE: +0.5 to +1.1 at ViT-B, +6.5 at L/14 where InfoNCE was unstable), similarity scale
  (learnable vs fixed temperature, +23.3), pooling (GPO, +2.6), multiple embeddings per sample (PVSE K=2 +6.3, DivE
  +1.1), negative-mining strength (none -> semi-hard +3.3), soft or graded targets from external similarity (CUSA
  +2.3 at B/32, listwise/BCLS about +1.0), regularizers on the embedding (VIB + PP + MSDA +1.2), and embedding
  geometry (oblique multi-token +0.8). Changes under 0.5: SWA (+0.1), prototype module (+0.45), probabilistic vs
  mean-only inference on the same trained model (0.0).
- **What moved PMRP**: in ECCV Caption Table 4, removing hard-negative mining (+2.3 vs hardest), a larger CLIP
  backbone (B/32 -> L/14 zero-shot +2.4), and CutMix pretraining (+0.94); PMRP moved opposite to mAP@R under the
  negative-mining change. No later paper reports ECCV-style PMRP.

## 5. Search limitations

- ar5iv and the CVF open-access PDFs (HTTP 403) were not reachable from this container; LaTeX sources replaced them.
  Papers without LaTeX on arXiv (e.g. "Integrating Language Guidance Into Image-Text Matching for Correcting False
  Negatives", TMM 2024; "Dynamic Soft Labeling for Visual Semantic Embedding", ICMR 2024; "Cross-Modal Retrieval with
  Noisy Correspondence via Consistency Refining and Mining", TIP 2024; MSRM, CVPR 2023) were not read, so any ECCV
  or PMRP numbers they contain are missing.
- Semantic Scholar's citation graph was rate-limited (HTTP 429) after two calls; only citations of ECCV Caption and
  PCME were enumerated, and Semantic Scholar's citation lists are incomplete.
- No PDF parser was available, so PDF-only sources (DITM 2505.09997) were assessed from the abstract only.
- Not verified: the published venue of CLIP4Clip; whether arXiv 2209.02127 is the TMLR 2024 paper; the publisher
  record of the listwise-ranking paper; the ICML 2026 proceedings entry for VACSR (taken from arXiv comments).
- Not read: LiT (Zhai et al. 2022), which locks the image tower during contrastive training rather than fine-tuning
  on COCO; ITRA's lock-image row is the closest LiT-style COCO data point found. No paper was found that reports
  WiSE-FT weight interpolation on COCO retrieval with numbers; PCME++'s SWA (+0.1 mAP@R) is the nearest
  weight-averaging result.
- Seen only as search snippets and not included: AOQ (arXiv 2003.03669) on batch size for triplet-loss VSE, and
  β-CLIP (arXiv 2512.12678) on batch size hurting fine-grained retrieval.
- Several leaderboard values are means of two directions computed here; they can differ from a paper's own
  rounding by up to 0.05.
