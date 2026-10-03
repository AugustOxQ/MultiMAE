> Supporting file for docs/reports/auto/v1/2026-10-03_lever_review.md. Phase 2 bibliographies are as written; V1/V2 list the corrections, and the report uses corrected values.

# B2: Representing one-to-many matches, and fusion at retrieval time, on top of a pretrained dual encoder

Phase 2 bibliography for theme B2 (ARS deep-research, lit-review mode). Last searched 2026-10-03.
No cross-theme synthesis and no project recommendations here; each source carries a one or two line relevance note.

Conventions used below.
- "COCO 5K R@1" in ECCV Caption and PCME++ tables is the mean of i2t and t2i R@1. "RSUM" in those two papers is the sum of the
  six COCO **1K** recalls (ECCV Caption Table 4 caption), so it is not on the same scale as MultiMAE's rsum (COCO 5K).
- Numbers marked "computed" are my arithmetic on table values (means of i2t/t2i, or sums of the six 5K recalls). Everything
  else is copied from the table named in the locator.
- PMRP values from the PCME paper (2021) and from the ECCV Caption paper (2022) are computed differently and sit on very
  different scales (about 30 to 46 vs about 47 to 58); they should not be compared across the two papers [reader-inferred].

---

## 1. Search strategy

**Sources searched.** arXiv full text through arxiv.org/html (all papers below parsed from HTML tables with a local
BeautifulSoup script, not from abstracts), ar5iv as fallback, ACL Anthology (TACL, ACL), AAAI OJS, PMLR, ICLR/ICML virtual
pages for venue checks, arXiv abstract-page "Comments" fields for venues, Semantic Scholar API (rate-limited after three
calls), web search (US index, "standard" and "extended" modes).

**Query strings (web search).**
1. `image-text retrieval "ECCV Caption" mAP@R soft label false negatives 2024`
2. `re-ranking CLIP top-k cross-encoder image-text retrieval COCO 5k ITM rerank gain dual encoder 2024 2025`
3. `probabilistic embeddings frozen CLIP post-hoc uncertainty retrieval 2025 arXiv ECCV Caption PMRP`
4. `"ECCV Caption" "PMRP" image-text matching 2025`
5. `multiple positives contrastive fine-tuning CLIP COCO captions same image soft targets ECCV Caption mAP@R improvement arXiv 2024 2025`
6. `BLIP-2 COCO retrieval "without re-ranking" OR "ITC only" OR "w/o ITM" R@1 dual encoder vs ITM reranking ablation`
7. `"ECCV Caption" BLIP ITC ITM rerank mAP@R R-Precision evaluation false negatives cross-encoder`
8. `set-based or multi-embedding image-text retrieval CLIP fine-tuning polysemous 2024 2025 ECCV Caption CxC results`
9. `arXiv 2025 image-text retrieval "ECCV Caption" mAP@R CLIP ViT-B fine-tuned COCO false negatives pseudo-positives new method`
10. Venue checks: SoftCLIP AAAI 2024; LoopITR venue; FILIP ICLR 2022 / BLIP ICML 2022; "Preventing Representation Collapse ..." arXiv id.
Plus known-item retrieval of the papers named in the task (ALBEF, BLIP, BLIP-2, X-VLM, MaskVLM, FILIP, PCME, PCME++, ProLIP,
PVSE, DivE, ProbVLM) and backward snowballing from ELIP, CUSA, PCME++ and LoopITR reference lists.

**Date range.** 2019 to 2026 (PVSE 2019 is the oldest included; newest included is GroVE, UAI 2025; ICLR 2026 and ICML 2026
items were screened).

**Inclusion.** (a) Retrieval-time fusion or re-ranking of a dual encoder's candidates, with numbers for the dual-encoder
ranking and the re-ranked one, or with polysemy-aware metrics; (b) probabilistic, set-valued or text-conditioned
embeddings for image-text retrieval with a same-recipe deterministic baseline, or post-hoc uncertainty on frozen CLIP;
(c) multi-positive or soft-target contrastive training evaluated on COCO retrieval. Priority to CLIP-initialized or
CLIP-scale work and to papers reporting ECCV Caption mAP@R / R-P, PMRP or CxC.

**Exclusion.** Papers whose only retrieval numbers are on non-COCO domains; video retrieval; papers I could not open in full
text; secondary sources that only re-quote numbers.

**Counts.** Identified about 90 records (web-search hits, approximate, duplicates across queries not tracked precisely).
Screened 35 (title plus keyword scan of the full text for "ECCV Caption", "mAP@R", "PMRP", "CxC", "CLIP").
Full text read (tables plus targeted sections) 26. Included 20: 17 full entries and 3 short entries (BLIP-2, X-VLM, MaskVLM),
which only supply re-ranking settings. This is above the brief's 8 to 15 target because the task named most of these papers.

Screened in full text and excluded:
- Post-hoc Probabilistic VLMs / BayesVLM, Baumann et al., ICLR 2026 (arXiv 2412.06014): classification and active
  learning only; no image-text retrieval table (table captions checked).
- Maximal Matching Matters (MaxMatch), Alomari et al., ACL 2025 (arXiv 2506.21538): set-based embeddings, but non-CLIP
  backbones and no ECCV/PMRP/CxC. Reported COCO 5K results for ResNeXt+BERT: VSE∞ RSUM 468.9, Set Div 474.9 (Table 5).
- β-CLIP (arXiv 2512.12678, preprint): no polysemy-aware metrics.
- Multiplicity position paper, Chun et al., ICML 2026 Position Track (arXiv 2505.19614): secondary; re-quotes PCME++
  (VSE∞ 40.0 to 20.2 mAP@R from B/32 to L/14 vs PCME++ 40.1 to 42.1).
- Image-Text Retrieval with Binary and Continuous Label Supervision (BCLS, arXiv 2210.11319, preprint): VSE++ only, no
  CLIP. ECCV mAP@R VSE++ 20.8/38.3 to 21.8/39.2 (i2t/t2i, Table VII). Kept out to stay within the CLIP-era scope.
- LightningDOT (Sun et al., NAACL 2021) and Thinking Fast and Slow (Miech et al., CVPR 2021): pre-CLIP re-ranking. Read
  but superseded by Geigle et al. and LoopITR, which compare the same model with and without re-ranking. For reference,
  LightningDOT COCO 5K TR/IR R@1 60.1/45.8, and 74.2/57.4 after a separately trained OSCAR re-ranker (Table 1).
- Your Negative May not Be True Negative (FNE, arXiv 2308.04380) and Deep Boosting Learning (arXiv 2404.18114): no
  polysemy-aware metrics found in the full text.
- Descriptive Image-Text Matching with Graded Contextual Similarity (arXiv 2505.09997): HTML fetch returned an error page;
  not assessed.

**Coverage skew.** The polysemy-aware metrics are reported almost only by Chun et al.'s group (ECCV Caption, PCME, PCME++,
ProLIP) and by papers that copy their evaluation code (DivE, CUSA). Re-ranking work almost never reports ECCV Caption,
PMRP or CxC. The search was web-index based (US), with no OpenReview or Google Scholar crawl.

---

## 2. Annotated bibliography

### 2.1 Fusion or cross-encoder re-ranking of a dual encoder's top-K

#### [R1] Li et al., 2021. Align before Fuse: Vision and Language Representation Learning with Momentum Distillation (ALBEF)
- Citation: Junnan Li et al., NeurIPS 2021. arXiv 2107.07651.
- Tier: peer-reviewed (NeurIPS 2021; venue per Semantic Scholar record).
- Read scope: full text (arXiv HTML): Sec 5 (downstream retrieval protocol), Sec 6.5, Tables 1, 2, 6.
- Setting: from-scratch V+L pretraining (4M or 14M images), ViT-B/16 image encoder, BERT-base text and fusion layers;
  fine-tuned on COCO and Flickr30K with ITC + ITM.
- Change tested: at test time, s_itc picks top-k candidates and the ITM head re-ranks them. Baseline: the same fine-tuned
  model ranked by s_itc alone.
- Effect:
  - Flickr30K test, fine-tuned, mean of R@1/5/10 (Table 6): TR 97.30 (s_itc only) to 98.60 / 98.57 / 98.57 (ITM, k = 16 /
    128 / 256); IR 90.95 to 93.64 / 93.99 / 93.95. The authors call the ITM ranking "not sensitive to changes in k" (Sec 6.5).
  - COCO 5K, fine-tuned, re-ranked with k = 256 (Table 2): 4M TR/IR R@1 73.1/56.8; 14M 77.6/60.7. No s_itc-only COCO
    number is given.
  - Multi-positive ITC in fine-tuning (relevant to 2.3): "we change the ground-truth label of ITC to consider multiple
    positives in the queue, where each positive has a ground-truth probability of 1/#positives" (Sec 5, retrieval
    paragraph). No ablation of this choice is reported.
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: the only same-model ITC vs ITC+ITM table in the ALBEF family, and it is on Flickr, not COCO; it also
  documents the multi-positive ITC target that MultiMAE's InfoNCE does not use [reader-inferred].
- Method weaknesses: the k ablation is on Flickr30K, where mean recall is near ceiling (97 to 99), which compresses any
  re-ranking gain [reader-inferred]; the multi-positive ITC label is introduced without an ablation, so its effect is
  unknown [reader-inferred].

#### [R2] Li et al., 2022. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
- Citation: Junnan Li et al., ICML 2022 (PMLR 162). arXiv 2201.12086.
- Tier: peer-reviewed (ICML 2022).
- Read scope: full text (arXiv HTML): Sec 5.1 retrieval protocol, Tables 1 and 5. BLIP's ECCV Caption and PMRP numbers come
  from third-party re-evaluations [E1] Table 4 and [P3] Table C.1 (both read in full).
- Setting: from-scratch V+L pretraining on 14M or 129M images, ViT-B/16 or ViT-L/16; COCO fine-tuning with ITC + ITM.
- Change tested: top-k by ITC similarity, then ITM re-ranking; "k = 256 for COCO and k = 128 for Flickr30K" (Sec 5.1).
  BLIP reports no ITC-only ranking.
- Effect: COCO 5K fine-tuned, BLIP 129M ViT-B TR/IR R@1 81.9/64.3 (Table 5); ViT-L 82.4/65.1.
- Polysemy-aware metrics (third-party, [E1] Table 4, "fine-tuned BLIP"): ECCV mAP@R 40.52, R-P 48.43, ECCV R@1 90.99,
  CxC R@1 74.30, COCO 1K R@1 86.12, COCO 5K R@1 73.11, PMRP 57.17. The 5K R@1 of 73.11 equals BLIP's own 129M ViT-B
  (81.9 + 64.3)/2 = 73.1, so the re-evaluated checkpoint is probably the re-ranked 129M ViT-B model [reader-inferred].
  Against efficient dual encoders in the same tables: PCME++ ViT-B/16 (CLIP init, COCO fine-tuned) has 5K R@1 61.1 but
  ECCV mAP@R 42.1 ([P3] Table 1); VSE∞ (WSL grid) has 5K R@1 59.01, ECCV mAP@R 42.41, PMRP 57.65 ([E1] Table 4). The
  cross-attention re-ranked model leads COCO 5K R@1 by 12 to 14 points and trails both on mAP@R by 1.6 to 1.9
  points (computed).
- Relevance to MultiMAE: the largest R@1 gain in this theme comes with no ECCV mAP@R or PMRP gain over CLIP-initialized dual
  encoders, which bears on whether a fused scorer would move the project's polysemy metrics [reader-inferred].
- Method weaknesses: BLIP and the dual encoders differ in pretraining data (129M vs CLIP's 400M plus COCO fine-tuning) and
  architecture, so the mAP@R gap is a cross-system comparison, not a re-ranking ablation [reader-inferred]; the
  re-evaluating papers do not state k or whether ITM re-ranking was used for BLIP [reader-inferred from [E1] Sec 4.1
  wording]; ECCV Caption positives were proposed by five models (PVSE, VSRN, PCME, ViLT, CLIP B/32), not BLIP, which may bias mAP@R
  against models whose top ranks differ from the annotators' [author-acknowledged in [E1], Sec 5 and App. E, general bias
  discussion].

#### [R3] Short entries: ITM re-ranking settings in BLIP-2, X-VLM and MaskVLM
- **Li et al., 2023. BLIP-2** (ICML 2023; arXiv 2301.12597; venue per Semantic Scholar). Read scope: full text, Sec 4.3,
  Tables 5, 6. COCO fine-tuning with ITC + ITM + ITG; inference "first select k = 128 candidates based on the image-text
  feature similarity, followed by a re-ranking based on pairwise ITM scores". COCO 5K TR/IR R@1: ViT-L 83.5/66.3, ViT-g
  85.4/68.3 (Table 5). Ablation Table 6 is about the ITG loss (ITC+ITM 84.5/67.2 vs +ITG 85.4/68.3), not about re-ranking.
  No ITC-only number. Polysemy metrics: no. Weaknesses: not assessed beyond these tables.
- **Zeng et al., 2022. X-VLM** (ICML 2022; arXiv 2111.08276). Read scope: Sec 4 retrieval protocol, Tables 2, 4. k = 256
  (COCO) and 128 (Flickr) "following ALBEF"; fine-tuning also uses a multi-positive target (1/#positives). COCO 5K
  TR/IR R@1 81.2/63.4 (16M, Table 2). No ITC-only number. Polysemy metrics: no.
- **Kwon et al., 2023. MaskVLM** (ICLR 2023; arXiv 2208.02131; listed in the brief as known). Read scope: Tables 1, 5, 6,
  retrieval protocol. Top-k ITC then cross-modal encoder, following ALBEF. Table 5 (pretrained on CC 50% + COCO,
  fine-tuned on Flickr): ITC-only model IR/TR R@1 65.10/80.10 vs ITC+ITM 79.96/92.30 vs MLM+MIM only 76.08/90.30 vs all
  four losses 81.26/94.10. This is a pretraining-objective ablation, so it confounds the re-ranking head with what the
  extra losses teach the encoders [reader-inferred]. Polysemy metrics: no.
- Relevance to MultiMAE: none of the ALBEF-family papers isolate the re-ranking gain on COCO 5K; LoopITR [R5] and Geigle et
  al. [R4] are the clean sources [reader-inferred].

#### [R4] Geigle et al., 2022. Retrieve Fast, Rerank Smart: Cooperative and Joint Approaches for Improved Cross-Modal Retrieval
- Citation: Gregor Geigle, Jonas Pfeiffer, Nils Reimers et al., TACL 2022 (ACL Anthology 2022.tacl-1.29). arXiv 2103.11920.
- Tier: peer-reviewed (TACL).
- Read scope: full text (arXiv HTML): Tables 1, 3, 4, 8, 9 and method sections.
- Setting: OSCAR-initialized transformers (pre-CLIP), fine-tuned on COCO or Flickr30K. BE = bi-encoder; CE = cross-encoder;
  Sep+Coop = separately trained BE retrieves top-k, CE re-ranks; Joint+Coop = one weight-shared model trained for both roles.
- Change tested: cooperative re-ranking of the BE's top-k by a CE. Baseline: the BE alone (and Joint+BE, the same jointly
  trained model used as a bi-encoder only).
- Effect (COCO 5K, Table 1; rsum = sum of six 5K recalls, computed):
  - Joint+BE IR/TR R@1 52.5/66.7, rsum 472.2 to Joint+Coop 54.7/70.8, rsum 481.9 (same weights, +2.2 IR, +4.1 TR,
    +9.7 rsum).
  - BE 52.2/66.9 (rsum 472.4) to Sep+Coop 52.8/70.2 (rsum 478.6).
  - Full CE over all pairs (own reproduction, OSCAR) 52.6/69.3 (rsum 476.0): re-ranking the top-k matched or beat
    exhaustive cross-encoding.
  - k (Table 8, COCO 5K Joint+Coop): k = 10 / 20 / 50 gives IR R@1 54.8 / 54.7 / 54.6 and TR R@1 70.9 / 70.8 / 70.7.
  - Cost (Table 4, V100, embeddings not cached): COCO 5K evaluation takes 30 s (BE), 25 min (Coop), 50 h (CE). Per-query
    latency on 50k images (Table 3): 16 ms BE, 74 ms Coop, 2 min CE.
  - Score fusion (Table 9, Flickr): adding the BE score to the CE score did not help (Joint+Coop IR R@1 76.4; add λ=0.1 76.7,
    λ=0.9 74.6).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: gives a same-weights estimate (+2 to +4 R@1 on COCO 5K at k = 10 to 20) of what a fusion
  re-ranker adds over its own dual-encoder ranking, with cost [reader-inferred].
- Method weaknesses: OSCAR uses detector region features, and both roles share a pre-CLIP backbone, so the size of the
  re-ranking gain over a CLIP-initialized dual encoder is not established here [reader-inferred]; the authors note that
  RerankSmart's COCO results may benefit from pretraining data annotated on COCO images [author-acknowledged in LoopITR
  [R5] Table 9 note, not by Geigle et al.].

#### [R5] Lei et al., 2022. LoopITR: Combining Dual and Cross Encoder Architectures for Image-Text Retrieval
- Citation: Jie Lei, Xinlei Chen, Ning Zhang et al., arXiv preprint, 2022. arXiv 2203.05465.
- Tier: preprint (venue UNVERIFIED; arXiv page lists none).
- Read scope: full text (arXiv HTML): Tables 1 to 9, 12, 13 and Appendix B.
- Setting: ALBEF-style from-scratch pretraining on 4M images; one network with a dual-encoder head and a cross-encoder head;
  cross-encoder scores distilled into the dual encoder; COCO fine-tuning.
- Change tested: cross-encoder re-ranking of the dual encoder's top-k ("as in ALBEF"). Baseline: the same model's dual
  encoder.
- Effect:
  - COCO 5K test: dual encoder TR/IR R@1 67.6/51.7 (Table 9; rsum 471.9 computed) to cross-encoder re-ranked 75.1/58.0
    (Table 7; rsum 494.7 computed): +7.5 TR, +6.3 IR, +22.8 rsum.
  - Distillation from the cross head improves the dual head (COCO 5K val, Table 4): 62.64/46.78 to 65.00/50.53.
  - Cost (Flickr 1K test, one A100, Table 13): dual 89.6/77.2 in 11 s; cross re-rank of top 16 94.5/83.4 in 76 s; full
    cross-encoding 94.4/83.1 in 2342 s.
- Polysemy-aware metrics: CxC R@1 for the dual encoder only (Table 12): I→T 69.2, T→I 53.5. No ECCV/PMRP, and no CxC for the
  re-ranked model.
- Relevance to MultiMAE: the cleanest same-model COCO 5K number in this theme for what fusion re-ranking adds (+6 to +8 R@1);
  it is ALBEF-scale, not CLIP-initialized [reader-inferred].
- Method weaknesses: the dual and cross heads share a backbone trained with the cross-encoder's hard negatives and
  distillation, so the dual baseline is not an independent CLIP-style dual encoder [reader-inferred]; single run, no
  variance [reader-inferred].

#### [R6] Yao et al., 2022. FILIP: Fine-grained Interactive Language-Image Pre-Training
- Citation: Lewei Yao et al., ICLR 2022. arXiv 2111.07783.
- Tier: peer-reviewed (ICLR 2022).
- Read scope: full text (arXiv HTML): Sec 3.1, Tables 2 to 5.
- Setting: from-scratch contrastive pretraining (FILIP300M + YFCC100M + CC12M + CC3M); ablations on a YFCC100M subset with
  ViT-B/32.
- Change tested: cross-modal late interaction (token-wise max similarity between patch and word embeddings) in place of the
  global cosine in the contrastive loss. Baseline: same ablation stack without late interaction.
- Effect:
  - Zero-shot COCO (Table 4, YFCC subset): I2T R@1 29.2 to 30.5, T2I R@1 17.9 to 18.5 (rows "w/ back translation" to
    "w/ cross-modal late interaction").
  - Cost (Table 5): training 1.31 to 2.85 s/iter and 14.3 to 26.0 GB memory with all tokens in fp32; 1.39 s/iter and
    16.1 GB in the final setting (25% of tokens, fp16, dim 256).
  - Fine-tuned COCO 5K I2T/T2I R@1 78.9/61.2 vs ALBEF 77.6/60.7 (Table 3).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: late interaction keeps offline-indexable token embeddings; its measured gain at small scale was
  about +1 R@1, smaller than cross-encoder re-ranking [reader-inferred].
- Method weaknesses: the late-interaction ablation is zero-shot after small-scale pretraining, not fine-tuning a pretrained
  CLIP; storage and inference cost of per-token retrieval are not measured in the tables read [reader-inferred].

#### [R7] Zhan et al., 2025. ELIP: Enhanced Visual-Language Foundation Models for Image Retrieval
- Citation: Guanqi Zhan, Yuanpei Liu, Kai Han et al., CBMI 2025 (per arXiv comment). arXiv 2502.15682.
- Tier: peer-reviewed (CBMI 2025). `preprint>=2024` does not apply.
- Read scope: full text (arXiv HTML): Sec 3 to 6, Tables 1 to 8.
- Setting: frozen CLIP (ViT-B, "default backbone"), SigLIP, SigLIP-2 and BLIP-2. A small MLP maps the text query to visual
  prompt tokens inserted into the frozen image encoder, which then re-encodes the top-k images. Trained on a DataCompDR
  subset with hard-sample batching, 2 A40 GPUs, 144 GPU-hours (Table 8). Text-to-image only.
- Change tested: query-conditioned re-encoding of the top-k (k = 100 for COCO and Flickr with ELIP-C). Baseline: the frozen
  model's own ranking.
- Effect (zero-shot COCO 5K T2I, Table 2): CLIP R@1 40.16 to 45.61; SigLIP 54.21 to 61.03; SigLIP-2 56.87 to 62.91; BLIP-2
  (COCO fine-tuned, itself ITM re-ranked) 68.25 to 68.41. CLIP's own Recall@100 on COCO is 97.67 (Table 4), which bounds the
  re-ranker. On Flickr, a late-fusion baseline reaches 69.58 vs ELIP-C 72.30 vs CLIP 67.56 (Table 5).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: a 2025 example of a light, query-conditioned re-ranker on a frozen CLIP that adds about 5 R@1 on COCO
  5K T2I; it does not report any one-to-many metric [reader-inferred].
- Method weaknesses: zero-shot only and text-to-image only; the gain on an already re-ranked model (BLIP-2) is +0.16 R@1,
  suggesting most of the gain is the missing cross-modal interaction rather than something new [reader-inferred]; single
  run, no variance [reader-inferred].

### 2.2 Probabilistic, set-valued and text-conditioned embeddings

#### [E1] Chun et al., 2022. ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO
- Citation: Sanghyuk Chun et al., ECCV 2022. arXiv 2204.03359. (Listed in the brief as known; used here for the
  cross-encoder and set-embedding rows only.)
- Tier: peer-reviewed (ECCV 2022).
- Read scope: full text (arXiv HTML): Sec 4.1 to 4.2, Tables 4 and 5, Appendix E.
- Setting: re-evaluation of 25 public checkpoints on COCO 5K, CxC, ECCV Caption and PMRP.
- Change tested: none; benchmark paper. Relevant rows below.
- Effect (Table 4; mean of i2t and t2i):

  | Model | ECCV mAP@R | ECCV R-P | CxC R@1 | COCO 5K R@1 | PMRP |
  |---|---|---|---|---|---|
  | PVSE K=1 (single embedding) | 33.98 | 44.49 | 38.38 | 36.20 | 53.56 |
  | PVSE K=2 (two embeddings) | 40.26 | 49.92 | 40.18 | 38.13 | 55.52 |
  | PCME | 37.11 | 47.82 | 40.09 | 38.03 | 56.71 |
  | VSE∞ (WSL grid) | 42.41 | 51.43 | 60.79 | 59.01 | 57.65 |
  | CLIP ViT-B/32 zero-shot | 26.75 | 36.91 | 41.97 | 40.28 | 55.32 |
  | CLIP ViT-L/14 zero-shot | 27.98 | 37.80 | 48.14 | 46.44 | 57.70 |
  | ViLT fine-tuned (single-stream cross-encoder) | 34.58 | 44.27 | 53.72 | 52.18 | 57.63 |
  | VinVL fine-tuned (cross-encoder) | 40.81 | 49.55 | 67.76 | 66.39 | 54.72 |
  | BLIP fine-tuned (ITC + ITM re-rank) | 40.52 | 48.43 | 74.30 | 73.11 | 57.17 |

  - Kendall τ between metric rankings over the 25 models (Table 5): COCO 5K R@1 vs ECCV mAP@R 0.39; ECCV mAP@R vs PMRP 0.20;
    COCO 1K R@1 vs PMRP 0.45.
  - Cost: "VinVL takes 25 hours to compute the full pairwise ranks for the COCO Caption test split by a single A100 GPU core,
    while VSE++ only takes 1 minute" (Sec 4.1).
  - The authors' reading: "the best models on COCO and ECCV (e.g., BLIP, VinVL, and VSE∞) show inferior PMRP scores", and
    contrastive models without negative mining are "specialized to PMRP" (Sec 4.2, Fig. 5).
- Polysemy-aware metrics: yes (all of the above).
- Relevance to MultiMAE: the CLIP B/32 zero-shot PMRP here (55.32) equals the project's zero-shot PMRP, and the whole
  25-model PMRP range is 46.95 to 57.70, so PMRP differences of a few tenths sit inside a narrow band that cross-encoders do not
  widen [reader-inferred].
- Method weaknesses: PVSE, VSRN, PCME, ViLT and CLIP B/32 were the machine annotators that proposed candidate positives, which
  can favour them on ECCV metrics [author-acknowledged, App. E: "only three machine annotators (PVSE, PCME and VSRN) achieve
  better rankings on ECCV mAP@R compared to the COCO R@1 ranking"]; PMRP "only captures the existence or absence of the
  objects" [author-acknowledged, Sec 4.2]; models differ in data and architecture, so rows are not controlled comparisons
  [reader-inferred].

#### [P1] Song and Soleymani, 2019. Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval (PVSE)
- Citation: Yale Song, Mohammad Soleymani, CVPR 2019. arXiv 1906.04402.
- Tier: peer-reviewed (CVPR 2019).
- Read scope: full text (arXiv HTML): Table 2 (COCO). Polysemy metrics from [E1] Table 4 and PCME [P2] Table 3.
- Setting: ImageNet ResNet-152 + GloVe/Bi-GRU trained on COCO (not CLIP); K embeddings per sample from multi-head local
  attention plus residual global feature; multiple-instance-learning triplet loss with hardest negatives.
- Change tested: K = 2 embeddings vs K = 1 (same model family).
- Effect:
  - COCO 5K R@1 (PVSE Table 2): i2t 41.7 to 45.2, t2i 30.6 to 32.4.
  - ECCV Caption ([E1] Table 4, official weights): mAP@R 33.98 to 40.26 (+6.28), R-P 44.49 to 49.92, PMRP 53.56 to 55.52
    (+1.96), 5K R@1 36.20 to 38.13 (+1.93).
  - PCME-paper PMRP on COCO 5K ([P2] Table 3, i2t / t2i): 29.3 / 30.1 to 31.8 / 32.0.
  - Counter-evidence on CUB Caption ([P2] Table E.2, with hardest negatives): i2t R-P 22.34 (K=1) to 19.67 (K=2) to 18.38 (K=4).
- Polysemy-aware metrics: yes, via [E1] and [P2].
- Relevance to MultiMAE: the largest single-change ECCV mAP@R gain in this theme (+6.3) came from a second embedding, though in
  a weak pre-CLIP model and with PVSE as one of ECCV Caption's annotators [reader-inferred].
- Method weaknesses: PVSE was an ECCV Caption machine annotator, which can inflate its ECCV scores [author-acknowledged in
  [E1] App. E]; the K=1/K=2 rows are separately released checkpoints, not seeds of one recipe [reader-inferred]; on CUB
  more embeddings lowered R-P [reader-inferred from [P2] Table E.2].

#### [P2] Chun et al., 2021. Probabilistic Embeddings for Cross-Modal Retrieval (PCME)
- Citation: Sanghyuk Chun et al., CVPR 2021. arXiv 2101.05068.
- Tier: peer-reviewed (CVPR 2021).
- Read scope: full text (arXiv HTML): Tables 1, 3, D.2, E.1 to E.5.
- Setting: ResNet-152 + GloVe/Bi-GRU on COCO and CUB Caption (not CLIP). Gaussian embeddings with sampled match
  probability. Introduced PMRP (plausible matches by shared COCO object labels).
- Change tested: probabilistic embedding vs "PCME μ only" (deterministic version, same architecture).
- Effect:
  - COCO 5K (Table 3): PMRP i2t 34.0 to 34.1, t2i 34.3 to 34.4; R@1 i2t 43.5 to 44.2, t2i 31.7 to 31.9.
  - COCO 1K (Table 3): PMRP i2t 45.0 to 45.0, t2i 45.9 to 46.0; R@1 i2t 68.0 to 68.8.
  - CUB Caption R-P (Table 1 lower block): i2t 24.7 to 26.3, t2i 25.6 to 26.8; σ degrees of freedom 0 / 1 / 512 gave i2t
    R-P 24.7 / 25.7 / 26.3 (Table D.2).
  - Re-trained with CutMix-pretrained ResNet in [E1] Table 4: ECCV mAP@R 37.11 (official) to 41.74.
- Polysemy-aware metrics: yes (PMRP, CUB R-P; ECCV via [E1]).
- Relevance to MultiMAE: on COCO the probabilistic head moved PMRP by 0.1 over its own deterministic version, a reference for
  how small such effects can be [reader-inferred].
- Method weaknesses: PMRP here treats any caption sharing enough object labels as a match, so it rewards object overlap
  rather than full caption plausibility [author-acknowledged for PMRP in [E1] Sec 4.2; [E1] Table 2 also measures the PCME
  plausible-match labels at 65.3 / 56.6 I2T / T2I precision vs human labels]; single-run numbers [reader-inferred].

#### [P3] Chun, 2024. Improved Probabilistic Image-Text Representations (PCME++)
- Citation: Sanghyuk Chun, ICLR 2024. arXiv 2305.18171. (Listed in the brief as known; numbers re-derived from the tables.)
- Tier: peer-reviewed (ICLR 2024).
- Read scope: full text (arXiv HTML): Sec 2 to 3, App. B, C; Tables 1 to 5, B.1, C.1 to C.11.
- Setting: **fine-tunes pretrained CLIP** ViT-B/32, B/16 and L/14 (both towers, lr multipliers 0.01 visual and 0.1 text,
  visual tower frozen for 2 epochs) on COCO for 25 epochs, batch 128, AdamP, GPO pooling, 1024-d embeddings, SizeAugment;
  models selected by validation RSUM; 3 runs averaged (Table 1 caption, App. B.2, Table B.1). Large-scale variant trained
  from scratch on CC3M + CC12M + RedCaps.
- Change tested: closed-form sampled distance (CSD) between Gaussian embeddings with a log σ² head, VIB, pseudo-positives
  (PP: in-batch pairs closer than the positive treated as positives, weight α) and mixed-sample augmentation (MSDA:
  Mixup/CutMix with soft labels). Baselines with the same backbone, initialization and hyperparameters (tuned on VSE∞):
  InfoNCE (CLIP loss), PCME, VSE∞, DAA, P2RM.
- Effect (Table 1, ViT-B/32, means of 3 runs):

  | Method | ECCV mAP@R | ECCV R-P | ECCV R@1 | CxC R@1 | COCO 5K R@1 | RSUM (1K) |
  |---|---|---|---|---|---|---|
  | CLIP zero-shot | 26.8 | 36.9 | 67.1 | 42.0 | 40.3 | 471.9 |
  | InfoNCE fine-tune | 39.0 | 48.7 | 81.7 | 54.9 | 53.0 | 532.6 |
  | PCME | 39.1 | 48.9 | 81.4 | 54.7 | 53.0 | 532.0 |
  | VSE∞ | 40.0 | 49.5 | 83.1 | 57.1 | 55.2 | 536.5 |
  | PCME++ (μ only, deterministic) | 39.5 | 49.1 | 82.7 | 57.0 | 55.2 | 536.2 |
  | PCME++ | 40.1 | 49.7 | 83.1 | 56.8 | 55.1 | 537.0 |

  - Scale (Table 1): at ViT-L/14, InfoNCE 35.6 mAP@R / 45.9 5K R@1 and VSE∞ 20.2 / 22.7 vs PCME 41.2 / 61.9 and PCME++
    42.1 / 64.3. At ViT-B/16: InfoNCE 41.1 / 59.3, PCME++ 42.1 / 61.1.
  - Ablation (Table 3, B/32): none of VIB/PP/MSDA 38.9 mAP@R; VIB 39.3; PP alone 39.0; MSDA alone 39.0; VIB+PP 39.6; all
    40.1. RSUM 535.9 / 534.5 / 536.0 / 535.5 / 534.8 / 537.0.
  - PP weight (Table C.3): α 0.1 to 10 keeps mAP@R at 40.0 to 40.3 while 5K R@1 drops 54.8 to 52.6.
  - Inference (Table C.7, SWA model): mean-only retrieval 40.2 mAP@R vs CSD 40.2 (5K R@1 55.2 vs 55.5).
  - Standard errors (Tables C.10, C.11): B/32 InfoNCE i2t mAP@R 31.2 ± 0.1, PCME++ 32.3 ± 0.2; t2i 46.8 ± 0.5 vs
    47.8 ± 0.2.
  - Noisy correspondence (Table 2, B/32): at 50% noise InfoNCE 33.6 vs PCME++ 35.7 mAP@R.
  - Cross-encoders and others re-evaluated (Table C.1): BLIP 40.5 mAP@R / 73.1 5K R@1; VinVL 40.8 / 66.4; PCME++ B/16 SWA
    42.2 / 61.3.
- Polysemy-aware metrics: ECCV mAP@R, R-P, R@1 and CxC: yes. PMRP: not reported.
- Relevance to MultiMAE: the closest setting to the project (CLIP B/32 fine-tuned on COCO, 3 runs), where the probabilistic
  treatment adds about +1.1 ECCV mAP@R over InfoNCE and the probabilistic inference itself adds nothing; its InfoNCE
  baseline (39.0) sits above the project's contrastive fine-tune (36.94), with a different recipe [reader-inferred].
- Method weaknesses: hyperparameters were tuned on VSE∞ B/32 validation RSUM and applied to all methods, which can
  disadvantage some baselines (e.g., DAA collapsing at B/16, VSE∞ at L/14) [reader-inferred from Sec 3.1 text and Table 1];
  PP and MSDA are applied only to PCME++, since "applying PP and MSDA to triplet loss ... is non-trivial", so part of the
  gap to InfoNCE is the soft-label recipe rather than the Gaussian embedding [author-acknowledged, Sec 3.2]; a diagonal
  Gaussian may be insufficient [author-acknowledged, App. D]; the definition of the "μ only" row (trained without σ, or
  evaluated without σ) is not stated in the text I read, and it differs from the C.7 mean-only number (39.5 vs 40.2)
  [reader-inferred].

#### [P4] Kim et al., 2023. Improving Cross-Modal Retrieval with Set of Diverse Embeddings (DivE)
- Citation: Dongwon Kim, Namyup Kim, Suha Kwak, CVPR 2023 (Highlight). arXiv 2211.16761.
- Tier: peer-reviewed (CVPR 2023).
- Read scope: full text (arXiv HTML): Tables 1 to 5, App. B.1, B.2 (Table A1).
- Setting: non-CLIP backbones (ResNet-152, Faster R-CNN regions, ResNeXt-101 WSL) with Bi-GRU or BERT, trained on COCO /
  Flickr30K. A slot-attention-like set prediction module outputs K embeddings per sample; smooth-Chamfer set similarity.
- Change tested: set of K = 4 embeddings vs single-embedding VSE∞ (same backbone).
- Effect:
  - COCO 5K RSUM, ResNeXt-101 + BERT (Table 1): VSE∞ 468.9 (ensemble 474.8) vs DivE 474.9 (ensemble 482.0).
  - K sweep (Table 4, Flickr RSUM): K = 1 / 2 / 3 / 4 / 5 / 6: 492.6 / 495.5 / 497.4 / 500.8 / 498.4 / 499.3.
  - ECCV Caption and CxC (Table A1, "our best model" vs VSE∞): i2t mAP@R 34.8 to 36.0, t2i 50.0 to 51.0 (mean 42.4 to 43.5,
    computed); CxC R@1 i2t 67.9 to 72.3, t2i 53.7 to 55.5.
  - Cost: DivE costs more than VSE∞; the authors report that enlarging VSE∞'s embedding to match DivE's FLOPs lowered
    VSE∞'s Flickr RSUM by 10.8 points (App. B.1 text).
- Polysemy-aware metrics: yes (ECCV mAP@R, R-P, R@1; CxC R@1). PMRP: no.
- Relevance to MultiMAE: a recent set-embedding method gaining about +1.1 mean ECCV mAP@R over a strong single-vector
  baseline, of the same size as PCME++'s gain over InfoNCE [reader-inferred].
- Method weaknesses: "best model" in Table A1 may be the two-model ensemble while the VSE∞ row (42.4 mean, matching [E1]'s
  single official checkpoint) is a single model, which would inflate the gap [reader-inferred]; VSRN, a comparison row, was an
  ECCV Caption annotator, possibly inflating its numbers [author-acknowledged, App. B.2]; retrieval needs K×K similarities
  per pair [reader-inferred].

#### [P5] Upadhyay et al., 2023. ProbVLM: Probabilistic Adapter for Frozen Vision-Language Models
- Citation: Uddeshya Upadhyay, Shyamgopal Karthik, Massimiliano Mancini et al., ICCV 2023. arXiv 2307.00398.
- Tier: peer-reviewed (ICCV 2023).
- Read scope: full text (arXiv HTML): Sec 3, 4.1, Tables 3, 4. Table 1 (calibration) did not parse as a table; its numbers
  are taken from the Sec 4.1 text.
- Setting: **frozen CLIP** (ViT-B/32, ViT-B/16, RN-50) and BLIP; small adapters (generalized Gaussian per dimension)
  trained post hoc on COCO or CUB with intra- and cross-modal alignment objectives, 100 epochs, lr 1e-4.
- Change tested: post-hoc uncertainty on top of fixed embeddings. Baselines: PFE*, PCME*, TTDA adapted post hoc.
- Effect: retrieval rankings do not change: "all these models use the same underlying embeddings and achieve the same
  performance on the retrieval task" (Sec 4.1). Calibration (−S·R² between uncertainty level and R@1, higher is better),
  BLIP trained and evaluated on COCO: ProbVLM 0.80 vs PFE* 0.58, PCME* 0.62, TTDA 0.29 (Sec 4.1 text).
- Polysemy-aware metrics: no (calibration only).
- Relevance to MultiMAE: shows the post-hoc route gives an uncertainty score per query but, by construction, no change to
  COCO, ECCV or PMRP retrieval numbers [reader-inferred].
- Method weaknesses: calibration is scored against R@1 on single-positive COCO labels, so "uncertain" queries may simply be
  queries with unlabelled plausible matches [reader-inferred, in line with [P3] Fig. 3 discussion that uncertain samples have
  lower COCO R@1].

#### [P6] Venkataramanan et al., 2025. Probabilistic Embeddings for Frozen Vision-Language Models: Uncertainty Quantification with Gaussian Process Latent Variable Models (GroVE)
- Citation: Aishwarya Venkataramanan, Paul Bodesheim, Joachim Denzler, UAI 2025 (per arXiv comment). arXiv 2505.05163.
- Tier: peer-reviewed (UAI 2025).
- Read scope: full text (arXiv HTML): implementation details, App. C.1, Tables 7 to 10.
- Setting: **frozen CLIP ViT-B/32** and BLIP ViT-B; a GPLVM maps both modalities to a shared latent (Q = 5) and returns
  Gaussian embeddings; trained post hoc on COCO or CUB.
- Change tested: GroVE vs post-hoc PFE, PCME, PCME++, ProbVLM, TTDA. Baseline for retrieval: the deterministic CLIP cosine.
- Effect (Table 7, R@1, retrieval by minimum Wasserstein distance for probabilistic methods): COCO i2t deterministic 0.715
  vs GroVE 0.512, PCME++ 0.397, ProbVLM 0.303; t2i 0.515 vs 0.288, 0.125, 0.156. Inputs with more masking get higher
  predicted uncertainty (Fig. 6 right; values not read).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: ranking by post-hoc distributions on frozen CLIP lost 20 to 40 R@1 points against plain cosine
  in this table, so these adapters serve as uncertainty estimators, not retrievers [reader-inferred].
- Method weaknesses: the COCO protocol (1K or 5K) is not pinned down in what I read; 0.715 i2t R@1 for zero-shot B/32 suggests
  1K [reader-inferred]; the post-hoc PCME++ baseline is a re-implementation on frozen features, not the fine-tuned PCME++ of
  [P3] [reader-inferred].

#### [P7] Chun et al., 2025. Probabilistic Language-Image Pre-Training (ProLIP)
- Citation: Sanghyuk Chun, Wonjae Kim, Song Park et al., ICLR 2025 (per Semantic Scholar record and the brief).
  arXiv 2410.18857.
- Tier: peer-reviewed (ICLR 2025).
- Read scope: full text (arXiv HTML): Sec 3.3, Tables 1, 2, C.1, C.3, C.4, C.8.
- Setting: **from-scratch** pretraining on DataComp 1B (ViT-B/16, 1.28B or 12.8B samples seen); larger models fine-tuned from
  SigLIP / DFN CLIP with the probabilistic objective. Uncertainty from an extra [UNC] token. Ablations on CC3M + CC12M +
  RedCaps (96M samples).
- Change tested: inclusion loss L_inc(Z ⊂ Z_partial): the embedding of a partial input should include the embedding of the
  full input. Partial inputs mask "75% of the input tokens": text tokens replaced by [MASK], image patches dropped (as in MAE);
  masked copies are made for 12.5% of the batch (Sec 3.3). A second inclusion term makes image ⊂ text.
- Effect:
  - DataComp, 1.28B samples (Table C.4): no inclusion: retrieval 53.6, average 56.6; masked inclusion only: 53.2, 56.7;
    image-text inclusion only: 53.2, 57.0; both: 53.4, 57.3. ("Retrieval" is DataComp's average of Flickr, MSCOCO and
    WinoGAViL; e.g., ProLIP 71.13, 45.73, 42.12 averages to 53.0, Table C.1, computed.)
  - 96M samples (Table C.3): masked inclusion only raises HierarCaps recall 44.8 to 47.9 and ImageNet ZS 37.4 to 37.5; both
    terms 54.8 recall, ImageNet 37.0.
  - Occlusion (Table C.8): 0% to 10% occlusion moves mean σ_v from 0.0148 to 0.0153 while ImageNet ZS falls 74.6 to 73.2.
  - Against deterministic baselines at 1.28B samples (Table 1): retrieval average CLIP 53.4, SigLIP 53.4, ProLIP 53.0.
- Polysemy-aware metrics: no ECCV Caption, PMRP or CxC ("ECCV Caption" appears only in the references). HierarCaps
  (hierarchical caption retrieval) is reported (Table 2).
- Relevance to MultiMAE: the one source here that feeds masked inputs (75% token masking, MAE-style patch dropping) into a
  retrieval embedding; it used them to shape uncertainty, and the masked term left retrieval flat (53.6 vs 53.4 with both
  terms) [reader-inferred].
- Method weaknesses: no fine-tuning-on-COCO result and no polysemy-aware retrieval metric, so the effect of masked inclusion on
  one-to-many retrieval is untested [reader-inferred]; ablation tables are single runs [reader-inferred]; the authors note that
  learned uncertainty is "often counterintuitive" without the inclusion loss, especially "under noisy image-text
  correspondences" [author-acknowledged, Sec 3.3].

#### [P8] Lavoie et al., 2024. Modeling Caption Diversity in Contrastive Vision-Language Pretraining (Llip)
- Citation: Samuel Lavoie, Polina Kirichenko, Mark Ibrahim et al., ICML 2024 (per arXiv comment). arXiv 2405.00740.
- Tier: peer-reviewed (ICML 2024).
- Read scope: full text (arXiv HTML): Sec 3, 10.4, Tables 1, 2, 4.
- Setting: **from-scratch** pretraining on MetaCLIP 2.5B, 12.8B samples, batch 32K; ViT-B/32 to ViT-G/14.
- Change tested: the image encoder outputs K learnable "mixture tokens"; a cross-attention from the caption embedding sets
  their weights, so the image embedding depends on the caption it is compared with. Baselines: SigLIP and MetaCLIP with the
  same data and recipe.
- Effect (zero-shot COCO, Table 2): ViT-B/16 I2T R@1 SigLIP 59.7 to Llip 63.4, T2I 42.0 to 45.6; ViT-L/14 65.4 to 68.1 and
  48.1 to 50.6. Inference time is "slightly higher than CLIP for the same model size" (Sec 10.4).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: a middle ground between one vector and a fusion re-ranker (query-conditioned pooling over K stored
  image tokens) with about +3.6 R@1 at equal data, untested on ECCV Caption or PMRP [reader-inferred].
- Method weaknesses: from-scratch only, so the gain when fine-tuning a pretrained CLIP is unknown; the gallery must keep K
  tokens per image and re-pool per query, which breaks single-vector ANN search [reader-inferred from Sec 3].

### 2.3 Multi-positive and soft-target contrastive training

(PCME++'s pseudo-positives and MSDA are in [P3]; ALBEF's and X-VLM's 1/#positives ITC targets are in [R1] and [R3].)

#### [S1] Huang et al., 2024. Cross-Modal and Uni-Modal Soft-Label Alignment for Image-Text Retrieval (CUSA)
- Citation: Hailang Huang, Zhijie Nie, Ziqiao Wang et al., AAAI 2024 (per arXiv comment). arXiv 2403.05261.
- Tier: peer-reviewed (AAAI 2024).
- Read scope: full text (arXiv HTML): method, implementation details, Tables 1, 2, 5.
- Setting: **fine-tunes CLIP ViT-B/32 and ViT-L/14@336** (and SGRAF, X2VLM-base) on COCO and Flickr30K. Soft labels come from
  frozen uni-modal teachers computed offline: Unicom for images, Sentence-BERT (all-mpnet-base-v2) for texts.
- Change tested: InfoNCE plus cross-modal soft-label alignment (CSA, teacher similarities as soft targets for image-text
  logits) and uni-modal soft-label alignment (USA). Baseline: the same model "fine-tuned ... using InfoNCE".
- Effect:
  - COCO 5K, CLIP B/32 (Table 1): i2t R@1 56.3 to 57.3, t2i 42.8 to 44.2; RSUM (5K) 422.6 to 429.7.
  - ECCV Caption, CLIP B/32 (Table 2): i2t mAP@R 28.5 to 29.6, R-P 39.4 to 40.7, R@1 72.5 to 72.0; t2i mAP@R 41.7 to 45.2,
    R-P 50.8 to 53.6, R@1 83.0 to 85.7. Mean mAP@R 35.1 to 37.4 (computed).
  - CLIP ViT-L/14@336: mean mAP@R 39.15 to 40.6 (computed from Table 2).
  - Ablation, CLIP B/32 (Table 5): ECCV average of six metrics 52.6 (InfoNCE) to 54.5 (+CSA), 53.6 (+USA), 54.5 (both);
    COCO 5K RSUM 422.6 to 427.7 / 425.7 / 429.7.
  - Re-ranked model (X2VLM-base, dual encoder + fusion re-ranking, author checkpoint): ECCV mAP@R i2t 36.6, t2i 43.8
    (mean 40.2) vs fine-tuned CLIP B/32 35.1 and ViT-L/14 39.15; with CUSA 37.6 / 48.4. Its COCO 5K R@1 is 83.5 / 66.2
    (Table 1) against CLIP B/32's 56.3 / 42.8.
- Polysemy-aware metrics: yes (ECCV mAP@R, R-P, R@1). No PMRP or CxC table.
- Relevance to MultiMAE: same backbone and data as the project; soft targets from caption-caption similarity gave +2.3 mean
  ECCV mAP@R (mostly text-to-image, +3.5) for +1.2 mean COCO 5K R@1, and the CSA term alone carried the ECCV gain
  [reader-inferred].
- Method weaknesses: no seeds or variance reported in the tables I read [reader-inferred]; the InfoNCE baseline is weak
  relative to other CLIP B/32 fine-tunes (ECCV mAP@R mean 35.1 vs 39.0 for PCME++'s InfoNCE in [P3]), so gains may shrink
  against a tuned baseline [reader-inferred]; text soft labels from Sentence-BERT reward paraphrase-like captions, which may
  match ECCV Caption's verified positives more than true visual polysemy [reader-inferred].

#### [S2] Gao et al., 2024. SoftCLIP: Softer Cross-Modal Alignment Makes CLIP Stronger
- Citation: Yuting Gao, Jinfeng Liu, Zihan Xu et al., AAAI 2024 (AAAI 38(3), pp. 1860 to 1868). arXiv 2303.17561.
- Tier: peer-reviewed (AAAI 2024).
- Read scope: full text (arXiv HTML): Tables 3 and 5, method overview.
- Setting: **from-scratch** CLIP pretraining on YFCC15M-V2 (also CC3M / CC12M), RN50 and ViT-B/16.
- Change tested: soft cross-modal targets built from intra-modal self-similarity, replacing strict one-to-one targets.
  Baseline: the authors' own CLIP implementation on the same data.
- Effect (zero-shot COCO 5K, Table 5): RN50 I2T R@1 29.4 to 36.0, T2I 18.9 to 22.2; ViT-B/16 I2T 30.7 to 30.9, T2I 19.1
  to 19.2.
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: soft targets helped a ResNet a lot and a ViT-B/16 hardly at all in this table, a caution against
  assuming large gains for ViT CLIP fine-tuning [reader-inferred].
- Method weaknesses: from-scratch pretraining only; the RN50 and ViT-B/16 gaps differ by an order of magnitude with no
  explanation in the sections read [reader-inferred]; single run [reader-inferred].

---

## 3. Search limitations

- **No same-model re-ranking result on polysemy-aware metrics was found.** Re-ranked systems appear on ECCV Caption, PMRP
  and CxC only as whole checkpoints (BLIP, VinVL, ViLT in [E1]; BLIP and VinVL in [P3] Table C.1; X2VLM in [S1]). No paper I
  read reports one model's dual-encoder ranking and its ITM re-ranked ranking on ECCV mAP@R, PMRP or CxC. LoopITR reports CxC
  for its dual encoder only. This is an absence within my search, not proof of absence.
- **No COCO 5K ITC-only vs ITC+ITM table exists in ALBEF, BLIP, BLIP-2 or X-VLM** (tables and protocol text checked);
  ALBEF's k ablation is on Flickr30K.
- PDF-only sources could not be parsed (no pdftotext or PDF library in the environment, and I did not install one), so the
  ACL 2025 MaxMatch paper was read through its arXiv HTML version, and arXiv 2505.09997 (HTML error page) was not assessed.
- ProbVLM's calibration table (Table 1) did not parse; its numbers come from the paper's own text summary. Llip's
  compute-time figure and GroVE's masking-noise figure were not read numerically.
- Venues for SoftCLIP, BLIP and FILIP were confirmed by web search; ALBEF, BLIP-2 and ProLIP by Semantic Scholar records;
  the API then rate-limited, and DBLP returned no parseable response. LoopITR's venue is UNVERIFIED (treated as a preprint).
- The ELIP CLIP variant is described only as "ViT-B" (default backbone); the exact checkpoint was not identified.
- WebFetch (small-model summarizer) was used once on PCME++ at the start; every number in this file was then re-read from the
  HTML tables with my own parser, and the WebFetch summary was not used as a source (it had also wrongly stated that the PCME++
  baselines lack shared hyperparameters).
- No retrieved page contained instructions aimed at the reader; nothing to report on that front.
