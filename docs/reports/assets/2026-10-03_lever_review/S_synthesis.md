> Supporting file for docs/reports/auto/v1/2026-10-03_lever_review.md. Phase 3 synthesis as written, built from B1 to B4 as corrected by V1 and V2; the report condenses it and drops its hidden reference markers.

# S: Synthesis of masked objectives, one-to-many retrieval and polysemy (MultiMAE)

Phase 3 synthesis (ARS lit-review mode, integrative review), 2026-10-03, from `00_brief.md`, B1 to B4, V1 and V2. Literature numbers come from B1 to B4 as corrected by V1 and V2. "(computed)" marks my arithmetic, "[project]" marks MultiMAE results from the brief or the caller, and "row N" points to Section 3. Hidden ref and anchor markers follow each citation; slugs are built from first author, year and name, since the Phase 2 files have no citation_key. Prose and tables run to about 4,000 words; the manifest, YAML inventory and source key are outside that count. No audit step was run.

<details><summary>Claim intent manifest (machine-readable, outside the word count)</summary>

```json
{
  "manifest_version": "1.0",
  "manifest_id": "M-2026-10-03T00:00:00Z-s3ma",
  "emitted_by": "synthesis_agent",
  "emitted_at": "2026-10-03T00:00:00Z",
  "claims": [
    {"claim_id": "C-001", "claim_text": "Image-conditioned MLM carries most of the masked-objective retrieval gain across sources; pixel MIM added on top of MLM adds little or hurts.", "intended_evidence_kind": "empirical", "planned_refs": ["zhao2023mamo", "kwon2023maskvlm", "dou2022meter", "fujii2023bilma", "li2026macco"]},
    {"claim_id": "C-002", "claim_text": "The size of the image-conditioned MLM gain in a fine-tuned CLIP dual encoder is unsettled (IRRA vs BiLMa on the same configuration).", "intended_evidence_kind": "empirical", "planned_refs": ["jiang2023irra", "fujii2023bilma"]},
    {"claim_id": "C-003", "claim_text": "No searched source isolates a masked objective by ablation on ECCV Caption mAP@R, PMRP, CxC or VWSD.", "intended_evidence_kind": "empirical", "planned_refs": ["chun2022eccvcaption", "kwon2023flavavwsd", "chun2025prolip"]},
    {"claim_id": "C-004", "claim_text": "At CLIP B/32 fine-tuned on COCO, same-recipe loss-side one-to-many changes moved ECCV mAP@R by about +0.1 to +2.3.", "intended_evidence_kind": "empirical", "planned_refs": ["chun2024pcmepp", "huang2024cusa", "wei2026vacsr"]},
    {"claim_id": "C-005", "claim_text": "The project's contrastive baseline sits about 2.1 mAP@R below PCME++'s InfoNCE fine-tune at similar 5K rsum, with a bundled recipe difference.", "intended_evidence_kind": "empirical", "planned_refs": ["chun2024pcmepp"]},
    {"claim_id": "C-006", "claim_text": "PMRP is a weak instrument for sub-point differences.", "intended_evidence_kind": "empirical", "planned_refs": ["chun2022eccvcaption", "pishdad2022uncertainty"]},
    {"claim_id": "C-007", "claim_text": "Conditioning reconstruction on the pooled embedding beat vanilla masked objectives in a non-CLIP dual encoder (single runs).", "intended_evidence_kind": "empirical", "planned_refs": ["luo2022conlip", "tang2025tulip"]},
    {"claim_id": "C-008", "claim_text": "VWSD gains came from adding context to the query, not from masking.", "intended_evidence_kind": "empirical", "planned_refs": ["kritharoula2023vwsd", "ualberta2023vwsd", "bhattacharya2026vwsd"]},
    {"claim_id": "C-009", "claim_text": "Two rankings of candidate levers for the 250 GPU-hour budget: by expected gain on polysemy-aware metrics and by information value.", "intended_evidence_kind": "normative", "planned_refs": ["huang2024cusa", "chun2024pcmepp", "luo2022conlip", "kwon2023maskvlm", "verma2022uniform", "chun2025prolip"]}
  ],
  "manifest_negative_constraints": [
    {"constraint_id": "MNC-1", "rule": "No claim that masked objectives improved ECCV mAP@R in the project's setting."},
    {"constraint_id": "MNC-2", "rule": "Mechanisms behind the project's PMRP gain are stated only as hypotheses."},
    {"constraint_id": "MNC-3", "rule": "No comparison of PMRP values across the PCME and ECCV Caption definitions."},
    {"constraint_id": "MNC-4", "rule": "Expected-effect ranges for levers are labelled reader-inferred extrapolations, not predictions."}
  ]
}
```
</details>

## 1. Thematic synthesis

**Organizing frame [reader-inferred].** MultiMAE retrieves with CLIP's clean pooled embedding, which the masked losses reach only through shared tower weights (route 1). The literature has three other routes:
- Route 2: pooling built from token features (mean pooling, GPO).
- Route 3: decoders conditioned on the pooled embedding (ConLIP<!--ref:luo2022conlip--><!--anchor:section:Table%201-->, TULIP<!--ref:tang2025tulip--><!--anchor:section:3.3-->).
- Route 4: a loss on the masked view's own embedding (ProLIP<!--ref:chun2025prolip--><!--anchor:section:3.3-->).

Only route 1 has been measured in CLIP-scale dual encoders, with small effects. Most levers in Section 5 test routes 2 to 4.

### 1.1 Masked-objective design

**Where sources converge.** Image-conditioned MLM carries most of the gain, and pixel MIM adds little once MLM is present:
- MAMO T6<!--ref:zhao2023mamo--><!--anchor:section:Table%206-->: MLM gives about 77% / 70% of the full COCO gain (V1 #13).
- MaskVLM T5<!--ref:kwon2023maskvlm--><!--anchor:section:Table%205-->: MLM alone +1.2 / +2.0 zero-shot R@1; MIM alone −0.2 / +0.5.
- METER T7<!--ref:dou2022meter--><!--anchor:section:Table%207-->: MIM on top of MLM cost up to 6.28 IR R@1.
- BiLMa App. T5<!--ref:fujii2023bilma--><!--anchor:section:App.%20Table%205-->: pixel MIM cost 0.52 Rank-1.
- MACCO T9<!--ref:li2026macco--><!--anchor:section:Table%209-->: on our backbone and data, MLM and MIM were not additive.

Our validation losses agree: fusion lowers MLM loss, and MAE gains nothing from text [project].

Feature targets beat pixels wherever both were compared, though by under 1 point in fine-tuning (MaskCLIP T9<!--ref:dong2023maskclip--><!--anchor:section:Table%209-->, MAMO T9<!--ref:zhao2023mamo--><!--anchor:section:Table%209-->).

Every sweep with a retrieval readout found 15% text masking too low (Verma T4<!--ref:verma2022uniform--><!--anchor:section:Table%204-->, MaskVLM T7<!--ref:kwon2023maskvlm--><!--anchor:section:Table%207-->, M3AE Fig. 5<!--ref:geng2022m3ae--><!--anchor:section:Fig.%205-->). Bitton<!--ref:bitton2021dataefficient--><!--anchor:section:Table%201--> explains why: on captions of about 20 tokens, 36% of sentences get no mask and 45-50% of masked tokens are stop words or punctuation, yet content words are the ones that need the image.

Image ratios from 0.5 to 0.75 barely matter (MaskVLM T7<!--ref:kwon2023maskvlm--><!--anchor:section:Table%207-->; TIPS T11B<!--ref:maninis2025tips--><!--anchor:section:Table%2011B-->).

**Where they diverge, and why.**
- **Pixel MIM.** It helped as the sole auxiliary (VFE-TPS T6<!--ref:shen2025vfetps--><!--anchor:section:Table%206-->) and when caption-relevant patches were masked (SyCoCa T5<!--ref:ma2024sycoca--><!--anchor:section:Table%205-->). It hurt or did nothing on top of MLM with random masks (rows 4 and 5).
- **Loss weight.** MLM at weight 1 hurt MaskCLIP (T6f<!--ref:dong2023maskclip--><!--anchor:section:Table%206f-->), whose MLM is text-only and trained from scratch, but helped IRRA (T4<!--ref:jiang2023irra--><!--anchor:section:Table%204-->; row 3).

**Specific to MultiMAE.** Our MLM decoder reads the masked pass, so it sees only 25% of the patches [project docs]. IRRA<!--ref:jiang2023irra--><!--anchor:section:3.2-->, MaskVLM<!--ref:kwon2023maskvlm--><!--anchor:section:3--> and MACCO<!--ref:li2026macco--><!--anchor:section:3--> condition MLM on the full image. MaskVLM found that masking one modality at a time beats masking both (App. T6<!--ref:kwon2023maskvlm--><!--anchor:section:App.%20Table%206-->). The weight, ratio and target evidence comes from ITM or from-scratch models, and its transfer to a CLIP dual encoder is untested.

**Strength:** moderate for MLM over MIM (five sources; single runs; mostly ITM or from scratch). Weak for weights, ratios and targets in our setting.

### 1.2 Retrieval representation and one-to-many

**Where sources converge.** At CLIP B/32 on COCO, loss changes that relax the one-positive assumption (probabilistic matching, soft or pseudo-positive targets) gain about +0.1 to +2.3 mAP@R:

| Change | mAP@R gain | Source |
|---|---|---|
| PCME++ vs InfoNCE | +1.1 (3 runs) | T1<!--ref:chun2024pcmepp--><!--anchor:section:Table%201--> |
| CUSA | +2.3 | T2<!--ref:huang2024cusa--><!--anchor:section:Table%202--> |
| VACSR | +1.4 (over its own sigmoid baseline) | tab:ablation<!--ref:wei2026vacsr--><!--anchor:section:tab%3Aablation--> |
| pseudo-positives alone | +0.1 | PCME++ T3<!--ref:chun2024pcmepp--><!--anchor:section:Table%203--> |

The representation itself adds little:
- Probabilistic retrieval equals mean-only retrieval on the same trained model (PCME++ C.7<!--ref:chun2024pcmepp--><!--anchor:section:Table%20C.7-->).
- PCME's head moved PMRP by 0.1 (T3<!--ref:chun2021pcme--><!--anchor:section:Table%203-->).
- Post-hoc distributions on frozen CLIP leave rankings unchanged (ProbVLM<!--ref:upadhyay2023probvlm--><!--anchor:section:4.1-->) or worsen them (GroVE<!--ref:venkataramanan2025grove--><!--anchor:section:App.%20C-->).

**Where they diverge, and why.**
- **Multiple embeddings.** They gave +6.28 mAP@R in pre-CLIP PVSE (ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204-->) but about +1 on strong backbones (DivE<!--ref:kim2023dive--><!--anchor:section:Table%20A1-->, Sun<!--ref:sun2023topology--><!--anchor:section:tab%3Aclip_eccv-->), and they lowered CUB R-P (PCME<!--ref:chun2021pcme--><!--anchor:section:Table%20E.2-->). Gains shrink with backbone strength, and PVSE was an annotator (row 8).
- **Text-conditioned pooling.** It adds about 4 to 5 COCO R@1 from scratch (Llip<!--ref:lavoie2024llip--><!--anchor:section:Table%202-->, FLAIR<!--ref:xiao2025flair--><!--anchor:section:Table%201-->), with no ECCV or PMRP numbers.
- **Re-ranking and late interaction.** They raise R@K (Geigle<!--ref:geigle2022rerank--><!--anchor:section:Table%201-->, LoopITR<!--ref:lei2022loopitr--><!--anchor:section:Tables%207%20and%209-->, FILIP<!--ref:yao2022filip--><!--anchor:section:Table%204-->; the FILIP gain as corrected by V2).
- **Re-ranking on the polysemy metrics.** Re-ranked BLIP trails CLIP-initialized dual encoders on mAP@R (ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204-->; PCME++ T1<!--ref:chun2024pcmepp--><!--anchor:section:Table%201-->). The best COCO models have inferior PMRP (Sec 4.2<!--ref:chun2022eccvcaption--><!--anchor:section:4.2-->). No same-model comparison exists (row 7).

**Our setting (computed).** ALBEF's same-image multi-positive target<!--ref:li2021albef--><!--anchor:section:5--> would be a near no-op: at batch 128 over 567k pairs from 113k images, a same-image pair occurs about 0.06 times per batch. Soft targets must come from semantic similarity, as in CUSA<!--ref:huang2024cusa--><!--anchor:section:Table%205-->.

**Strength:** moderate to strong for PCME++'s +1.1 (3 runs, shared hyperparameters); moderate for the +0.1 to +2.3 band; weak for re-ranking and multi-embedding at CLIP scale.

### 1.3 Fine-tuning recipe

Our contrastive fine-tune reaches rsum 441.94 and mAP@R 36.94 [project]. PCME++'s InfoNCE B/32 baseline reaches 5K RSUM 444.1 and mAP@R 39.0 (T1<!--ref:chun2024pcmepp--><!--anchor:section:Table%201-->). R@K is similar, but the 2.06 mAP@R gap (computed) exceeds PCME++'s own method gain. CUSA's InfoNCE baseline is lower on both (422.6, 35.1<!--ref:huang2024cusa--><!--anchor:section:Tables%201%20and%202-->).

The PCME++ recipe differs from ours in many ways (App. B.2<!--ref:chun2024pcmepp--><!--anchor:section:App.%20B.2-->):
- GPO pooling with new 1024-d heads;
- visual lr 5e-6 with layer-wise decay 0.7, text lr 5e-5;
- the visual tower frozen for 2 epochs;
- 25 epochs (about 111k steps against our 44k);
- temperature initialized at 1.0.

Only pooling is ablated (C.6<!--ref:chun2024pcmepp--><!--anchor:section:Table%20C.6-->), against an unnamed pooling (row 9). The fixed-temperature collapse (15.7, VACSR<!--ref:wei2026vacsr--><!--anchor:section:tab%3Aloss-->) is compared with a copied PCME++ row, so it shows only that a scale of 1 fails (V2 issue 4).

Evidence on batch and schedule is thin. One ResNet-50 documentation page (ITRA<!--ref:itra-docs--><!--anchor:section:sections%201-4-->) finds batch 128 → 512 worth about +16.3 5K RSUM (computed; lr scaled with batch), 10 → 15 epochs worth +0.45 mean recall, and lr 1e-5 to 2e-5 best. CLIP4Clip<!--ref:luo2022clip4clip--><!--anchor:section:4.4--> finds batch 128 ≈ 256 on video. Neither measured ECCV or PMRP.

**Where sources diverge.** Tower learning rates span 5e-7 (MACCO<!--ref:li2026macco--><!--anchor:section:4-->) to 5e-5 (PCME++ text<!--ref:chun2024pcmepp--><!--anchor:section:Table%20B.1-->). No source ablates PCME++'s 10:1 text-to-visual ratio.

**Strength:** weak to moderate. One consequence [reader-inferred]: a masked-objective gain over our baseline may shrink over a baseline as strong as PCME++'s.

### 1.4 What has moved ECCV mAP@R and PMRP

**ECCV mAP@R.** At CLIP B/32 on COCO, the same-recipe changes rank as follows (B3 Δ table as corrected by V2):

| Change | mAP@R change | Caveat |
|---|---|---|
| GPO pooling (PCME++ C.6<!--ref:chun2024pcmepp--><!--anchor:section:Table%20C.6-->) | +2.6 | unnamed baseline |
| CUSA soft labels<!--ref:huang2024cusa--><!--anchor:section:Table%202--> | +2.3 | weak baseline |
| VACSR<!--ref:wei2026vacsr--><!--anchor:section:tab%3Aablation--> | +1.4 | own sigmoid baseline |
| VIB+PP+MSDA (PCME++ T3<!--ref:chun2024pcmepp--><!--anchor:section:Table%203-->) | +1.2 | |
| PCME++ vs InfoNCE (T1<!--ref:chun2024pcmepp--><!--anchor:section:Table%201-->) | +1.1 | 3 runs |
| SWA (T1) | +0.1 | |
| Probabilistic inference (C.7<!--ref:chun2024pcmepp--><!--anchor:section:Table%20C.7-->) | 0.0 | |

Moving from B/32 to B/16 under InfoNCE adds +2.1 (PCME++ T1<!--ref:chun2024pcmepp--><!--anchor:section:Table%201-->). Larger pre-CLIP effects (PVSE K=2, semi-hard mining, CutMix; ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204-->) did not recur at CLIP scale where tested.

None of these changes is a masked objective. ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204--> scores MLM-pretrained ViLT and VinVL, and Kwon et al.<!--ref:kwon2023flavavwsd--><!--anchor:quote:we%20adopted%20two%20SOTA%20zero-shot%20ITM%20models%2C%20CLIP%20and%20FLAVA--> run FLAVA on VWSD, but neither ablates the masked objective. Within the searched literature, our 3-seed contrast appears to be the only within-recipe measurement (V1's narrowed claim A): +0.02 mAP@R and +0.39 PMRP [project]. On mAP@R it is a null result.

**PMRP is the weakest instrument.**
- Its pseudo-positives are 65.3% / 56.6% precise against human labels (T2<!--ref:chun2022eccvcaption--><!--anchor:section:Table%202-->).
- Its Kendall τ with mAP@R is 0.20 (T5<!--ref:chun2022eccvcaption--><!--anchor:section:Table%205-->).
- It records object presence only (Sec 4.2<!--ref:chun2022eccvcaption--><!--anchor:section:4.2-->).
- Two definitions are in use (Pishdad<!--ref:pishdad2022uncertainty--><!--anchor:section:tab%3Aresults_coco-->).
- No later paper in the search reports ECCV-style PMRP.
- T4 and App. D.3<!--ref:chun2022eccvcaption--><!--anchor:section:App.%20D.3--> disagree (row 2). Our 55.31 reproduction supports T4 for CLIP B/32.

**Where our numbers sit (computed).** Fine-tuning raised PMRP by 1.17, and multilearner added a third of that (0.39). Multilearner is still 0.82 below zero-shot CLIP L/14, inside a 25-model band of 10.75. The gain is detectable at a seed std of about 0.05, but it is not large.

**Strength:** moderate for the B/32 ranking; weak for every PMRP contrast.

### 1.5 Masking and polysemy

**One-to-many.** Masked inputs have served as "less specific" views only at pretraining scale:
- ProLIP: masked inclusion alone moved DataComp retrieval by −0.4 and HierarCaps recall by +3.1 (C.3, C.4<!--ref:chun2025prolip--><!--anchor:section:Tables%20C.3%2C%20C.4-->, corrected by V1).
- COSMOS: a masked-text view lost to sentence crops (Supp. T17<!--ref:kim2025cosmos--><!--anchor:section:Supp.%20Table%2017-->).
- A-CLIP: a single random 50% view fell below full-image CLIP, while an attention-selected one rose above it (T2a<!--ref:yang2023aclip--><!--anchor:section:Table%202a-->).

Pretrained CLIP already orders captions by generality (HierarCaps<!--ref:alper2024hierarcaps--><!--anchor:section:Table%201-->). MERU<!--ref:desai2023meru--><!--anchor:section:Table%201--> and HyCoCLIP<!--ref:pal2025hycoclip--><!--anchor:section:Table%202--> encode generality with crops and phrases rather than masks. No study measures masked views on ECCV or PMRP.

**Lexical ambiguity.** On VWSD, the gains came from adding context:
- LLM definitions: +4.79 accuracy (Kritharoula T3<!--ref:kritharoula2023vwsd--><!--anchor:section:Table%203-->, computed).
- Definitions plus translation: 60.5 → 69.1 English Hit@1 (UAlberta<!--ref:ualberta2023vwsd--><!--anchor:section:Table%203-->).
- Prompting: Bhattacharya<!--ref:bhattacharya2026vwsd--><!--anchor:section:Table%20IV-->, though its 0.6250 is unattainable on 463 items (V1 #77).

Image augmentation did not help (Bhattacharya<!--ref:bhattacharya2026vwsd--><!--anchor:section:Table%20V-->), and no VWSD system found in the search used MLM. CLIP's text encoder superposes the senses of a polysemous word (White & Cotterell<!--ref:white2022schrodinger--><!--anchor:section:abstract-->, abstract only). On 463 English items, Hit@1 near 60% has a standard error of about 2.3 points (computed).

**Interpretation [reader-inferred].** Masking removes context, while VWSD rewards adding it. A masked model's plausible route to sense resolution is image-conditioned MLM supplying the word the image implies. Two hypotheses fit our "PMRP up, mAP@R flat" result:
- **H1, object grounding.** The MLM learns object words from pixels; content words need the image most (Bitton<!--ref:bitton2021dataefficient--><!--anchor:section:Table%201-->), and PMRP rewards object-label overlap.
- **H2, softer similarity.** The auxiliary losses lower hard-negative pressure, which ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204--> ties to higher PMRP (App. D.3 reverses this).

Existing checkpoints can test both (E0).

**Strength:** weak; no direct test exists.

**Convergence map.**
- Moderate: MLM over MIM (5 sources); loss-side one-to-many gains of +0.1 to +2.3 mAP@R (3 sources at B/32); re-ranking raises R@K (4 sources).
- Emerging: 15% MLM is too low (4 sources, ITM or from scratch); pooled-conditioned reconstruction (1 source).
- Gap: a masked objective ablated on ECCV, PMRP or VWSD (none outside this project).

## 2. Evidence table

| # | Lever | Concrete change | Best-evidence effect | Setting | Sources | Strength (reason) | Verification |
|---|---|---|---|---|---|---|---|
| 1 | Image-conditioned MLM | MLM decoder over image tokens, dropped at test | CUHK Rank-1 68.19 → 71.23 over InfoNCE; 70.52 → 73.38 over SDM+ID; BiLMa rerun 73.01 → 73.16 | CLIP B/16 FT, person retrieval, dual encoder | IRRA T4<!--ref:jiang2023irra--><!--anchor:section:Table%204-->, BiLMa T2<!--ref:fujii2023bilma--><!--anchor:section:Table%202--> | moderate: sign agrees, size conflicts | confirmed (V1 #34, #37) |
| 2 | Pixel MIM on top of MLM | add a text-conditioned pixel decoder | Flickr ZS IR/TR R@1 66.08/78.10 → 62.12/76.90; COCO FT −0.8/+0.2 | CLIP-ViT + RoBERTa, ITM; VLP, ITM | METER T7<!--ref:dou2022meter--><!--anchor:section:Table%207-->, MAMO T6<!--ref:zhao2023mamo--><!--anchor:section:Table%206--> | moderate: both use ITM; BiLMa<!--ref:fujii2023bilma--><!--anchor:section:App.%20Table%205--> agrees in an FT dual encoder | METER added by V1 |
| 3 | Pixel MIM alone | text-guided SimMIM as the only auxiliary | Rank-1 70.61 → 72.16 | CLIP B/16 FT, person | VFE-TPS T6<!--ref:shen2025vfetps--><!--anchor:section:Table%206--> | weak: one run | venue corrected; ratio claim not found (V1 #42) |
| 4 | MIM target | pixels → features | Flickr ZS 57.3/41.1 → 62.3/41.4; COCO FT 75.8/59.1 → 76.6/59.2 | from scratch; VLP ITM | MaskCLIP T9<!--ref:dong2023maskclip--><!--anchor:section:Table%209-->, MAMO T9<!--ref:zhao2023mamo--><!--anchor:section:Table%209--> | moderate: small in FT | confirmed |
| 5 | MLM weight | 1 → 0.05 | Flickr ZS 51.7/32.1 → 70.1/45.6 (CLIP 52.9/32.8) | from scratch, text-only MLM | MaskCLIP T6f<!--ref:dong2023maskclip--><!--anchor:section:Table%206f--> | weak for FT | confirmed |
| 6 | Text mask rate | 15% → 30-60% | COCO FT R@1 +3.00/+4.11; Flickr FT +1.0/+1.6 | ITM models | Verma T4<!--ref:verma2022uniform--><!--anchor:section:Table%204-->, MaskVLM T7<!--ref:kwon2023maskvlm--><!--anchor:section:Table%207--> | moderate direction, weak transfer | Verma seeds are fine-tuning only (V1 #20) |
| 7 | MLM image access | mask one modality per pass, so the MLM sees the full image | COCO FT R@1 59.5/76.0 → 60.1/76.3 | VLP, ITM | MaskVLM App. T6<!--ref:kwon2023maskvlm--><!--anchor:section:App.%20Table%206--> | weak: one pair | confirmed |
| 8 | Pooled conditioning | decoders read the [CLS] instance embedding | COCO FT T2I/I2T R@1 46.8/62.7 (contrastive), 46.3/62.3 (vanilla) → 47.5/63.4 | non-CLIP dual encoder, 5.3M pairs | ConLIP T1<!--ref:luo2022conlip--><!--anchor:section:Table%201--> | weak: single runs | added and verified by V1 |
| 9 | Attentive MIM masking | mask the top-50% caption-similar patches | COCO ZS mTR/mIR 14.4/15.2 → 18.3/18.4 | CoCa from scratch | SyCoCa T5<!--ref:ma2024sycoca--><!--anchor:section:Table%205--> | weak: low recall | confirmed |
| 10 | Masked view as a general view | inclusion loss on 75%-masked inputs | DataComp retrieval 53.6 → 53.2; HierarCaps 44.8 → 47.9 | from scratch | ProLIP C.3, C.4<!--ref:chun2025prolip--><!--anchor:section:Tables%20C.3%2C%20C.4--> | weak | corrected (V1 #55) |
| 11 | Soft targets | unimodal-teacher similarities as targets | mAP@R 35.1 → 37.4; 5K RSUM 422.6 → 429.7 | CLIP B/32 FT on COCO | CUSA<!--ref:huang2024cusa--><!--anchor:section:Tables%201%2C%202%2C%205--> | moderate: weak baseline, one run | recipe not found (V2) |
| 12 | Probabilistic loss | PCME++ vs InfoNCE | mAP@R 39.0 → 40.1; 5K RSUM 444.1 → 452.1 (computed in B3) | CLIP B/32 FT, 3 runs | PCME++ T1<!--ref:chun2024pcmepp--><!--anchor:section:Table%201--> | moderate to strong | confirmed |
| 13 | Pooling | GPO vs an unnamed pooling | mAP@R 37.4 → 40.0 | CLIP B/32 FT, PCME++ loss | PCME++ C.6<!--ref:chun2024pcmepp--><!--anchor:section:Table%20C.6--> | weak: baseline unknown | baseline not found (V2) |
| 14 | Recipe bundle | PCME++ recipe vs ours | mAP@R 36.94 [project] vs 39.0 | CLIP B/32 FT | PCME++ B.1<!--ref:chun2024pcmepp--><!--anchor:section:Table%20B.1--> | weak for attribution | confirmed |
| 15 | Batch, epochs | 128 → 512 with lr scaled; 10 → 15 epochs | mean recall 72.14 → 74.85; 73.98 → 74.43 | ResNet-50 CLIP FT | ITRA<!--ref:itra-docs--><!--anchor:section:sections%201-4--> | weak: grey literature | page re-fetched (V2) |
| 16 | Re-ranking; late interaction | cross-encoder over the top-k; token max-similarity | COCO 5K rsum 471.9 → 494.7 (computed in B2); ZS R@1 25.0/14.7 → 30.5/18.5 | ALBEF-scale; from scratch | LoopITR<!--ref:lei2022loopitr--><!--anchor:section:Tables%207%20and%209-->, FILIP T4<!--ref:yao2022filip--><!--anchor:section:Table%204--> | moderate for R@K; no ECCV/PMRP data | FILIP corrected (V2) |
| 17 | Multiple embeddings | K = 1 → 2 | mAP@R 33.98 → 40.26 (PVSE); 42.4 → 43.5 (DivE) | pre-CLIP | ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204-->, DivE<!--ref:kim2023dive--><!--anchor:section:Table%20A1--> | weak: annotator bias | confirmed |
| 18 | Query context (VWSD) | LLM definitions added to the query | VWSD accuracy 63.28 → 68.07 | zero-shot CLIP | Kritharoula T3<!--ref:kritharoula2023vwsd--><!--anchor:section:Table%203--> | moderate; no masking | confirmed |

## 3. Contradictions and resolutions

| # | Claim A | Claim B | Resolution |
|---|---|---|---|
| 1 | IRRA T4<!--ref:jiang2023irra--><!--anchor:section:Table%204-->: image-conditioned MLM +2.86 Rank-1 over SDM+ID | BiLMa T2<!--ref:fujii2023bilma--><!--anchor:section:Table%202-->: +0.15 over its own SDM+ID run | **Unresolved on size; the sign agrees.** The labs' SDM+ID runs differ by 2.49 (computed), about the size of IRRA's gain. The likely gain is small, positive and baseline-dependent, in line with our +2.36 rsum (p=0.11). |
| 2 | ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204-->: PMRP falls as mining gets harder (56.67 → 54.37) | Same paper, App. D.3<!--ref:chun2022eccvcaption--><!--anchor:section:App.%20D.3-->: the means rise (44.84 → 47.17) | **Partly unresolved.** 11 of 25 rows conflict (V2). D.3's i2t column may be mislabelled, and our reproduction matches T4. Use T4 for CLIP rows, but not the mining trend. |
| 3 | MaskCLIP T6f<!--ref:dong2023maskclip--><!--anchor:section:Table%206f-->: MLM at weight 1 falls below CLIP | IRRA<!--ref:jiang2023irra--><!--anchor:section:Table%204-->, MaskVLM<!--ref:kwon2023maskvlm--><!--anchor:section:Table%205-->, MAMO<!--ref:zhao2023mamo--><!--anchor:section:Table%206-->: unit-weight MLM helps | **Resolved (conditional).** MaskCLIP's MLM is text-only on a from-scratch shared encoder, so it competes with contrastive learning. The others are image-conditioned, as ours is. |
| 4 | VFE-TPS T6<!--ref:shen2025vfetps--><!--anchor:section:Table%206-->: pixel MIM +1.55 Rank-1 | BiLMa App. T5<!--ref:fujii2023bilma--><!--anchor:section:App.%20Table%205-->, METER<!--ref:dou2022meter--><!--anchor:section:Table%207-->, SyCoCa random<!--ref:ma2024sycoca--><!--anchor:section:Table%205-->: it hurts | **Resolved (conditional).** It helps as the sole auxiliary or with caption-relevant masks. On top of MLM with random masks it is useless (MACCO T9<!--ref:li2026macco--><!--anchor:section:Table%209-->). V1 drops B1's ratio explanation. |
| 5 | METER T7<!--ref:dou2022meter--><!--anchor:section:Table%207-->: MIM hurts | VL-BEiT T4<!--ref:bao2022vlbeit--><!--anchor:section:Table%204-->: MIM +1.0/+1.6 FT R@1 | **Resolved (conditional).** VL-BEiT's MIM is unimodal, with its own ImageNet-22K data and tokenizer targets, so objective and data are confounded. |
| 6 | CUSA<!--ref:huang2024cusa--><!--anchor:section:Table%202-->: soft targets +2.3 mAP@R | PCME++ T3<!--ref:chun2024pcmepp--><!--anchor:section:Table%203-->: pseudo-positives +0.1 | **Resolved (conditional).** CUSA uses external teachers and has a baseline 3.9 mAP@R weaker (computed). Caveat: CUSA is a single run. |
| 7 | Geigle<!--ref:geigle2022rerank--><!--anchor:section:Table%201-->, LoopITR<!--ref:lei2022loopitr--><!--anchor:section:Tables%207%20and%209-->: re-ranking +9.7 to +22.8 rsum | ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204-->: re-ranked BLIP trails dual encoders on mAP@R | **Unresolved.** Systems and metrics differ, and no same-model comparison exists (V2 Sec 4). |
| 8 | ECCV Caption T4<!--ref:chun2022eccvcaption--><!--anchor:section:Table%204-->: PVSE K=2 +6.28 mAP@R | PCME T E.2<!--ref:chun2021pcme--><!--anchor:section:Table%20E.2-->: K=2 lowers CUB R-P (22.34 → 19.67) | **Resolved (conditional).** CUB positives are class-level, PVSE was an annotator, and on strong backbones the gain is about +1. |
| 9 | PCME++ C.6<!--ref:chun2024pcmepp--><!--anchor:section:Table%20C.6-->: GPO +2.6 mAP@R | PCME++ never names the pooling used without GPO | **Unresolved.** The baseline is unknown and the loss is PCME++'s, so this is no evidence for a CLIP-pooled InfoNCE fine-tune. Lever R3 tests it. |

#### Cross-paper tension inventory (#262)

```yaml
cross_paper_tensions:
  - {pair_id: CP-001, paper_a: jiang2023irra, paper_b: fujii2023bilma, candidate_basis: "shared construct/outcome/measure", overlap_topic: "image-conditioned MLM gain over SDM+ID, CLIP B/16 on CUHK-PEDES Rank-1", a_finding: "70.52 -> 73.38", a_evidence_pointer: "B1 5.1, IRRA Table 4", b_finding: "73.01 -> 73.16", b_evidence_pointer: "B1 5.2, BiLMa Table 2", pair_assessment: contradiction, resolution_status: flagged_unresolved, scholar_confirmation: pending}
  - {pair_id: CP-002, paper_a: dong2023maskclip, paper_b: jiang2023irra, candidate_basis: "opposite finding direction", overlap_topic: "MLM at loss weight 1 next to a contrastive loss", a_finding: "weight 1 falls below CLIP", a_evidence_pointer: "B1 1.1, MaskCLIP Table 6f", b_finding: "unit-weight MLM adds Rank-1", b_evidence_pointer: "B1 5.1, IRRA Table 4", pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis, resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 3", scholar_confirmation: pending}
  - {pair_id: CP-003, paper_a: shen2025vfetps, paper_b: fujii2023bilma, candidate_basis: "opposite finding direction", overlap_topic: "pixel MIM as an auxiliary in a CLIP person-retrieval fine-tune", a_finding: "+1.55 Rank-1", a_evidence_pointer: "B1 5.3, VFE-TPS Table 6", b_finding: "-0.52 Rank-1", b_evidence_pointer: "B1 5.2, BiLMa App. Table 5", pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis, resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 4", scholar_confirmation: pending}
  - {pair_id: CP-004, paper_a: dou2022meter, paper_b: bao2022vlbeit, candidate_basis: "opposite finding direction", overlap_topic: "adding MIM to a masked VL objective, Flickr retrieval", a_finding: "MIM lowers ZS R@1", a_evidence_pointer: "V1 Sec 4, METER Table 7", b_finding: "MIM adds +1.0/+1.6 FT R@1", b_evidence_pointer: "B1 4.3, VL-BEiT Table 4", pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis, resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 5", scholar_confirmation: pending}
  - {pair_id: CP-005, paper_a: huang2024cusa, paper_b: chun2024pcmepp, candidate_basis: "shared construct/outcome/measure", overlap_topic: "soft or pseudo-positive targets, CLIP B/32 FT, ECCV mAP@R", a_finding: "+2.3", a_evidence_pointer: "B2 S1, CUSA Table 2", b_finding: "+0.1 (PP alone)", b_evidence_pointer: "B2 P3, PCME++ Table 3", pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis, resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 6", scholar_confirmation: pending}
  - {pair_id: CP-006, paper_a: lei2022loopitr, paper_b: chun2022eccvcaption, candidate_basis: "agent-noted cross-cluster", overlap_topic: "does cross-encoder re-ranking improve COCO retrieval quality", a_finding: "+22.8 rsum", a_evidence_pointer: "B2 R5, LoopITR Tables 7 and 9", b_finding: "re-ranked BLIP below dual encoders on mAP@R", b_evidence_pointer: "B2 E1, ECCV Caption Table 4", pair_assessment: conditional_difference, resolution_status: flagged_unresolved, scholar_confirmation: pending}
  - {pair_id: CP-007, paper_a: chun2022eccvcaption, paper_b: chun2021pcme, candidate_basis: "opposite finding direction", overlap_topic: "PVSE K=2 vs K=1", a_finding: "+6.28 mAP@R (COCO)", a_evidence_pointer: "B3 leaderboard, ECCV Caption Table 4", b_finding: "-2.67 R-P (CUB, computed)", b_evidence_pointer: "B2 P1, PCME Table E.2", pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis, resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 8", scholar_confirmation: pending}
  - {pair_id: CP-008, paper_a: chun2025prolip, paper_b: kim2025cosmos, candidate_basis: "shared construct/outcome/measure", overlap_topic: "masked views as training views and retrieval", a_finding: "masked inclusion alone -0.4 retrieval avg", a_evidence_pointer: "V1 row 55, ProLIP Table C.4", b_finding: "masked-text view below sentence crops", b_evidence_pointer: "V1 row 69, COSMOS Supp. Table 17", pair_assessment: no_material_conflict, resolution_status: not_applicable, scholar_confirmation: pending}
```

**Coverage note.** The corpus holds B1 (18 entries, plus METER and ConLIP from V1), B2 (20), B3 (15) and B4 (16 core and 9 short), with overlaps across files. I considered 10 candidate pairs (shared construct, opposite direction, cross-cluster) and list 8. The other two are within-paper conflicts (rows 2 and 9). This is a **scoped advisory scan, not complete pairwise detection**; bibliographic coupling was not used, and the scholar confirms each `resolution_pointer`.

## 4. Gap analysis

Tags: **[research]** = candidate contribution on the broad question; **[engineering]** = recipe; **[evaluation]** = measurement.

1. **[research]** Why does a masked objective move PMRP (+0.39) but not mAP@R (+0.02)? Is it object grounding (H1) or softer similarity (H2)? No source tests either.
2. **[research]** Does the effect grow with how strongly the MLM depends on the image (25% vs 100% of patches visible; random vs content-word masks)? Sweeps exist only for ITM or from-scratch models.
3. **[research]** Does routing reconstruction through the retrieval embedding (routes 2 and 3) turn the masked signal into one-to-many gains? The only test is ConLIP<!--ref:luo2022conlip--><!--anchor:section:Table%201-->: non-CLIP, R@1 only.
4. **[research]** Can masked captions or images, used as under-specified views with inclusion or multi-positive targets, teach one-to-many structure in a fine-tune? ProLIP<!--ref:chun2025prolip--><!--anchor:section:Tables%20C.3%2C%20C.4--> tried this only from scratch, and never on ECCV or PMRP.
5. **[research]** Does a masked objective add anything on top of a loss that already models one-to-many matching? No 2×2 exists.
6. **[research, lexical]** Does masked COCO fine-tuning change VWSD Hit@1 or MRR relative to contrastive fine-tuning? No data exist.
7. **[evaluation]** Do PMRP differences of 0.1 to 0.5 mean anything, given that T4 and D.3 conflict and PMRP is object-only?
8. **[engineering]** Which recipe element explains the 2.06 mAP@R gap to PCME++'s InfoNCE? What do batch size and schedule do to ECCV and PMRP?
9. **[evaluation, tangential]** How does a model's dual-encoder ranking compare with its own ITM re-ranking on ECCV, PMRP or CxC? This is unreported.
10. **[research, theory]** No framework links reconstruction to ranking under multiplicity. Inclusion (ProLIP<!--ref:chun2025prolip--><!--anchor:section:3.3-->) and entailment (MERU<!--ref:desai2023meru--><!--anchor:section:Table%201-->, HierarCaps<!--ref:alper2024hierarcaps--><!--anchor:section:Table%201-->) have not been connected to masked reconstruction in any source found.

## 5. Ranked candidate levers (~250 GPU-hours)

**Costs and detection [project].**
- Per seed on one A6000: contrastive 4.2 h, fusion_none 6.9 h, multilearner 7.2 h.
- Masked-side arms reuse the existing 3-seed baselines.
- Recipe and loss changes go to both arms, at 11.4 h per seed pair (computed).
- 3 seeds detect about +4.3 rsum, +0.6 mAP@R and +0.15 PMRP.
- Effect ranges are reader-inferred extrapolations, not predictions.

| ID | Lever (change) | Expected Δ mAP@R / PMRP / rsum (evidence) | Mechanism tested; what a null teaches | Novelty (where searched) | Code | GPU-h, 1 seed / 3 seeds | Baseline gets it? |
|---|---|---|---|---|---|---|---|
| E0 | Diagnostics on existing checkpoints: VWSD (463 English items); PMRP by direction and object category; learned logit scale per arm; a masked-caption probe (does similarity fall with the mask ratio?) | none (measurement) | Separates H1 from H2; first reading of a masked fine-tune on lexical ambiguity | VWSD after masked fine-tuning not found (V1 A4, A5) | small | ~0 (minutes) | all arms evaluated |
| M1 | MLM reads the clean pass's full image tokens, with and without stop-gradient as in MACCO<!--ref:li2026macco--><!--anchor:section:3--> | 0 to +0.5 / 0 to +0.4 / 0 to +4 (MaskVLM App. T6<!--ref:kwon2023maskvlm--><!--anchor:section:App.%20Table%206-->; Section 2 rows 1, 7) | Dose-response of image conditioning; a flat PMRP weakens the image-conditioning account | Only in ITM-based VLP pretraining; none found in a CLIP dual encoder (V1 claim B) | small | 7.2 / 21.6 | no |
| M2 | Text masking at 30-60%, or content-word masking | 0 to +0.5 / 0 to +0.4, largest under H1 / 0 to +4 (Verma T4<!--ref:verma2022uniform--><!--anchor:section:Table%204-->, Bitton T1<!--ref:bitton2021dataefficient--><!--anchor:section:Table%201-->) | Sharp test of H1: masking object words should raise PMRP the most | No dual-encoder CLIP sweep found (V1 claim B) | config (rate); small (content words) | 7.2 / 21.6 per setting | no |
| M3 | Decoders conditioned on the other modality's pooled embedding (MLM on the image, MAE on the text), so masked words cannot leak through the clean text embedding [reader-inferred] | unknown / unknown / 0 to +4 (ConLIP T1<!--ref:luo2022conlip--><!--anchor:section:Table%201-->, +0.7 / +0.7 R@1) | Route 3: a null means this route is not the bottleneck. V1 does not say which embedding ConLIP used, so re-read it first. | Not found for CLIP or on ECCV/PMRP (V1 query B3) | small to moderate | 7.2 / 21.6 | no |
| M4 | Caption-relevant MAE masking, with patch scores from the clean pass | unknown / unknown / −2 to +3 (SyCoCa T5<!--ref:ma2024sycoca--><!--anchor:section:Table%205-->) | Can the image side be made to use the text? Today our MAE ignores it. | From scratch only | small | 7.2 / 21.6 | no |
| M5 | MAE off (MLM only); then feature targets | ±0.3 / ±0.2 / −2 to +2 (Section 2 rows 2, 4) | The decomposition any paper claim needs | Done outside our setting | config; moderate for feature targets | ≤7.2 / ≤21.6 (MAE-off cost not timed) | no |
| M6 | Masked views as under-specified positives (inclusion, or soft multi-positive targets), with and without reconstruction | −0.5 to +1 / unknown / −3 to +1 (ProLIP C.3, C.4<!--ref:chun2025prolip--><!--anchor:section:Tables%20C.3%2C%20C.4-->; COSMOS Supp. T17<!--ref:kim2025cosmos--><!--anchor:section:Supp.%20Table%2017-->) | Route 4 and the broad question: does masking model under-specification usefully? | From scratch only; never on ECCV/PMRP (B4 Q8, Q15, Q16; V1 A6) | moderate | ≤6.9 + 7.2 / ≤42.3 for two arms (not timed) | its contrastive version is the control |
| R1 | CUSA-style soft targets from semantic similarity, crossed 2×2 with masking | +0.1 to +2.3 / unknown / 0 to +7 (CUSA T1, T2<!--ref:huang2024cusa--><!--anchor:section:Tables%201%20and%202-->; PCME++ T3<!--ref:chun2024pcmepp--><!--anchor:section:Table%203-->) | Does masking add anything once the loss models one-to-many matching? | CUSA without masking; combination not found (B4 Sec 2.6) | small, plus an offline teacher pass (not timed) | 11.4 / 34.2 | yes |
| R2 | PCME++ learning rates: text 10× visual, layer-wise decay, visual tower frozen for 2 epochs | 0 to +2 / unknown / 0 to +3 (gap 2.06, but the recipe is bundled) | Does the masked effect survive a stronger baseline? | Used by PCME++<!--ref:chun2024pcmepp--><!--anchor:section:App.%20B.2-->, never ablated | small | 11.4 / 34.2 | yes |
| R3 | Mean pooling (config) first, GPO later (code) | −1 to +2.6 / unknown / sign unknown (PCME++ C.6<!--ref:chun2024pcmepp--><!--anchor:section:Table%20C.6-->) | Route 2: with mean pooling, token-level reconstruction shapes the retrieval vector directly | PCME++ C.6 only | config; larger for GPO | 11.4 / 34.2 | yes |
| R4 | Global batch 384-512 (bf16 or 3 GPUs) | unknown / unknown, possibly lower / 0 to +16 (ITRA<!--ref:itra-docs--><!--anchor:section:sections%201-4-->, ResNet-50) | Hard-negative pressure against multiplicity: R@K and PMRP may move apart | Not measured on ECCV/PMRP (B3) | config | ~11.4 / ~34.2 (throughput not measured) | yes |
| R5 | 15 epochs | small / unknown / 0 to +3 (ITRA<!--ref:itra-docs--><!--anchor:section:sections%201-4-->) | Schedule only | Engineering | config | 17.1 / 51.3 (computed) | yes |
| R6 | ITM head over the fusion module, re-ranking the top-k at test | unknown / possibly lower / +10 to +23 (Geigle<!--ref:geigle2022rerank--><!--anchor:section:Table%201-->, LoopITR<!--ref:lei2022loopitr--><!--anchor:section:Tables%207%20and%209-->; pre-CLIP) | Does test-time fusion help one-to-many matching? (gap 9) | Not found (V2 Sec 4) | larger | ≥7.2 / ≥21.6, plus evaluation (not timed) | yes: an ITM head too |

**(a) Ranked by expected gain on ECCV mAP@R and PMRP.**
1. **R1:** the largest same-backbone, same-data gain (+2.3). A PCME++-style loss (+1.1, larger code) is the fallback.
2. **R2:** could close part of the 2.06 gap.
3. **R3:** caveated, but mean pooling is config-only.
4. **M1:** PMRP side.
5. **M2.**
6. **M3.**
7. **M6.**
8. **M4.**
9. **R5.**
10. **R4:** R@K up, but the polysemy metrics may fall.
11. **M5:** about 0.
12. **R6:** no evidence of a gain on these metrics.

**(b) Ranked by information value for the broad question.**
1. **E0:** free; separates H1 from H2; first VWSD reading.
2. **M1:** dose-response.
3. **M2:** sharp test of H1.
4. **M6:** directly tests "masking models under-specification"; novel.
5. **M3:** route 3.
6. **R1 × masking:** does a masked claim survive a one-to-many-aware loss?
7. **R3 × masking:** route 2.
8. **M4.**
9. **M5:** necessary, but not novel.
10. **R2.**
11. **R6:** tangential.
12. **R4, R5.**

**Budget fit (computed).**
- E0 + M1 + M2 (one setting) + M3 + M6 (two arms) + R1 (2×2, reusing the existing arms) + R3 = 0 + 21.6 + 21.6 + 21.6 + 42.3 + 34.2 + 34.2 = 175.5 GPU-hours at 3 seeds.
- That leaves about 74.5 for R2 (34.2) or a second M2 setting.
- Reusing the existing baselines is valid only if the code changes leave them reproducible; otherwise add 11.4 h per seed pair.
- One-seed screening suits only R1 to R3 on mAP@R. The masked-side levers need 3 seeds.

## 6. Limitations of this review

- **No source re-read;** this rests on the Phase 2 reads and verifications.
- **Evidence quality.** Most evidence is single-run, from scratch or ITM re-ranked. Only IRRA<!--ref:jiang2023irra--><!--anchor:section:Table%204-->, BiLMa<!--ref:fujii2023bilma--><!--anchor:section:Table%202-->, VFE-TPS<!--ref:shen2025vfetps--><!--anchor:section:Table%206-->, CUSA<!--ref:huang2024cusa--><!--anchor:section:Table%201-->, PCME++<!--ref:chun2024pcmepp--><!--anchor:section:Table%20C.10--> and MACCO<!--ref:li2026macco--><!--anchor:section:Table%2015--> fine-tune a pretrained CLIP, and only PCME++ reports variance on our metrics.
- **Effect ranges are directional.** They map Rank-1 or R@1 results onto rsum and mAP@R with no valid conversion.
- **PMRP rests on one paper**, and that paper is internally inconsistent.
- **Coverage skews:** a US web index, arXiv-heavy reads, journals and ACL PDFs under-sampled.
- **Costs** marked "not timed" are bounds, and project numbers were not re-derived here.
- **Anchors** come from the bibliographies' locators (no page anchors). The contradiction search was scoped, not exhaustive.

<details><summary>Source key (outside the word count)</summary>

A-CLIP: Yang et al. 2023, ICCV. ALBEF: Li et al. 2021, NeurIPS. Bhattacharya et al. 2026, arXiv 2602.06799. BiLMa: Fujii and Tarashima 2023, ICCV Workshops. Bitton et al. 2021, Findings of EMNLP. CLIP4Clip: Luo et al. 2022, Neurocomputing. ConLIP: Luo et al. 2022, Findings of EMNLP. COSMOS: Kim et al. 2025, CVPR. CUSA: Huang et al. 2024, AAAI. DivE: Kim et al. 2023, CVPR. ECCV Caption: Chun et al. 2022, ECCV. FILIP: Yao et al. 2022, ICLR. FLAIR: Xiao et al. 2025, CVPR. Geigle et al. 2022, TACL. GroVE: Venkataramanan et al. 2025, UAI. HierarCaps: Alper and Averbuch-Elor 2024, ECCV. HyCoCLIP: Pal et al. 2025, ICLR. IRRA: Jiang and Ye 2023, CVPR. ITRA: documentation page, n.d. (grey literature). Kritharoula et al. 2023, EMNLP. Kwon et al. 2023, ACL (FLAVA on VWSD). Llip: Lavoie et al. 2024, ICML. LoopITR: Lei et al. 2022, arXiv. M3AE: Geng et al. 2022, arXiv. MACCO: Li et al. 2026, ACL. MAMO: Zhao et al. 2023, SIGIR. MaskCLIP: Dong et al. 2023, CVPR. MaskVLM: Kwon et al. 2023, ICLR. MERU: Desai et al. 2023, ICML. METER: Dou et al. 2022, CVPR. PCME: Chun et al. 2021, CVPR. PCME++: Chun 2024, ICLR. Pishdad et al. 2022, arXiv. ProbVLM: Upadhyay et al. 2023, ICCV. ProLIP: Chun et al. 2025, ICLR. Sun 2023, arXiv 2209.02127. SyCoCa: Ma et al. 2024, ICML. TIPS: Maninis et al. 2025, ICLR. TULIP: Tang et al. 2025, arXiv. UAlberta at SemEval-2023 Task 1, 2023. VACSR: Wei et al. 2026, ICML. Verma et al. 2022, arXiv. VFE-TPS: Shen et al. 2025, Knowledge-Based Systems. VL-BEiT: Bao et al. 2022, arXiv. White and Cotterell 2022, arXiv.
</details>
