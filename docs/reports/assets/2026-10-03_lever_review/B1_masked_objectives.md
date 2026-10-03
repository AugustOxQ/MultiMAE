> Supporting file for docs/reports/auto/v1/2026-10-03_lever_review.md. Phase 2 bibliographies are as written; V1/V2 list the corrections, and the report uses corrected values.

# B1: Masked-reconstruction auxiliary objectives combined with a contrastive loss

Theme B1 of the MultiMAE integrative review (ARS deep-research, Phase 2, lit-review mode). Bibliography agent output, 2026-10-03. Numbers were read from the papers' own text or tables; every number carries a locator. No synthesis across themes.

## 1. Search strategy

**Sources searched.** Web search (general engine; hits came from arXiv, CVF open access, OpenReview, PMLR, ACL Anthology, mlanthology, ECVA), arXiv abstract pages (title, authors, date, and the comments field for venue), arXiv full-text PDFs (text extracted locally with pypdf), and arXiv or ar5iv HTML for tables whose check marks were lost in PDF text extraction. Semantic Scholar and OpenReview search were not queried directly.

**Query strings** (14 searches, last run 2026-10-03):
1. `fine-tuning pretrained CLIP with masked language modeling auxiliary loss image-text retrieval COCO Flickr30k`
2. `TIPS text-image pretraining with spatial awareness ICLR 2025 masked image modeling self-distillation ablation retrieval`
3. `uniform masking prevails vision-language pretraining masking rate retrieval`
4. `IRRA implicit relation reasoning cross-modal masked language modeling CLIP fine-tuning text-to-image person retrieval ablation`
5. `"ECCV Caption" OR "PMRP" masked language modeling image-text retrieval mAP@R`
6. `fine-tune pretrained CLIP with masked image modeling auxiliary loss retrieval Flickr30K COCO 2024 2025 arXiv`
7. `BiLMa bidirectional local-matching text-based person re-identification masked image modeling IRRA`
8. `text-based person retrieval CLIP masked image modeling cross-modal reconstruction ablation mask ratio IRRA improvement 2024`
9. `CLIP fine-tuning image-text retrieval MSCOCO Flickr30K auxiliary masked language modeling cross-attention decoder discarded at inference 2025`
10. `arXiv 2025 2026 contrastive language-image pretraining masked reconstruction objective ablation zero-shot retrieval "masked" "reconstruction" CLIP auxiliary loss weight`
11. `"fine-tuning" CLIP "MS-COCO" retrieval "masked image modeling" OR "masked language modeling" auxiliary objective improves rsum ViT-B/32`
12. `"Crisscrossed Captions" OR "ECCV Caption" evaluation masked autoencoder vision-language model retrieval results`
13. and 14. Venue checks for TULIP and SyCoCa.

Plus direct arXiv lookups of the examples named in the task (MaskCLIP, MILAN, M3AE, A-CLIP, SigLIP 2, TULIP, TIPS, SILC, FLAVA, VL-BEiT, MaskVLM, MAMO).

**Date range.** Publications 2021 to 2026 (newest included: MACCO, arXiv 2026-06-11). Last searched 2026-10-03.

**Inclusion criteria.** (a) A masked reconstruction or masked prediction objective (image MIM/MAE, text MLM, or a cross-modal version) used alongside, or ablated against, an image-text contrastive or matching objective; or a masking design ablated in a vision-language model. (b) At least one reported number that isolates a design choice (loss weight, target, ratio, strategy, which modality is reconstructed or conditions the reconstruction) on retrieval, or on the paper's primary metric when no retrieval ablation exists. (c) The number is readable in the full text.

**Exclusion criteria.** Token dropping for efficiency with no reconstruction target, unless the masking strategy was ablated on retrieval (A-CLIP kept for that reason; FLIP, cluster masking 2405.08815 and word-frequency text masking 2412.16148 excluded). Image-only MIM with no text or contrastive term (MVP 2203.05175, MILAN 2208.06049, SemMAE, AttMask). Withdrawn papers (TIR-SE, 2307.09059, withdrawn on arXiv over an author dispute). Retrieval paradigms far from a dual encoder (LexLIP 2302.02908, sparse lexicon retrieval). Self-distillation without masked prediction (SILC 2310.13355; its masked successors TIPS and SigLIP 2 are included). Works from the 2026-10-02 scan were re-read only where new numbers were needed (MaskCLIP, MaskVLM, MAMO re-verified; Weers et al., FLIP, MAP not re-read).

**Counts.** Identified: about 120 search result links (with duplicates) plus 12 named examples. Screened by title and abstract: about 45 unique papers. Full text read: 22 (MaskCLIP, Verma et al., Bitton et al., M3AE, A-CLIP, MILAN, TIPS, SigLIP 2, TULIP, SILC (partial), FLAVA, VL-BEiT, MaskVLM, MAMO, SyCoCa, IRRA, DetailCLIP, BiLMa, VFE-TPS, TIR-SE (partial, then excluded), LexLIP (partial, then excluded), MACCO). Included: 18. The count exceeds the 8 to 15 target because sub-theme 5 needed four fine-tuning sources to be answered at all.

**Coverage skew.**
- Most retrieval evidence on masked objectives comes from 2022 to 2023 from-scratch pretraining, often with fusion encoders whose retrieval is reranked by an ITM head (MaskVLM, MAMO, VL-BEiT, Verma et al.). Dual-encoder zero-shot retrieval ablations exist only in MaskCLIP, SyCoCa, FLAVA, A-CLIP and TIPS (Flickr only).
- Fine-tuning an already pretrained CLIP with masked objectives turned up only in text-to-person retrieval (IRRA, BiLMa, VFE-TPS) and compositionality (MACCO, no retrieval). No source fine-tunes CLIP with MLM or MAE and reports COCO or Flickr retrieval.
- No included source reports ECCV Caption mAP@R or R-Precision, PMRP, CxC, or VWSD.

## 2. Annotated bibliography

Abbreviations: R@1 = Recall@1; I2T/TR = image-to-text (text retrieval); T2I/IR = text-to-image (image retrieval); ZS = zero-shot; FT = fine-tuned; MLM = masked language modeling; MIM = masked image modeling.

### Sub-theme 1. Weighting the masked loss against the contrastive loss

#### 1.1 MaskCLIP
- **Citation:** Dong, X. et al. (2023). MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining. CVPR 2023. arXiv:2208.12262.
- **Tier:** peer-reviewed (CVPR 2023). Re-verified from the 2026-10-02 scan.
- **Read scope:** full text (arXiv PDF): Sec 3.3 to 3.4, 4.1, 4.4, Tables 1, 5, 6a to 6f, 7, 9, App. B.
- **Setting:** from-scratch pretraining. ViT-B/16 image encoder and a 12-layer, 512-wide text encoder. YFCC-15M, 25 epochs, batch 4096. Image branch: 75% random masking; a 1-layer decoder predicts EMA-teacher features, mapped to soft codewords by an online quantizer and trained with cross-entropy. Text branch: 20% of tokens masked (Sec 4.1), text-only BERT-style MLM (not image-conditioned) through a small text decoder. Total loss L_I + L_T + λ L_Dist + β L_MLM (Eq. 8), defaults λ = β = 0.05.
- **Change tested and baseline:** CLIP trained identically (Tables 1 and 5). One-at-a-time ablations of each weight, the target and the components.
- **Effect:**
  - MLM weight β (Table 6f; ImageNet ZS top-1; Flickr30K ZS I2T/T2I R@1): β = 1: 36.5; 51.7/32.1. β = 0.1: 44.3; 69.2/45.9. β = 0.05 (default): 44.5; 70.1/45.6. β = 0.01: 43.2; 70.6/45.6. CLIP baseline: 37.6; 52.9/32.8 (Table 1). At β = 1 all three numbers fall below plain CLIP.
  - Distillation weight λ (Table 6e; ImageNet ZS/linear/FT): λ = 1: 38.5/68.2/82.5. λ = 0.1: 44.4/73.5/83.5. λ = 0.05: 44.5/73.7/83.6. λ = 0.01: 43.6/73.0/83.4.
  - Target chain (Table 9; Flickr30K ZS I2T/T2I R@1): CLIP+MAE with pixel targets 57.3/41.1 -> feature targets 62.3/41.4 -> EMA teacher 65.0/41.6 -> adding MLM 70.1/45.6. Plain CLIP 52.9/32.8 (Table 1).
  - Removing one term (Table 6a; Flickr30K I2T/T2I): without MLM 65.0/41.6; without distillation 65.4/40.5; full 70.1/45.6.
  - Text decoder depth (Table 6d): 0 layers (MLM read directly from the text encoder output) 65.2/44.1; 4 layers 70.1/45.6.
  - COCO 5k ZS I2T/T2I R@1 (Table 5): CLIP 27.5/17.7 -> MaskCLIP 41.4/25.5.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** this is the only sweep found that crosses an equal 1:1 weighting (MultiMAE's setting) with smaller weights; at β = 1 the MLM pulled retrieval below plain CLIP. Its MLM is text-only, whereas MultiMAE's MLM reads image tokens through fusion.
- **Method weaknesses:**
  - The weight sweeps vary one weight at a time in from-scratch training on 15M pairs. Whether the gap between 1 and 0.05 carries over to fine-tuning an already converged CLIP at lr 1e-5 was not tested [reader-inferred].
  - One run per cell with no variance, so gaps of about 1 point (β = 0.05 vs 0.01) may be seed noise [reader-inferred].
  - The explanation for the β = 1 failure, that the extra tasks "mislead the model to a wrong converge direction", is offered as a suspicion with no diagnostic [author-acknowledged, Sec 4.4 "Distillation & MLM loss weight"].

#### 1.2 SyCoCa
- **Citation:** Ma, Z. et al. (2024). SyCoCa: Symmetrizing Contrastive Captioners with Attentive Masking for Multimodal Alignment. ICML 2024 (PMLR 235). arXiv:2401.02137.
- **Tier:** peer-reviewed (ICML 2024, per the PMLR listing). The arXiv v1 text was read; it was not compared against the PMLR version.
- **Read scope:** full text (arXiv PDF, plus arXiv HTML for the Table 5 check marks): Sec 3.3 to 3.4, 4.1, 4.5, Tables 1, 5, 6, 7.
- **Setting:** from-scratch pretraining of CoCa-Base (open_clip implementation) with an added image decoder. Main results on CC12M (20 epochs, batch 2048); ablations on CC3M. Loss L_ITC + λ_IC L_IC + λ_TM L_TM, with λ_IC = 2 and λ_TM = 1. TG-MIM reconstructs the pixels of masked patches conditioned on the caption. Attentive masking scores each patch by its maximum token-wise similarity to the caption. For TG-MIM it masks the top 50% of patches (r_h); for the captioning input it masks the bottom 50% (r_l).
- **Change tested and baseline:** CoCa (ITC + captioning) trained identically.
- **Effect:**
  - TG-MIM weight λ_TM (Table 7; CC3M; ZS mean of R@1/5/10; Flickr mTR/mIR, COCO mTR/mIR): 0.1: 42.1/32.4, 18.3/18.2. 0.5: 42.8/32.2, 18.2/18.2. 1.0: 42.6/32.4, 18.3/18.4. 2.0: 40.5/31.3, 17.7/17.8.
  - Objectives and masking (Table 5; COCO mTR/mIR): ITC only 13.5/14.8. CoCa 16.1/16.0. CoCa with attentive masking on the captioning input only 17.5/18.3. +MIM with random masking 15.8/16.3. +MIM with attentive masking 15.2/16.2. +TG-MIM with random masking 14.4/15.2. +TG-MIM with attentive masking (SyCoCa) 18.3/18.4. On Flickr: CoCa 37.5/28.7, TG-MIM random 35.8/26.4, SyCoCa 42.6/32.4.
  - Mask ratios (Table 6): COCO mTR 17.5 to 18.6 and mIR 17.3 to 18.8 across r_l and r_h in {25, 50, 75}%.
  - Main result (Table 1; CC12M; COCO ZS I2T/T2I R@1): CoCa 16.3/15.3 -> SyCoCa 18.7/17.2.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** this is the closest from-scratch analogue to MultiMAE's text-conditioned image reconstruction. With random masking, text-guided MIM fell below the no-MIM CoCa baseline; it helped only when the patches the caption describes were masked. Weights from 0.1 to 1.0 gave the same result and 2.0 was worse.
- **Method weaknesses:**
  - The ablations ran on CC3M at low absolute recall (COCO mTR about 18) with no seeds reported, so differences of 0.1 to 1 point are within likely run-to-run variance [reader-inferred].
  - Attentive masking of the captioning input alone, with no MIM, already captures most of the gain (COCO 17.5/18.3 vs 18.3/18.4), so part of the effect credited to TG-MIM comes from captioning-side masking [reader-inferred, Table 5].
  - The masks come from the model's own similarity scores, which are noisy early in from-scratch training. No warm-up analysis is given [reader-inferred].

#### 1.3 DetailCLIP
- **Citation:** Monsefi, A. K. et al. (2025). DetailCLIP: Detail-Oriented CLIP for Fine-Grained Tasks. SSI-FM Workshop at ICLR 2025 (arXiv comments). arXiv:2409.06809.
- **Tier:** peer-reviewed workshop. The arXiv comment says "Accepted in SSI-FM Workshop of ICLR 2025", while the PDF header reads "Published as a conference paper at ICLR 2025". Flag: preprint>=2024 venue ambiguity.
- **Read scope:** full text: Sec 3.2.3 to 3.2.4, 4.1, Tables 1 and 2, App. A.4 Table 5.
- **Setting:** from-scratch, YFCC-15M, 25 or 50 epochs. L = α1 L_CLS + α2 L_Patch + α3 L_Rec + L_CLIP (Eq. 10). L_Rec is pixel reconstruction of masked patches. The two distillation terms use an EMA teacher, and attention-based token removal keeps the 50% of patches with the highest teacher attention. All α = 1 by default.
- **Change tested and baseline:** A-CLIP, MaskCLIP, SLIP and CLIP at the same data and epochs. Baseline numbers are taken from A-CLIP (Table 2 caption).
- **Effect:**
  - Pixel-reconstruction weight α3 (Table 5; ImageNet ZS, 25 epochs): α3 = 0: 42.9. α3 = 1: 43.9. α3 = 2: 43.2. With α1 = α2 = 0 and α3 = 1: 42.6. All weights 0.5: 43.3.
  - Retrieval against the strongest baseline (Table 2; ZS R@1, Flickr I2T/T2I and COCO I2T/T2I): at 25 epochs A-CLIP 62.7/42.1 and 38.0/23.2, DetailCLIP 62.8/42.2 and 38.3/22.9. At 50 epochs A-CLIP 66.7/43.2 and 39.8/24.4, DetailCLIP 65.9/44.7 and 39.8/24.9.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** a pixel-reconstruction weight of 1 added 1.0 ImageNet point, and doubling it gave most of that back. The reconstruction term was never ablated on retrieval, and retrieval only matched A-CLIP.
- **Method weaknesses:**
  - The weight ablation reports ImageNet zero-shot only, so retrieval sensitivity to the weight is unknown [reader-inferred].
  - Baselines were copied from A-CLIP, not rerun [author-stated, Table 2 caption].
  - Single runs, with Table 2 gaps of 0.1 to 0.8 points [reader-inferred].

#### 1.4 SigLIP 2
- **Citation:** Tschannen, M. et al. (2025). SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv preprint arXiv:2502.14786.
- **Tier:** preprint; preprint>=2024.
- **Read scope:** full text: Sec 2.2 to 2.5, 3.1, Table 1. Other sections skimmed.
- **Setting:** from-scratch pretraining at WebLI scale: sigmoid loss plus LocCa decoder losses. Self-distillation (from SILC) and masked prediction (from TIPS) join only at 80% of training, with the teacher initialized from the student. Masked prediction replaces 50% of embedded patches with mask tokens in the student, and the target is the EMA teacher's per-patch features on the same global view. Loss weights are 1 (self-distillation) and 0.25 (masked prediction), each further multiplied by 0.25, 0.5, 1.0 and 0.5 for B, L, So400m and g. The extra losses use extra augmented views, while the SigLIP and LocCa losses use the original image "to ensure that data augmentation does not negatively impact the image-text alignment" (Sec 2.3).
- **Change tested and baseline:** SigLIP (v1). No ablation isolates masked prediction.
- **Effect:** the full recipe compared with SigLIP (Table 1; B/16 at 256 px; COCO ZS T2I/I2T R@1): 47.4/65.1 -> 53.2/69.7. The authors attribute the B-size gains largely to distillation by data curation (Sec 3.1).
- **Polysemy-aware metrics:** no (XM3600 multilingual retrieval only).
- **Relevance to MultiMAE [reader-inferred]:** a design data point for sub-themes 1 and 2. The masked loss has a feature target and a small weight, joins late in training, is scaled by model size, and is kept off the views the contrastive loss sees.
- **Method weaknesses:**
  - Several recipe changes are bundled (decoder losses, curation distillation, multilingual data), so the 47.4 -> 53.2 gain cannot be attributed to masked prediction [author-acknowledged in part, Sec 3.1].
  - The weights and the 80% start point are given without a sweep [reader-inferred].

### Sub-theme 2. Reconstruction target (pixels, features, tokens, labels)

MaskCLIP Table 9 (entry 1.1) and BiLMa Table 5 (entry 5.2) also compare targets.

#### 2.1 MAMO
- **Citation:** Zhao, Z., Guo, L., He, X., Shao, S., Yuan, Z., Liu, J. (2023). MAMO: Masked Multimodal Modeling for Fine-Grained Vision-Language Representation Learning. SIGIR 2023. arXiv:2210.04183.
- **Tier:** peer-reviewed (SIGIR 2023). Re-verified.
- **Read scope:** full text (PDF, plus arXiv HTML for the Table 6 check marks): Sec 3, 4.2, 4.5, Tables 6 to 10, Fig. 6.
- **Setting:** VLP pretraining with BERT-base and ViT-B/16 plus a fusion encoder. Data: CC3M, SBU, VG and COCO (4.1M images, 9.7M pairs), 40 epochs; ablations run 10 epochs. The loss is an equal-weight sum of MRM, MIM, MLM, ITC and ITM (Eq. 6). MRM is implicit: it regresses EMA-target latents at masked positions. Text is masked at 25% and images at 75%; MLM sees the image through the fusion encoder.
- **Change tested and baseline:** ITC + ITM with the same pretraining.
- **Effect:**
  - Objectives (Table 6; FT R@1 with ITM reranking; Flickr TR/IR and COCO TR/IR):
    - ITC+ITM: 93.5/80.7 and 71.9/55.2.
    - +MLM: 93.2/83.7 and 75.5/58.0.
    - +MLM+MIM: 94.0/83.5 and 74.7/58.2.
    - +MRM: 94.1/81.4 and 71.8/55.5.
    - +MRM+MLM: 95.2/83.6 and 75.8/59.0.
    - All: 95.5/84.3 and 76.6/59.2.
  - MIM target (Table 9; full model):
    - Raw pixels: 93.1/83.9 and 75.8/59.1.
    - DALL-E tokens: 95.2/84.0 and 76.5/58.1.
    - Momentum (EMA) features: 95.5/84.3 and 76.6/59.2.
  - MIM decoder depth (Table 10): from an MLP to 6 blocks, every result falls within 0.8 points.
  - Masking ratio (Fig. 6, COCO mean recall): higher text ratios hurt and higher image ratios generally help. Exact values appear only in the plot.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** image-conditioned MLM carried almost all of the gain over ITC+ITM on COCO (+3.6 TR, +2.8 IR R@1). Adding MIM on top changed COCO by -0.8 TR and +0.2 IR. Feature targets beat pixels by only 0.8 TR and 0.1 IR on COCO. This matches MultiMAE's validation-loss reading that MLM carries the effect.
- **Method weaknesses:**
  - Retrieval is fine-tuned and reranked by the ITM head, so the masked objectives also shape the cross-encoder used at test time. MultiMAE retrieves with a dual encoder [reader-inferred, Sec 4.3 protocol].
  - The ablations ran for 10 epochs with a single seed, so differences below about 0.5 R@1 cannot be interpreted [reader-inferred].
  - MRM used alone "may collapse into trivial solutions" [author-acknowledged, Sec 4.5].

#### 2.2 TIPS
- **Citation:** Maninis, K.-K. et al. (2025). TIPS: Text-Image Pretraining with Spatial Awareness. ICLR 2025. arXiv:2410.16512.
- **Tier:** peer-reviewed (ICLR 2025, camera-ready on arXiv).
- **Read scope:** full text: Sec 3.2, 4.1 to 4.2, Tables 1, 10, 11, App. A.2.
- **Setting:** from-scratch pretraining, ViT-B for the ablations. Data: curated WebLI, 116M pairs; ViT-B trained 70 epochs at batch 16k. Loss ½(L_CLIP + L̂_CLIP) + α L_distill + β L_mask with α = 1 and β = 2 (App. A.2). The masked loss is iBOT-style: student mask tokens are matched to the EMA teacher's prototype distribution over 32k prototypes for each unmasked patch. Masking is random at 75%.
- **Change tested and baseline:** CLIP trained identically on noisy web captions (Table 1, row A).
- **Effect:**
  - Losses (Table 1, block C; Flickr ZS I2T/T2I R@1): CLIP 79.1/62.9. +self-distillation 81.5/67.0. +self-distillation +MIM 82.6/67.6. Pascal segmentation: 64.4 -> 70.3 -> 75.9.
  - Masking ratio (Table 11B; full TIPS): 75%: 89.2/77.3. 50%: 90.5/78.0. 25%: 90.5/77.9. NYUv2 depth RMSE: 0.478, 0.501, 0.533.
  - Blockwise masking (Table 11C): 89.2/77.3 at 75%, the same as random.
  - Training CLIP first and MIM second, EVA-style (Table 11E): 78.0/65.9, against CLIP's 79.1/62.9.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** with an EMA teacher, feature-target MIM added +1.1/+0.6 Flickr R@1 on top of self-distillation. A lower masking ratio was slightly better for retrieval (+1.3/+0.7 at 50%), while 75% favored dense tasks.
- **Method weaknesses:**
  - MIM was never ablated without self-distillation, so its effect alongside CLIP alone is unknown [reader-inferred, Table 1C].
  - The weights α = 1 and β = 2 were not swept [reader-inferred, App. A.2].
  - Single runs at the 116M scale with no variance for about 1-point differences [reader-inferred].

#### 2.3 TULIP
- **Citation:** Tang, Z. et al. (2025). TULIP: Towards Unified Language-Image Pretraining. arXiv preprint arXiv:2503.15485 (v2).
- **Tier:** preprint; preprint>=2024. No venue found.
- **Read scope:** full text: Sec 3.3, 4.1, Tables 1 and 5.
- **Setting:** continued training from pretrained SigLIP: "we initialize all model variant weights from their respective SigLIP models". Data: 500M samples from DataComp-1B; batch 49,152; lr 1e-5. Added on top: image-image and text-text contrastive losses with generative augmentation (GeCo), and a reconstruction regularizer. The image side is MAE-style pixel reconstruction that uses the embedding as a bottleneck; the text side is a T5-style causal decoder seeded with the text embedding. L = λ_c L_cont + λ_r L_recons (Eq. 8 to 9); weight values not found in the sections read.
- **Change tested and baseline:** SigLIP, the initialization.
- **Effect:**
  - Adding reconstruction (Table 5; So/14; MMVP / LLaVA-Bench after LLaVA training): with I/I and T/T contrastive 17.4/82.3; +reconstruction 18.2/82.1. For B/14: 14.4/81.3 -> 15.8/80.8. No retrieval ablation of reconstruction.
  - Full TULIP vs SigLIP (Table 1; B/16 at 224 px; ZS R@1): COCO T2I/I2T 47.2/64.5 -> 54.2/70.1; Flickr 77.9/89.6 -> 81.8/93.9.
- **Polysemy-aware metrics:** no (Winoground, Table 3).
- **Relevance to MultiMAE [reader-inferred]:** the only large-scale case found that adds reconstruction while continuing from a pretrained contrastive model. Its reconstruction reads the pooled embedding, so the gradient reaches the retrieval vector directly. MultiMAE's decoders read token-level features.
- **Method weaknesses:**
  - The retrieval gains bundle I/I and T/T contrastive losses, GeCo and re-captioned data, so the share due to reconstruction is unknown [reader-inferred, Table 1 vs Table 5].
  - λ_i and λ_t were not located in the sections read (UNVERIFIED whether the appendix gives them) [reader-inferred].

### Sub-theme 3. Masking ratio and masking strategy

MaskVLM Table 7 (4.1), MAMO Fig. 6 (2.1), TIPS Table 11 (2.2), SyCoCa Tables 5 and 6 (1.2), VFE-TPS Fig. 6 (5.3) and MACCO Table 12 (5.4) also bear on this sub-theme.

#### 3.1 Uniform masking (Verma et al.)
- **Citation:** Verma, S. et al. (2022). Uniform Masking Prevails in Vision-Language Pretraining. arXiv preprint arXiv:2212.05195.
- **Tier:** preprint (no venue in the arXiv metadata or search results).
- **Read scope:** full text: Sec 3 to 5, Limitations, Table 4 (column order checked against the arXiv HTML).
- **Setting:** ViLT (single-stream, 135M parameters) pretrained from scratch on 4M pairs (VG, COCO, SBU, CC). Text masking rate in {15, 30, 45, 60, 75}% crossed with five strategies (uniform, whole word, noun-verb, span, PMI). Each configuration was pretrained, then fine-tuned with 3 seeds. Retrieval uses the ITM head (cross-encoder).
- **Change tested and baseline:** 15% masking under each strategy.
- **Effect:**
  - Change in FT R@1 from 15% to 60% masking, as mean ± s.e.m. over 3 seeds (Table 4):
    - Uniform: Flickr image +3.73±0.03, text +3.20±0.66; COCO image +3.00±0.10, text +4.11±0.72.
    - Whole word: Flickr +3.75 and +1.57; COCO +3.16 and +3.63.
    - Noun-verb: Flickr +1.11 and -1.80; COCO +0.62 and +0.01.
  - Best rate (Sec 4.1): 60% for Flickr retrieval, 75% for COCO retrieval.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** the only multi-seed sweep of MLM rate found with retrieval as a readout. 15% was the worst rate tested, and 60% lifted COCO R@1 by 3 to 4 points while shrinking the gaps between strategies. MultiMAE uses 15%.
- **Method weaknesses:**
  - Retrieval is ITM cross-encoder scoring on the same fusion transformer that MLM trains, so the MLM-retrieval link may be tighter than in a dual encoder, where MLM reaches the towers only through gradients [reader-inferred].
  - Only the deltas are tabulated; absolute values appear in plots [reader-inferred].
  - "We cannot test all masking strategies" [author-acknowledged, Limitations].

#### 3.2 Data-efficient MLM (Bitton et al.)
- **Citation:** Bitton, Y., Stanovsky, G., Elhadad, M., Schwartz, R. (2021). Data Efficient Masked Language Modeling for Vision and Language. Findings of EMNLP 2021. arXiv:2109.02040.
- **Tier:** peer-reviewed (Findings of EMNLP 2021).
- **Read scope:** full text: Sec 2 to 4, Table 1, Sec 4.1 results.
- **Setting:** probing uses the published LXMERT. The downstream comparison re-pretrains LXMERT for 7 epochs on 10% to 100% of its data with these strategies: baseline 15% random; one object word per caption (Objects); one content word, with content words chosen 80% of the time.
- **Change tested and baseline:** LXMERT with 15% random MLM.
- **Effect:**
  - Statistics (Sec 2.2): captions run about 20 tokens, so at 15% masking 36% of LXMERT sentences get no masked token. 45% to 50% of masked tokens are stop words or punctuation.
  - Image necessity (Table 1; published LXMERT; Accuracy@5 with vs without the image): baseline MLM 89% vs 78% (gap 10 points); stop words and punctuation 98% vs 96% (2); content words 76% vs 56% (20).
  - Downstream (Sec 4.1): with 10% of the pretraining data, Objects masking adds 0.72 to 0.86 on VQA and GQA and 4% on NLVR2. With 100% of the data the VQA and GQA gains are minor; NLVR2 gains 1.08.
- **Polysemy-aware metrics:** no; retrieval not evaluated.
- **Relevance to MultiMAE [reader-inferred]:** this quantifies why 15% random MLM on short captions uses the image so little. COCO captions are short, so the same arithmetic applies to MultiMAE's 15% MLM.
- **Method weaknesses:**
  - No retrieval evaluation, and the gains shrink with more data or epochs [author-acknowledged, Sec 4.1 and footnote 10].
  - Pretraining was cut to 7 epochs [author-acknowledged, footnote 8].

#### 3.3 M3AE
- **Citation:** Geng, X. et al. (2022). Multimodal Masked Autoencoders Learn Transferable Representations. arXiv preprint arXiv:2205.14204.
- **Tier:** preprint (no venue in the arXiv metadata).
- **Read scope:** full text: Sec 3 to 4, Fig. 5, App. Table 2.
- **Setting:** one shared encoder over image patches and text tokens, trained only by masked prediction (no contrastive loss). Data: CC12M; ViT-L for 50 epochs in the ratio ablation. Image mask 75%; text loss weight 0.5 against an image weight of 1 (App. Table 2).
- **Change tested and baseline:** 15% text masking.
- **Effect (Fig. 5; ImageNet linear probe):** text mask 15%: 45.9. 50%: 61.0. 75%: 64.1. 90%: 63.9.
- **Polysemy-aware metrics:** no; no retrieval.
- **Relevance to MultiMAE [reader-inferred]:** when text is reconstructed with the image visible, the best text-masking rate moved far above BERT's 15% (+18.2 linear probe at 75%). This agrees with Verma et al. and MaskVLM.
- **Method weaknesses:**
  - No contrastive loss and no retrieval readout, so the transfer to a CLIP fine-tune is indirect [reader-inferred].
  - The linear probe measures only the image side of the shared encoder [reader-inferred].

#### 3.4 A-CLIP (Attentive Mask CLIP)
- **Citation:** Yang, Y. et al. (2023). Attentive Mask CLIP. ICCV 2023, pp. 2771 to 2781. arXiv:2212.08653.
- **Tier:** peer-reviewed (ICCV 2023).
- **Read scope:** full text: Sec 3.3, 4.2, Tables 1 to 4.
- **Setting:** from-scratch, ViT-B/16, YFCC-15M, 25 epochs, batch 4096. Masking here removes image tokens from the contrastive input; nothing is reconstructed. An EMA encoder scores how relevant each patch is to the caption, and online-EMA SSL heads (SimCLR, BYOL) are added.
- **Change tested and baseline:** CLIP on full images with the same data.
- **Effect (Table 2a; ZS ImageNet; Flickr I2T/T2I; COCO I2T/T2I R@1):**
  - CLIP, full images: 37.6; 51.4/32.6; 27.9/17.6.
  - Random masking, one view at 50%: 35.0; 48.8/32.5; 28.9/16.6. Two views at 50%: 38.0; 54.6/34.4; 31.1/18.7.
  - Attentive masking, one view at 50%: 39.5; 57.6/36.6; 34.2/19.8. Two views at 50%: 41.3; 59.3/38.4; 35.1/21.3.
  - Selection rule (Table 2b): keeping the most relevant 50% ("low") gives 41.3; 59.3/38.4; 35.1/21.3. Masking the most relevant 50% ("high", AttMask-style) gives 28.5; 42.6/29.0; 23.5/13.6.
  - CLIP+MAE, i.e. MaskCLIP (Table 3): 42.7; 60.0/38.8; 34.1/21.2.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** in the contrastive branch, removing caption-relevant patches collapsed retrieval (COCO I2T 35.1 -> 23.5) and keeping them helped. This is the opposite direction to SyCoCa's reconstruction branch, which masks the relevant patches and rebuilds them from text. MultiMAE's contrastive loss sees clean images, so only the reconstruction-side finding applies to it directly.
- **Method weaknesses:**
  - Attentive selection costs close to 30% more training time unless the half-resolution EMA input is used [author-acknowledged, Sec 4.2].
  - Single runs [reader-inferred].

### Sub-theme 4. Image-conditioned MLM vs image MAE, and which modality is reconstructed

MAMO Table 6 (2.1), SyCoCa Table 5 (1.2), BiLMa Table 2 (5.2) and MACCO Table 9 (5.4) also separate the two.

#### 4.1 MaskVLM
- **Citation:** Kwon, G. et al. (2023). Masked Vision and Language Modeling for Multi-modal Representation Learning. ICLR 2023. arXiv:2208.02131.
- **Tier:** peer-reviewed (ICLR 2023). Re-verified.
- **Read scope:** full text: Sec 3 to 4.5, Table 5, App. A.2 to A.4 (Tables 6 and 7).
- **Setting:** VLP initialized from an ImageNet-pretrained ViT-B/16 and RoBERTa (not CLIP), with ALBEF-style cross-modality encoders. Ablations pretrain on CC 50% + COCO for 30 epochs. MLM masks 30% of text tokens and predicts them conditioned on the unmasked image. MIM masks 60% of the image in 32×32 blocks and reconstructs RGB pixels with an L1 loss, through a decoder of 3 cross-attention blocks that attend to the unmasked text. Losses are unweighted (App. A.2).
- **Change tested and baseline:** ITC + ITM.
- **Effect:**
  - Objectives (Table 5; Flickr30k R@1 IR/TR, FT | ZS):
    - ITC: 65.10/80.10 | 55.08/68.40.
    - ITC+ITM: 79.96/92.30 | 69.50/82.40.
    - +MLM: 80.34/92.00 | 70.74/84.40.
    - +MIM: 80.12/91.50 | 69.26/82.90.
    - +MLM+MIM: 81.26/94.10 | 71.18/85.60.
    - MLM+MIM without ITC/ITM: 76.08/90.30 FT (ZS not possible).
  - Masking ratio, image/text (Table 7; FT Flickr R@1 IR/TR): 0.5/0.3: 81.32/93.30. 0.6/0.3: 81.26/94.10. 0.7/0.3: 81.82/93.60. 0.6/0.15: 80.30/92.50.
  - One modality masked at a time vs both together (Table 6; COCO 5k FT R@1 IR/TR): one 60.1/76.3; both 59.5/76.0; ALBEF 56.8/73.1.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** this separates image-conditioned MLM from text-conditioned pixel MIM. Alone, MLM added +1.2 IR and +2.0 TR zero-shot R@1, and MIM changed them by -0.2 IR and +0.5 TR. Together they added +1.7 IR and +3.2 TR. Text masking at 30% beat 15% by +1.0/+1.6 FT R@1, and image ratios from 0.5 to 0.7 performed the same.
- **Method weaknesses:**
  - Retrieval reranks the top-k candidates with the cross-modality encoders, which the masked losses also train [reader-inferred, Sec 4.1].
  - Single runs, so the MLM-only vs MIM-only gaps (at most 1.5 R@1) are not replicated [reader-inferred].
  - Training starts from unimodal encoders, not an aligned CLIP [reader-inferred].

#### 4.2 FLAVA
- **Citation:** Singh, A. et al. (2022). FLAVA: A Foundational Language And Vision Alignment Model. CVPR 2022. arXiv:2112.04482.
- **Tier:** peer-reviewed (CVPR 2022).
- **Read scope:** full text: Sec 3.2 to 3.3, 4, Table 4.
- **Setting:** from-scratch on PMD (70M pairs) with a ViT-B/16 image encoder, a transformer text encoder and a multimodal encoder. MMM masks image patches (BEiT block masking, dVAE token targets) and 15% of text tokens, and predicts both from the multimodal encoder. The global contrastive loss runs on unmasked inputs. ITM is also used.
- **Change tested and baseline:** FLAVA_C, contrastive only, with the same data and architecture.
- **Effect (Table 4; ZS R@1 from the contrastive embeddings):**
  - FLAVA_C -> FLAVA_MM (+MMM +ITM): COCO TR 43.08 -> 43.48 and IR 37.59 -> 38.46; Flickr TR 68.30 -> 69.30 and IR 60.56 -> 63.16.
  - Full FLAVA (adds unimodal data and DINO/MLM initialization): COCO 42.74/38.38; Flickr 67.70/65.22.
  - FLAVA_MM vs FLAVA_C macro averages (Sec 4): multimodal +2.86, NLP +9.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** at 70M pairs, cross-modal masked modeling on top of a contrastive loss moved dual-encoder zero-shot COCO R@1 by less than 1 point, with ITM bundled in.
- **Method weaknesses:**
  - MMM and ITM were added together, so MMM's own effect is not isolated [reader-inferred, Table 4 columns 3 vs 4].
  - Adding unimodal tasks slightly lowered the macro average; the authors attribute this to harder optimization and round-robin sampling with no curriculum [author-acknowledged, Sec 4].

#### 4.3 VL-BEiT
- **Citation:** Bao, H., Wang, W., Dong, L., Wei, F. (2022). VL-BEiT: Generative Vision-Language Pretraining. arXiv preprint arXiv:2206.01127.
- **Tier:** preprint.
- **Read scope:** full text: Sec 2 to 3.4, Tables 4 and 5 (Table 4 check marks from the arXiv HTML).
- **Setting:** from-scratch training of a shared Mixture-of-Modality-Experts (MoME) Transformer with no contrastive pretraining. MVLM masks 50% of text and 40% of the image (block-wise, BEiT v2 tokenizer targets). Unimodal MIM runs on ImageNet-22K and unimodal MLM on Wikipedia and BookCorpus. Ablations run 40 epochs. Retrieval is fine-tuned with ITC and ITM; at inference a dual encoder picks the top-k candidates and the fusion encoder reranks them (Sec 3.2).
- **Change tested and baseline:** MVLM only.
- **Effect (Table 4; FT Flickr30k TR/IR R@1):** MVLM 91.2/75.8. +MIM 92.2/77.4. +MIM+MLM 92.2/77.9.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** on top of a cross-modal masked objective, unimodal MIM on extra images helped retrieval more (+1.0/+1.6) than unimodal MLM (+0/+0.5). Here the MIM target is a tokenizer, not pixels.
- **Method weaknesses:**
  - Objective and data are confounded: each unimodal task brings its own corpus [reader-inferred].
  - No contrastive pretraining, and the retrieval readout is reranked [reader-inferred].

### Sub-theme 5. Masked objectives while fine-tuning an already pretrained CLIP

#### 5.1 IRRA
- **Citation:** Jiang, D., Ye, M. (2023). Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval. CVPR 2023. arXiv:2303.12501.
- **Tier:** peer-reviewed (CVPR 2023).
- **Read scope:** full text: Sec 3, 4.1 to 4.3, Tables 1, 4, 5.
- **Setting:** fine-tunes the full pretrained CLIP ViT-B/16 and CLIP text transformer for text-to-person retrieval on CUHK-PEDES (80,412 descriptions of 13,003 identities), ICFG-PEDES and RSTPReid. 60 epochs; lr 1e-5 for CLIP and 5e-5 for new modules; Adam with cosine decay; one RTX 3090. IRR is an image-conditioned MLM: 15% of text tokens are masked (BERT 80/10/10), the masked text tokens query the image tokens through one cross-attention layer and a 4-layer transformer, and an MLP predicts the tokens. The interaction encoder is unused at inference, so retrieval is dual-encoder. Loss L_irr + L_sdm + L_id with unit weights.
- **Change tested and baseline:** the same CLIP fine-tuned with InfoNCE (Table 4, No. 0).
- **Effect (Table 4; Rank-1 on CUHK / ICFG / RSTP):**
  - InfoNCE baseline: 68.19 / 56.74 / 54.05. With IRR: 71.23 / 60.96 / 57.90, a gain of +3.04 / +4.22 / +3.85.
  - +SDM: 70.42 / 60.45 / 57.20. +SDM+IRR: 72.81 / 63.27 / 59.25. +SDM+ID: 70.52 / 61.03 / 58.65. Full IRRA: 73.38 / 63.46 / 60.20.
  - CUHK mAP (Table 1): 61.12 (baseline) -> 66.13.
  - Interaction module (Table 5; CUHK Rank-1): co-attention 73.28, merged attention 73.21, IRR module 73.38.
- **Polysemy-aware metrics:** no ECCV/PMRP/CxC. mAP and mINP are reported (Table 1).
- **Relevance to MultiMAE [reader-inferred]:** the strongest evidence found that an image-conditioned MLM head, discarded at test time, improves a fine-tuned CLIP dual encoder: +3 to +4 Rank-1 over InfoNCE fine-tuning on three datasets. It shares MultiMAE's 15% rate, cross-attention fusion and two trained towers, but the domain is pedestrian descriptions.
- **Method weaknesses:**
  - Masking random single tokens teaches word-level, not phrase-level, semantics [author-acknowledged, Sec 4.3].
  - Single runs with no variance. BiLMa's rerun of the configuration without IRR (SDM+ID) scored 73.01 Rank-1 on CUHK against IRRA's 70.52, which would shrink the IRR gain on that dataset to +0.15 (entry 5.2) [reader-inferred].

#### 5.2 BiLMa
- **Citation:** Fujii, T., Tarashima, S. (2023). BiLMa: Bidirectional Local-Matching for Text-based Person Re-identification. ICCV 2023 Workshops (CLVL). arXiv:2309.04675.
- **Tier:** peer-reviewed workshop (ICCVW 2023).
- **Read scope:** full text including the supplementary in the arXiv version: Sec 3 to 4, Tables 1, 2, 5, App. A.5 to A.6. A.6 is plots only. Table 2 column positions were read with layout-mode PDF extraction.
- **Setting:** IRRA (the fine-tuned CLIP ViT-B/16 above) plus MIM decoded through the shared interaction encoder with text visible. L = L_id + L_sdm + α L_mlm + β L_mim, with α = 1 and a 0.15 MLM rate. The MIM rate and β were grid-searched per dataset, and the best result is reported (Sec 4.1).
- **Change tested and baseline:** IRRA, both as their own run without MLM or MIM (Table 2) and as IRRA's published numbers (Table 5).
- **Effect:**
  - MIM target (App. Table 5; CUHK Rank-1/mAP; mask 0.15; β = 1): without MIM (IRRA's published result) 73.38/66.13. Pixel-level 72.86/65.61. Patch-mean RGB 73.07/66.01. Feature-level (KL) 73.52/66.20. SemMIM, which predicts human-parser part labels, 74.03/66.57.
  - Components (Table 2; Rank-1 on CUHK / ICFG / RSTP): neither 73.01 / 63.09 / 59.50. MLM only 73.16 / 63.60 / 59.05. SemMIM only 73.55 / 63.08 / 59.40. Both 74.03 / 63.83 / 61.20.
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** a direct test of adding image MIM to an image-conditioned MLM inside a CLIP fine-tune. Pixel targets hurt (-0.52 Rank-1) and only semantic-label targets helped (+0.65). In their own runs the MLM-only gain on CUHK was just +0.15.
- **Method weaknesses:**
  - The "without MIM" row of Table 5 uses IRRA's published 73.38, not their own MLM-only run (73.16), so the paper mixes two baselines [reader-inferred].
  - Pixel, patch and feature MIM were tried only at mask rate 0.15 and β = 1, while SemMIM got a per-dataset grid search, so the target comparison did not get equal tuning [reader-inferred, App. A.5 vs Sec 4.1].
  - The method depends on an external human parser, and the influence of parser errors is left to future work [author-acknowledged, Sec 5].

#### 5.3 VFE-TPS
- **Citation:** Shen, W. et al. (2024). Enhancing Visual Representation for Text-based Person Searching. arXiv preprint arXiv:2412.20646.
- **Tier:** preprint; preprint>=2024.
- **Read scope:** full text: Sec 3.2 to 3.3, 4.3, 4.5, Tables 5 and 6. Figs. 5 and 6 are plots without tabulated values.
- **Setting:** fine-tunes CLIP ViT-B/16 and the CLIP text Transformer on CUHK-PEDES, ICFG-PEDES and RSTPReid; 60 epochs, lr 1e-5, batch 100, one RTX 3090. TG-MIM zeroes random patches (SimMIM-style); image tokens query the text tokens through one multi-head cross-attention layer, then a conv layer and PixelShuffle predict raw pixels under an L1 loss (Alg. 1). An identity-supervised loss (IS-GVFC) is also added. Both auxiliary heads are unused at inference.
- **Change tested and baseline:** the same model without either auxiliary task.
- **Effect:**
  - Auxiliary tasks (Table 6; CUHK Rank-1/mAP): none 70.61/63.12; +TG-MIM 72.16/63.99; +IS-GVFC 71.61/64.24; both 72.47/64.26.
  - Image-only MAE and SimMIM also improve over the baseline, but less than TG-MIM (Fig. 5, bars only).
  - mAP peaks at a masking ratio of 0.5 to 0.6 for all three (Fig. 6; Sec 4.5.3 text).
- **Polysemy-aware metrics:** no.
- **Relevance to MultiMAE [reader-inferred]:** a counterpoint to BiLMa. Text-conditioned pixel MIM gave +1.55 Rank-1 when it was the only token-level auxiliary (no MLM) and used 0.5 to 0.6 masking rather than 0.15.
- **Method weaknesses:**
  - Table 5 puts fully fine-tuned CLIP (CMPM loss) at 66.78 Rank-1, while Table 6's no-auxiliary baseline is 70.61; the gap is not explained in the sections read [reader-inferred].
  - MAE and SimMIM comparisons appear only as bar heights; single runs [reader-inferred].

#### 5.4 MACCO
- **Citation:** Li, W., Huang, Z., Tian, X. (2026). Cross-Modal Masked Compositional Concept Modeling for Enhancing Visio-Linguistic Compositionality. ACL 2026 Main Conference (per arXiv comments). arXiv:2606.13288.
- **Tier:** peer-reviewed per the authors' arXiv note (ACL 2026 main); proceedings not checked.
- **Read scope:** full text: Sec 3 to 4, Sec 6 (Limitations), App. D to F, Tables 1, 3, 9, 10, 12, 15. Tables 9 and 12 verified from the arXiv HTML.
- **Setting:** fine-tunes both encoders of OpenAI CLIP ViT-B/32 on about 110k COCO image-text pairs. 5 epochs, batch 256, lr 5e-7 for CLIP and 1e-3 for the predictors, AdamW with weight decay 0.2, one A100. Masking targets scene-graph "compositional concepts" in both modalities (the ablation uses random 75% image / 15% text masking instead).
  - MLM: a text predictor with 2 cross-attention layers from the masked text tokens to the full image features (stop-gradient on the image side) and a vocabulary head.
  - MIM: pixel MSE on masked patches, conditioned on the text (stop-gradient on the masked image features).
  - Two masked-augmented contrastive losses: masked inputs serve as soft negatives across modalities (MCA), and masked and full views are contrasted within each modality (MIR).
  - The predictors are removed at inference.
- **Change tested and baseline:** CLIP-FT, a contrastive-only fine-tune on the same COCO data.
- **Effect:**
  - Losses (Table 9; mean of ARO, SugarCrepe and VL-Checklist): CLIP 64.6; CLIP-FT 66.1. Single additions: +MLM 68.2; +MIM 68.5; +MLM+MIM 68.2; +MCA+MIR only 68.7. MLM with MCA+MIR 72.8; MIM with MCA+MIR 70.5; all four 73.4.
  - Masking strategy (Table 12): random masking without the auxiliary losses 67.5; random with them 71.2; concept masking without them 68.2.
  - Variance (Table 15; 4 seeds), e.g. ARO-Relation: CLIP-FT 64.4±0.40 vs MACCO 73.5±0.60.
  - ImageNet-style zero-shot average over 11 datasets (Table 3): CLIP 59.5, CLIP-FT 57.9, MACCO 58.0.
- **Polysemy-aware metrics:** no retrieval and no ECCV/PMRP/CxC; compositional benchmarks only (other).
- **Relevance to MultiMAE [reader-inferred]:** same backbone and training data as MultiMAE (CLIP B/32 fine-tuned on COCO). MLM and MIM each added about 2 points on compositional benchmarks, but together they added nothing beyond either alone (68.2) unless the masked views also entered the contrastive loss. No COCO retrieval numbers are reported.
- **Method weaknesses:**
  - No image-text retrieval evaluation, so the effect on R@K or PMRP is unknown [reader-inferred].
  - Requires scene-graph and phrase pre-processing and adds training-time predictors [author-acknowledged, Sec 6].
  - A CLIP lr of 5e-7 for 5 epochs keeps the model near its initialization, and the masked-objective effects may depend on that regime [reader-inferred].

## 3. Search limitations

- Semantic Scholar and OpenReview were not queried through their own search interfaces, only through a general web search. Workshop papers and non-arXiv venues are therefore under-sampled.
- Read in part or not in full, or screened only: EVA / EVA-CLIP (2211.07636, 2303.15389; a sequential MIM-then-CLIP design, whose successive variant is covered by TIPS Table 11E), MVP, BEiT-3, FD-CLIP (2205.14141), SLIP (seen only through the MaskCLIP and A-CLIP tables), SemMAE, AttMask (its "mask the most relevant" rule appears as A-CLIP's "high" selection), VLMAE (2208.09374), EVE (2308.11971), ECLIP (OpenReview 5IFFLY4JR24), UTA (2405.19009), and Wettig et al. "Should You Mask 15%" (40% MLM, text only; cited by Verma et al. and MaskVLM). Weers et al., MAP and FLIP were not re-read because they were covered on 2026-10-02.
- "ACAV" in the task list could not be matched to a masking method (ACAV100M is an audio-visual dataset) and was not pursued.
- Several values exist only in figures: MAMO Fig. 6, VFE-TPS Figs. 5 and 6, BiLMa Figs. 6 to 8, and the absolute values behind Verma et al.'s deltas. They were not digitized, so the entries above report only values printed in text or tables.
- PDF text extraction loses check-mark columns. Tables were re-read from the arXiv HTML (MAMO Table 6, VL-BEiT Table 4, SyCoCa Table 5, MACCO Tables 9 and 12, Verma et al. Table 4) or with layout-mode extraction (BiLMa Table 2). A WebFetch summarizer misreported MAMO Table 6 (check marks) and MaskCLIP Table 6a (a linear-probe value); both summaries were discarded in favor of direct extraction.
- Venue status:
  - TULIP, Verma et al., M3AE, VL-BEiT and VFE-TPS: no venue found, recorded as preprints.
  - DetailCLIP: the arXiv comment and the PDF header disagree (workshop vs conference).
  - MACCO: venue taken from the arXiv comment; proceedings not checked.
  - SyCoCa: the arXiv v1 text was read, not the ICML camera-ready.
- None of the included papers report results over seeds, except Verma et al. (3 seeds, s.e.m.) and MACCO (4 seeds, std).
- No instruction-like or injected text was encountered in the retrieved pages.
