> Supporting file for docs/reports/auto/v1/2026-10-03_lever_review.md. Phase 2 bibliographies are as written; V1/V2 list the corrections, and the report uses corrected values.

# B4: Novelty sweep, masked models and polysemy in image-text retrieval

Bibliography agent, ARS deep-research Phase 2 (lit-review mode). Theme B4. Written 2026-10-03.
Scope: what already exists where masking, generative objectives and polysemy meet, and which mechanisms are
new for a fine-tuned CLIP + fusion + masked-reconstruction model. No cross-theme synthesis, no recommendations.

## 1. Search strategy

**Sources searched.** Web search (standard and extended modes) over arXiv, CVF open access, ICLR/ICML/NeurIPS
proceedings, ACL Anthology, OpenReview, mlanthology.org (venue confirmation). Full text read through
arxiv.org/html and ar5iv.labs.arxiv.org. ACL Anthology PDFs could not be parsed (see Section 3).

**Date range.** 2019 to 2026, weighted to 2023-2026. Last searched 2026-10-03.

**Query strings** (Q-numbers are referenced in the "where searched" column of the mechanism table, Section 2.6):
- Q1 `masked image modeling polysemy image-text retrieval ECCV Caption mAP@R 2024` (extended)
- Q2 `visual word sense disambiguation CLIP 2024 2025 benchmark ambiguous query image retrieval` (extended)
- Q3 `Llip learning to model the diversity of visual descriptions arXiv 2405.00740`
- Q4 `hyperbolic vision-language entailment cones generic-to-specific caption hierarchy retrieval 2025` (extended)
- Q5 `one-to-many image-text retrieval benchmark 2025 multiple valid matches false negatives evaluation dataset` (extended)
- Q6 `ambiguous text query image retrieval benchmark multiple interpretations 2024 2025 dataset underspecified` (extended)
- Q7 `CLIP polysemous words analysis homonym sense embedding vision-language model 2024` (extended)
- Q8 `masked caption partial text "less specific" inclusion probabilistic embedding uncertainty vision-language 2025` (extended)
- Q9 `PCME++ ECCV Caption mAP@R PMRP results table new methods 2025 probabilistic cross-modal embedding` (extended)
- Q10 `"ECCV Caption" mAP@R masked language modeling image-text matching 2025 arXiv` (extended)
- Q11 `cross-modal masked distillation predict text features from masked image vision-language pretraining 2024 2025` (extended)
- Q12 `TULIP towards unified language-image pretraining reconstruction 2025 arXiv retrieval`
- Q13 `polysemous image-text retrieval benchmark 2025 ambiguous captions multiple meanings dataset CLIP evaluation homonym retrieval` (extended)
- Q14 `COCO image-text retrieval false negatives re-annotation new benchmark 2024 2025 additional positive captions human verified` (extended)
- Q15 `masked modeling one-to-many correspondence image-text retrieval uncertainty 2025 "masked" "one-to-many" cross-modal retrieval` (extended)
- Q16 `ProLIP follow-up 2025 2026 probabilistic vision-language masked inclusion ECCV Caption evaluation fine-tuning CLIP uncertainty` (extended)
- Q17 `visual word sense disambiguation fine-tuning CLIP masked language modeling context SemEval 2023 Task 1 first place FCLL system description`
- Q18 `SemEval-2023 Task 1 visual word sense disambiguation "masked language model" OR "fill-mask" context generation system`
- Q19 `CLIP text encoder homonym "superposition" of senses ambiguous word analysis paper`
- Q20-Q29: venue and identity checks (Kritharoula EMNLP 2023; PolCLIP ACL 2024; COSMOS/FLAIR CVPR 2025; TIPS ICLR 2025;
  MERU ICML 2023; HyCoCLIP ICLR 2025; DreamLIP ECCV 2024; A-CLIP ICCV 2023; CapPa NeurIPS 2023; HierarCaps ECCV 2024).

**Inclusion criteria.** (a) Uses masking, partial inputs, captioning/reconstruction, multi-positive or
text-conditioned matching in a CLIP-like image-text model and reports retrieval or a fine-grained/compositional
or hierarchy measure; or (b) studies lexical ambiguity / polysemy with CLIP-like models; or (c) defines a
one-to-many or ambiguity benchmark. Numbers had to be readable from the paper's own HTML tables.
**Exclusion criteria.** Withdrawn papers; video-only or domain-specific retrieval (surveillance, medical,
remote sensing, documents); classification-uncertainty methods without retrieval; papers already covered by the
2026-10-02 scan unless they answer a B4-specific question (ProLIP masked inclusion, PCME++ MSDA ablation, MAP
polysemy evaluation check); secondary reviews whose numbers I could not trace to a primary source.

**Counts.** 29 queries; about 270 result links identified (many duplicates); about 75 unique candidate papers
screened by title and snippet; 30 read in targeted full text (specific sections and tables via HTML); 8 read
at abstract level. Included: 16 core entries (full annotation), 9 short secondary entries, 6 benchmark cards
(Section 2.4). Excluded after reading: 5 (listed at the end of Section 2).

**Coverage skew.** arXiv-HTML heavy, so arXiv-hosted ML venues are over-represented relative to ACL/ACM/IEEE
journals. All numbers came through a page-to-text extraction step; several first parses mislabeled columns,
and I re-read those tables in raw-cell form on ar5iv (noted per entry). 2026 work is arXiv-only.

## 2. Annotated bibliography

### 2.1 Sub-theme 1: masking or partial inputs as a model of ambiguity, generality or hierarchy

**[1] ProLIP (masked-inclusion detail only; source already known)**
- Citation: Chun, S. et al. (2025). Probabilistic Language-Image Pre-training. ICLR 2025. arXiv:2410.18857.
- Tier: peer-reviewed (ICLR 2025).
- Read scope: full text (arxiv.org/html): Sec 3.3, Sec 4.1, Sec 4.3 (Figs 7-8), App B.1, App C.3, App D.
- Setting: from-scratch pretraining on DataComp-1B (1.28B seen samples main, 12.8B scaled; Table 1); ViT-B/16 main.
- Change tested: Gaussian embeddings from an [UNC] token plus an inclusion loss that includes (a) image-text
  inclusion and (b) original-in-masked inclusion: 75% of input tokens masked (text: [MASK] tokens; image: token
  dropping), applied to 12.5% of samples in a mini-batch (Sec 3.3), weight alpha_2 = 0.001 (App B.1).
- Effect: no table isolates the masked-inclusion term on retrieval or zero-shot accuracy (App C.3, Table C.5
  ablates only the stability hyperparameters epsilon and c). Fig 8: more than 70% of images satisfy inclusion
  with their masked versions. Fig 7: short captions get large uncertainty.
- Polysemy-aware metrics: none in the ProLIP paper (HierarCaps used for hierarchy). Zero-shot ECCV Caption
  numbers for ProLIP B/16 appear in LongProLIP, entry [2].
- Relevance to MultiMAE: masked inputs are already used as "less specific" views for an uncertainty/inclusion
  objective, but with no reconstruction, at pretraining scale, and with a tiny loss weight. [reader-inferred]
- Method weaknesses: the 0.001 weight on 12.5% of samples means any retrieval effect of the masked-inclusion
  term cannot be separated from the rest of the objective [reader-inferred]; diagonal covariance simplification
  [author-acknowledged, App D].

**[2] LongProLIP**
- Citation: Chun, S., Yun, S. (2025). LongProLIP: A Probabilistic Vision-Language Model with Long Context Text.
  arXiv preprint, arXiv:2503.08048.
- Tier: preprint, `preprint>=2024` (an OpenReview attachment exists; venue UNVERIFIED).
- Read scope: full text (arxiv.org/html v1): method, Table C.1.
- Setting: fine-tunes the ProLIP ViT-B/16 checkpoint from 64 to 256 text tokens (positional interpolation) on
  mixes of ShareGPT4V (1.2M) with HYPE + DFN-medium data (S24M, S128M, SHD128M).
- Change tested and baseline: long-context fine-tuning vs the pretrained ProLIP.
- Effect (Table C.1, zero-shot, ECCV Caption mAP@R I2T / T2I): ProLIP 28.9 / 39.2; S24M 30.5 / 41.0;
  S128M 29.6 / 37.2; SHD128M 29.2 / 40.1. DataComp 38-task average: 63.3 -> 60.5 / 58.7 / 63.3. Urban-1k
  average 55.5 -> 91.3 (S128M).
- Polysemy-aware metrics: yes, ECCV Caption mAP@R (numbers above). No PMRP, CxC.
- Relevance to MultiMAE: the only model trained with a masked-input inclusion loss for which I found ECCV
  Caption numbers; data-mix changes alone move ECCV mAP@R by -2.0 to +1.8. [reader-inferred]
- Method weaknesses: the text does not say whether the masked-inclusion loss is kept during fine-tuning (our
  reading), so the ECCV changes cannot be attributed to it [reader-inferred]; no variance estimates in Table C.1
  as parsed [reader-inferred]; long-context vs zero-shot trade-off [author-acknowledged, results discussion].

**[3] MERU**
- Citation: Desai, K., Nickel, M., Rajpurohit, T., Johnson, J., Vedantam, R. (2023). Hyperbolic Image-Text
  Representations. ICML 2023, PMLR 202:7694-7731. arXiv:2304.09172.
- Tier: peer-reviewed (ICML 2023).
- Read scope: full text (arxiv.org/html v3): method (entailment loss), Table 1, traversal analysis.
- Setting: from-scratch on RedCaps (12M pairs); ViT-S/16, B/16, L/16; CLIP baseline trained identically.
- Change tested: Lorentz-model hyperbolic embeddings plus an entailment-cone loss (text entails its image);
  generic concepts drift toward the origin.
- Effect (Table 1, zero-shot R@5, CLIP -> MERU): B/16 COCO text->image 32.9 -> 33.2, COCO image->text
  41.4 -> 41.8, Flickr text->image 40.3 -> 41.1, Flickr image->text 50.2 -> 48.1. L/16 COCO text->image
  31.7 -> 32.6, image->text 40.6 -> 41.9.
- Polysemy-aware metrics: no. Hierarchy shown qualitatively by image-to-origin traversals.
- Relevance to MultiMAE: entailment ordering is a second formalization (besides variance) of "a generic caption
  fits many images"; no masking involved. [reader-inferred]
- Method weaknesses: from-scratch on a 12M noisy corpus gives low absolute numbers, and differences of -2.1 to
  +1.3 R@5 come from single runs [reader-inferred]; hierarchy evidence mostly qualitative [reader-inferred].

**[4] HyCoCLIP**
- Citation: Pal, A., van Spengler, M., di Melendugno, G. M. D., Flaborea, A., Galasso, F., Mettes, P. (2025).
  Compositional Entailment Learning for Hyperbolic Vision-Language Models. ICLR 2025 (oral). arXiv:2410.06912.
- Tier: peer-reviewed (ICLR 2025).
- Read scope: full text (arxiv.org/html): method, Table 2, limitations.
- Setting: from-scratch on GRIT (20.5M grounded pairs, 35.9M boxes); CLIP and MERU retrained on GRIT, batch 768.
- Change tested: image boxes and noun-phrase text boxes act as "general" parents of the full image/caption;
  hierarchical compositional contrastive loss plus entailment-cone loss (inter- and intra-modal).
- Effect (Table 2, smallest-backbone rows; one parse labelled them ViT-S/16, another B/16, so backbone label
  UNVERIFIED; CLIP / MERU / HyCoCLIP): COCO text R@5 71.4 / 72.3 / 72.0; COCO image R@5 57.4 / 57.4 / 58.4;
  Flickr text R@5 93.6 / 93.5 / 92.6. Hierarchical metrics (TIE, LCA, J) improve in direction per Table 2;
  exact values UNVERIFIED (backbone row ambiguity).
- Polysemy-aware metrics: no (no ECCV Caption, CxC, PMRP).
- Relevance to MultiMAE: "a part of an image or caption is more general than the whole" is the closest
  published analogue of treating a masked view as a generic query; it uses crops and phrases, not masks.
  [reader-inferred]
- Method weaknesses: needs box annotations at training time [author-acknowledged, limitations]; "may not be
  optimal for tasks like large-scale retrieval" [author-acknowledged, limitations]; retrieval differences of
  about 1 R@5 without variance [reader-inferred].

**[5] HierarCaps and Radial Embeddings**
- Citation: Alper, M., Averbuch-Elor, H. (2024). Emergent Visual-Semantic Hierarchies in Image-Text
  Representations. ECCV 2024 (oral). arXiv:2407.08521.
- Tier: peer-reviewed (ECCV 2024).
- Read scope: full text (arxiv.org/html) for method and dataset; Table 1 re-read raw on ar5iv.
- Setting: probes pretrained CLIP B/L, OpenCLIP, ALIGN; text-encoder-only fine-tuning (image tower frozen) on
  HierarCaps train, 1 epoch, batch 8, lr 1e-7, with a regularizer toward the pretrained embeddings.
- Change tested: Radial Embedding: the empty-string embedding is the "entailment root"; generality is distance
  from the root; entailment is an exterior angle. Baseline: the same model before fine-tuning.
- Effect (Table 1, raw; before -> after FT): CLIP-B HierarCaps P 0.14 -> 0.15, R 0.36 -> 0.47, tau_d 0.89 ->
  0.99; COCO text->image R@1 0.30 -> 0.31, R@5 0.55 -> 0.56. CLIP-L R 0.37 -> 0.44, tau_d 0.88 -> 0.97;
  COCO R@1 0.36 -> 0.36, R@5 0.60 -> 0.61. (A first parse mislabeled the BREEDS column as COCO; corrected.)
- Polysemy-aware metrics: HierarCaps (generic-to-specific hierarchy), not ECCV/CxC/PMRP.
- Relevance to MultiMAE: pretrained CLIP already carries a generality axis (distance from the empty caption), so
  "masked caption sits between the root and the full caption" is testable on existing checkpoints.
  [reader-inferred]
- Method weaknesses: one linear hierarchy per image oversimplifies branching descriptions [author-acknowledged,
  limitations]; dual encoders only [author-acknowledged]; train hierarchies are LLM-generated and NLI-filtered,
  so label noise is possible [reader-inferred].

**[6] A-CLIP (masked-view consistency)**
- Citation: Yang, Y. et al. (2023). Attentive Mask CLIP. ICCV 2023. arXiv:2212.08653.
- Tier: peer-reviewed (ICCV 2023).
- Read scope: full text (ar5iv): Sec 3, Table 1, Table 2a.
- Setting: from-scratch on YFCC-15M, ViT-B/16, 25 epochs.
- Change tested: keep 50% of image tokens chosen by EMA-encoder [CLS] attention (vs random), plus
  online-to-EMA distillation and SimCLR/SimSiam between masked views. Baselines: CLIP, SLIP, MaskCLIP.
- Effect: Table 1 (Flickr I2T/T2I R@1, COCO I2T/T2I R@1, IN-1K 0-shot): CLIP 51.4/32.6, 27.9/17.6, 37.6;
  MaskCLIP 60.0/38.8, 34.1/21.2, 42.7; A-CLIP 62.7/42.1, 38.0/23.2, 43.9. Table 2a: random 1x50% mask
  48.8/32.5, 28.9/16.6, 35.0 vs attentive 1x50% 57.6/36.6, 34.2/19.8, 39.5.
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: which tokens are masked decides whether a masked view stays a valid positive; random
  50% masking in the Table 2a ablation (Flickr I2T 48.8) sits below the Table 1 CLIP baseline (51.4), though the two tables may differ in setup. [reader-inferred]
- Method weaknesses: gains mix masking with two extra self-supervised losses and come from a 15M from-scratch
  setting [reader-inferred].

### 2.2 Sub-theme 2: masked or generative objectives on newer encoders; multi-positive and cross-modal masked training

**[7] SigLIP 2**
- Citation: Tschannen, M. et al. (2025). SigLIP 2: Multilingual Vision-Language Encoders with Improved
  Semantic Understanding, Localization, and Dense Features. arXiv preprint, arXiv:2502.14786.
- Tier: preprint, `preprint>=2024`.
- Read scope: full text (arxiv.org/html): training recipe section; Table 1 re-read raw on ar5iv.
- Setting: from-scratch on WebLI (10B images, 12B alt-texts, 109 languages), 40B examples seen.
- Change tested: SigLIP loss plus LocCa decoder (captioning, referring expressions, grounded captioning), plus
  self-distillation (1 global teacher view, 8 local student views) and masked prediction ("replace 50% of the
  embedded image patches in the student network with mask tokens and train the student to match the features of
  the teacher at masked locations"), both added at 80% of training; also data-mix and multilingual changes.
  Baseline: SigLIP.
- Effect (Table 1, B/16 at 256 px, SigLIP -> SigLIP 2): COCO T->I R@1 47.4 -> 53.2, I->T 65.1 -> 69.7;
  Flickr T->I 78.3 -> 81.7, I->T 91.1 -> 94.4; XM3600 T->I 22.5 -> 40.7. (A first parse assigned these
  columns wrongly; settled by the raw ar5iv header.)
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: large-scale evidence that adding masked feature prediction and a captioning decoder is
  compatible with better retrieval, but with no ablation isolating the masked term. [reader-inferred]
- Method weaknesses: several simultaneous changes (decoder, distillation, masking, data) are not ablated
  separately on retrieval (our reading) [reader-inferred].

**[8] TIPS**
- Citation: Maninis, K.-K., Chen, K., Ghosh, S., et al. (2025). TIPS: Text-Image Pretraining with Spatial
  Awareness. ICLR 2025. arXiv:2410.16512.
- Tier: peer-reviewed (ICLR 2025, per mlanthology).
- Read scope: full text (arxiv.org/html v2): method, Table 1 (raw), Table 3 (raw).
- Setting: from-scratch on about 116M curated WebLI images with noisy web captions and PaliGemma captions.
- Change tested (Table 1 ablation): (C) CLIP + self-distillation vs CLIP + self-distillation + masked image
  modeling (prototype-based recovery of masked patch semantics).
- Effect (Table 1): adding MIM: Flickr I->T 81.5 -> 82.6, T->I 67.0 -> 67.6; Pascal VOC segmentation 70.3 ->
  75.9; NYUv2 depth RMSE 0.589 -> 0.511; ImageNet KNN 79.1 -> 79.0. For scale, the caption change alone
  (noisy-caption CLIP -> both captions, dual embedding): Flickr I->T 79.1 -> 88.7, T->I 62.9 -> 77.1.
- Polysemy-aware metrics: no. The dual [CLS] design gives one image two embeddings, one per caption style.
- Relevance to MultiMAE: a controlled ablation where MIM added on top of contrastive + distillation moved
  retrieval by +0.6 to +1.1 and dense tasks by much more. [reader-inferred]
- Method weaknesses: single-run ablation rows with no variance shown [reader-inferred]; synthetic
  captions lack object detail, motivating the dual design [author-acknowledged, Sec 3].

**[9] Cap / CapPa**
- Citation: Tschannen, M., Kumar, M., Steiner, A., Zhai, X., Houlsby, N., Beyer, L. (2023). Image Captioners
  Are Scalable Vision Learners Too. NeurIPS 2023 (oral). arXiv:2306.07915.
- Tier: peer-reviewed (NeurIPS 2023).
- Read scope: full text (arxiv.org/html): Sec 3, Sec 5, Table 6.
- Setting: from-scratch on 1B English WebLI pairs; CLIP* trained on the same data as baseline.
- Change tested: captioning-only pretraining; CapPa adds parallel prediction (decoder input is all [MASK]
  tokens) for 75% of training examples (Sec 3).
- Effect (Table 6, ARO, B/16: Attribution / Relation / Order-Flickr / Order-COCO): CLIP* 53.2 / 39.7 / 45.5 /
  37.0; Cap 88.9 / 86.6 / 99.1 / 99.0; CapPa 85.7 / 86.7 / 99.2 / 98.8; text-only "blind decoder" 83.7 /
  86.2 / 98.8 / 98.7. Retrieval: CapPa is reported below CLIP* on COCO retrieval via LiT (Sec 4.2, Table 4;
  numbers not recorded).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: image-conditioned masked token prediction is the generative cousin of MultiMAE's MLM
  decoder; its compositional advantage over CLIP is large but mostly reproduced by a text-only decoder.
  [reader-inferred]
- Method weaknesses: ARO is largely solvable from language priors (blind decoder row) [reader-inferred];
  decoder-based scoring makes zero-shot retrieval expensive [author-acknowledged, Sec 5].

**[10] Llip**
- Citation: Lavoie, S., Kirichenko, P., Ibrahim, M., Assran, M., Wilson, A. G., Courville, A., Ballas, N.
  (2024). Modeling Caption Diversity in Contrastive Vision-Language Pretraining. ICML 2024 (PMLR 235).
  arXiv:2405.00740.
- Tier: peer-reviewed (ICML 2024).
- Read scope: full text (arxiv.org/html): method, Table 2 (re-read verbatim).
- Setting: from-scratch on MetaCLIP 2.5B (12.8B seen); ViT-B/32 to G/14; "All methods are pre-trained with the
  same dataset and use the same pre-training recipe" (Table 2 caption context).
- Change tested: the vision encoder outputs K learnable mixture tokens; a text-conditioned cross-attention mixes
  them into the final image embedding, so an image's embedding depends on the caption it is compared with.
- Effect (Table 2, R@1, MetaCLIP -> Llip_64): B/16 COCO I->T 59.4 -> 63.4, T->I 41.4 -> 45.6; Flickr I->T
  85.9 -> 90.1, T->I 70.5 -> 75.1. G/14 COCO I->T 66.7 -> 72.7, T->I 49.6 -> 54.2.
- Polysemy-aware metrics: no, despite the one-to-many motivation.
- Relevance to MultiMAE: the most direct "one image, many captions" mechanism in a dual encoder; never combined
  with masking or tested on ECCV Caption/PMRP. [reader-inferred]
- Method weaknesses: gallery image embeddings depend on the query, so retrieval needs per-pair cross-attention
  [reader-inferred]; the authors average queries over templates to cut zero-shot classification cost
  [author-acknowledged, Sec on zero-shot evaluation].

**[11] DreamLIP**
- Citation: Zheng, K., Zhang, Y., Wu, W., Lu, F., Ma, S., Jin, X., Chen, W., Shen, Y. (2024). DreamLIP:
  Language-Image Pre-training with Long Captions. ECCV 2024. arXiv:2403.17007.
- Tier: peer-reviewed (ECCV 2024).
- Read scope: full text (arxiv.org/html); component ablation table re-read raw on ar5iv.
- Setting: from-scratch on CC3M, CC12M, YFCC15M re-captioned by MLLMs (30M merged); baseline CLIP on original
  short captions.
- Change tested: long captions, sub-captions sampled as multiple positives, MLLM short captions, and a
  subcaption-specific grouping loss (sub-caption to local patches).
- Effect (component ablation table, ViT-B/16, CC3M; COCO text-retrieval R@1 / COCO image-retrieval R@1):
  original captions 14.8 / 11.5; long captions direct 32.7 / 23.0; full DreamLIP 42.8 / 30.4. The labels of
  the intermediate rows (sampling, short captions) could not be parsed reliably, so the multi-positive
  sampling step is not isolated here.
- Polysemy-aware metrics: no (ARO and SugarCrepe in a later table, not read).
- Relevance to MultiMAE: most of the gain comes from caption content; multiple positives per image is the
  training-side encoding of one-to-many. [reader-inferred]
- Method weaknesses: MLLM hallucination grows with caption length [author-acknowledged, limitations]; baseline
  uses short original captions, so data and loss changes are mixed [reader-inferred].

**[12] FLAIR**
- Citation: Xiao, R., Kim, S., Georgescu, M.-I., Akata, Z., Alaniz, S. (2025). FLAIR: VLM with Fine-grained
  Language-informed Image Representations. CVPR 2025. arXiv:2412.03561.
- Tier: peer-reviewed (CVPR 2025).
- Read scope: full text (arxiv.org/html latest and ar5iv v1): method, Table 1 (raw, v1), Table 5, Table 11.
- Setting: from-scratch on 30M re-captioned images (DreamLIP captions); ViT-B/16. "For fair comparison, we
  reproduce CLIP and SigLIP on all re-captioned datasets under identical training configurations."
- Change tested: text-conditioned attention pooling (global text embedding queries local image tokens) plus a
  multi-positive sigmoid loss over K = 8 sampled sub-captions.
- Effect: Table 1 (ar5iv v1, R@1): COCO T2I SigLIP 46.6 -> FLAIR 51.2, I2T 62.6 -> 67.3; DOCCI-FG T2I
  18.9 -> 23.0, I2T 46.3 -> 53.7. A later arXiv HTML version shows FLAIR COCO 53.3 / 68.0, so values are
  version-dependent. Table 5 (CC3M-recap, COCO T2I / I2T R@1): global loss only 28.3 / 40.1; text-conditioned
  loss only 32.0 / 45.6; full 37.7 / 51.6. Table 11: K = 2 -> 8 sub-captions, COCO T2I 36.4 -> 37.7.
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: text-conditioned image pooling is a concrete way to let one image answer different
  captions differently; shown only from scratch with synthetic captions. [reader-inferred]
- Method weaknesses: per-pair pooling cost at retrieval [reader-inferred]; still behind CLIP trained on much
  more data on global tasks [author-acknowledged, limitations].

**[13] COSMOS**
- Citation: Kim, S., Xiao, R., Georgescu, M.-I., Alaniz, S., Akata, Z. (2025). COSMOS: Cross-Modality
  Self-Distillation for Vision Language Pre-training. CVPR 2025, pp. 14690-14700. arXiv:2412.01814.
- Tier: peer-reviewed (CVPR 2025).
- Read scope: full text (arxiv.org/html v2): method, Table 1 (CC3M rows), Table 5 (verbatim).
- Setting: from-scratch, 3M to 30M pairs with synthetic long captions; ViT-B/16.
- Change tested: text cropping (global = 1 to 5 sentences, local = 1 sentence), image crops, EMA teacher,
  cross-attention modules producing cross-modal embeddings for a cross-modality self-distillation loss.
  "Notably, the cross-attention module is not involved in the inference process." No masking of patches or
  tokens anywhere in the method (checked).
- Effect (Table 5, CC3M, COCO I2T / T2I R@1): CLIP 15.0 / 10.7; + image aug 17.5 / 12.7; + image self-distill
  20.6 / 14.6; + text aug 50.4 / 37.5; + text self-distill 51.0 / 38.0; + cross-attention (= COSMOS)
  53.1 / 40.1. Table 1 CC3M: CLIP 40.2 / 27.2 vs COSMOS 53.1 / 40.1 (the Table 1 CLIP baseline's caption
  source is not stated in what I read; UNVERIFIED).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: same structural idea as MultiMAE's fusion (cross-modal modules used only during
  training, acting through gradients into the towers); the cross-attention step added +2.1 / +2.1 R@1 at CC3M.
  [reader-inferred]
- Method weaknesses: increments of about 2 points per row from single runs [reader-inferred]; most of the gain
  comes from the long synthetic captions [reader-inferred].

**[14] MACCO (cross-modal masked concept reconstruction on fine-tuned CLIP)**
- Citation: Li, W., Huang, Z., Tian, X. (2026). Cross-Modal Masked Compositional Concept Modeling for Enhancing
  Visio-Linguistic Compositionality. arXiv preprint, arXiv:2606.13288.
- Tier: preprint, `preprint>=2024`.
- Read scope: full text (arxiv.org/html v1): method, Table 1 (raw), Table 3 (raw), limitations. Appendix D
  ablation tables (Tables 9-12) not retrieved.
- Setting: fine-tunes OpenAI CLIP ViT-B/32 on about 110k COCO pairs, 55 epochs, batch 256, lr 5e-7 (CLIP) and
  1e-3 (predictors); also B/16, L/14, SigLIP.
- Change tested: relation/attribute concepts masked in text (located with GroundingDINO) and in the image
  (scene-graph parser); masked text reconstructed from full image features (MLM) and masked patches from full
  text features (MIM); masked instances used as soft negatives in the contrastive loss; intra-modal
  regularizer; predictors removed at inference. Baseline: CLIP-FT (contrastive fine-tune on the same COCO data).
- Effect (Table 1, CLIP / CLIP-FT / MACCO): ARO-Relation 58.7 / 64.3 / 73.1; ARO-Attribute 62.7 / 66.2 /
  68.5; ARO-Order 54.1 / 49.1 / 76.0; SugarCrepe Relation 68.8 / 71.1 / 77.1; SugarCrepe Attribute 70.8 /
  77.7 / 79.1; VL-Checklist Relation 63.6 / 60.9 / 70.2; VALSE Relation 70.1 / 69.3 / 75.3; What's-up 41.8 /
  41.4 / 43.2. Table 3: zero-shot classification average 59.5 / 57.9 / 58.0; linear probe 80.1 / 80.0 / 79.7.
  Image-text retrieval recall is not reported (checked).
- Polysemy-aware metrics: no.
- Relevance to MultiMAE: the closest published setup found (same backbone, same COCO data, mask one modality
  and reconstruct from the other, decoders dropped at test); its retrieval and one-to-many behaviour are
  untested. [reader-inferred]
- Method weaknesses: targeted masking depends on external parsers [author-acknowledged, limitations]; no
  retrieval numbers, so any R@K cost is unknown [reader-inferred]; ARO is partly solvable by text priors (see
  [9]) [reader-inferred].

### 2.3 Sub-theme 3: lexical polysemy and visual word sense disambiguation

**[15] SemEval-2023 Task 1: Visual Word Sense Disambiguation (task and system results)**
- Citation: Raganato, A., Calixto, I., Ushio, A., Camacho-Collados, J., Pilehvar, M. T. (2023). SemEval-2023
  Task 1: Visual Word Sense Disambiguation. Proc. SemEval-2023, pp. 2227-2234. ACL Anthology 2023.semeval-1.308.
  Dataset counts and per-language results taken from a participant paper: UAlberta at SemEval-2023 Task 1,
  arXiv:2306.14067 (Sec 3, Table 3).
- Tier: peer-reviewed (SemEval shared-task workshop).
- Read scope: overview: abstract only (PDF unparseable here); UAlberta: full text Sec 3 and Table 3.
- Setting: given a target word plus minimal context (mostly two-word phrases), pick one of 10 candidate images.
  Train: silver set of 12,869 English instances. Test: 968 instances = 463 English + 305 Italian + 200 Farsi.
  Metrics HIT@1 and MRR. 96 submissions, 40 beat the zero-shot CLIP baseline; systems "often" used generative
  models and data augmentation (overview abstract).
- Change tested and baseline: zero-shot CLIP baseline HIT@1 60.5 (En) / 22.6 (It) / 28.5 (Fa) (UAlberta
  Table 3).
- Effect (UAlberta Table 3, HIT@1): translate-to-English system Tr 61.1 / 59.3 / 43.0; Tr + InstructGPT
  definitions 69.1 / 63.3 / 40.0. Top-ranked FCLL: average H@1 72.56, MRR 82.22 (from the FCLL abstract via
  search; its anthology id UNVERIFIED).
- Polysemy-aware metrics: this is the benchmark.
- Relevance to MultiMAE: the standard lexical-ambiguity benchmark has 463 English test items, so a COCO
  fine-tune would be judged on a small set. [reader-inferred]
- Method weaknesses: overview not read in full, so task-design weaknesses are not assessed (read scope:
  abstract_only for the overview); the training set is silver (automatically built) per UAlberta Sec 3
  [reader-inferred].

**[16] Kritharoula et al. (LLM context completion for VWSD)**
- Citation: Kritharoula, A., Lymperaiou, M., Stamou, G. (2023). Large Language Models and Multimodal Retrieval
  for Visual Word Sense Disambiguation. EMNLP 2023. arXiv:2310.14025.
- Tier: peer-reviewed (EMNLP 2023).
- Read scope: full text (arxiv.org/html): Table 3, Table 6, Table 8.
- Setting: zero-shot CLIP/ALIGN retrievers; LLM-generated phrase enhancement; captioning (GiT) for
  text-text matching; learning-to-rank (LGBMRanker) trained on the VWSD train set (12,869 train, 463 test,
  10 images each; Table 8).
- Change tested and baseline: CLIP with the original phrase vs CLIP with GPT-3 "meaning_of" enhanced phrases;
  final LTR combination.
- Effect: Table 3, CLIP (with penalty): accuracy 63.28 / MRR 76.27 -> GPT-3 "meaning_of" 68.07 / 80.08.
  Table 6 best LTR (ALIGN + GiT-L greedy captions + all LLM prompts): 79.35 / 87.23. LLMs under 7B gave
  "only marginal" gains. No masked language modeling used (checked).
- Polysemy-aware metrics: VWSD (above).
- Relevance to MultiMAE: on VWSD the gains come from adding context to the short query (context completion),
  the opposite direction from masking it. [reader-inferred]
- Method weaknesses: the "penalty" baseline (63.28) differs from the official CLIP baseline (60.48 HIT@1 cited
  by other teams), so cross-paper comparison needs care [reader-inferred]; GPT-3 knowledge can leak sense
  information [reader-inferred].

**Short secondary entries (sub-theme 3)**
- PolCLIP: Yang, Q., Li, Y., Wang, X., Wang, F. L., Hao, T. (2024). PolCLIP: A Unified Image-Text Word Sense
  Disambiguation Model via Generating Multimodal Complementary Representations. ACL 2024 (Long), pp.
  10676-10690. Peer-reviewed. Read scope: abstract only. Setting: CLIP-based model for textual and visual WSD
  that "simulates" a diffusion model to produce implicit visual sense representations and a captioner to give
  implicit text for images; new image-sense dataset. Effect (abstract): +2.22% HR@1 on Visual-WSD and +2.53 F1
  on textual WSD over prior state of the art. Polysemy-aware: VWSD. Relevance: generative "fill in the missing
  modality" for senses, without masking. [reader-inferred] Weaknesses: not assessed (read scope: abstract_only).
- Bhattacharya, S. et al. (2026). Visual Word Sense Disambiguation with CLIP through Dual-Channel Text
  Prompting and Image Augmentations. arXiv:2602.06799. `preprint>=2024`. Read scope: full text (Tables II-V,
  X). Zero-shot CLIP B/32 (LAION) on the 463 English test items: vanilla MRR 0.7227 / Hit@1 0.5810 (Table III)
  -> full method 0.7590 / 0.6220 (Table II); prompting only 0.7506 / 0.6250 (Table IV); image augmentations
  only 0.7225 / 0.5745 (Table V); translation degrades (Table X). Relevance: on VWSD, text-side context helps
  and image-side augmentation does not. [reader-inferred] Weaknesses: large validation-test gap (0.851 -> 0.727
  MRR) when combining components [author-acknowledged, Table VII discussion].
- White, J. C., Cotterell, R. (2022). Schrödinger's Bat: Diffusion Models Sometimes Generate Polysemous Words
  in Superposition. arXiv:2211.13095. Preprint (2022). Read scope: abstract only. Finding: the CLIP text
  encoder used by Stable Diffusion encodes polysemous words as a superposition of meanings that linear edits can
  shift. Relevance: one CLIP text vector for an ambiguous word is already a sense mixture. [reader-inferred]
  Weaknesses: not assessed (read scope: abstract_only).
- Cekinmez, J., Wu, A. J., Marjieh, R., Griffiths, T. L. (2026). Where did the ambiguity go? Examining how
  multimodal models interpret polysemous words. arXiv:2608.00410. `preprint>=2024`. Read scope: full text
  (Figs 2-5, Tables 1-3). 100 English polysemous words (+25 Turkish, +25 French), 17 text-to-image and 15 text
  models, 540 human participants. Sense-distribution normalized entropy: image models 0.10, text models 0.25,
  humans 0.47 (Fig 2, Table 1); FLUX diffusion models depict several senses in one image in 6.6% of samples vs
  0.9% for others (Fig 5). No retrieval models evaluated. Relevance: generators collapse to dominant senses;
  not evidence about CLIP retrieval. [reader-inferred] Weaknesses: LLM judge (GPT-5.4) and closed sense
  inventories [author-acknowledged, limitations].
- Search for masked language modeling in VWSD systems (Q17, Q18, plus reading [15], [16], HKUST arXiv:2311.18273,
  Bhattacharya [above]): no VWSD system found that used MLM; context augmentation came from LLM definitions,
  glosses (WordNet/BabelNet) and translation.

### 2.4 Sub-theme 4: benchmarks for one-to-many retrieval and ambiguity

| Benchmark (source) | Modality and size | What counts as a positive | How polysemic | Read scope |
|---|---|---|---|---|
| ECCV Caption (Chun et al., ECCV 2022, arXiv:2204.03359; known) | COCO Karpathy-test subset. Abstract: x3.6 positive image-to-caption and x8.5 caption-to-image associations vs COCO. Query counts (1,261 image / 1,332 caption queries) seen only in search snippets: UNVERIFIED here; AAHR Table 3 caption also says ECCV Caption test set | Machine-proposed (5 models) and human-verified extra pairs; mAP@R, R-P | One-to-many from missing positives in COCO; not lexical ambiguity | abstract |
| CxC (Parekh et al., EACL 2021, arXiv:2004.15020; known) | COCO; 267,095 human similarity judgments | Graded image-image, caption-caption and image-caption similarity | Graded similarity, not sense ambiguity | abstract |
| PMRP and CUB Caption (Chun et al., PCME, CVPR 2021, arXiv:2101.05068) | CUB Caption: 11,788 images of 200 bird classes, 10 captions each (Reed et al.); 150 train/val classes, 50 test classes | CUB: any same-class pair; PMRP: COCO pairs whose object-class vectors differ in at most zeta in {0,1,2} positions | CUB: extremely one-to-many by construction (class-level), fine-grained; PMRP: proxy | full text (benchmark section, Tables 2-3). Table 2 CUB R-P i2t/t2i: VSE0 22.4 / 22.6 -> PCME 26.3 / 26.8 |
| MRW (Song & Soleymani, PVSE, CVPR 2019, arXiv:1906.04402) | 50K video (GIF)-sentence pairs from social media ("my reaction when") | Paired reaction sentence | Loose, affective text-video relation; video not images | abstract |
| HierarCaps (Alper & Averbuch-Elor, ECCV 2024, entry [5]) | 73K train images (auto hierarchies), 1K manually curated test items, 4 tiers | Every tier of an image's generic-to-specific chain | Generality hierarchy, not one-to-many retrieval in COCO's sense | full text |
| INQUIRE (Vendrow et al., NeurIPS 2024 D&B, arXiv:2411.02537) | 5M iNat24 images, 250 expert queries, 33,000 labeled matches | All exhaustively labeled matches per query | Many positives per query (expert, natural world); best models below 50 mAP@50 (abstract) | abstract |
| SemEval-2023 VWSD (entry [15]) | 463 En / 305 It / 200 Fa test, 10 images each | The one image showing the intended sense | Lexical ambiguity with minimal context | see [15] |
| LaViSA (Lee, Inadumi, Yoshino, 2026, arXiv:2606.19552, `preprint>=2024`) | 700 structurally ambiguous sentences, 1,503 disambiguated sentences, DALL-E 3 images (cartoon and photorealistic) | Multiple choice between 2-3 interpretations | Structural (attachment, scope, ellipsis) ambiguity; only MLLMs evaluated (Gemini 3.1 Pro 88.9 per-trial accuracy, Table 2) | full text |

Short secondary entries (sub-theme 4)
- AAHR: Chen, J., Gao, Y., Ge, M., Li, M. (2025). Ambiguity-Aware and High-Order Relation Learning for
  Multi-Grained Image-Text Matching. arXiv:2507.09256. `preprint>=2024`. Read scope: full text (method, Table 3).
  Frozen CLIP B/32 global features + Faster R-CNN regions + BERT, trained on COCO; prototypes for soft
  positives; no masking. Table 3 ECCV Caption (I2T mAP@R / R-P / R@1, T2I mAP@R / R-P / R@1): ESA 31.5 / 43.0 /
  74.6, 49.3 / 56.9 / 90.2; AAHR 34.2 / 44.8 / 81.3, 49.7 / 57.4 / 90.3. Relevance: 2025 matching papers still
  report ECCV mAP@R; soft-positive handling moved I2T more than T2I here. [reader-inferred] Weaknesses: single
  runs vs numbers copied from other papers [reader-inferred].
- PCME++ MSDA ablation (Chun, ICLR 2024, arXiv:2305.18171; known source, B4-specific detail). Read scope: Table 3
  (verbatim), MSDA section. Fine-tunes CLIP ViT-B/32 on COCO, 25 epochs; MSDA mixes 25% of images with
  Mixup/CutMix (lambda ~ Beta(2,2)) with softened labels. Table 3 (columns read as ECCV mAP@R, ECCV R-P, ECCV
  R@1, CxC R@1, COCO 1K R@1, COCO 5K R@1, RSUM): no VIB/PP/MSDA 38.9 / 48.6 / 82.2 / 56.7 / 75.2 / 54.9 / 535.9;
  MSDA only 39.0 / 48.6 / 82.1 / 56.4 / 74.9 / 54.6 / 535.5; all three 40.1 / 49.7 / 83.1 / 56.8 / 75.4 /
  55.1 / 537.0. Relevance: the only case found where partial (cut or mixed) visual inputs were tested on ECCV
  Caption in a CLIP-B/32-on-COCO fine-tune; alone they moved mAP@R by +0.1. [reader-inferred] Weaknesses: no
  variance shown in Table 3 as parsed [reader-inferred].

### 2.5 Short secondary entries (sub-themes 1-2)

- TULIP: Tang, Z., Lian, L., Eisape, S., Wang, X., Herzig, R., Yala, A., Suhr, A., Darrell, T., Chan, D. M.
  (2025). TULIP: Towards Unified Language-Image Pretraining. arXiv:2503.15485. `preprint>=2024`. Read scope:
  full text (method, Table 5). Initialized from SigLIP; 500M DataComp-1B samples (20% re-captioned); adds
  image-image and text-text contrastive terms, reconstruction (MAE-style image decoder with the embedding as
  bottleneck; T5-based causal text decoder from the text embedding; one modality per step) and generative
  augmentation (GeCo). Table 5 (MMVP / LLaVA): SigLIP 5.9 / 81.1; + I/I and T/T contrastive 17.4 / 82.3;
  + reconstruction 18.2 / 82.1; + GeCo 20.3 / 81.9. Retrieval: two parses of the retrieval table were mutually
  inconsistent, so no retrieval numbers are reported here (UNVERIFIED). Relevance: reconstruction on a
  pretrained contrastive model added +0.8 MMVP. [reader-inferred] Weaknesses: loss weights not given (as
  parsed); no retrieval ablation [reader-inferred].
- Masked Diffusion Captioning: Feng, C., Wei, Z., Owens, A. (2025). Masked Diffusion Captioning for Visual
  Feature Learning. Findings of EMNLP 2025. arXiv:2510.26799. Peer-reviewed. Read scope: full text (Table 2,
  Sec 5). From-scratch on CC12M / Recap-DataComp subsets; image-conditioned masked diffusion LM (mask ratio t
  in [0.5, 1]). Table 2 ARO-Relation: CLIP 53.6, autoregressive captioner 82.7, MDC 84.6. No retrieval or
  one-to-many analysis. Weaknesses: academic scale (about 10M pairs) [author-acknowledged, Sec 5].
- HiMo-CLIP: Wu, R. et al. (2025). HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language
  Alignment. arXiv:2511.06653. `preprint>=2024`. Read scope: method and results summary (table locators not
  recorded, so no numbers given). Fine-tunes CLIP B/16 and L/14 on ShareGPT4V (1.2M); in-batch PCA components
  of text embeddings act as partial descriptions; a monotonicity-aware loss asks fuller text to align more
  strongly; evaluated on long-text retrieval and a new HiMo-Docci set, not ECCV/CxC. Relevance: "partial text
  should score lower than full text" is a property masked captions could be tested against. [reader-inferred]
  Weaknesses: not assessed beyond read scope.
- MAP (Ji et al., CVPR 2023; known): checked for B4. No ECCV Caption, CxC, PMRP or CUB evaluation; Table 5
  ablates task pairs on VQA/SNLI-VE/NLVR2 only, so distributional MLM's effect on retrieval is not isolated.

### 2.6 Mechanism record: combined with masked reconstruction? evaluated on a polysemy-aware benchmark?

| Candidate mechanism | Sources | Combined with masked reconstruction? | Polysemy-aware evaluation? | Where searched for "not found" |
|---|---|---|---|---|
| Masked input as a more general view (inclusion of original in masked) | [1] ProLIP, [2] LongProLIP | No: masking used for inclusion, no reconstruction | Only zero-shot ECCV mAP@R of the whole model ([2] Table C.1); the masked term is never ablated on ECCV/PMRP | Q8, Q10, Q15, Q16; full text of [1], [2] |
| Distributional MLM (uncertainty-aware MLM) | MAP (known) | Yes (MLM is the objective) | No | full text of MAP (Sec 2.5) |
| Hyperbolic entailment cones (generic near origin) | [3] MERU, [4] HyCoCLIP; Hyper3-CLIP arXiv:2608.29313 (read, from scratch on GRIT, no masking, HierarCaps only) | Not found | No (hierarchy metrics, HierarCaps) | Q4, Q8, Q15; full text of [3], [4], Hyper3-CLIP |
| Radial embedding / monotonicity of partial text | [5] HierarCaps, HiMo-CLIP | Not found | HierarCaps only | Q4, Q8; full text of [5], HiMo-CLIP |
| Masked-view consistency / masked feature prediction | [6] A-CLIP, [7] SigLIP 2, [8] TIPS | Masked prediction yes (feature targets) | No | Q1, Q10, Q11; full text of [6]-[8] |
| Captioning or parallel masked token prediction | [9] CapPa, Masked Diffusion Captioning, SigLIP 2 LocCa | Yes | No (ARO/SugarCrepe only) | Q10, Q11; full text |
| Text-conditioned image embedding (many embeddings per image) | [10] Llip, [12] FLAIR | Not found | No | Q3, Q5, Q15; full text of [10], [12] |
| Multi-positive sub-captions from LLM recaptioning | [11] DreamLIP, [12] FLAIR, [13] COSMOS | Not found ([13] explicitly has no masking) | No | Q5, Q11, Q14; full text of [11]-[13] |
| Cross-modal masked reconstruction (mask one modality, predict from the other) on fine-tuned CLIP | [14] MACCO (plus known Weers 2023, MaskVLM, MAMO) | Yes | No (compositionality only; no retrieval recall reported) | Q1, Q10, Q11, Q15; full text of [14] |
| Training-only cross-modal modules acting through gradients | [13] COSMOS (no masking), [14] MACCO (masking) | [14] yes | No | as above |
| Mixed or partial visual inputs with soft labels | PCME++ MSDA (known) | No | Yes: ECCV mAP@R +0.1 alone (Table 3) | Q9; full text of PCME++ Table 3 |
| Context completion for ambiguous words (LLM definitions) | [15], [16], Bhattacharya 2026 | No (no VWSD system with MLM found) | VWSD yes | Q2, Q7, Q17, Q18; full text of [16], UAlberta, HKUST |

Excluded after reading: Descriptive Image-Text Matching with Graded Contextual Similarity (arXiv:2505.09997,
withdrawn by the authors); Dynamic Distribution-Aware Uncertainty Tracking (arXiv:2608.09011, ACM MM 2026,
classification failure detection, no retrieval); Bridging Lexical Ambiguity and Vision: A Mini Review on VWSD
(arXiv:2602.01193; secondary, and its headline "MRR 95.77 / HIT@1 92.00" for Setitra et al. 2025 could not be
traced to a primary source: UNVERIFIED); Hyper3-CLIP (arXiv:2608.29313; kept only in the mechanism table, its
baseline UNCHA was not checked); HKUST at SemEval-2023 (arXiv:2311.18273; used only for the cited official
CLIP baseline of 60.48 HIT@1 / 73.88 MRR).

## 3. Search limitations

- ACL Anthology PDFs (SemEval-2023 overview, PolCLIP) could not be parsed: one was unreadable binary to the
  fetch tool and the other exceeded the size limit; no local PDF tools were available. Overview dataset counts
  therefore come from participant papers.
- Every number passed through an extraction step. First parses mislabeled columns in SigLIP 2 Table 1,
  HierarCaps Table 1 and FLAIR Table 1 (version difference); these were re-read as raw cells on ar5iv. TULIP's
  retrieval table and HyCoCLIP's backbone row stayed inconsistent and are marked UNVERIFIED.
- Not read: SILC, CoCa, LaCLIP, Recap-DataComp, Unmasked Token Alignment, cluster masking, Semantics-enhanced
  Cross-modal MIM (arXiv:2403.00249), PHyCLIP (arXiv:2510.08919), Ukrainian VWSD benchmark (UNLP 2024; snippet
  says 87 -> 381 instances, UNVERIFIED), FCLL system paper (only its abstract numbers via search).
- MACCO's appendix ablations (MLM vs MIM vs random masking) were not retrievable from the HTML.
- No direct Semantic Scholar or OpenReview API queries; IEEE/ACM journals (TPAMI, TMM, ACM MM) only through web
  search, so journal-only work on one-to-many matching may be missed.
- "Not found" claims cover the queries and full texts listed in Section 2.6 only.
- No retrieved page contained text that tried to direct the agent.
