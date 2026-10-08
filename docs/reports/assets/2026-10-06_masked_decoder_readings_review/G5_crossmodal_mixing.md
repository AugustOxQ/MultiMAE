> Supporting file for docs/reports/auto/v1/2026-10-06_masked_decoder_readings_review.md. Phase 2 bibliography as written; V2 lists the corrections, and the report uses corrected values.

# G5: cross-modal mixing augmentation (bibliography, Phase 2)

Searched 2026-10-06. Agent: G5. Extraction note: for MixGen, PCME++ and Geodesic, numbers were re-read from the raw arXiv LaTeX source (e-print)
downloaded to `dl_G5/` (the coordinator's method note); for every other entry, page text was read through a WebFetch
summarizer (arxiv.org/html, ar5iv, CVF/NeurIPS pages), not through a local PDF reader (`pdftotext` is not installed;
PDF-only fetches of Geodesic, LeMDA, MixMAE, MixSiam-type papers returned binary). Numbers for those other entries are therefore
"as transcribed by the fetcher" and should be re-checked before citing; where two fetches of the same table disagreed, this is stated. Nothing is filled in
from memory.

## 1. Search strategy

- **Sources:** open web via WebSearch (standard and extended), arXiv html/ar5iv, CVF open access (WACV 2023W page
  returned 403; venue confirmed elsewhere, see MixGen entry), NeurIPS/ICLR/AAAI/PMLR proceedings listings,
  amazon.science, Pith. Semantic Scholar and ACL Anthology were not queried directly.
- **Queries (Q1...Q14):**
  - Q1 MixGen Hao WACV 2023 image interpolation text concatenation
  - Q2 PCME++ mixed sample data augmentation MSDA
  - Q3 geodesic multimodal mixup contrastive Oh NeurIPS 2023
  - Q4 MixMAE Mixed and Masked Autoencoder CVPR 2023
  - Q5 image-text mixup augmentation vision-language retrieval 2025 mixed images multiple captions one-to-many
  - Q6 LeMDA learning multimodal data augmentation in feature space
  - Q7 i-Mix / MixCo / Un-Mix
  - Q8 TiMix text-aware image mixing
  - Q9 blended / superimposed images ambiguity multiple interpretations CLIP
  - Q10 mix-and-separate / unmixing objective, vision-language
  - Q11 cross-modal mixup + masked autoencoder, image-text, reconstruct both
  - Q12 mixup image-text contrastive multi-positive soft labels CLIP fine-tuning 2024-2026
  - Q13 VLMixer cross-modal CutMix
  - Q14 composite images / two captions / multi-concept retrieval; MixGen follow-ups noisy correspondence 2024-2026;
    "transparency prediction" self-supervised pretext
- **Date range:** 2020 to 2026-10; last searched 2026-10-06.
- **Inclusion:** the source mixes samples (pixels, patches, embeddings, tokens, features) with a stated label or
  target rule, or reconstructs/separates mixed sources, or uses blended/ambiguous images to probe multiple readings.
  **Exclusion:** mixing used only for classification with no label-rule detail and no link to contrastive or
  reconstruction objectives; papers where only an unreadable PDF was available and no abstract-level facts were
  confirmable (Dataset Augmentation by Mixing Visual Concepts, arXiv 2412.15358: generative concept mixing, PDF
  unreadable, excluded); VWSD with CLIP + image augmentations (arXiv 2602.06799: augmentations are crops, flips,
  jitter, no blending; excluded); M3AE, VLMAE (masked cross-modal reconstruction without mixing; belong to other themes).
- **Counts (approximate):** identified about 45 distinct candidates, screened about 30 by title/abstract, full text
  (via fetcher) of 14, included 14 (12 with full-text reads, 2 abstract-only).
- **Coverage skew:** strong for 2020-2023 mixing papers; thin for 2025-2026 (only searched, little found). English
  only. The decisive negative finding is in section 4.

## 2. Annotated bibliography

### (a) Image-text mixing for vision-language training

**A1. MixGen (the author's idea, nearest published form)**
- Citation: Hao, Zhu, Appalaraju, Zhang, Zhang, Li, Li. 2023. "MixGen: A New Multi-Modal Data Augmentation."
  arXiv 2206.08358 (v1 2022-06-16, v3 2023-01-09).
- Venue as verified: WACV 2023 Workshop on Pretraining Large Vision and Multimodal Models (oral). Confirmed by the
  arXiv abs page ("Oral presentation at WACV 2023 Pretraining Large Vision and Multimodal Models Workshop"), by
  amazon.science ("WACV 2023 Workshop on Pretraining Large Vision and Multimodal Models") and by the CVF listing
  `openaccess.thecvf.com/content/WACV2023W/Pretrain/...` seen in search results (the CVF page itself returned 403, so
  the page header was not read). Workshop paper, no journal reference. Tier: peer-reviewed (workshop).
- Read scope: full text via ar5iv: Sec 3 method, Tables 1, 2, 3, 4, 7. Numbers below re-read from the raw LaTeX source (arXiv e-print, `mixgen.tex`, tables
  `tab:ft_retrieval`, `tab:zs_retrieval`, `table:general_model`, downstream-task table, `tab:mixgen_variants`,
  `table:variants`).
- Setting: pre-training from scratch of vision-language models (ALBEF main; ViLT, CLIP, TCL for compatibility), data
  3M setting = 3.3M unique images / 4.4M pairs (COCO, VG, SBU, CC), plus a 1M setting (COCO, VG, SBU).
- Exactly how it mixes: image `I_k = lambda*I_i + (1-lambda)*I_j` with fixed lambda = 0.5 (pixel interpolation, i.e.
  50% opacity blend); text `T_k = concat(T_i, T_j)` (both full captions kept, in order); M = B/4, i.e. a quarter
  of each batch is replaced by such generated pairs, built from pairs within the batch. Labelled by the paper as
  semantic-preserving "both-at-once". Not read: the exact loss handling of the generated pair (assumed to be one more
  positive pair for each objective) [reader-inferred].
- Change tested / strongest baseline: same model and data without MixGen (ALBEF-base 3M; other architectures 1M).
- Effect (all from the 3M-image setting, baseline = the authors' own ALBEF-base retrained on 3M, not the gray 4M
  published ALBEF row). Fine-tuned retrieval, Table "Fine-tuned image-text retrieval": COCO 5K TR R@1 72.5 -> 74.2,
  IR R@1 55.8 -> 57.3, RSUM 485.6 -> 491.8 (+6.2); Flickr30K 1K TR R@1 95.1 -> 94.8, IR R@1 81.4 -> 82.4, RSUM 569.5 ->
  570.9 (+1.4). Zero-shot Flickr30K: TR R@1 91.1 -> 91.6, IR R@1 75.7 -> 77.2, RSUM 552.8 -> 558.1 (+5.3). So the abstract's
  "+6.2%" and "+5.3%" are RSUM points. Compatibility table (1M images, RSUM, Flickr-Ft / COCO-Ft / Flickr-Zs): ViLT
  232.2 / 172.5 / 135.5 -> +9.2 / +17.2 / +14.9; CLIP 538.2 / 450.5 / 485.3 -> +2.8 / +4.1 / +6.8; ALBEF 555.7 / 470.7 /
  518.0 -> +5.3 / +7.0 / +6.3; TCL 561.5 / 487.0 / 537.3 -> +2.1 / +3.2 / +2.2. Downstream (ALBEF 3M, baseline -> MixGen):
  VQA test-dev 74.38 -> 74.51, test-std 74.51 -> 74.79; NLVR2 dev 79.47 -> 80.23, test-P 80.05 -> 80.94; SNLI-VE val
  79.49 -> 80.05, test 79.69 -> 80.05; weakly-supervised grounding on RefCOCO+ +0.9 per abstract (table not
  read). Variant ablation (1M ALBEF, RSUM Flickr-Ft / COCO-Ft / Flickr-Zs; baseline 555.7 / 470.7 / 518.0; Table
  "Ablation of MixGen design"): MixGen 561.0 / 477.7 / 524.3; (a) lambda ~ Beta(0.1,0.1) 553.2 / 467.4 / 512.9;
  (b) blend images, keep one text (T_i or T_j) 557.2 / 470.0 / 516.2; (c) pick one image, concat texts 555.4 / 472.3 /
  518.2; (d) proportional token split 555.2 / 463.7 / 506.8; (e) concat then keep random 50% of tokens, lambda
  Beta(0.1,0.1) 559.2 / 477.1 / 523.7. The paper also has an input-level vs embedding-level mixing comparison
  (input-level better) and a ratio study (M = B/4 best at B = 512), both in a figure, not read numerically.
- Multiplicity/polysemy measured? No. Standard recall metrics only; no ECCV Caption, no multi-reference measure.
- Framing: both-at-once (single concatenated caption). Variant (b), blend with only one of the two captions, is the
  closest thing to either-one in the paper (a random single caption per blended image, not a separate target for each
  source) and gives about no gain on COCO-Ft (470.0 vs 470.7).
- Mix space and ratio: pixels, lambda = 0.5, quarter of each batch.
- Combined with masked reconstruction? ALBEF's pre-training includes MLM on its text (not stated whether mixed pairs go
  through it; not read). No image reconstruction.
- Relevance to MultiMAE / H-b: [reader-inferred] the direct precedent for the author's blend-and-concatenate idea;
  it shows a retrieval gain only in the pre-training regime and says nothing about one-to-many matching or decoder
  behaviour. Ablation (b) and (c) suggest the concat text matters (both-at-once), a point relevant to the either-one contrast.
- Method weaknesses: [author-acknowledged, limitations section] small scale (no LAION-400M), only four models and five
  tasks, no captioning or generation. [reader-inferred] gains are RSUM sums and per-recall changes are mixed (Flickr30K fine-tuned TR R@1 falls 95.1 -> 94.8), the ablation is
  one run per cell with no seed variance visible in the tables, and the baseline is a 3M-image retrain that sits
  below the published 4M ALBEF on most cells; 1M-setting ablations may not transfer.

**A2. PCME++ (re-read for G5: what MSDA does and its isolated effect)**
- Citation: Chun. 2024. "Improved Probabilistic Image-Text Representations." ICLR 2024; arXiv 2305.18171 (v5
  2024-04-09). Tier: peer-reviewed. First author given as in the arXiv listing; author names were not extracted by the
  fetch, so UNVERIFIED beyond "Chun".
- Read scope: full text via arxiv html, MSDA Sec 3.3, and the raw LaTeX (`main.tex`, `tables/loss_ablation.tex`) for Table 3 and the hyper-parameter paragraph; the appendix ablation on MSDA batch-wise mixing was not read.
- Setting: training from a pretrained CLIP ViT-B/32 probabilistic head setup on COCO Caption; also ViT-B/16, L/14.
- Change tested: mixed sample data augmentation on **images only** (Mixup and CutMix); text is not mixed ("MSDA for
  the textual modality is not straightforward"). 25% of the mini-batch images are mixed with lambda ~ Beta(2,2), Mixup or CutMix chosen at random per mini-batch.
  Label rule: the pair-wise matching indicator of a mixed image and the original caption is smoothed to
  m_vt = lambda (the pair-wise probabilistic loss takes a soft label in [0,1]); this is a soft target per
  (mixed image, source caption) pair, i.e. an **either-one / soft multi-positive** framing, pixel or patch space.
  The paper notes batch-softmax losses cannot take this directly (mixed samples would enter the denominator).
- Effect (Table 3 (`loss_ablation.tex`, caption "Effect of optimization methods"), ViT-B/32, COCO; columns ECCV mAP@R, ECCV R-P, ECCV R@1, CxC R@1, COCO 1K R@1, 5K R@1, RSUM; ECCV mAP@R shown):
  base (no VIB/PP/MSDA) 38.9; MSDA only 39.0 (RSUM 535.9 -> 535.5, i.e. no gain); PP only 39.0; VIB only 39.3;
  VIB+PP 39.6; VIB+PP+MSDA 40.1 (RSUM 537.0). So MSDA alone is within noise at +0.1 mAP@R and its contribution
  inside the bundle is +0.5 over VIB+PP (39.6 -> 40.1), single table, seeds not read.
  This extends the old B2 entry, which gave only the bundle "VIB + pseudo-positives + MSDA 38.9 -> 40.1".
- Multiplicity measured: ECCV Caption mAP@R and R-Precision (numbers above), CxC R@1.
- Framing: either-one (soft target lambda per source caption). Mix space: pixels / image regions, lambda ~ Beta(2,2),
  25% of batch. Combined with masked reconstruction: no.
- Relevance to MultiMAE / H-b: [reader-inferred] the only mixing method in this list evaluated on ECCV mAP@R, with
  an isolated row; the isolated gain is null, so published evidence that mixing helps one-to-many retrieval is weak.
- Method weaknesses: [reader-inferred] one backbone in the ablation table; differences of 0.1-0.5 mAP@R are
  reported without seed variance in the text read; the label is the mixing coefficient, which assumes a pixel-area
  share equals semantic weight.

**A3. Geodesic Multi-Modal Mixup**
- Citation: Oh, So, Byun, Lim, Shin, Jeon, Song. 2023. "Geodesic Multi-Modal Mixup for Robust Fine-Tuning." NeurIPS
  2023 (main track); arXiv 2203.03897. Tier: peer-reviewed.
- Read scope: full text via arxiv html plus the raw LaTeX source (`main.tex`) for Table 1 and the implementation details.
- Setting: fine-tuning CLIP ViT-B/32 on Flickr30k (30K train) and COCO (118K), 9 epochs, batch 128.
- Change tested: mix **L2-normalised image and text embeddings** (geodesic interpolation on the hypersphere,
  lambda ~ Beta(0.5,0.5)) to build **hard negatives** that enter the contrastive denominator only (not the numerator).
  Baselines include naive fine-tuning, temperature variants, i-Mix, Un-Mix.
- Effect (Table `tab:retrieval_multimodal`, top-1 recall, re-read from the LaTeX source; the full method row is
  labelled m^3-Mix, which adds uni-modal mixups to the hard-negative term m^2-Mix): COCO FT i->t 36.7, t->i 36.9; multimodal mixup 41.0 / 39.9; FT with
  temperature 0.05 is 40.2 / 41.6 and multimodal mixup at temperature 0.05 is 40.4 / 42.0, so part of the gain is a
  temperature effect. Flickr30k FT 81.2 / 80.7; mixup 82.3 / 82.7. i-Mix 34.0 / 34.6 and Un-Mix 38.8 / 33.4 on COCO sit
  at or below plain FT. The low absolute COCO values suggest a non-standard protocol; not checked.
- Multiplicity measured: no.
- Framing: neither; the mixed point is a hard negative ("not either-one, not both"). Mix space: embeddings, lambda
  ~ Beta(0.5,0.5). Combined with masked reconstruction: no (a captioning loss with CoCa is mentioned in search results only).
- Relevance: [reader-inferred] shows that image-text mixing in embedding space can be used with the opposite label to
  the "both" reading; evidence against assuming mixing implies multi-positive.
- Method weaknesses: [author-acknowledged, limitations] over-lifting uniformity/alignment hurts retrieval in some cases.
  [reader-inferred] the temperature confound above; embedding mixing has no pixel-level counterpart.

**A4. LeMDA**
- Citation: Liu, Zhang, Smola, et al. 2023. "Learning Multimodal Data Augmentation in Feature Space." ICLR 2023;
  arXiv 2212.14453. Tier: peer-reviewed. Full author list not extracted (UNVERIFIED beyond "Liu").
- Read scope: full text via ar5iv, summary-level only (no tables extracted).
- Setting: supervised fine-tuning on 8 datasets (Hateful Memes, Food101, SNLI-VE, Petfinder, others); classification.
- Change tested: a learned augmentation network (VAE-like) that generates augmented multimodal latent features, trained
  adversarially with the task network; baselines include Mixup, Manifold Mixup, MixGen, EDA, TrivialAugment; reported
  gains of up to about 6% accuracy on Hateful Memes and Petfinder.
- Multiplicity measured: no. Framing: not applicable (learned, not a defined mix). Mix space: features.
  Combined with masked reconstruction: no.
- Relevance: [reader-inferred] a baseline family comparison that includes MixGen; no retrieval, so not informative for H-b.
- Method weaknesses: [author-acknowledged] 30-50% throughput loss; late-fusion architectures only, early-fusion open.

**A5. TiMix**
- Citation: Jiang et al. 2024. "TiMix: Text-Aware Image Mixing for Effective Vision-Language Pre-training." AAAI 2024;
  arXiv 2312.08846. Tier: peer-reviewed. Author list beyond "Jiang" not extracted.
- Read scope: full text via arxiv html (Sec 3, ablations at summary level).
- Setting: pre-training mPLUG-style model on up to 14M images (COCO, VG, CC, CC12M, SBU); retrieval and others.
- Change tested: CutMix with side ratio gamma ~ U(0.25, 0.75), choosing the source region most relevant to its text and
  the target region least relevant; **soft labels** from area share, s_y = hw/HW, s_x = 1 - s_y over the two source
  texts; plus a patch-text alignment (PTA) task built from COCO/VG object labels.
- Effect: mPLUG-TiMix (14M) COCO R@1 82.3 (text retrieval) and 65.2 (image retrieval), reported better than BLIP (129M).
  Ablation: without PTA, random region mixing gives negligible gain; PTA alone helps less than full TiMix (no
  numbers extracted).
- Multiplicity measured: no. Framing: either-one (soft target per source text). Mix space: image regions, gamma
  0.25-0.75, relevance-guided. Combined with masked reconstruction: no.
- Relevance: [reader-inferred] a retrieval-focused example of "each source caption is a separate soft target"; again
  no one-to-many metric.
- Method weaknesses: [reader-inferred] relevance scoring depends on a pre-trained model's patch-text scores; the
  isolated effect of mixing is entangled with PTA in the fetched text.

**A6. VLMixer**
- Citation: Wang et al. 2022. "VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix." ICML 2022
  (PMLR 162:22680-22690); arXiv 2206.08919. Tier: peer-reviewed.
- Read scope: abstract only (search result summary).
- Setting: pre-training with unpaired text-only and image-only corpora; five downstream tasks.
- Change tested: cross-modal CutMix replaces visually-grounded words in a sentence by image patches of similar
  semantics, trained with contrastive learning between uni-modal and multi-modal views.
- Effect, multiplicity, framing, mask link: not assessed (read scope: abstract_only). Mix space: tokens to patches.
- Relevance: [reader-inferred] a token-level cross-modal substitution, a different mixing axis from pixel blending.
- Method weaknesses: not assessed (read scope: abstract_only).

**A7. MixReorg**
- Citation: Cai et al. (authors beyond first UNVERIFIED). 2023/2024. "MixReorg: Cross-Modal Mixed Patch Reorganization
  is a Good Mask Learner for Open-World Semantic Segmentation." arXiv 2308.04829 (v2, 2024-03). Tier: preprint (venue
  not verified; read as an arXiv version).
- Read scope: full text via arxiv html, summary-level.
- Setting: vision-language pre-training from image-text pairs for zero-shot semantic segmentation.
- Change tested: patches from several images are mixed into one image, each patch keeping its source image's text
  (patch-to-text correspondence preserved); a "mixing restoration" stage restores the source images from mixed patches
  with contrastive loss between restored features and text; segmentation masks for mixed images.
- Effect: zero-shot mIoU PASCAL VOC 50.5, PASCAL Context 25.4, COCO 23.6, ADE20K 10.1 (multi-scale), better than
  GroupViT by 5.0 to 6.2 points; ablation: contextual mixing removed 19.3 -> 18.0 on COCO.
- Multiplicity: no. Framing: unmixing plus either-one (each patch region tied to its own source text). Mix space: patches.
  Combined with masked reconstruction: restoration of source images from mixed patches is a mixing-restoration task,
  not pixel MAE.
- Relevance: [reader-inferred] closest cross-modal "recover each source from a mixed image, conditioned on text"
  precedent, but on segmentation, not retrieval.
- Method weaknesses: [author-acknowledged] extra training compute from mixing; gap to pixel-level supervision.

**A8. M3ixup**
- Citation: "M3ixup: A multi-modal data augmentation approach for image captioning." Pattern Recognition, 2024
  (ScienceDirect S0031320324006927; authors not retrieved, UNVERIFIED). Tier: peer-reviewed (page returned 403).
- Read scope: abstract only (from search snippet: mixes visual features, sentences and the objective).
- Setting, effect, multiplicity, framing, mask link, weaknesses: not assessed (read scope: abstract_only).
- Relevance: [reader-inferred] sentence-level mixing in captioning; left in as the only caption-side mixing hit.

### (b) Mixing in contrastive SSL with soft or multi-positive targets (the "either-one" framing)

**B1. i-Mix**
- Citation: Lee, Zhu, Sohn, Li, Shin, Lee. 2021. "i-Mix: A Domain-Agnostic Strategy for Contrastive Representation
  Learning." ICLR 2021 (iclr.cc 2021 listing); arXiv 2010.08887. Tier: peer-reviewed. (Author list as I recall it; not
  extracted by the fetch, UNVERIFIED.)
- Read scope: full text via ar5iv, summary level.
- Setting: from-scratch SSL; SimCLR, MoCo v2, BYOL; CIFAR, ImageNet-100/1k; also speech, tabular.
- Change: Mixup or CutMix of inputs with the **virtual instance labels mixed in the same proportion**
  (loss on (Mix(x_i,x_j;lambda), lambda*v_i + (1-lambda)*v_j)); lambda ~ Beta(alpha,alpha). This is the
  either-one / soft multi-positive target.
- Effect (linear eval, Table 1/2): CIFAR-10 93.5 -> 96.1 (MoCo v2), CIFAR-100 71.6 -> 78.1, ImageNet-100 84.1 -> 87.0,
  ImageNet-1k 70.9 -> 71.3 (small at scale). 
- Multiplicity: no (classification linear probe). Framing: either-one. Mix space: input pixels (or CutMix regions).
  Combined with masked reconstruction: no.
- Relevance: [reader-inferred] origin of "mixed input, mixed virtual target" for contrastive objectives; the
  construct the author's either-one variant would parallel for captions.
- Method weaknesses: [author-acknowledged] smaller benefit at large scale; some mixing operators underperform in transfer.

**B2. Un-Mix**
- Citation: Shen, Liu, Xu, Xing, et al. 2022. "Un-Mix: Rethinking Image Mixtures for Unsupervised Visual
  Representation Learning." AAAI 2022; arXiv 2003.05438. Tier: peer-reviewed. (Author order not verified.)
- Read scope: full text via ar5iv, summary level.
- Change: mixed image I = lambda*I_1 + (1-lambda)*I_2, lambda ~ Beta(1,1); soft "distance" to each source = lambda or 1-lambda;
  loss adds the mixture terms for both orderings to the base loss. Either-one.
- Effect: CIFAR-10 SimCLR 91.80 -> 92.35; CIFAR-100 BYOL 91.73 -> 94.20; ImageNet MoCo v2 200 ep 67.5 -> 68.6;
  VOC detection MoCo v2 AP50 82.4 -> 83.0 (as transcribed). Framing: either-one. Space: pixels, lambda ~ U(0,1)
  at gamma = 1.0. Masked reconstruction: no.
- Multiplicity: no. Relevance: [reader-inferred] another soft-pair rule; appears as a weak baseline in A3
  (COCO fine-tuning t->i 33.4 vs FT 36.9).
- Method weaknesses: [author-acknowledged] one extra forward pass (about one third extra cost).

**B3. MixCo**
- Citation: Kim, Lee, Shin. 2020. "MixCo: Mix-up Contrastive Learning for Visual Representation." arXiv 2010.06300.
  Tier: preprint (no venue verified). Author list not extracted (UNVERIFIED).
- Read scope: full text via ar5iv, summary level.
- Change: "semi-positive" pairs from mixing a positive and negative image; target similarity weighted lambda and 1-lambda,
  lambda ~ U(0,1). Either-one. Effect: TinyImageNet ResNet-18 (100 ep) 35.79 -> 42.65; ResNet-34 42.15 -> 45.54;
  ResNet-50 45.82 -> 49.96. Multiplicity: no. Masked reconstruction: no.
- Relevance: [reader-inferred] same either-one target; more helpful for small models in the paper's own framing.
- Method weaknesses: [author-acknowledged] SimCLR results incomplete, limited theory; extra memory and time (x1.29, x1.16).

### (c) Mixing combined with masked reconstruction; unmixing

**C1. MixMAE**
- Citation: Liu, Huang, Zheng, Liu, Li. 2023. "MixMAE: Mixed and Masked Autoencoder for Efficient Pretraining of
  Hierarchical Vision Transformers." CVPR 2023; arXiv 2205.13137. Tier: peer-reviewed (CVF listing seen in search).
- Read scope: full text via ar5iv, Sec 3, Tables 2, 8, 9, 18 at summary level.
- Setting: from-scratch SSL, Swin-B/W14 etc., ImageNet-1K pre-training.
- Change tested: the masked patches of image 1 are replaced by visible patches of image 2 (patch-space mixing with
  a binary mask M, default 75% masking by mixing 4 images, ablation 2 to 5 images and ratios 0.5 to 0.8); the encoder
  sees the mixed image; **dual reconstruction** recovers both originals (loss on each image's own masked patches,
  Eq. 2), after unmixing the hidden states and filling [MASK] tokens. No [MASK] token in the encoder.
- Effect: 300 ep Swin-B/W14 ImageNet top-1 84.8 vs BEiT 83.2 (Table 2); 600 ep 85.1; dual reconstruction on vs off
  +0.6 ImageNet-1K top-1 and +2.6 ADE20K mIoU (Table 9); 85 vs 116 GPU-hours against SimMIM (Table 18).
- Multiplicity: no. Framing: unmixing (recover each source). Mix space: patches, ratio about 0.5 for two images.
  Combined with masked reconstruction: it is a masked reconstruction (mixing replaces the mask).
- Relevance: [reader-inferred] "recover each source from a blend" with a reconstruction head, but image-only; the mix
  is a masking device, not a source of ambiguity to model.
- Method weaknesses: [reader-inferred] the unmixing is easy because sources are spatially disjoint patches; it would
  not carry over to 50% opacity pixel blends. Hierarchical ViTs only.

**C2. MixedAE**
- Citation: Chen, Liu, et al. (authors beyond first UNVERIFIED). 2023. "Mixed Autoencoder for Self-Supervised
  Visual Representation Learning." CVPR 2023 (CVF PDF read); arXiv id not confirmed (a guess at 2303.17152 resolved to
  the right paper on ar5iv, treated as UNVERIFIED).
  Tier: peer-reviewed.
- Read scope: full text via ar5iv, Secs 3.1, 3.2, 4.1, Tables 3 and 7 at section level.
- Change: several images mixed into one (mixing ratio r = 0.25, matching MAE's 0.75 masking); a "homologous recognition"
  auxiliary task makes each query attend to and discriminate patches of its own source image (TopK + contrastive loss).
- Effect: reaches 83.9% ImageNet top-1 per the fetch; the baseline MAE value and ablation table values were not
  extracted. UNVERIFIED comparison.
- Framing: unmixing (source identity as an explicit auxiliary task). Space: patches, r = 0.25. Combined with masked
  reconstruction: MAE-style with mixed inputs.
- Multiplicity: no. Relevance: [reader-inferred] the source-attribution task is what would stop a model from blending
  two readings into one.
- Method weaknesses: not assessed beyond the above (numbers not extracted).

**C3. Mix-up SSL for contrast-agnostic applications**
- Citation: "Mix-up Self-Supervised Learning for Contrast-agnostic Applications." arXiv 2204.00901 (2022); an ICME 2022
  file name appeared in search results, venue UNVERIFIED. Authors not extracted. Tier: preprint (venue not confirmed).
- Read scope: abstract only (search snippet); the PDF fetch returned binary.
- Change per snippet: cross-domain mix-up; reconstruction of the target image plus prediction of the mixing ("transparency")
  ratio from the mixed sample, so the model separates the contents of different images.
- Effect, multiplicity, weaknesses: not assessed (read scope: abstract_only).
- Framing: unmixing, with a ratio-recovery head. Space: pixels (alpha blend). Masked reconstruction: no, image reconstruction.
- Relevance: [reader-inferred] the nearest published object to "50% opacity blend then reconstruct", but image-only;
  could not be verified here.

### (d) Mixed or ambiguous images as probes of several readings (near-misses)

**D1. Evaluating Vision-Language Models on Bistable Images**
- Citation: arXiv 2405.19423 (May 2024), authors not extracted (UNVERIFIED). Tier: preprint>=2024.
- Read scope: full text via arxiv html, summary level.
- Setting: 12 VLMs, 6 architectures (CLIP variants, Idefics, LLaVA-1.5, mPLUG-Owl, InstructBLIP, BLIP-2); bistable
  stimuli (Rubin vase, duck-rabbit etc.), loss-ranking classification and open-ended generation.
- Effect: strong one-sided preferences, e.g. about 80% preference for "two faces" in the Rubin vase; five of six
  models prefer "two faces"; Idefics family and LLaVA-1.5-13b most balanced.
- Multiplicity measured: preference between two known readings, per image. Not mixing-based.
- Framing: not applicable. Relevance: [reader-inferred] a protocol for measuring which of two known readings a model
  picks and how balanced it is; transferable to blended images whose two readings are known exactly.
- Method weaknesses: [author-acknowledged] static images vs human perceptual switching; annotators were students at one
  institution; larger models untested.

**D2. Vision-Language Asymmetry in Bistable Image Captioning**
- Citation: Agate. 2026. arXiv 2606.08031 (submitted 2026-06-06). Tier: preprint>=2024 (2026).
- Read scope: summary page (Pith), not full text.
- Setting: LLaVA-1.6-7B; sparse-autoencoder probing and causal steering on duck-rabbit and similar images.
- Effect: 72% (50 of 69) of ambiguous stimuli show both interpretation features active in the vision tower;
  steering flips captions only for default-dominant images (33% flip rate), not for force-balanced young/old at any
  tested coefficient, which the author reads as commitment happening in the language layers.
- Multiplicity measured: co-activation of both interpretations' features in the encoder.
- Relevance: [reader-inferred] bears on H-a vs H-b: an encoder can carry both readings while the generating side commits
  to one; no mixing involved.
- Method weaknesses: not assessed beyond the Pith summary (read scope: summary only; one model).

## 3. Closest-to-H-b list (H-b x mixing)

1. **MixGen (A1):** blend at lambda 0.5 and concatenate captions, the author's idea; shows retrieval gain in
   pre-training (COCO RSUM +6.2 with ALBEF), gains vanish when text is not concatenated (Table 7 variants b, c). Does
   not test several readings, ECCV mAP@R, or any decoder.
2. **PCME++ MSDA (A2):** the only mixing method scored on ECCV mAP@R with an isolated row: 38.9 -> 39.0, null alone; the
   soft target lambda per source caption is a ready-made either-one rule. Image-only mixing, no reconstruction.
3. **MixMAE / MixedAE (C1, C2) with Mix-up SSL (C3):** unmixing objectives that recover each source from a blend
   (patch-level for C1, C2; pixel alpha plus ratio head for C3). Image-only, no text, no ambiguity evaluation.
4. **TiMix (A5) and MixReorg (A7):** per-source soft text targets and source-restoration with text, the nearest
   to either-one and unmixing in a vision-language setting; evaluated on retrieval and segmentation, not one-to-many.

## 4. Search limitations

- **Absence claims (bounded):** no paper found in Q1 to Q14 that (i) blends two images and reconstructs, or generates,
  each source caption from the blend in a vision-language model, (ii) combines mixing with MLM or MAE-style cross-modal
  reconstruction, or (iii) evaluates mixing on one-to-many retrieval beyond PCME++'s ECCV mAP@R row. Not found in Q1 to Q14
  is not proof of absence; Semantic Scholar citation chains from MixGen, and Google Scholar, were not run.
- **Unverified:** several author lists (PCME++, LeMDA, TiMix,
  MixCo, MixedAE, MixReorg, bistable images, Mix-up SSL), the MixedAE arXiv id, the venue of MixCo, MixReorg and C3.
  PCME++ first author recorded as "Chun" only.
- **Access:** the CVF open access MixGen page and PDF returned 403 (venue confirmed through three other sources);
  ScienceDirect (M3ixup) 403; several PDFs (Geodesic, LeMDA, MixMAE, MixCo-era papers) returned binary through the
  fetcher; ar5iv/html pages were used instead. No local PDF text extraction was available.
- **Not searched in depth:** 2025-2026 follow-ups of MixGen; audio-visual mix-and-separate work (Sound of Pixels style);
  the generative concept-mixing line (arXiv 2412.15358, excluded as unreadable); Cut-Mix-Unmix variants.
- **Fetcher caveat:** numbers for entries other than MixGen, PCME++ and Geodesic are summarizer transcriptions. A first
  fetcher pass of MixGen was wrong on the baseline rows (it used the gray 4M ALBEF row as baseline); the raw-source
  re-read above supersedes it, which is why the other entries need the same check before citing.
