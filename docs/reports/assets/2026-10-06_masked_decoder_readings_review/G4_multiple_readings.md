> Supporting file for docs/reports/auto/v1/2026-10-06_masked_decoder_readings_review.md. Phase 2 bibliography as written; V2 lists the corrections, and the report uses corrected values.

# G4: getting several readings out of a model (generation time or representation)

Last searched 2026-10-06. Note on read scope: WebFetch returns a small-model summary of the page, not raw text. "Read scope" below says what was fetched; numbers are as reported by that summary of the paper unless marked, and each carries the locator the summary gave (many have none; marked "locator not returned"). Anything I could not read is marked.

## 1. Search strategy

- Tools: WebSearch (standard and extended), WebFetch on arxiv.org/html, ar5iv, arxiv.org/pdf (PDF fetches of 1612.00197, 2311.01052, 1903.12020 failed to parse; I re-fetched via ar5iv or html where possible).
- Queries:
  - Q1 diverse image captioning CVAE latent variable Div-2 oracle
  - Q2 discrete diffusion image captioning DDCap Bit Diffusion masked diffusion
  - Q3 resilient / annealed multiple choice learning winner-take-all collapse
  - Q4 multiple embeddings cross-modal retrieval MCL winner-take-all one-to-many
  - Q5 Resilient MCL (Letzelter, NeurIPS 2023)
  - Q6 Rupprecht multiple hypotheses ICCV 2017
  - Q7 stochastic MCL Lee 2016
  - Q8 Guzman-Rivera MCL NIPS 2012
  - Q9 Self-CIDEr Wang and Chan
  - Q10 Diverse Beam Search
  - Q11 masked diffusion VLM (LLaDA-V, MMaDA, Dimple)
  - Q12 diffusion captioning sample diversity 2023-2025
  - Q13 CapPa parallel prediction
  - Q14 multi-hypothesis / WTA embeddings for image-text retrieval 2024-2025
  - Q15 text-conditioned image embedding, multiple interpretations, generative loss selects slot
  - Q16 multiple embeddings per image with captioning decoder
- Date range: 2012 to 2026-10; effort on 2023-2026.
- Inclusion: method or metric gives several outputs or readings per input, or evaluates coverage of a reference set, or explains sampling from a masked predictor, or a collapse problem and fix for winner-take-all training. Exclusion: single-output captioners, pure LLM diffusion without vision, document-retrieval multi-vector (ColBERT-style) work, which is about efficiency and not readings.
- Counts: about 45 identified from result lists, about 28 screened by title, 19 fetched, 17 included.
- Coverage skew: captioning diversity is heavily 2017-2020; masked diffusion and MCL are 2022-2025. For (d) the search found little (see Section 4).
- Already covered in B1-B4 and used only for facts the old entry lacks: Cap/CapPa (B4), PVSE, DivE (B2); MaxMatch is new here and builds on DivE.

## 2. Annotated bibliography

### (a) Diverse image captioning

**A1. Vijayakumar et al., 2016/2018, "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models" (arXiv:1610.02424; later AAAI 2018 version "Diverse Beam Search for Improved Description of Complex Scenes").**
- Tier: preprint (the AAAI 2018 version is peer-reviewed).
- Read scope: arxiv html summary (method, headline numbers, limits); per-table locators not returned.
- Setting: decoding-time change on pretrained captioners, COCO and PASCAL-50S; also translation and visual question generation.
- Change: beam budget split into G groups, dissimilarity penalty between groups. Baseline: standard beam search and prior diverse M-best methods.
- Effect: oracle SPICE@20 up 7 to 9% over beam search; nearly 300% more distinct 4-grams; human preference 83% on complex images vs 50% on simple (as reported; locator not returned).
- Multiplicity measured: oracle (best-of-20 vs references) and distinct n-gram counts.
- How readings are represented: samples (beams), no latent. Assignment to a training target: none, since decoding only. Coverage or diversity: as above.
- Relevance [reader-inferred]: sets the decoding-only baseline: diversity from a single-peaked model by search, no model of readings.
- Weaknesses: penalty weight lambda tuned per task [author-acknowledged, summary of limitations]; oracle scores reward any diverse list, including diverse errors [reader-inferred].

**A2. Dai et al., 2017, "Towards Diverse and Natural Image Descriptions via a Conditional GAN" (ICCV 2017; arXiv:1703.06029).**
- Tier: peer-reviewed (ICCV 2017, venue from memory; not shown in fetched page, UNVERIFIED).
- Read scope: arxiv html summary.
- Setting: train from scratch on COCO / Flickr30k; generator takes noise z plus image, evaluator scores relevance.
- Change vs baseline: G-GAN vs G-MLE (same generator trained by maximum likelihood).
- Effect: self-retrieval R@1 with 5,000 COCO images 14.30% (G-GAN) vs 5.06% (G-MLE) (Table 2); user study 61% preference for G-GAN (Fig 5).
- Multiplicity measured: not directly; no Div-n reported (summary says no type-token metric); distinctiveness measured by self-retrieval.
- Representation: samples via noise z. Assignment: none, adversarial evaluator. Coverage: none against a reference set.
- Relevance [reader-inferred]: shows that "diverse" can mean "distinctive per image", not "several readings of one image".
- Weaknesses: incorrect details rise (colors, counts) [author-acknowledged]; no set-level diversity metric [reader-inferred].

**A3. Wang, Schwing, Lazebnik, 2017, "Diverse and Accurate Image Description Using a Variational Auto-Encoder with an Additive Gaussian Encoding Space" (NeurIPS 2017; arXiv:1711.07068).**
- Tier: peer-reviewed (NeurIPS 2017, venue from memory, UNVERIFIED on the fetched page).
- Read scope: arxiv html summary (Tables 1 and 3 cited).
- Setting: from scratch, COCO, LSTM decoder.
- Change: AG-CVAE, latent prior mean is the sum of cluster means of detected objects. Baseline: LSTM and vanilla CVAE.
- Effect: oracle (100 samples) BLEU-4 0.557, CIDEr 1.517 (Table 1); uniqueness 76.4% (AG-CVAE) vs 11.8% (CVAE) with 20 samples (Table 3); novelty 79.5%.
- Multiplicity measured: oracle best-of-N, uniqueness, novelty.
- Representation: latent samples. Assignment: object labels from the image set the prior component; a reading is not learned per target.
- Relevance [reader-inferred]: reading assignment from an external attribute (objects), the opposite of H-b where the visible context picks it.
- Weaknesses: detected objects often missing from captions; trouble mentioning more than 2 to 3 objects [author-acknowledged]. Different papers report different AG-CVAE uniqueness (47.4% in the COS-CVAE table below), so numbers across papers are not comparable [reader-inferred].

**A4. Mahajan and Roth, 2020, "Diverse Image Captioning with Context-Object Split Latent Spaces" (NeurIPS 2020; arXiv:2011.00966).**
- Tier: peer-reviewed (NeurIPS 2020).
- Read scope: arxiv html summary.
- Setting: from scratch, COCO, Faster R-CNN object features; pseudo-supervision from captions of similar images.
- Change: latent z split into context and object parts. Baselines: Seq-CVAE, AG-CVAE.
- Effect (COCO, best-of-100 oracle where applicable; locator not returned): Div-1 0.39 vs Seq-CVAE 0.33 vs AG-CVAE 0.23; Div-2 0.57 vs 0.48 vs 0.32; m-BLEU-4 0.53 vs 0.64 vs 0.70 (lower is more diverse); oracle CIDEr 1.893 vs 1.695 vs 1.517; uniqueness 96.3% vs 84.2% vs 47.4%.
- Multiplicity measured: Div-1/2, m-BLEU, oracle CIDEr, uniqueness, novel captions.
- Representation: latent samples (two-part). Assignment: unpaired context pseudo-targets from retrieved captions of other images.
- Relevance [reader-inferred]: the strongest numbers for the CVAE route, and a reference point for what "oracle over N samples" buys.
- Weaknesses: needs similar contextual descriptions in training data; gains shrink with consensus re-ranking [author-acknowledged]; oracle metrics grow with N regardless of whether samples are distinct readings [reader-inferred].

**A5. Wang and Chan, 2019, "Describing like Humans: On Diversity in Image Captioning" (CVPR 2019; arXiv:1903.12020).**
- Tier: peer-reviewed (CVPR 2019).
- Read scope: ar5iv summary.
- Change: Self-CIDEr, a set-level diversity metric: kernel matrix of pairwise CIDEr between captions, eigenvalues, div = -log_m(r), r = sqrt(l1) / sum sqrt(li).
- Effect: human annotations show both high accuracy and high diversity; CIDEr-reward-trained models have low diversity; mixing cross-entropy and CIDEr reward (lambda=5) narrows the gap. Human validation Spearman 0.617 with diversity judgments.
- Multiplicity measured: this is the measure (semantic topics per caption set); no per-model numbers retained from the summary.
- Relevance [reader-inferred]: a usable set-level scorer for "how many readings does this caption set contain", including for 5 human captions of an ArtELingo painting.
- Weaknesses: depends on CIDEr quality [author-acknowledged]; CIDEr is n-gram based, so two paraphrases look different and two readings sharing words look same [reader-inferred].

**A6. Wang et al., 2023, "DiffCap: Exploring Continuous Diffusion on Image Captioning" (arXiv:2305.12144), preprint.**
- Tier: preprint.
- Read scope: arxiv html summary.
- Setting: frozen CLIP ViT-B/32 image features, BERT-based continuous diffusion over token embeddings, COCO.
- Change vs strongest baseline: non-autoregressive diffusion vs BLIP (autoregressive).
- Effect: inter-distinct uni-/bi-gram 46.61% / 77.33% vs BLIP 6.19% / 14.75%; Self-BLEU uni-gram 71.74 vs BLIP 100 (lower is more diverse); BLEU-4 31.6 vs BLIP 40.4 (accuracy is lower).
- Representation: samples from different noise draws. Assignment: none.
- Relevance [reader-inferred]: a non-autoregressive captioner is the closest sampling analogue to our parallel MLM; diversity is bought with accuracy.
- Weaknesses: long sampling time; low noise does not mean close meaning for a token [author-acknowledged]; BLIP baseline is deterministic beam search, so the comparison to a sampling method is unfair on diversity [reader-inferred].

### (b) Masked and discrete-diffusion generators, iterative unmasking

**B1. Zhu et al., 2022, "Exploring Discrete Diffusion Models for Image Captioning" (DDCap; arXiv:2211.11694), preprint.**
- Tier: preprint.
- Read scope: ar5iv html summary.
- Setting: COCO Karpathy, with and without 4M-pair pretraining.
- Change vs baseline: discrete diffusion with length prediction, best-first inference (top-K recovered tokens frozen at each step), T=20 steps. Baseline: autoregressive Transformer of the same setup.
- Effect: CIDEr 117.8 vs 112.8 (no pretraining); 125.1 with 4M pretraining; caption infilling of masked adjectives CIDEr 230.3 vs AR 203.5.
- Multiplicity measured: none; no Div-n or Self-CIDEr reported. Authors note repeated words.
- Representation: samples (different starting noise). Sampling: start from all [MASK], iterative bidirectional unmasking, best tokens kept.
- Relevance [reader-inferred]: the closest published analogue to using a masked caption predictor as a captioner conditioned on an image; the infilling result is the only test of a decoder given a partly visible caption.
- Weaknesses: repetition [author-acknowledged]; no test of sample diversity, so it does not show several readings [reader-inferred].

**B2. Feng, Wei, Owens, 2025, "Masked Diffusion Captioning for Visual Feature Learning" (arXiv:2510.26799), preprint>=2024.**
- Tier: preprint>=2024.
- Read scope: arxiv html summary (Tables 1 to 3 cited).
- Setting: ViT-L/14, Recap-DataComp-10M, image-conditioned masked diffusion language model with randomly sampled mask ratio; used as a representation learner.
- Change vs strongest baseline: masked diffusion captioning vs autoregressive captioning and CLIP variants.
- Effect: linear-probe average 83.1% vs autoregressive captioning 83.3% (Table 1); compositionality (ARO, SugarCrepe) 84.6 to 98.8 with a heuristic likelihood variant vs CLIP 53.6 to 72.8 (Table 2); captioning CIDEr 24.1 to 64.7 by specified length (Table 3).
- Sampling: fully masked start, greedy unmasking of the single most confident masked position per step.
- Multiplicity measured: none ("sample diversity not explicitly evaluated" per the summary).
- Relevance [reader-inferred]: the same setting as our 80% text-masking MLM (varied mask ratio, image-conditioned), and it uses the model's likelihood for scoring, which is the decoder-scoring idea in H-b.
- Weaknesses: academic scale (about 10M pairs) [author-acknowledged]; greedy confidence unmasking is nearly deterministic, so it does not sample readings [reader-inferred].

**B3. Chang et al., 2022, "MaskGIT: Masked Generative Image Transformer" (CVPR 2022; arXiv:2202.04200).**
- Tier: peer-reviewed (CVPR 2022, from memory, UNVERIFIED on the fetched page; page lists arXiv Feb 2022 only).
- Read scope: arxiv html summary.
- Setting: VQ tokens, ImageNet 256x256, class-conditional (image generation, not text).
- Change: iterative decoding: predict all masked tokens in parallel, keep the most confident, re-mask the rest, with a cosine schedule over 8 steps (vs 256 for autoregressive). Baseline: VQGAN.
- Effect: FID 6.18, IS 182.1 vs VQGAN FID 15.78, IS 78.3; Classification Accuracy Score 63.14 top-1. (Locators not returned.)
- Multiplicity measured: sample variety via FID and CAS; no per-input diversity.
- Relevance [reader-inferred]: the reference recipe for sampling from a masked predictor (confidence unmasking plus a schedule); randomness enters through token sampling and temperature, which text MLM scoring here does not use.
- Weaknesses: semantic shifts in long outpainting, artifacts on faces [author-acknowledged]; image domain, class condition only [reader-inferred].

**B4. Tschannen et al., 2023, "Image Captioners Are Scalable Vision Learners Too" (CapPa; NeurIPS 2023; arXiv:2306.07915).** Used only for facts the B4 entry may lack.
- Tier: peer-reviewed (NeurIPS 2023).
- Read scope: search-result abstract/snippets only; not fetched in full.
- Facts: decoder input is replaced by all [MASK] tokens and causal mask dropped; the prediction type (autoregressive vs parallel) is sampled per training example; parallel prediction on 75% of examples by default; no change of objective or decoder, no extra cost.
- Multiplicity measured: not assessed (read scope: snippets).
- Relevance [reader-inferred]: parallel prediction from all [MASK] is exactly our 80%-text-masking regime, so the MLM there equals CapPa's Pa mode.
- Weaknesses: not assessed (read scope: abstract and snippets only).

### (c) Multiple-choice learning and winner-take-all

**C1. Guzman-Rivera, Batra, Kohli, 2012, "Multiple Choice Learning: Learning to Produce Multiple Structured Outputs" (NeurIPS 2012).**
- Tier: peer-reviewed.
- Read scope: abstract only (search result).
- Change: train M predictors to minimize the loss of the best one (oracle loss), max-margin formulation. Tasks: image co-segmentation and protein side-chain prediction; outperforms M-best MAP.
- Effect: not assessed with numbers (read scope: abstract_only).
- Relevance [reader-inferred]: origin of "the best of M readings gets the gradient", the assignment rule if H-b slots compete for each target.
- Weaknesses: not assessed (read scope: abstract_only).

**C2. Lee et al., 2016, "Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles" (NeurIPS 2016; arXiv:1606.07839).**
- Tier: peer-reviewed.
- Read scope: ar5iv summary.
- Setting: deep ensembles trained from scratch/fine-tuned: CIFAR-10 CNNs, FCN segmentation on PASCAL VOC, CNN+RNN captioning on MSCOCO.
- Change: winner-take-gradient: each example backpropagates only to the member with lowest loss. Baseline: independent ensemble, plus earlier MCL.
- Effect: captioning oracle CIDEr-D (M=5) 0.922 vs 0.831 independent ensemble; segmentation about 7% relative oracle improvement; members specialize (e.g. by class clusters) (locators not returned).
- Multiplicity measured: oracle error over members; interpretability of specialization.
- Representation: M ensemble members. Assignment: winner by lowest loss on that target.
- Relevance [reader-inferred]: the only entry in this theme that reports WTA on captioning, a direct precedent for member-per-reading with a generative loss.
- Weaknesses: not given in the summary; oracle metric rewards the best of M and does not test whether the others are valid [reader-inferred].

**C3. Rupprecht et al., 2017, "Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses" (ICCV 2017; arXiv:1612.00197).**
- Tier: peer-reviewed.
- Read scope: ar5iv summary.
- Change: multiple-hypothesis prediction with a meta loss and relaxed winner-take-all: winner weight 1-eps, others eps/(M-1), eps=0.05; 1% random drop of predictions. Fixes starvation (hypotheses never winning, which get zero gradient when all targets fall in one Voronoi cell).
- Effect: LSP pose 59.7% (single) -> 61.2% (5 hypotheses) -> 62.8% (10); VOC2007 83.8 -> 85.1 (9 hyp.); COCO 65.2 -> 67.8 (5 hyp.); VOC2012 segmentation mean IoU 70.3 (4 hyp.) vs 69.1 (MCL) at a quarter of the parameters. Performance falls for M >= 9 in classification [author-acknowledged].
- Representation: M hypothesis heads. Assignment: nearest hypothesis to the target gets most of the weight.
- Relevance [reader-inferred]: the minimal collapse fix (soft winner weights) that H-b slots would need.
- Weaknesses: more hypotheses than discrete outcomes adds noise [author-acknowledged]; gains are for top-1 accuracy of a single chosen output, not coverage of references [reader-inferred].

**C4. Letzelter et al., 2023, "Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis" (NeurIPS 2023; arXiv:2311.01052).**
- Tier: peer-reviewed.
- Read scope: arxiv html summary (final 2,050 characters unread).
- Change: add scoring heads that predict each hypothesis's probability of being the winner, trained with a binary loss; Voronoi interpretation. Fixes overconfidence of rare-mode hypotheses. Baselines: sMCL, independent ensembles.
- Effect: best Earth Mover's Distance (EMD) on multimodal synthetic and sound-source-localization splits (ANSYN, RESYN, D1 to D3); oracle error slightly worse than plain WTA (numbers not retained).
- Multiplicity measured: oracle error and EMD to the true conditional distribution.
- Relevance [reader-inferred]: gives a way to weight readings (the scores), which H-b needs if "which readings are plausible" is to be read off.
- Weaknesses: number of hypotheses tuned per dataset; oracle-EMD trade-off; Voronoi reading assumes train and test distributions match [author-acknowledged].

**C5. Perera et al., 2024, "Annealed Multiple Choice Learning: Overcoming limitations of Winner-takes-all with annealing" (NeurIPS 2024; arXiv:2407.15580).**
- Tier: peer-reviewed, 2024.
- Read scope: arxiv html v3 summary (first 100,000 of 179,175 characters).
- Change: hard min replaced by temperature-dependent softmin with cooling; entropy-constrained view with phase transitions where hypotheses split. Baselines: vanilla MCL, Relaxed-WTA.
- Effect: better distortion on 10 UCI regressions; WSJ0-mix speech separation 2-speaker 16.85 +- 0.13 SI-SDR vs PIT 16.88, 3-speaker 10.00 +- 0.21 vs 10.01 (matches, does not exceed).
- Representation: hypothesis heads. Assignment: soft, annealed to hard.
- Relevance [reader-inferred]: a collapse remedy that needs no extra labels; but results are on regression and audio.
- Weaknesses: the temperature schedule is hard to tune; slow cooling lengthens training [author-acknowledged].

### (d) Set-valued and multi-vector cross-modal representations (beyond PVSE, DivE, Llip)

**D1. Alomari et al., 2025, "Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval" (ACL 2025; arXiv:2506.21538).**
- Tier: peer-reviewed (ACL 2025; the ACL Anthology link appeared in results, "2025.acl-long.1533").
- Read scope: arxiv html v1 summary.
- Setting: from scratch with ResNet152 + Bi-GRU, Faster R-CNN + BERT; COCO 1K/5K, Flickr30k.
- Change: Maximal Pair Assignment Similarity between embedding sets via Hungarian one-to-one matching, plus global discriminative loss and intra-set divergence loss. Baseline: SetDiv (DivE-style).
- Effect (rSum): Flickr30k ResNet+Bi-GRU 442.6 -> 469.9; COCO 5K 377.7 -> 398.95; Flickr30k Faster R-CNN+BERT 524.2 -> 527.1.
- Multiplicity measured: Recall@K and rSum only; collapse shown qualitatively ("Smooth-Chamfer leads to set collapse"); no coverage of several valid matches.
- Representation: fixed-size set of embeddings. Assignment: Hungarian one-to-one matching of set elements, not a generative loss.
- Relevance [reader-inferred]: shows the collapse failure of set embeddings and an assignment fix; the evaluation is standard recall, so it would not detect extra readings.
- Weaknesses: constrained COCO/Flickr30k "concrete objects and actions"; two-modality only; fails on visually homogeneous content [author-acknowledged].

## 3. Closest-to-H-b list

1. **Lee et al. 2016 (C2)**: WTA over M generative captioner members; oracle CIDEr-D 0.922 vs 0.831 for independent ones. Establishes that competitive assignment yields several captions per image; does not test the readings as readings, and members are separate models.
2. **Feng et al. 2025 (B2)** with **DDCap (B1)**: image-conditioned masked captioners with varied mask ratio and confidence unmasking, scored by likelihood. They establish that such a predictor works as a captioner and a scorer; neither measures sample diversity.
3. **Mahajan and Roth 2020 (A4)** with Self-CIDEr (A5): the best measured coverage numbers for captioning diversity (oracle CIDEr 1.893, Div-2 0.57) and a set-level scorer. They do not use masking or retrieval.
4. Rupprecht 2017 and Resilient MCL (C3, C4): the collapse problem and fixes (soft winners, scores) that any slot-per-reading design would inherit; neither involves images and text together.

## 4. Search limitations

- Sub-theme (d): I did not find (Q4, Q14 to Q16) a retrieval method where a reconstruction or generative loss chooses which embedding slot a target belongs to, nor a text-conditioned image embedding scored on coverage of several valid matches, beyond Llip (already covered). This is an absence bounded by those queries and by search-engine results.
- Not read in full text: C1 (abstract only), B4 (snippets), and all PDF fetches that failed to parse (1612.00197 and 2311.01052 and 1903.12020 PDFs; HTML versions used instead). The ACL 2025 paper "Modeling Uncertainty in Composed Image Retrieval" appeared in Q15 but its PDF did not parse; not included.
- Seen but excluded: AdaptiveEmbed (arXiv:2608.25412, Aug 2026; multi-vector capacity per sample; the id and date come only from the arxiv abstract summary and the content is about retrieval efficiency, not readings; not read in full text, UNVERIFIED); CLARA (arXiv:2606.18992, composed image retrieval ambiguity with conformal coverage and diffusion prototypes; June 2026 preprint; author list and affiliation looked odd in the summary and I could not verify them; not included); Dimple and LLaDA-V (parallel decoding of multimodal diffusion LMs; reports benchmark accuracy, nothing on sample diversity per the summary).
- The "Winner-takes-all learners are geometry-aware conditional density estimators" (Letzelter et al., arXiv:2406.04706, ICML 2024 per my memory, UNVERIFIED) was read in summary (first 100,000 of 202,445 characters): it argues WTA models can estimate conditional densities and that mixture density networks collapse to single modes; not given its own entry because its experiments are synthetic, UCI and audio only.
- Venues marked "from memory" (A2, A3, B3) were not confirmed on the fetched pages.
- Most diversity numbers across captioning papers (Div-n, uniqueness, oracle CIDEr) use different sample counts and re-ranking settings; do not compare across entries (A3 vs A4 AG-CVAE uniqueness differs: 76.4% vs 47.4%).
