> Supporting file for docs/reports/auto/v1/2026-10-06_masked_decoder_readings_review.md. Phase 2 novelty sweep as written; V3 lists the corrections, and the report uses corrected values.

# N: Novelty sweep for H-b (masked cross-modal decoder as a model of several readings)

Bibliography agent, ARS deep-research Phase 2 (lit-review mode), theme N (novelty, crossing G1-G5 with masked
cross-modal modelling and polysemy). Written 2026-10-06 20:04. Extends
`docs/reports/assets/2026-10-03_lever_review/B4_novelty_polysemy.md`; B4 entries are not repeated (ProLIP, PCME++ MSDA,
MAP, CapPa, MACCO, VWSD systems, Cekinmez et al. 2026 are cited below only for a fact the H-b question needs).
No synthesis across themes and no project recommendations beyond the per-source relevance lines.

H-b split into sub-claims (from the task):
- **S1** a masked or generative cross-modal model's predictive distribution represents several readings (bimodal or
  high-entropy predictions on ambiguous inputs);
- **S2** that spread is measured against human readings in image-text settings;
- **S3** masked or generative models are compared with contrastive dual encoders on ambiguity or polysemy;
- **S4** the decoder or its distribution is used at retrieval time to cover several readings;
- **S5** masking teaches under-specification or ambiguity, beyond ProLIP's masked inclusion;
- **S6** mixing or synthetic ambiguous images combined with masked reconstruction for polysemy.

## 1. Search strategy

**Sources searched.** Web search (standard and extended) over arXiv, OpenReview, CVF open access, ACL Anthology,
mlanthology, Springer; full text fetched as raw arXiv HTML with curl into
`scratchpad/hb_review/dl_N/` and read with grep/sed after a local HTML-to-text pass (helper script at
`scratchpad/h2t.py`, outside the download folder). Numbers below come from those raw files unless marked otherwise.

**Date range.** 2019 to 2026, weighted to 2024-2026 (11 of the 16 included sources are 2024-2026). Last searched
2026-10-06.

**Queries** (E = extended mode, S = standard):
- Q1 `masked vision-language model predictive distribution ambiguity multiple interpretations image caption entropy 2025` (E)
- Q2 `captioning model distribution vs human caption variation annotator disagreement image description 2024 2025 arXiv` (E)
- Q3 `generative vs discriminative vision-language models ambiguity polysemy evaluation contrastive dual encoder vs generative scoring 2024 2025` (E)
- Q4 `ambiguous images bistable vision-language models multiple interpretations CLIP 2025 arXiv` (E)
- Q5 `pluralistic alignment vision-language models perspective-aware multimodal subjective annotation 2025` (E)
- Q6 `ambiguity-aware evaluation vision-language models benchmark ambiguous visual questions multiple valid answers 2025 2026` (E)
- Q7 `image captioning models capture human variation in descriptions distribution of references calibration "human variation" captioning 2024 2025` (E)
- Q8 `AmbiBench ambiguous images vision-language models human interpretations arXiv 2026 Ma` (S)
- Q9 `ambiguous text-to-image retrieval query multiple interpretations diverse results set retrieval 2025 arXiv` (E)
- Q10 `human label variation vision-language models distribution of human interpretations images Plank 2025` (E)
- Q11 `unified multimodal discrete diffusion masked image text joint inpainting diversity UniDisc 2025` (S)
- Q12 `vision-language model caption diversity human-like mode collapse sampling diversity compared to human captions 2025` (E)
- Q13 `diffusion classifier vs CLIP ambiguous images multiple interpretations generative classifier human-like 2024 2025` (E)
- Q14 `masked language modeling vs contrastive vision-language ambiguous captions underspecified retrieval comparison "underspecified" image-text 2024 2025` (E)
- Q15 `mixup blended images two captions vision-language ambiguous synthetic images multiple readings reconstruction 2024 2025` (E)
- Q16 `visual anagrams multi-view illusions evaluate CLIP recognizes both interpretations benchmark 2024 2025` (E)
- Q17 `"masked" cross-modal reconstruction polysemy OR "multiple interpretations" OR "one-to-many" vision-language decoder uncertainty retrieval 2025 2026 arXiv` (E)
- Q18 `emotion distribution prediction artwork vision-language model annotator distribution KL ArtEmis captioning model entropy 2025` (S)
- Q19 `"Visual Semantic Entropy" vision-language models visually ambiguous inputs 2026 arXiv` (S)
- Q20 `uncertainty of image captioning models correlates with human caption disagreement semantic entropy ambiguous images 2024 2025 2026` (E)
- Q21 `image retrieval represent each image by multiple generated captions sampled from captioner multi-vector set retrieval one-to-many 2025` (E)
- Q22 `underspecification vision-language "underspecified" captions images many depictions Hutchinson scene description depiction ambiguity text-to-image` (S)
- Q23 `visual word sense disambiguation generative model likelihood scoring vs CLIP captioning model BLIP image-to-text likelihood polysemous 2024 2025` (E)
- Q24 `"ambiguous" captions image-text matching generative model represents multiple meanings "contrastive" fails one embedding per caption 2025 2026 study` (E)
- Q25 `CLIP multi-object representation bias dominant object image encoder text encoder CVPR 2025 analysis` (S)
- Q26 `masked image modeling vs contrastive CLIP representations ambiguous images bistable illusions comparison self-supervised 2024 2025` (S)
- Q27 `MixGen image-text mixing masked language modeling vision-language pretraining interpolated images concatenated captions ablation ITM MLM` (S)
- Q28 `hybrid images or visual anagrams used as training data or evaluation for image-text retrieval two valid captions both readings CLIP 2025 2026` (E)
- Q29 `image context disambiguates masked word prediction multimodal transformer polysemous word entropy reduction visual context MLM probing` (E)
- Q30 `perspectivist multimodal dataset disaggregated annotator labels image-text disagreement vision-language model soft labels 2025` (E)
- Q31 `vision-language models distribution of interpretations images humans pluralistic "multiple valid" interpretations image meaning subjective captions metaphor 2025 2026 benchmark` (E)
- Q32 `verb sense clusters visual activity recognition ambiguity evaluation vision-language clustering each image four sense clusters perspective` (S)
- Q33 `masked diffusion vision-language model LaViDa OR LLaDA-V OR MMaDA caption diversity vs autoregressive multiple plausible completions 2025` (E)
- Q34 `incomplete or partial caption query uncertainty grows with masked tokens probabilistic embedding image-text retrieval 2024 2025 "partial" text query vision-language` (E)
- Q35 `masked autoencoder deterministic decoder predicts mean blurry reconstruction aleatoric uncertainty multiple plausible completions stochastic MAE 2024 2025` (E)
- Q36 `image ambiguity score from human caption variability predicting caption diversity captioning model entropy correlates with inter-annotator caption variation 2024 2025 2026` (E)
- Q37 `"one-to-many" OR "multiple valid captions" masked token prediction cross-modal decoder distribution evaluation against references "coverage" vision-language 2026 arXiv` (E)
- Q38 `"AmbiBench" "What Do VLMs See" ambiguous images camouflaged bistable hybrid multi-view benchmark arXiv` (E)
- Q39 `text-to-image retrieval ambiguous queries benchmark CLIP versus generative MLLM reranking multiple interpretations recall diversity 2025 2026` (E)
- Q40 `ArtELingo-28 OR ArtELingo emotion captioning model predicted emotion distribution vs annotators diversity of explanations 2024 2025` (S)
- Q41-Q44: identity and venue checks (Pezzelle ACL 2023; FOCUS ACL 2025; arXiv comment fields of 2606.31407 and
  2503.20853; IC3 ACL Anthology page).

**Inclusion criteria.** (a) A generative, masked or fused vision-language model whose output distribution is
examined on ambiguous, polysemous or multi-reading inputs; or (b) a comparison of such a model with a contrastive dual
encoder on ambiguity; or (c) model spread compared with human readings in an image-text task; or (d) a generator's
samples or likelihood used at retrieval time to cover several readings; or (e) masking, under-specification or
mixing as a source of ambiguity in image-text training. Numbers had to be readable in the paper's own text or tables.

**Exclusion criteria.** Classification-only uncertainty work without image-text modelling; text-only perspectivism;
themes covered in depth by G1-G5 unless the source crosses them with masking or polysemy; sources whose full text I
could not open, except as short abstract-level entries marked so.

**Counts.** 44 queries; about 390 result links identified (many duplicates); about 95 unique candidates screened by
title and snippet; 13 downloaded and read in targeted full text (raw HTML); 7 read at abstract level. Included: 10
core entries and 6 short secondary entries (16). Excluded after reading: 7 (end of Section 2).

**Coverage skew.** arXiv-HTML heavy. OpenReview (ICLR 2026 submissions) and ACL Anthology PDFs could not be opened,
so 2026 workshop and ACL work is under-read. Much of the 2025-2026 ambiguity literature studies autoregressive MLLMs
(LLaVA family) on illusion stimuli; masked cross-modal decoders are almost absent from it, which is itself a finding
for S1-S3 but also means the evidence base is narrow.

## 2. Annotated bibliography

### 2.1 Generative decoders on ambiguous images: is more than one reading kept? (S1, S3)

**[N1] Panagopoulou, Melkin, Callison-Burch 2024, bistable images**
- Citation: Panagopoulou, A., Melkin, C., Callison-Burch, C. (2024). Evaluating Vision-Language Models on Bistable
  Images. CMCL 2024 workshop (ACL Anthology 2024.cmcl-1.2). arXiv:2405.19423.
- Tier: peer-reviewed (workshop).
- Read scope: full text (raw arxiv html v1): Sec 3-4.7, Fig 2-6 captions.
- Setting: inference only on pretrained models. 12 models in 6 families: CLIP B/32, B/16, LAION B/32 (contrastive);
  Idefics-9b (base, instruct), LLaVA-1.5 7b/13b, mPLUG-Owl 7b, InstructBLIP FlanT5-xl, BLIP-2 (OPT-2.7b, OPT-6.7b,
  FlanT5-xl). 29 bistable images x 116 manipulations (brightness, tint, rotation) = 3,364 images (Sec 1, 3.1).
- Change tested and baseline: classification between the two interpretation labels (generative models by
  negative-log-likelihood ranking of each label, Sec 3.2); free captioning; prompt and synonym variation. Human
  baseline: 3 annotators' first-seen interpretation (Sec 4.5) and Takashima et al. (2012), n = 70 (Sec 4.6).
- Effect: numbers are in figures only (Fig 2, 3, 5). Text: "a pronounced preference for one interpretation over
  another among the models", except Idefics and LLaVA-1.5-13b, which are "more balanced" (abstract, Sec 4.1); CLIP
  variants show "almost consistently opposite preferences to the LLM based generative models" on some images and are
  the most sensitive to brightness, tint and rotation (Sec 4.1-4.2); "most models only comment on a single
  interpretation" when captioning (Sec 4.7, Fig 6); "limited correspondence" with human first interpretations
  (Sec 4.5); no model reproduced the Takashima human pattern (Sec 4.6, Fig 5). Sec 5: five of six model families
  prefer "two faces" (Rubin vase), "cube seen from above" (Necker) and "duck" (duck-rabbit); only Young-Old Woman
  is "more neutral and mixed"; "CLIP emerged as a model with strong, variable preferences, while LLaVa is more
  neutral".
- Multiplicity/polysemy measured? Yes: probability of the favoured interpretation per image (Fig 2), and whether a
  caption names both readings (Fig 6). No entropy summary in text.
- H-b sub-claims touched: S1 (contradicts for most generative VLMs: one reading dominates; partly supports for
  Idefics and LLaVA-1.5-13b), S3 (direct contrastive-vs-generative comparison on ambiguity: neither keeps both
  readings reliably; they differ in which reading and in sensitivity), S2 (contradicts alignment with humans).
- Relevance to MultiMAE / H-b: the only found head-to-head of CLIP and likelihood-scored generative VLMs on
  multi-reading images; it suggests a decoder's likelihood is not automatically "plural". [reader-inferred]
- Method weaknesses: two-label NLL ranking forces a binary choice, so it cannot show "both readings at once"
  [reader-inferred]; famous illusions may be memorized from web data [reader-inferred; raised by N2, Sec 5
  limitations]; 3 annotators, all students at one US institution [author-acknowledged, Sec 4.5]; human switching over
  time has no analogue in static models [author-acknowledged, Sec 5].

**[N2] Takatsuki et al. 2026, next-token distribution on bistable images**
- Citation: Takatsuki, R., Doi, T., Watahiki, A., Seth, A. K., Yanaka, H. (2026). (How) Do MLLMs Report Bistable
  Images Like Humans? arXiv:2609.13254. "Accepted to EMNLP 2026 (Main Conference)" per the paper's thanks note.
- Tier: peer-reviewed per author statement (EMNLP 2026), `preprint>=2024` on arXiv.
- Read scope: full text (raw arxiv html v1): Sec 3.1-3.3, Sec 4.3 opening, Fig 2-5 captions.
- Setting: inference and mechanistic analysis on pretrained LLaVA-1.5-7B/13B, LLaVA-v1.6 (Vicuna, Mistral),
  Llama3-LLaVA-Next-8B; behavioural checks on Qwen2-VL-7B, SmolVLM, IDEFICS2-8B, InstructBLIP. Stimuli: Harper's
  duck-rabbit plus 50 synthetic duck-rabbit-like Visual Anagrams (rotation-dependent) with matched two-animal controls.
- Change tested and baseline: next-token probabilities after the prefix "I see a" under rotation, red-circle cues and
  linguistic cues ("It starts with 'r'"); beam-search mass on single- vs multiple-animal continuations. Baseline:
  matched two-animal control images.
- Effect: for 37-44 of 50 anagrams (by model) the preferred reading reverses at a rotation angle, fitted with a
  logistic psychometric curve on normalized first-token probabilities (Sec 3.2.1); linguistic cues shift the
  competition but cannot induce an absent animal on non-ambiguous controls (Sec 3.2.2); multiple-animal continuations
  get "∼10% of beam mass near the ambiguity boundary" vs "≥98%" for two-animal controls (Sec 3.3, Fig 5); models
  answer "1 object" with p = 0.88-0.99 on the ambiguous stimuli (Sec 3.3).
- Multiplicity/polysemy measured? Yes: probability split between readings along a stimulus continuum; beam mass on
  enumerating both readings. Human comparison is qualitative, against published human results (rotation:
  Moreno-Sanchez et al. 2016; fixation maps: Hsu and Chen 2025), not new human data.
- H-b sub-claims touched: S1 (supports in an "either-or" form: the decoder's next-token distribution holds two
  competing readings whose balance is set by visual and textual context; contradicts the "both at once" form: reports
  are exclusive), S6 (synthetic ambiguous images with exactly known readings are usable as a probe), S2 (open:
  qualitative only).
- Relevance to MultiMAE / H-b: closest found evidence that a decoder's distribution at one slot can carry alternative
  readings selected by partial context, which is the mechanism H-b expects from visible caption tokens; shown for an
  autoregressive MLLM, not a masked decoder. [reader-inferred]
- Method weaknesses: static object-category ambiguity only [author-acknowledged, Limitations]; famous stimuli may be
  memorized, mitigated by anagrams [author-acknowledged, Sec 1-2]; results "somewhat prompt-sensitive"
  [author-acknowledged, Sec 3.3]; continuation classes assigned by an LLM judge [reader-inferred].

**[N3] Agate 2026, vision-side superposition vs language-side commitment**
- Citation: Agate, A. (2026). Vision-Language Asymmetry in Bistable Image Captioning. arXiv preprint,
  arXiv:2606.08031.
- Tier: preprint, `preprint>=2024`.
- Read scope: full text (raw arxiv html v1): abstract, Sec 1, 3, 4.1-4.4.
- Setting: LLaVA-1.6-Vicuna-7B; TopK sparse autoencoder (k = 32, 65,536 features) trained on CLIP ViT-L layer 22
  (the layer LLaVA reads). 83 bistable stimuli (29 from N1, 53 from AmbiBench, 1 rendered), SDXL pure-aspect controls.
- Change tested and baseline: 40 sampled captions per stimulus under neutral vs forced-choice prompts (3,320
  captions, Qwen3-8B judge); per-aspect SAE feature pools compared with thresholds from opposite-aspect controls;
  steering at layer 22.
- Effect: mean caption dominance |P(A) - P(B)| = 0.558; 38 of 83 stimuli default-dominant (Sec 4.1, Fig 1); both
  aspect feature pools active on 50 of 69 stimuli (72%), including 12/12 duck-rabbit (Sec 4.3, Fig 2); steering flips
  duck-rabbit captions to rabbit in 33.3% at alpha = 16 but flips 0/7 force-balanced young/old stimuli at any alpha
  (Sec 4.4); a Necker cube gives 40/40 aspect-agnostic captions under a neutral prompt and a near-even split under
  forced choice (Sec 1).
- Multiplicity/polysemy measured? Yes: caption dominance per stimulus, superposition rate of patch features.
- H-b sub-claims touched: S1 (both readings present in the vision features that a decoder reads; the decoder's
  sampled output commits, so S1 depends on how the distribution is read out: forced choice at a slot shows a split,
  free sampling hides it), S3 (the contrastive tower holds both readings in patch features; nothing on pooled
  embeddings).
- Relevance to MultiMAE / H-b: MultiMAE's MLM decoder reads CLIP patch tokens, the level at which this paper finds
  superposition; masked-slot prediction resembles its forced-choice probe more than free captioning does.
  [reader-inferred]
- Method weaknesses: one VLM and one SAE run [author-acknowledged, Sec 5]; no random-feature or permuted-aspect
  baselines yet [author-acknowledged, Sec 5]; thin controls for some groups (n = 8) [author-acknowledged, Sec 3 / App A];
  single-author preprint without human data [reader-inferred].

**[N4] Visual Semantic Entropy (VSE)**
- Citation: Huy, T. D., Nguyen, T., Chowdhury, T., Yadav, A., To, M.-S., Liao, Z., Verjans, J. W., Phan, V. M. H.
  (2026). Visual Semantic Entropy: Do Vision Language Models Recognize Visual Ambiguity? arXiv:2606.31407;
  arXiv comments: "Accepted at ECCV2026".
- Tier: peer-reviewed per author statement (ECCV 2026), `preprint>=2024`.
- Read scope: full text (raw arxiv html v1): Sec 3 analysis (Fig 3), Table 2, Sec 5 adversarial results,
  limitations.
- Setting: inference on Qwen2.5-VL, Gemma3, LLaVA-NeXT, Qwen3-VL, InternVL3.5; VQA benchmarks (AOKVQA, MMVet, VILP,
  VLM-are-biased, others).
- Change tested and baseline: semantic entropy over answers sampled under image perturbations (text fixed) vs
  standard Semantic Entropy (SE) and 10 other estimators.
- Effect: AOKVQA AUC (Table 2), SE -> VSE: Qwen2.5-VL 0.702 -> 0.783; Gemma3 0.732 -> 0.778; LLaVA-NeXT 0.622 ->
  0.724; Qwen3-VL 0.746 -> 0.798; InternVL3.5 0.775 -> 0.792. VILP (Qwen2.5-VL): strongest baseline 0.535 -> VSE
  0.650; VLM-are-biased: 0.783 -> 0.826 (Sec 5 text). Fig 3: visually confident samples show low SE "despite the
  visual ambiguity".
- Multiplicity/polysemy measured? Only indirectly: AUC for predicting wrong answers; no human ambiguity labels.
- H-b sub-claims touched: S1 (cautions: sampled-output diversity underestimates input ambiguity when the visual
  embedding is confident, so entropy of samples is a weak readout of readings), S2 (open: no human labels).
- Relevance to MultiMAE / H-b: if H-b is measured by decoder entropy, this predicts under-counting on images the
  encoder is sure about; perturbing the image (here, re-drawing the patch mask) is the remedy they found.
  [reader-inferred]
- Method weaknesses: "visual ambiguity" is proxied by correctness, not human disagreement [reader-inferred, from
  Sec 4 setup]; temperature sensitivity and dependence on the semantic distance model [author-acknowledged,
  Limitations].

**Short secondary entries (2.1)**
- Wang, X., Nalisnick, E. (2025). Are vision language models robust to uncertain inputs? arXiv:2505.11804,
  `preprint>=2024`. Read scope: abstract only. Proposes caption diversity under sampling as a label-free signal of a
  VLM's internal uncertainty on ambiguous or anomalous inputs. S1: supports that sample diversity tracks ambiguity,
  in tension with N4. Weaknesses: not assessed (read scope: abstract_only).
- AmbiBench: Ma et al. (2026), "What Do VLMs See? Benchmarking Vision-Language Models on Ambiguous Images",
  OpenReview R2dCGaqzYW (ICLR 2026 submission; decision UNVERIFIED). Read scope: secondary only (OpenReview blocked;
  HF dataset card; N3's description). 9 ambiguity types including Bistable, Hybrid, Multi-view and Mixed (HF card);
  2,238 images per N3 Sec 2. Headline numbers seen only in search snippets: UNVERIFIED. S1/S6 benchmark candidate.
  Weaknesses: not assessed.

### 2.2 Model spread vs human readings in image-text tasks (S2)

**[N5] Lan, Frassinelli, Plank 2024, human uncertainty in VQA**
- Citation: Lan, J., Frassinelli, D., Plank, B. (2024). Mind the Uncertainty in Human Disagreement: Evaluating
  Discrepancies between Model Predictions and Human Responses in VQA. arXiv preprint, arXiv:2410.02773.
- Tier: preprint, `preprint>=2024` (no venue stated in v1).
- Read scope: full text (raw arxiv html v1): Sec 3-5, Tables 1-2 text, Fig 3 discussion.
- Setting: fine-tunes BEiT3-base and LXMERT on VQA 2.0 (official recipes); evaluates on held-out validation samples
  with 10 annotators each and per-annotator confidence (yes / maybe / no); samples split into thirds by human
  uncertainty (HUD): 1,083 / 1,083 / 1,082 for BEiT3 (Sec 4).
- Change tested and baseline: accuracy vs three distributional metrics (TVD, KL, entropy calibration error EntCE);
  temperature scaling toward accuracy (t = 0.6 for BEiT3) vs toward human distributions (t = 2.0).
- Effect (overall set, Sec 5 text on Tables 1-2): BEiT3 TVD 0.436 -> 0.340, KL 1.701 -> 0.647, EntCE 0.541 -> 0.333
  after calibrating toward humans; calibrating toward accuracy raises BEiT3 KL 1.701 -> 3.232. Accuracy LXMERT 0.67
  vs BEiT3 0.84. Both models fit the human distribution best on the high-uncertainty third and worst on the low third.
- Multiplicity/polysemy measured? Yes: distance between model and human answer distributions.
- H-b sub-claims touched: S2 (contradicts that a strong model's spread matches humans; supports that it can be
  partly aligned by a global temperature). BEiT3 is pretrained with masked data modelling over images, text and pairs
  (a fact from the BEiT-3 paper, Wang et al. 2023, not re-read here), but in this study the spread comes from a
  fine-tuned answer classifier, not a masked decoder, so S1 stays open.
- Relevance to MultiMAE / H-b: the nearest protocol for H-b's measurement step (model distribution vs annotator
  distribution with TVD/KL/entropy error), and a warning that accuracy-tuned models are too peaked. [reader-inferred]
- Method weaknesses: BEiT3 and LXMERT use different validation subsets [author-acknowledged, Sec 4]; only temperature
  scaling tried [author-acknowledged, Limitations]; HUD bins are equal thirds of self-reported confidence, so "high"
  uncertainty still averages 0.64 confidence [reader-inferred].

**[N6] Yao, Jarvis, Jiang 2026, verb sense clusters**
- Citation: Yao, L. H., Jarvis, N., Jiang, T. (2026). Towards Robust Evaluation of Visual Activity Recognition:
  Resolving Verb Ambiguity with Sense Clustering. Findings of EACL 2026 (ACL Anthology 2026.findings-eacl.32,
  preview). arXiv:2508.04945.
- Tier: peer-reviewed (Findings of EACL 2026).
- Read scope: full text (raw arxiv html v2): Sec 3-4.2, Tables 2, 3, 5, 6 text.
- Setting: imSitu (504 verbs). Clusters of image-verb pairs built from GPT-4o mini or Llama-3.2-90B verb predictions
  and K-Means. Models: supervised ResNet-50, CLIP and Llama-3.2-11B encoders with linear heads; zero-shot MLLMs.
- Change tested and baseline: cluster-based accuracy vs exact-match gold accuracy and WordNet-synset accuracy;
  human correctness labels on 100 images as the reference.
- Effect: images fall in 4.44 (GPT) / 3.61 (Llama) clusters on average; 95% / 71% of images in more than one cluster
  (Table 2). CLIP-encoder Top-1: gold 46.2, WN-synset 48.4, cluster 61.5 (Table 3). On the 100-image subset, CLIP
  Top-1 gold 50 vs human-judged 70, cluster 64 (Table 5).
- Multiplicity/polysemy measured? Yes: number of valid perspectives per image; agreement with human judgments.
- H-b sub-claims touched: S2 (supports that images carry about four valid readings in a labelled image-text task and
  that single-gold scoring under-counts them; says nothing about model distributions).
- Relevance to MultiMAE / H-b: an ambiguity-aware scoring recipe (count a prediction as right if it falls in any
  valid reading cluster) that could be reused for decoder outputs. [reader-inferred]
- Method weaknesses: clusters are derived from MLLM predictions, so "perspectives" are partly model-defined
  [reader-inferred]; human check on 100 images only [reader-inferred]; imSitu only [author-acknowledged, Sec 1].

**Pointer to B4 (fact needed here).** Cekinmez, Wu, Marjieh, Griffiths (2026), arXiv:2608.00410 (B4 entry): sense
distribution normalized entropy for polysemous words, image generators 0.10, text models 0.25, humans 0.47 (B4,
Fig 2 / Table 1). For S2 this is the only found work comparing a generative model's distribution over readings
with a human distribution over readings, and it finds generators far too peaked; it tests text-to-image generators,
not masked cross-modal decoders or retrieval.

### 2.3 A generator's distribution used at retrieval time (S4)

**[N7] IC3, Image Captioning by Committee Consensus**
- Citation: Chan, D. M., Myers, A., Vijayanarasimhan, S., Ross, D. A., Canny, J. (2023). IC3: Image Captioning by
  Committee Consensus. EMNLP 2023 (main). arXiv:2302.01328.
- Tier: peer-reviewed (EMNLP 2023).
- Read scope: full text (raw arxiv html): Sec 3, Tables 5-8, App C.5.
- Setting: inference only. K = 10 captions sampled from BLIP, BLIP-2 or OFA (temperature tuned), summarized into one
  caption by GPT-3; evaluated on COCO Karpathy test and Flickr30K test.
- Change tested and baseline: the summarized caption vs the base model's beam-search caption, and vs a human
  reference caption; CLIP text-to-image retrieval MRR as an "indexing" proxy.
- Effect (Table 5, COCO MRR): BLIP 0.501 -> 0.734; OFA 0.586 -> 0.748; BLIP-2 0.589 -> 0.746; human reference
  0.593 -> 0.776 with IC3. Table 6 (Flickr MRR): BLIP 0.499 -> 0.736; OFA 0.658 -> 0.806. Table 7 (COCO exact noun
  recall against all references): BLIP 0.266 -> 0.353; reference 0.255 -> 0.552. App C.5: correlation of candidate
  set diversity (Self-BLEU) with MRR r = 0.059-0.094.
- Multiplicity/polysemy measured? Partly: noun and verb recall against all 5 references (coverage of annotator
  viewpoints); no ambiguity or reading-level measure.
- H-b sub-claims touched: S4 (supports that sampling a decoder's distribution and pooling the samples covers more of
  what several annotators say and helps retrieval; but it merges readings into one caption, the opposite of keeping
  them apart), S2 (partly: coverage of several references).
- Relevance to MultiMAE / H-b: a concrete "sample the decoder, then aggregate" retrieval pipeline; for H-b the
  aggregation step would need to keep a set rather than summarize. [reader-inferred]
- Method weaknesses: GPT-3 summarizer cost and hallucination propagation [author-acknowledged, Sec 5]; CLIP-MRR
  rewards specificity, not coverage of alternative readings [reader-inferred]; the text's "up to 84.46%" Flickr gain
  for BLIP does not match the Table 6 cells as parsed (0.499 -> 0.736 is +47.5%) [reader-inferred].

**[N8] UniDisc, unified multimodal discrete diffusion**
- Citation: Swerdlow, A., Prabhudesai, M., Gandhi, S., Pathak, D., Fragkiadaki, K. (2025). Unified Multimodal
  Discrete Diffusion. arXiv preprint, arXiv:2503.20853.
- Tier: preprint, `preprint>=2024` (venue UNVERIFIED; arXiv comments give only a project page).
- Read scope: full text (raw arxiv html v1): Sec 3-4.3, Fig 3, 5, 7 (table cells), Fig 13 caption.
- Setting: from scratch; one bidirectional transformer over joint image (LFQ) and text (BPE) tokens trained by
  masking (absorbing-state discrete diffusion), 115M-1.4B parameters; 18M DataComp1B pairs for the retrieval study.
- Change tested and baseline: masked discrete diffusion vs an autoregressive (Chameleon-style) model trained on the
  same data; retrieval by model likelihood p(x_img | x_txt), p(x_txt | x_img) or joint.
- Effect (Fig 7 cells; 1 correct among 16 candidates, others random): DataComp joint retrieval AR 0.17 -> UniDisc
  0.64; text retrieval 0.85 -> 0.85; image retrieval 0.96 -> 0.95; Winoground text 0.24 -> 0.31, image 0.25 -> 0.27,
  joint 0.06 -> 0.20. Fig 5b: at matched generative perplexity, UniDisc samples have higher entropy than AR. Fig 3:
  13.2x more compute for UniDisc to reach AR's loss.
- Multiplicity/polysemy measured? No (token-level sample entropy only).
- H-b sub-claims touched: S4 (supports that a masked cross-modal generative model can retrieve by likelihood),
  S3 (open: compared with AR, not with a contrastive dual encoder), S1 (open: diversity not tied to readings).
- Relevance to MultiMAE / H-b: the closest found "masked joint image-text model used as a retriever"; it shows the
  scoring step is feasible but says nothing about multiple readings. [reader-inferred]
- Method weaknesses: 16-way retrieval with random negatives is far easier than COCO or ECCV Caption retrieval
  [reader-inferred]; retrieval accuracy depends strongly on CFG and denoising steps (Fig 13) [author-acknowledged,
  Sec 4.3]; training inefficiency [author-acknowledged, Fig 3].

**Short secondary entry (2.3)**
- AMER: Chen, H.-T., Liu, X., Ravfogel, S., Choi, E. (2025). Beyond Single Embeddings: Capturing Diverse Targets
  with Multi-Query Retrieval. arXiv:2511.02770, `preprint>=2024`. Read scope: abstract only. Text-only: an
  autoregressive model emits several query vectors when relevant documents form several clusters; abstract reports
  4x over a single embedding on synthetic data and 4-21% gains on real data, larger when targets are dissimilar.
  S4: supports "generate several embeddings per query" outside vision; G4 covers set embeddings in depth.
  Weaknesses: not assessed (read scope: abstract_only).

### 2.4 Under-specification, masking and dual encoders (S5)

**[N9] Pezzelle 2023, semantic underspecification**
- Citation: Pezzelle, S. (2023). Dealing with Semantic Underspecification in Multimodal NLP. ACL 2023 (long),
  ACL Anthology 2023.acl-long.675. arXiv:2306.05240.
- Tier: peer-reviewed (ACL 2023; position paper with two proofs of concept).
- Read scope: full text (raw arxiv html): Sec 3 (both PoCs), Fig 1-3 captions.
- Setting: pretrained CLIP, CLIPScore; 100 COCO image-caption pairs (top CLIPScore of 495), each rewritten into up to
  6 valid underspecified versions (pronouns, quantifiers, locatives, demonstratives, "They are doing something here").
- Change tested and baseline: CLIPScore of valid underspecified captions vs the original caption (PoC 1) and vs 10
  random captions of other images (PoC 2).
- Effect: replacing the subject with "they" lowers CLIPScore by about 16 points on average; the fully underspecified
  sentence by about 40 points (Sec 3, Fig 2); for 82 of 100 images at least one unrelated caption scores above the
  valid fully underspecified one (Sec 3, PoC 2).
- Multiplicity/polysemy measured? Indirectly: a less specific caption fits more images, and CLIP penalizes it.
- H-b sub-claims touched: S5 (supports the premise that a contrastive dual encoder mishandles under-specified text,
  the kind a masked caption is; leaves open whether a masked decoder handles it better), S3 (one side only).
- Relevance to MultiMAE / H-b: a masked caption is an under-specified caption; this is direct evidence that CLIP's
  single-point score mis-ranks such text, which H-b claims a decoder distribution would not. [reader-inferred]
- Method weaknesses: one annotator (the author) wrote the variants, 100 items, CLIPScore only, "for illustrative
  purposes" [author-acknowledged, Sec 3]; result size depends on the number of random captions sampled
  [author-acknowledged, PoC 2].

**Short secondary entries (2.4)**
- FOCUS: Zhou, K., Lai, E., Mouratidis, K., Jiang, J. (2025). FOCUS: Evaluating Pre-trained Vision-Language Models on
  Underspecification Reasoning. ACL 2025 (long), 2025.acl-long.1337. Read scope: abstract only (anthology PDF not
  parsed). Probing set of underspecified sentences with image contexts; VLMs "still fall short" even when the image
  could resolve the ambiguity. S5: supports that current VLMs do not exploit context to resolve under-specification.
  Weaknesses: not assessed (read scope: abstract_only).
- D-MLM: Gowda, S. N., Zhang, R., Gu, X., Weng, Y., Yang, L. (2025). Distribution-Based Masked Medical
  Vision-Language Model Using Structured Reports. arXiv:2507.21794 (Springer chapter; proceedings UNVERIFIED),
  `preprint>=2024`. Read scope: Sec 2 method, table captions (Tables 1-7). Chest X-ray pretraining in which tokens
  and patches are Gaussian and masked tokens and patches are predicted with a KL loss (a medical port of MAP's
  distributional masked modelling). Evaluated only on classification, grading and segmentation (Tables 1-5); no
  ambiguity or human-reading measure. S1/S5: masked + distributional modelling exists, but its "ambiguity" is never
  measured. Weaknesses: claims of capturing ambiguity are untested against any reading or disagreement data
  [reader-inferred].

### 2.5 Mixing combined with a masked objective (S6)

**[N10] MixGen inside ALBEF (crossing fact only; G5 covers MixGen in depth)**
- Citation: Hao, X., Zhu, Y., Appalaraju, S., Zhang, A., Zhang, W., Li, B., Li, M. (2023; arXiv v3 2023-01-09). MixGen: A New Multi-Modal
  Data Augmentation. WACV 2023 Workshops (Pretrain). arXiv:2206.08358.
- Tier: peer-reviewed (workshop).
- Read scope: full text (raw arxiv html): Algorithm 1, Sec 2.1, Sec 3 setup, Tables 6-7.
- Setting: ALBEF pretraining on COCO + VG + SBU (1M setting for ablations); M = B/4 batch items replaced by mixed
  pairs, lambda = 0.5; "we ... update the minibatch. We then forward the updated batch to the network without
  modifying any of their original training settings" (Sec 3), so the mixed pairs also go through ALBEF's masked
  language modelling loss.
- Change tested and baseline: blended image + concatenated captions ("both at once") vs ALBEF; variant (b) blended
  image + one of the two captions picked uniformly; variant (c) one image + concatenated captions (Table 6).
- Effect (Table 7, RSUM; ALBEF / MixGen / variant (b) / variant (c)): COCO-Ft 470.7 / 477.7 / 470.0 / 472.3;
  Flickr-Ft 555.7 / 561.0 / 557.2 / 555.4; Flickr-Zs 518.0 / 524.3 / 516.2 / 518.2.
- Multiplicity/polysemy measured? No.
- H-b sub-claims touched: S6 (mixing + masked reconstruction has already co-occurred, but only with the conjunction
  target, and never evaluated for polysemy; the "either one" variant used a single sampled caption as the only
  positive, not two valid targets, and did not help).
- Relevance to MultiMAE / H-b: the closest found precedent for "blend, then train with MLM"; the "either reading,
  selected by visible tokens" design is untested. [reader-inferred]
- Method weaknesses: single-run ablations without variance [reader-inferred]; variant (b) gives a noisy single target
  rather than a multi-positive or multi-target loss, so it does not test "either one" properly [reader-inferred].

### 2.6 Short secondary entry (S3 mechanism)
- CLIP Under the Microscope: Abbasi, R., Nazari, A., Sefid, A., et al. (2025). CLIP Under the Microscope: A
  Fine-Grained Analysis of Multi-Object Representation. CVPR 2025, pp. 9308-9317. arXiv:2502.19842. Read scope:
  abstract and CVF page only. CLIP's text encoder favours the first-mentioned object and its image encoder the largest
  object (ComCO dataset). S3: a mechanism by which one pooled point keeps a dominant reading; no comparison with a
  generative model. Numbers not recorded (abstract-level). Weaknesses: not assessed (read scope: abstract_only).

**Excluded after reading.** Singh and Pakrashi 2026, arXiv:2609.34506 (model uncertainty vs CIFAR-10H / FER+
disagreement; CNN classifiers only, no image-text model; G2 territory). Lu et al. 2026, arXiv:2602.21175 (LLM query
enrichment for quality control; no reading coverage). Long et al. 2025, DAR, arXiv:2501.15379 (diffusion-augmented
interactive retrieval; G4 territory, no ambiguity measure in abstract). Sanders et al. 2022, SQUID-E,
arXiv:2210.03102 (event classification with human uncertainty; pre-2024, not image-text decoding). arXiv:2008.09688
(GAN-art ambiguity from crowd descriptions, 2020; no model). DUST, arXiv:2402.12486 (text-only LMs). LaViDa
(2505.16839), LLaDA-V (2505.16933), MMaDA: masked-diffusion VLMs, abstracts report captioning and infilling but no
reading-coverage or ambiguity analysis.

## 3. Closest-to-H-b list

Per sub-claim, the closest work found, what it establishes, and what it leaves open.

- **S1 (decoder distribution holds several readings).** Closest: N2 Takatsuki et al. 2026 (next-token distribution
  of an autoregressive MLLM splits between two readings and is tilted by visual and textual context; reports stay
  exclusive, about 10% beam mass on naming both), with N3 Agate 2026 (both readings present in CLIP patch features,
  caption sampling commits; dominance 0.558) and N1 (most VLMs prefer one reading). N4 VSE adds that sample entropy
  under-reads ambiguity when the encoder is confident. Status: **partly established** for autoregressive MLLMs on
  illusion stimuli, in the "either-or at one slot" form; **open** for a masked cross-modal decoder and for natural
  multi-reading data (paintings, polysemous captions). No paper found tests the masked-token distribution of a
  cross-modal masked model for several readings (Q1, Q11, Q17, Q29, Q33, Q37).
- **S2 (spread vs human readings, image-text).** Closest: N5 Lan et al. 2024 (VQA answer distributions vs 10
  annotators; a masked-pretrained BEiT3 with a classifier head is too peaked, KL 1.701, improved to 0.647 by
  temperature 2.0) and the B4 entry Cekinmez et al. 2026 (generator sense entropy 0.10 vs human 0.47). N6 shows about
  four valid readings per imSitu image. Status: **established in the negative direction** for classifiers and
  generators (model spread narrower than human spread); **open** for masked cross-modal decoders and for caption- or
  emotion-level human distributions such as ArtELingo (G2/G3 may hold closer work).
- **S3 (masked/generative vs contrastive on ambiguity).** Closest: N1 Panagopoulou et al. 2024 (CLIP vs
  likelihood-scored generative VLMs on 29 bistable images: both mostly pick one reading; they differ in which and in
  sensitivity) and N9 Pezzelle 2023 (CLIP penalizes valid under-specified captions; no generative arm). Status:
  **open**: no comparison of a masked cross-modal decoder with a contrastive dual encoder on VWSD, ECCV Caption-style
  one-to-many retrieval, or multi-emotion images was found (Q3, Q13, Q14, Q23, Q24, Q26, Q39; B4 found no VWSD
  system using MLM).
- **S4 (decoder used at retrieval to cover readings).** Closest: N8 UniDisc (masked joint image-text model retrieves
  by likelihood; 0.64 vs AR 0.17 joint 16-way accuracy) and N7 IC3 (sampled captions pooled for retrieval; COCO MRR
  0.501 -> 0.734, but merged into one caption). AMER does multi-vector generation for text. Status: **components
  established** (decoder scoring, sampling for coverage; G1/G4 hold more); **open** for keeping per-reading sets from
  a masked decoder and testing them on multi-reading retrieval (Q9, Q21, Q37, Q39).
- **S5 (masking teaches under-specification beyond ProLIP).** Closest: ProLIP's masked inclusion (B4) remains the
  only training-side use found; N9 and FOCUS show dual encoders and VLMs mishandle under-specified text; D-MLM masks
  with Gaussian tokens but never measures ambiguity. Status: **open** beyond ProLIP (Q14, Q22, Q34, Q35). A design
  note: a pixel decoder trained with mean-squared error predicts the conditional mean, so only a categorical (MLM)
  head can show more than one reading in MultiMAE [reader-inferred].
- **S6 (mixing or synthetic ambiguous images + masked reconstruction for polysemy).** Closest: N10 MixGen inside
  ALBEF (mixed pairs already trained with MLM, conjunction target; single-caption variant (b) COCO-Ft RSUM 470.0 vs
  ALBEF 470.7 vs MixGen 477.7; no polysemy evaluation) and N2's Visual Anagrams as synthetic stimuli with known
  readings (evaluation only). Status: **open**: no work found that trains with blended or illusion images and treats
  each source caption as a separate valid masked-reconstruction target, or evaluates such training for polysemy
  (Q15, Q16, Q27, Q28).

**Overall novelty-risk judgment for H-b: medium-low** [reader-inferred]. Reasons:
1. The full chain (masked cross-modal decoder distribution, measured against human readings, compared with a
   contrastive dual encoder, used for retrieval) was not found as a whole in Q1-Q40, nor in B4's Q1-Q29.
2. The risk is not "already done" but "already half-answered elsewhere": autoregressive MLLM work (N2, N3, N1)
   already shows that a decoder's slot distribution can split between readings while its outputs commit to one, so
   a reviewer will ask what masking adds over next-token probabilities of an off-the-shelf captioner (G1 baseline).
3. Decoder-likelihood retrieval (N8, G1) and probabilistic embeddings (ProLIP, PCME++, B2/B4) are established, so S4
   novelty rests on the per-reading set and the multi-reading benchmark, not on decoder scoring itself.
4. The direction of prior evidence on S2 is negative (N5, Cekinmez et al.): model spread tends to be narrower than
   human spread, so H-b carries real risk of a null or negative result, which would still be reportable.

Per sub-claim risk [reader-inferred]: S1 medium (AR-MLLM precedent), S2 low-medium (protocols exist, masked decoders
untested), S3 low, S4 medium (pieces exist), S5 low, S6 low.

## 4. Search limitations

- OpenReview pages and PDFs (AmbiBench, ICLR 2026 submissions) returned a verification page; AmbiBench numbers are
  therefore UNVERIFIED and taken only from N3's description and the HF dataset card.
- ACL Anthology PDFs (FOCUS, CMCL proceedings version of N1) were not parsed; N1 was read from arXiv v1, which may
  differ from the workshop version.
- N1 reports its main quantities only in figures, so no per-image probabilities are given here.
- IC3's "up to 84%" text claim could not be reconciled with its Table 6 cells as parsed.
- Not read beyond snippets: Illusory VQA (arXiv:2412.08169), IllusionBench+ (arXiv:2501.00848), MUCAR
  (arXiv:2506.17046), ClearVQA ("Teaching VLMs to Ask", arXiv:2507.13773), VLM-UQBench (arXiv:2602.09214), "Semantic
  and Expressive Variation in Image Captions Across Languages" (CVPR 2025), "Does CLIP perceive art the same way we
  do?" (arXiv:2505.05229), ArtELingo-28 (arXiv:2411.03769). The last two are likely in G3's scope.
- Absence claims are bounded: "no masked cross-modal decoder evaluated for several readings" covers Q1, Q11, Q17,
  Q29, Q33, Q35, Q37 and the full texts of N1-N10; "no masked-vs-contrastive comparison on polysemy" covers Q3, Q13,
  Q14, Q23, Q24, Q26, Q39; "no mixing + masked reconstruction for polysemy" covers Q15, Q16, Q27, Q28.
- No Semantic Scholar API queries; journal-only work (TPAMI, TMM, ACM MM) reached only through web search.
- No retrieved page contained text that tried to direct the agent.
- Files left behind: raw downloads in `scratchpad/hb_review/dl_N/` (about 3 MB) and `scratchpad/h2t.py`.
