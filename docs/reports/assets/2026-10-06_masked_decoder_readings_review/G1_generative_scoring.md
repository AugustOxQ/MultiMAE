> Supporting file for docs/reports/auto/v1/2026-10-06_masked_decoder_readings_review.md. Phase 2 bibliography as written; V1 lists the corrections, and the report uses corrected values.

# G1: scoring image-text matches with a generative model (Phase 2, lit-review mode)

Last searched 2026-10-06. Caveat on reading: all page reads went through the WebFetch tool, which returns a small-model digest of the page, not raw text. Numbers below are as the digest reported them, with table locators as reported; numbers marked (x2) were returned identically by two separate fetches. No number was filled in by me. Anything not read is marked.

Verification status (after the coordinator's method note): numbers in entries 1 (Lin ICML 2024), 5 (MASS) and 10 (DiffusionITM) were re-read from raw arxiv html downloaded to scratchpad/hb_review/dl_G1/ and corrected where the first digest was wrong. Entries 2, 3, 4, 6, 8, 9, 11, 12, 13 still carry digest-level numbers (not re-read from raw text) and should be treated as UNVERIFIED at the table-cell level.

## 1. Search strategy

- Tools: WebSearch (standard and extended), WebFetch on arxiv.org/html pages. Sources hit: arXiv, ACL Anthology, ICML/ICLR/AAAI/CVF proceedings pages, Semantic Scholar-style aggregators.
- Date range: no lower bound (foundations back to 2020), emphasis 2023 to 2026.
- Queries:
  - Q1 VisualGPTScore language priors; Q2 VQAScore; Q3 Salazar MLM scoring PLL; Q4 generative classifier vs CLIP compositionality Winoground captioner likelihood 2024 2025.
  - Q5 re-ranking top-k with captioner likelihood; Q6 VLM masked-LM pseudo-log-likelihood image-text matching; Q7 Intriguing properties of generative classifiers; Q8 generative VLM multiple valid captions / short generic captions / length normalization.
  - Q9 MASS fetch; Q10 LLM-based reranker GPT-2 rescoring soft labels; Q11 DiffusionITM; Q12 Surface Form Competition PMI_DC; Q13 ECCV Caption/CxC with generative captioner likelihood; Q14 CapPa likelihood classification.
  - Q15 2025 generative likelihood vs ITM ambiguity; Q16 PLL cost approximations (maskless, union mask); Q17 captioner likelihood zero-shot retrieval CoCa/BLIP-2; Q18 MLLM yes-probability reranking; Q19 generative vs discriminative language-prior calibration; Q20 false negatives with generative likelihood.
- Inclusion: a paper that scores image-text (or text-text) matches with a likelihood, PLL, PMI-corrected likelihood or answer probability, or that compares such a score with a contrastive one, or that costs/approximates PLL. Exclusion: papers using generative likelihood only as a training signal; generic MLLM rerankers without a likelihood score (Rank-Nexus 2601.20623, Sugar 2411.00304, "A Paragraph is Worth a Thousand Captions" 2608.05260, Agarwal et al. 2605.24702 had no generative-vs-contrastive comparison from one model; screened and dropped).
- Counts: about 75 identified in result lists, about 30 screened by title/snippet, 20 full-text or html fetched, 13 included.
- Coverage skew: strong on Lin et al. line and diffusion classifiers; weak on one-to-many metrics (see Sec. 4); sub-theme (e) is thin.

## 2. Annotated bibliography

### (a) Likelihood as match score, generative vs discriminative

**1. Lin, Chen, Pathak, Zhang, Ramanan, 2024. Revisiting the Role of Language Priors in Vision-Language Models. ICML 2024 (PMLR 235). arXiv 2306.01879.**
- Tier: peer-reviewed (ICML 2024).
- Read scope: full-text digest of arxiv.org/html v4, Tables 1 and 3, Sec on PMI and cost; last 9.7k characters not read.
- Setting: zero-shot, no fine-tuning; BLIP (ViT-L, LAION-114M) main, BLIP-2 and LLaVA-1.5 secondary; image-to-text retrieval and compositional benchmarks.
- Change tested / strongest baseline: VisualGPTScore = P(text|image) from the captioning head, vs ITCScore and ITMScore of the same BLIP; then alpha-debiasing by the text prior P(text) (equals PMI^(1/alpha), their Eq 8); the paper's strongest baseline is BLIP ITM.
- Effect (all numbers below re-read from the raw arxiv.org/html file, grep on the downloaded page; locators as in the paper). Table 1, ARO columns Rel/Attr/COCO-Order/Flickr-Order: blind text-only BLIP 87.6/80.7/98.6/99.1; BLIP-ITC 63.1/81.6/34.3/41.7; BLIP-ITM 58.7/90.3/45.1/51.3; VisualGPTScore (alpha=0) 89.1/95.3/99.4/99.5. Table 1 SugarCrepe (Replace/Swap/Add): BLIP-ITC 85.8/73.8/85.7; ITM 88.7/81.3/87.6; VisualGPTScore alpha=0 93.3/91.0/91.0; val-tuned debiased 95.1/92.4/97.4. Note ARO order sets are won by a blind LM (98.6/99.1), so these gains are partly a language-prior effect. Table 3a, image-to-text, full candidate set (raw numbers re-read): COCO 5k R@1 ITM 71.9, VisualGPTScore alpha=0 19.7, alpha=1 46.2, alpha*=val-tuned 48.0; Flickr30K R@1 ITM 88.8, alpha=0 34.6, alpha=1 58.7, alpha* 63.6. Winoground text score: alpha=0 27.5, alpha=1 33.7, alpha* 36.6, ITM 35.5. EqBen: 9.6 raw, 19.8 debiased. The paper states generative scores "still lag" ITM on the two large retrieval sets.
- Multiplicity measured? No: digest reports the paper does not address several acceptable captions per image. Single-GT recall only.
- Score type: likelihood, PMI-corrected (alpha-exponent on a prior estimated by Monte-Carlo or by a Gaussian-noise image).
- Candidate set and cost: full candidate set (5k images/25k captions in COCO); one decoder forward pass per (image, caption) pair, "same cost as ITMScore" (causal vs bidirectional mask). No ITC prefilter reported in the digest.
- Same-model contrastive comparison: yes (ITC and ITM of BLIP). Result: generative wins on adversarial-caption sets (ARO, SugarCrepe), loses badly on COCO/Flickr retrieval raw (19.7 vs 71.9 R@1), recovers about half the gap after debiasing, still below ITM.
- Relevance [reader-inferred]: this is the direct template for scoring a MultiMAE decoder; shows raw likelihood is dominated by the text prior, and gives the prior-correction recipe needed before any one-to-many claim.
- Weaknesses: alpha tuned on validation split of each benchmark [author-acknowledged, Table 3 reports alpha*val separately]; prior estimated from the same model, error in P(text) distorts rankings when candidate captions are generic or very specific [reader-inferred]; BLIP's ITM head is trained with hard negatives on the same data, so the comparison is not neutral [reader-inferred].

**2. Lin, Chen, Pathak, Ramanan, et al., 2024. Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore). ECCV 2024. arXiv 2404.01291.**
- Tier: peer-reviewed (ECCV 2024).
- Read scope: digest of arxiv.org/html first 100k of 135k characters; Tables 1, 3, 4 and Sec 3.
- Setting: off-the-shelf VQA models (InstructBLIP, LLaVA-1.5, CLIP-FlanT5, GPT-4V baselines); scoring, not training.
- Change tested / baseline: score = P("Yes" | image, "Does this figure show {text}?") instead of caption likelihood or CLIP cosine; baselines CLIPScore, BLIPv2 ITM, VIEScore (GPT-4V).
- Effect: Winoground (Table 1) text/image CLIP-FlanT5 60.0/57.5 vs CLIPScore 27.8/11.5, VIEScore 40.8/39.3. Table 4 AUROC vs human judgement: DrawBench 85.3 vs CLIPScore 49.1; COCO-T2I 85.0 vs 63.7. GenAI-Bench 1,600 prompts.
- Multiplicity measured? No.
- Score type: other (answer probability, a verification-style ITM-like score from a generative model); sidesteps the text prior since the text is in the prompt, not the target. The digest says it is contrasted with VisualGPTScore in Sec 3.
- Candidate set and cost: pairwise scoring; one MLLM forward per pair (not an embedding lookup). No retrieval top-k protocol in the digest.
- Same-model contrastive comparison: CLIP-FlanT5 contains a CLIP vision tower but its score is not compared with that tower's cosine; compared against separate CLIP models. Not the same-model design.
- Relevance [reader-inferred]: shows the prior problem can be avoided by conditioning on the text as input rather than predicting it; not a decoder-of-masked-tokens score, so only indirectly relevant to H-b.
- Weaknesses: evaluates generation-alignment with human ratings, not retrieval with many positives [reader-inferred]; the answer probability is calibrated by instruction tuning, which a masked-reconstruction decoder does not have [reader-inferred].

**3. Tschannen, Kumar, Steiner, Zhai, Houlsby, Beyer, 2023. Image Captioners Are Scalable Vision Learners Too (Cap/CapPa). NeurIPS 2023. arXiv 2306.07915.** (B4 covers Cap/CapPa; cited here only for facts the old entry lacks: likelihood scoring numbers.)
- Tier: peer-reviewed (NeurIPS 2023).
- Read scope: digest of first 100k of 116k characters; ARO and SugarCrepe tables, zero-shot sentence.
- Setting: from-scratch pretraining, ViT-B/16 up to larger; likelihood of class name or caption as score.
- Change / baseline: score by log-likelihood (no prior correction stated in the digest) vs contrastive CLIP trained with the same batch size, and OpenCLIP G/14.
- Effect: ARO (VG-Attr/VG-Rel/Flickr-Order/COCO-Order): Cap B/16 88.9/86.6/99.1/99.0; CapPa 85.7/86.7/99.2/98.8; CLIP* 53.2/39.7/45.5/37.0. SugarCrepe Replace/Swap/Add: CapPa B/16 89.95/88.71/84.35; OpenCLIP G/14 96.67/88.07/74.75.
- Multiplicity measured? No.
- Score type: likelihood (raw). Candidate set and cost: authors state class names "have to be scored again for every new image", so Cap/CapPa are less efficient zero-shot classifiers than CLIP.
- Same-model contrastive comparison: no (separate CLIP trained alongside, not the same model).
- Relevance [reader-inferred]: the CapPa masked-prediction head with parallel decoding is the closest published analogue of our non-causal MLM decoder; this entry adds that the paper reports no retrieval-by-likelihood with many positives in the parts read.
- Weaknesses: ARO/SugarCrepe negatives are text-only adversarial, so a language-prior-sensitive score can win there for non-visual reasons (Lin et al. show a blind LM gets 87.6 on ARO) [reader-inferred, grounded in entry 1].

**4. Rambhatla and Misra (names UNVERIFIED), 2023. SelfEval: Leveraging the discriminative nature of generative models for evaluation. arXiv 2311.10708. Venue not verified.**
- Tier: preprint (venue UNVERIFIED).
- Read scope: digest of arxiv.org/html; Sec 3.2, 4.1, App C and F.
- Setting: zero-shot diffusion models used as classifiers via p(image|text), real image-text benchmarks (TIFA, CLEVR, ARO, Winoground).
- Change / baseline: ELBO-based conditional likelihood with uniform class prior vs CLIP.
- Effect: Winoground text score about 28-29 for diffusion models vs CLIP 8-12.75 (digest, Table 5 level, exact table row UNVERIFIED); counting about 25 vs CLIP 67.8; text-corruption about 24 vs CLIP 51.
- Multiplicity: No.
- Score type: likelihood (ELBO); no text-prior correction (uniform prior). Cost: O(N C T), N=10 trials, T=100 timesteps, C candidates; authors state cost grows linearly with candidates and is hard to scale to thousands of classes (App F).
- Same-model contrastive comparison: no (separate CLIP).
- Relevance [reader-inferred]: gives a cost figure for sampling-based likelihood scoring, the diffusion analogue of PLL's many-forward-passes problem.
- Weaknesses: uniform prior assumed, so sensitivity to prior not studied [author-acknowledged per digest].

### (b) Prior correction and debiasing

**5. MASS: Overcoming Language Bias in Image-Text Matching. AAAI 2025. arXiv 2501.11469.** (Author list not captured by the fetch; first author UNVERIFIED.)
- Tier: peer-reviewed (AAAI 2025, per search listing).
- Read scope: digest of arxiv.org/html: Eq 6, Tables 1-4, Appendix D, limitations.
- Setting: inference-time, no training; OFA (33M, 180M, 930M), BLIP-2 and LLaVA as alternates; CLIP-base (86M) baseline.
- Change / baseline: MASS(c,x) = mean over tokens of log[p(x_t|x_<t,c) / p(x_t|x_<t,c_empty)] with a black image as the marginal proxy; baseline is raw token likelihood (TL), plus ITM/ITC.
- Effect (OFA large; numbers re-read from the raw html, grep on the downloaded page). Table 1 VALSE counting, Adv. set: ITM 72.0, TL 66.8, MASS 76.7. Table 2 MS COCO gender-bias test, R@1: text-to-image CLIP ITC 36.5, OFA ITM 35.3, TL 38.0, MASS 55.6; image-to-text CLIP ITC 56.9, ITM 42.1, TL 33.4, MASS 57.3, Bias@1 TL 6.6 -> MASS 3.5 (an earlier draft of this entry mislabelled 38.0/55.6 as image-to-text; corrected). Table 3 Winoground group score: OFA large ITM 9.5, TL 15.8, MASS 20.3; with BLIP-2 T5 the order flips: ITM 15.5, TL 11.0, MASS 18.0. Table 4 SVO-Probes group score OFA large: ITM 39.7, TL 12.5, MASS 43.0 (so MASS beats ITM by 3.3 there, TL is far below it).
- Multiplicity measured? Partly: recall counts a hit when any ground-truth pair is retrieved (Sec 4.3); no ECCV mAP@R or coverage measure.
- Score type: PMI-corrected (length-normalised, per-token, black-image marginal). Candidate set and cost: two forward passes per pair, 0.55 s per sample for OFA-large (App D), vs 0.27 s for one pass.
- Same-model contrastive comparison: TL vs MASS vs ITM from OFA large on Winoground group score (ITM 9.5 < TL 15.8 < MASS 20.3), but with BLIP-2 T5 ITM (15.5) beats TL (11.0) and MASS (18.0) is only slightly above ITM; on COCO the MASS image-to-text R@1 (57.3) is at CLIP-ITC level (56.9), not a different model comparison. So the benefit over a discriminative score depends on the backbone.
- Relevance [reader-inferred]: a cheap, per-token PMI that works with a decoder reading the image; the black-image marginal is a practical stand-in for the prior of a masked decoder (feed all patches masked).
- Weaknesses: the black image is a proxy for the true marginal; it can leave residual prior or add a distribution-shift error when the model never saw blank inputs [reader-inferred]; single-GT recall on COCO [reader-inferred]; limitations in Broader Impact say effect varies with base model [author-acknowledged].

**6. Holtzman, West, Shwartz, Choi, Zettlemoyer, 2021. Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. EMNLP 2021. arXiv 2104.08315.**
- Tier: peer-reviewed (EMNLP 2021).
- Read scope: digest of arxiv.org/html; definition, 13 datasets, GPT-3 table, cost.
- Setting: zero-shot multiple choice with GPT-2 (125M-1.6B) and GPT-3 (2.7B-175B); text-only.
- Change / baseline: PMI_DC = P(y|x, domain)/P(y|domain), prior estimated with a domain prompt; baselines raw LM, length-normalised, contextual calibration.
- Effect (GPT-3 175B): COPA 85.2 (LM) / 82.8 (length-norm) / 89.2 (PMI_DC); CommonsenseQA 61.0 / 57.4 / 66.7; ARC-Challenge 40.2 / 43.2 / 45.5. Best or tied on 62.5% of splits vs 18.75% for length-norm. Fails on HellaSwag, ARC-Easy, BoolQ.
- Multiplicity: this is the mechanism where several valid strings split probability mass, the text-side version of several valid captions; not measured on images.
- Score type: PMI-corrected. Cost: one extra forward pass per option (doubling).
- Same-model contrastive: not applicable.
- Relevance [reader-inferred]: establishes that correcting by the prior can hurt when the prior is informative (failure on 3 datasets), which bounds how far an H-b score can lean on PMI.
- Weaknesses: text-only; domain-prompt prior is hand-chosen [reader-inferred].

### (c) Masked-LM scoring

**7. Salazar, Liang, Nguyen, Kirchhoff, 2020. Masked Language Model Scoring. ACL 2020. arXiv 1910.14659.**
- Tier: peer-reviewed (ACL 2020).
- Read scope: search snippets and ACL abstract; full text NOT read in this run. Rest of the entry limited accordingly.
- Setting: pretrained MLMs (BERT, RoBERTa, XLM); PLL = sum over tokens of log P(token | all other tokens), one masked copy per token; used to rescore ASR and NMT n-best lists and to judge linguistic acceptability.
- Effect (from abstract snippet): RoBERTa rescoring reduces an end-to-end LibriSpeech model's WER by up to 30% relative, up to +1.7 BLEU on low-resource NMT; PLLs beat GPT-2 scores on BLiMP (+10 points on island effects and NPI licensing).
- Multiplicity: not measured. Score type: PLL. Candidate set and cost: n-best lists, O(tokens) forward passes per candidate (confirmed by entry 8). The abstract-level claim that maskless scoring is possible via fine-tuning came from a search summary and is UNVERIFIED.
- Same-model contrastive: not applicable.
- Relevance [reader-inferred]: it is the standard recipe for scoring a candidate with a masking model; the re-ranking use case (n-best, not full-corpus) matches H-b's re-ranking variant.
- Weaknesses: not assessed (read scope: abstract_only).

**8. Kauf and Ivanova, 2023. A Better Way to Do Masked Language Model Scoring. ACL 2023 (short). arXiv 2305.10588.** (Author names from the ACL listing; not shown in the digest, so UNVERIFIED.)
- Tier: peer-reviewed (ACL 2023 short).
- Read scope: digest of arxiv.org/html, Sec 4-5, BLiMP table.
- Setting: BERT and RoBERTa base/large; text-only.
- Change / baseline: PLL-word-l2r masks the target token and all later tokens of the same word; baseline is original PLL.
- Effect (BLiMP, Table in Sec 5): BERT-base 84.2 -> 84.7; BERT-large 84.8 -> 85.0; RoBERTa-base 85.4 -> 86.7; RoBERTa-large 86.5 -> 87.5. Original PLL inflates scores for multi-token words (inverse length relation fails; word-frequency relation breaks).
- Multiplicity: no. Score type: PLL. Cost: O(n) forward passes per sentence; no saving.
- Relevance [reader-inferred]: a masked decoder predicting several tokens of one word in parallel faces the same inflation; relevant to any PLL on caption tokens (subword leakage), and our content-word masking (M2b) partly avoids it.
- Weaknesses: BLiMP is acceptability, not retrieval [reader-inferred].

**9. McCarter, Bhattacharya, Ober, Elliott, 2026. How to make the most of your masked language model for protein engineering. arXiv 2603.10302.** preprint>=2024.
- Tier: preprint. Read scope: digest of first 100k of 119.7k characters.
- Setting: protein MLMs (ESM2-650M, AbLang2, ProteinMPNN); not vision.
- Change: masked-marginal approximation of PLL for sequences one edit away from a template, reusing the template's conditional probabilities; cost O(L^4) -> O(L^3) amortised.
- Effect: Pearson / Spearman with exact PLL (Table 1) ESM2-650M 0.964/0.940, AbLang2 0.869/0.869, ProteinMPNN 0.944/0.934.
- Multiplicity: no. Score type: PLL (approximate).
- Relevance [reader-inferred]: shows a cheaper PLL exists when candidates share most tokens with a template; captions of one image rarely share tokens, so little transfer. Tangential, included for sub-theme (c) cost only.
- Weaknesses: domain is proteins; the approximation holds only for near-identical candidates [reader-inferred].

### (a, continued) Diffusion classifiers as comparison point

**10. Krojer, Poole-Dayan, Voleti, Pal, Reddy, 2023. Are Diffusion Models Vision-And-Language Reasoners? (DiffusionITM, GDBench). NeurIPS 2023. arXiv 2305.16397.**
- Tier: peer-reviewed (NeurIPS 2023).
- Read scope: digest of arxiv.org/html; Sec 3.1, Sec 5, Tables 1-3.
- Setting: zero-shot Stable Diffusion 1.5/2.1, 250 noise samples; GDBench of 7 tasks (Winoground, CLEVR, ARO, ImageCoDe, SVO, Flickr30K, Pets) plus bias.
- Change / baseline: image-text score = denoising error with text minus error without text (a prior/modality-asymmetry correction); baseline CLIP RN50x64 (and OpenCLIP).
- Effect (raw html re-read; Table 1a, image retrieval, Flickr Img / Winoground Image): CLIP RN50x64 71.9 / 13.8; OpenCLIP ViT-L/14 79.9 / 11.25; naive Diffusion Classifier (no unconditional subtraction) 11.1 / 0.25; DiffusionITM 61.5 / 10.2; HardNeg-DiffusionITM 66.1 / 12.3. So on Flickr image retrieval DiffusionITM is below both CLIPs and the uncorrected score is near chance (an earlier draft quoted 86.2 and Winoground 52.3; 86.2 is the SVO-Obj column, now corrected; text-retrieval Table 1b columns were not parsed cleanly, so no text-retrieval number is claimed). Bias effect sizes (Table 3 last row, lower is less biased): SD2.1 0.65, SD1.5 0.79, CLIP RN50x64 1.09, OpenCLIP 1.26.
- Multiplicity: no.
- Score type: PMI-corrected-style (unconditional baseline subtraction). Cost: Flickr30K text-retrieval validation took 68 min on one RTX A6000 with 10 noise samples vs about 4 min for OpenCLIP ViT-L/14 (Runtime paragraph, confirmed in raw html). The authors say they envision such discriminators as "slow retrievers applied to a small set of candidates" provided by a fast one, i.e. re-ranking (sub-theme e). Naive likelihood gave near-chance image retrieval ("modality asymmetry").
- Same-model contrastive comparison: no (CLIP separate); SD's own CLIP text encoder is shared but only as conditioning.
- Relevance [reader-inferred]: the cleanest evidence that uncorrected generative scores can be at chance on retrieval, and a ~17x cost penalty for sampling-based scoring.
- Weaknesses: single-GT retrieval; text-encoder bias inherited from SD's CLIP [reader-inferred].

**11. Jaini, Clark, Geirhos, 2024. Intriguing Properties of Generative Classifiers. ICLR 2024. arXiv 2309.16779.** (Author order from memory, UNVERIFIED; the digest did not list authors.)
- Tier: peer-reviewed (ICLR 2024).
- Read scope: digest of arxiv.org/html, Sec 1-2, Table 1, Figs 3-4, App C.
- Setting: zero-shot Imagen, Stable Diffusion, Parti (exact autoregressive likelihood from one forward pass) as classifiers; 17 OOD datasets; 52 discriminative models; human psychophysics.
- Effect: OOD accuracy Imagen 0.71, SD 0.69, CLIP 80-prompt 0.71, CLIP 1-prompt 0.55; shape bias Imagen 99% vs CLIP 57-80% (Table 1); error consistency with humans (Cohen's kappa) Imagen 0.31 vs ViT-22B 0.26.
- Multiplicity: no, though it studies agreement with human error patterns, not distributions over several labels.
- Score type: likelihood (ELBO) and exact likelihood for Parti. Cost: one reconstruction per class; authors call it slow and "not practical".
- Same-model contrastive: no.
- Relevance [reader-inferred]: supports "generative scores resist shortcuts", but evidence is classification; no multiplicity.
- Weaknesses: model sizes, data and text encoders differ between generative and discriminative contenders [author-acknowledged].

**12. Pearson, Boulbarss, Wray, Lewis, 2025. Evaluating Compositional Generalisation in VLMs and Diffusion Models. arXiv 2508.20783.** preprint>=2024.
- Tier: preprint. Read scope: digest of arxiv.org/html, results tables.
- Setting: fine-tuning with 20-40 images per class; Diffusion Classifier (Stable Diffusion) vs CLIP vs ViLT; Extended Concept Binding Benchmark (single-object, two-object, relational), ZSL and GZSL; 200 noise samples.
- Effect: Relational GZSL OOD test DC-FT 38.25 vs CLIP 34.75; single-object OOD DC-FT 99.47 vs CLIP 91.21. Authors conclude the generative classifier does not inherently solve composition.
- Multiplicity: no. Score type: likelihood (ELBO). Same-model contrastive: no.
- Relevance [reader-inferred]: counter-evidence to "generative scoring is better"; the advantage is task- and setup-dependent.
- Weaknesses: synthetic shapes and colours, small fine-tuning sets [reader-inferred].

**13. Fathi, Ayar, Abdolali, Javanmardi, Tsukada, Javanmardi, 2026. How Do Diffusion Classifiers Decide? A Bias-Centric Evaluation (ASOB-Bench). arXiv 2607.03831.** preprint>=2024.
- Tier: preprint. Read scope: digest of arxiv.org/html, Tables 1-3.
- Setting: diffusion classifiers (also SD3) vs OpenCLIP ViT-H/14 with the same text encoder.
- Effect: attribute binding better than OpenCLIP (Table 1: natural-colour CAB 0.714 vs 0.8775; two-object accuracy 0.212 vs 0.112); size-order shortcut worse (Table 2: accuracy gap 36.75 points vs 22.91); background changes cost 15-19 points vs about 2 for CLIP (Table 3).
- Multiplicity: no. Score type: reconstruction-error likelihood proxy. Cost: noise-prediction pass per candidate over 50 timesteps (not quantified).
- Same-model contrastive: closest to same-text-encoder comparison; result mixed.
- Relevance [reader-inferred]: pixel-aggregated scoring weights large regions regardless of meaning, the same issue a MAE pixel-loss score would have.
- Weaknesses: classification of synthetic scenes [reader-inferred].

(Li et al., ICCV 2023, "Your Diffusion Model is Secretly a Zero-Shot Classifier": the page returned HTTP 403; only a search-result title was seen, so it is excluded rather than annotated.)

## 3. Closest-to-H-b list

1. Lin et al. 2024 (entry 1): same model scored by likelihood, by ITC and by ITM; shows raw likelihood fails on COCO/Flickr retrieval (R@1 19.7 vs 71.9) and that prior debiasing recovers part (46.2 to 63.6) but not parity. It does not measure several valid matches and uses a causal decoder, not a masked-reconstruction one.
2. MASS (entry 5): per-token PMI with a content-free image marginal, same-model TL vs ITM comparison on Winoground, recall that accepts any of several GT pairs on COCO; costs two passes per pair. Not a masked decoder; no ECCV mAP@R.
3. Salazar et al. 2020 plus Kauf and Ivanova 2023 (entries 7-8): the PLL recipe and its known inflation failure with multi-token words; they establish cost (one pass per token per candidate) and the re-ranking use, but contain nothing multimodal. DiffusionITM (entry 10) is the next closest for "naive generative score is at chance, correction rescues it".

## 4. Search limitations

- No paper found (Q5, Q13, Q15, Q17, Q18, Q20) that scores COCO or CxC or ECCV Caption with a generative likelihood and reports ECCV mAP@R, R-Precision or any multi-positive coverage measure against a contrastive score from the same model. Absence bounded by those queries and by what WebSearch indexes.
- No paper found (Q6) that applies Salazar-style PLL to a multimodal masked model for image-text retrieval; PLL appears only in text-only work in the results. MLM-in-VLM entries returned were training-objective papers, not scoring papers.
- Re-ranking a dual encoder's top-k with a generative likelihood (sub-theme e): results were MLLM listwise or yes/no rerankers for RAG and universal retrieval, which I did not read in enough depth to include; BLIP's ITC-then-ITM re-rank is covered in the 2026-10-03 B2 bibliography, not repeated.
- Maskless or amortised PLL (Q16): only a search-summary mention of "maskless scoring" and a union-mask approximation; neither primary source was opened (UNVERIFIED).
- Author lists were missing for MASS, Kauf and Ivanova, Jaini et al., and SelfEval because the digest tool did not return them; marked UNVERIFIED. Venue of SelfEval not confirmed.
- Digest-level reading: table values may carry transcription errors; the Lin et al. COCO/Flickr numbers were confirmed across two fetches, others came from one.
