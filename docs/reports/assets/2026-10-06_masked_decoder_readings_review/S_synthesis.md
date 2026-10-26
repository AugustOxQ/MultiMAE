> Supporting file for docs/reports/literature/2026-10-06_masked_decoder_readings_review.md. Phase 3 synthesis as written, built from G1 to G5 and N as corrected by V1 to V3; the report condenses it. The report corrects two points: sMCL led beam search at M = 2 to 4 and trailed only at M = 5 (V2 rows 8, 9), and Cekinmez et al.'s 0.47 and 0.73 are in Fig 2 and Table 2, not Table 1.
> The final review (tests/20261003_ml_improve/review_2026-10-06_masked_decoder_readings_review.md) superseded V3 problem 1 (Panagopoulou et al.) and row 24 (UniDisc), which this synthesis follows; the report uses the final review's values.

# S: Synthesis for H-b, the masked cross-modal decoder as a model of several readings (MultiMAE)

Phase 3 synthesis (ARS deep-research, lit-review mode, integrative review), written 2026-10-06 from 20:14. Inputs: `00_brief.md`; the bibliographies G1 to G5 and N; the verification files V1 to V3, whose corrections override the bibliographies everywhere below; the 2026-10-03 synthesis (structure only); the 2026-10-06 Stages 1 and 2 report (Summary, sections 4 and 5); facts the lead agent verified locally (marked [lead]); and a reading of `mmae/models/{model,decoders,masking,fusion}.py` (marked [code]).

**Conventions.**
- A number is verifier-checked against the paper's raw text unless it carries "(digest-level)": then the only read was a small-model page summary, and the number should be re-read before it goes into a report.
- "(N's raw read)" marks numbers N read from raw HTML that V3 did not re-check.
- "(computed)" is my arithmetic. [reader-inferred] marks my judgments. [project] marks MultiMAE results from the brief or the Stages report.
- Locators are the papers' own. "V1 row 12" points to a verifier's claim table; "V2 problem 8" to its list of load-bearing problems.

**Terms used throughout.**
- *H-b*: the claim that a cross-modal masked reconstruction model's decoder distribution represents several readings of an image or caption, which a contrastive dual encoder cannot, and that this is measurable and usable for retrieval. *S1 to S6*: its sub-claims, defined in Section 1.
- *Reading*: one valid interpretation of an input. On ArtELingo we recommend the annotator's emotion label as the unit (Section 5.0).
- *MLM decoder*: our masked-language-model head. Learned per-position queries cross-attend to a fused memory built from the masked-pass image tokens (13 of 49 patches of CLIP B/32, "the 25% view") and the masked caption encoded by CLIP's causal text tower; the fusion learners are bidirectional transformers [code]. It predicts every caption position in parallel, so it returns per-position marginals given (image view, visible tokens, caption length) [lead].
- *JSD, KL, TVD*: Jensen-Shannon distance, Kullback-Leibler divergence KL(human‖model), total variation distance between a model's and the annotators' label distributions. *EntCE*: the gap between model and human entropy per item. *RankCS*: agreement of the model's class ranking with the vote ranking (Baan et al. 2022). *Split-half ceiling*: agreement between two random halves of the annotators.
- *PMI score*: a likelihood divided by a prior likelihood (here, the caption's likelihood under a content-free image). *PLL*: pseudo-log-likelihood, the sum of log-probabilities of each token when it is masked.
- *AL-28*: ArtELingo-28, WikiArt paintings annotated in 28 languages. *LDL*: label distribution learning, predicting a per-item vote distribution.
- *Coverage@k*: the share of a painting's distinct readings that have at least one of its own captions in the top k retrieved.
- *Par-cap*: a plain parallel captioner (full clean image, 100% text masking, no partial caption context). *AR-cap*: an autoregressive captioner. *ML-15, ML-80*: `fusion_multilearner` at 15% and 80% text masking.

## 0. Summary

1. **Established.** A decoder's slot distribution can split between two readings and be tilted by visual or textual context, but only for autoregressive multimodal LLMs on illusion stimuli (Takatsuki et al. 2026: reversal in 37 to 44 of 50 anagrams), while their free outputs mostly commit to one reading (Panagopoulou et al. 2024; Agate 2026: caption dominance 0.558).
2. **Established.** The protocols for model spread against vote distributions (JSD, KL, TVD, EntCE; Nie 2020, Baan 2022, Lan 2025), and a consistent direction: models are more peaked than people (BEiT3 KL 1.701, brought to 0.647 by one global temperature; generator sense entropy 0.10 against 0.47 for humans).
3. **Established.** Generative scores work as re-rankers. They need a prior correction only when they rank the modality they generate: raw caption likelihood gave COCO image-to-text R@1 19.7 against ITM's 71.9, but text-to-image 55.6 against 54.8 (Lin et al. 2024).
4. **Contradictions resolved by the verifiers.** MixGen's ablation cannot show that the concatenated caption matters. Both-at-once reports do occur (Raven-Bear: 43 to 51% of beam mass). ArtECulture's agreement figures come from an agreement-filtered set. AL-28 has about 100 votes per painting, not 140.
5. **Open.** No source tests a masked cross-modal decoder's distribution against human readings, against a dual encoder, or at retrieval time. No source trains with pixel blends and an either-one caption target under masked reconstruction, or uses blends as ambiguity probes.
6. **Novelty risk after verification.** S1 medium, S2 low to medium, S3 low, S4 medium, S5 low, S6 low; overall medium-low [reader-inferred]. The risk is concentrated in one reviewer question: what does masking add over a captioner's next-token probabilities?
7. **First three designs** [reader-inferred]:
   - (D1+D2) a zero-training COCO probe suite, about 8 GPU-h: decoder scoring on ECCV Caption with prior correction, and a blend probe with exactly known readings. It builds the readout code every later design needs.
   - (D4) an ArtELingo triad, about 31 GPU-h: contrastive, ML-80 and Par-cap, with an emotion slot in the MLM target, scored against held-out AL-28 vote distributions (median 108 votes per painting).
   - (D6+D7) masking-specific readouts on those checkpoints, about 3 GPU-h: view sampling and context-selected readings. These are the only measurements where masking can beat a captioner by construction.
8. **Budget.** GPU-hours are not the constraint: the full sequence in Section 7 uses about 126 to 155 of about 400 (up to about 200 with COCO captioner mirrors). Code and review time are.

## 1. Thematic synthesis by H-b sub-claim

### 1.1 S1: the decoder's predictive distribution holds several readings

**Converging evidence comes from autoregressive models on illusions.**
- *Takatsuki et al. (2026; EMNLP 2026 per the authors).* They read LLaVA-family next-token probabilities after the prefix "I see a" on 50 synthetic duck-rabbit-like Visual Anagrams. The preferred reading reversed at some rotation for 37 to 44 of 50 anagrams, depending on the model, along a logistic psychometric curve (Sec 3.2; V3 rows 1, 5). Textual cues shifted the balance but could not induce an absent animal on unambiguous controls (Sec 3.2.2). The distribution at one slot therefore held two competing readings whose balance followed visual and textual context.
- *Their reports were mostly exclusive.* About 10% of beam mass named both animals near the ambiguity boundary, against at least 98% for two-animal controls (Sec 3.3, Fig 5; LLaVA-1.5-7B, 8 beams, continuations classified by an LLM judge; V3 row 2). Models answered "1 object" with p = 0.88 to 0.99 (Table 2).
- *The counter-case V3 added.* On the Raven-Bear figure-ground stimulus, three of five LLaVA models put 43 to 51% of beam mass on continuations naming both readings (App. B.4.1; V3 row 4).
- *Agate (2026; ICML 2026 workshop).* Both readings' sparse-autoencoder features were active in CLIP ViT-L layer-22 patch activations on 50 of 69 bistable stimuli (Sec 4.3). The split was uneven by group: 12/12 for duck-rabbit, 7/16 for face-vase, 1/4 for Schroeder stairs (V3 row 7). Sampled captions committed, with mean dominance |P(A) − P(B)| = 0.558 (Sec 4.1). A Necker cube gave 40/40 aspect-agnostic captions under a neutral prompt but a near-even split under forced choice (Sec 1).
- *Panagopoulou et al. (2024; CMCL workshop).* Twelve VLMs scored 29 bistable images in 3,364 manipulated versions. Most preferred one interpretation, and most named a single interpretation in free captions (Sec 4.7). Idefics and LLaVA-1.5-13b were more balanced.

**The readout matters.**
- *Sample entropy can under-report ambiguity.* Visual Semantic Entropy (Huy et al. 2026; ECCV 2026 per the authors) argued this happens when the visual representation is confident, and that perturbing the image helps. Its gain over the strongest baseline per model was +0.009 to +0.073 AUROC on A-OKVQA (Table 2; V3 row 12). The "visual confidence" account was tested with a LogitLens proxy on misanswered adversarial items, without human ambiguity labels (V3 row 13).
- *Generators are far more peaked than people over word senses.* Cekinmez et al. (2026; COLM workshop) measured normalised sense entropy of 0.10 for image models, 0.25 for text models and 0.47 for humans (Fig 2, Table 1). Models asked to predict the human distribution estimated about 0.73 (Table 2; V3 row 17).
- *Caption diversity is easy to buy and hard to interpret.* DiffCap's inter-distinct bigrams were 77.33% against BLIP's 14.75%, but BLIP decoded deterministically, so the gap measured determinism (V2 row 13). On COCO, beam search over one model reached oracle CIDEr-D 0.943 at M = 5, above stochastic multiple-choice learning's 0.922 (V2 row 9).
- *Masked and discrete-diffusion captioners work as captioners, but nobody measured their sample diversity.* DDCap reached CIDEr 117.8 against 112.8 for an autoregressive model (V2 row 20). Neither DDCap nor Masked Diffusion Captioning (MDC) reported sample diversity (V2 row 21). MDC's greedy confidence unmasking is nearly deterministic, and DDCap acknowledged repeated words.

**What it means for our decoder** [reader-inferred from [code] and [lead]].
- *It is a parallel predictor.* Our MLM decoder is a CapPa-style parallel predictor (CapPa fed all-[MASK] decoder inputs and dropped the causal mask; V2 row 26). It returns per-position marginals, not a joint distribution.
- *A reading-bearing slot can show a split directly.* An emotion word, or the identity of a blended source, gives the forced-choice readout that Agate and Takatsuki found more revealing than free sampling.
- *A whole reading needs iterative unmasking.* Independent marginals can mix two readings across positions; DDCap's repetition is a symptom of the same parallel decoding.
- *At 80% masking the decoder is already trained in the under-specified regime.* It predicts most of the caption from 13 of 49 patches: MLM loss rose 1.569 → 3.053 nats, perplexity 4.8 → 21.2 (Stages report, Table 7) [project]. A spread is expected there.

**Strength.** Moderate that an autoregressive MLLM's slot distribution can split between two readings on illusion stimuli: two 2026 papers, one model family, one stimulus class. Weak for natural images. None for masked cross-modal decoders: not found in N's Q1, Q11, Q17, Q29, Q33 and Q37, or in V3's three extra searches.

### 1.2 S2: that spread is measured against human readings

**The protocols converge.**
- *ChaosNLI* collected 100 labels per NLI item (Nie et al. 2020, Table 5).
  - On SNLI, models reached JSD about 0.23 (BERT-base 0.2345, RoBERTa-base 0.2294), against 0.383 for uniform chance and 0.0614 for humans. Their KL of 0.47 to 0.53 was close to chance (0.5457; V1 problem 4).
  - On MNLI, model KL ran 0.6652 to 0.8845 against 0.3559 for chance; on αNLI models were also at or worse than chance (V1 rows 32 to 35).
- *Baan et al. (2022, Table 1)* showed that ECE (expected calibration error) punishes a perfect oracle: 0.25, against RoBERTa's 0.14. They proposed the instance-level EntCE, RankCS and DistCE (a TVD). Temperature scaling cut ECE 0.14 → 0.03 while DistCE moved only 0.26 → 0.22 (V1 rows 37, 38).
- *Lan et al. (2025; AAAI 2025, N's "preprint" corrected by V3)* carried this to VQA.
  - BEiT3's answer distribution sat at TVD 0.436, KL(H‖M) 1.701 and EntCE 0.541 against 10 annotators (Table 1).
  - One global temperature fitted to humans (t = 2.0) brought KL to 0.647; scaling for accuracy raised it to 3.232 (Table 2; V1 rows 39 to 42; V3 row 15). Fit was best on the most uncertain third of items (BEiT3 KL 2.150 → 1.408 from the least to the most uncertain third; V1 row 40).
- *MultiEmo-Bench (2026 preprint)* gave an image-emotion ceiling: human split-half PLCC 0.700 (Table 4; 0.716 in Sec 3 under a different protocol), against 0.634 for GPT-4o. Its split halves hold about 10 votes each, so the ceiling understates a full 20-vote human (V1 rows 49 to 51).
- *Wang and Chan (2019)* validated set-level caption-diversity metrics against human diversity judgements at about 0.6 (Self-CIDEr, overall Spearman 0.617; V2 row 19).

**The direction is negative.**
- Every comparison found models too peaked, or no better than chance on divergence (Nie; Lan; Cekinmez's 0.10 against 0.47).
- CNNs did not show human-like uncertainty without soft labels (Peterson et al. 2019, digest-level).
- Uma et al. (2021) concluded that soft-label training helps when many high-quality judgements per item exist; with few or noisy ones, gold plus soft labels did best (conclusion only).

**The human side on art.**
- *ArtEmis* found a strong majority emotion for only 45.6% of paintings, and at least one positive and one negative emotion for 61% (Sec 3.2; V1 row 52). Every painting had at least 5 annotators.
- *ArtECulture's figures are not an outside check.* Its α of 0.306 to 0.324 and majority shares of 70 to 76% were computed on images kept only when each of three cultures had a majority (V1 problem 1), so they overstate ArtELingo-wide agreement. Even on that set, cross-culture Fleiss κ was 0.088.
- *AL-28 is the dense set.* The paper says about 140 votes per image, but its own totals give 100 (200K on 2,000; V1 problem 2). Our local CSV has 173,745 rows on 1,658 paintings, median 108 per painting [lead].

**Absence, narrowed by V1.**
- No paper compares a captioner's sampled meaning clusters or distribution with human caption or emotion distributions on still images or paintings. Wang and Chan compared scalar set diversity on COCO. VEGAS (2026, video) found that Gemini captions "cover only a small portion of the semantic variation among human annotations" (V1, claim G2-1).
- Yao et al. (2026; Findings of EACL) found 3.61 to 4.44 verb-sense clusters per imSitu image (Table 2; N's raw read), a reminder that single-gold scoring under-counts readings.

**What it means** [reader-inferred]. ArtELingo's own five votes per painting are too few for distribution-level tests.
- *A five-vote sample shows many labels even for a peaked painting* (computed): five draws from a uniform distribution over the 9 labels give 4.0 distinct labels on average. A painting with one emotion at 50% and the rest spread evenly still gives 3.2.
- *The observed counts are no higher.* In the retrieval test split the mean number of distinct emotions is at least 3.46 [lead counts; computed]. That count is over all of each painting's test captions, up to 12, so the comparison is approximate.
- *So the distinct-emotion count does not separate a reading spread from sampling noise.* The dense AL-28 votes are needed for distribution-level claims.

**Strength.** Strong for the protocols. Moderate for "models are too peaked": four settings, none of them a masked decoder. Absent for masked decoders on art.

### 1.3 S3: generative or masked models against contrastive dual encoders on ambiguity

**The one head-to-head on multi-reading images is weaker than first reported.**
- Panagopoulou et al. scored CLIP by similarity and 11 generative VLMs by label likelihood; most models of both kinds preferred one reading.
- N's statement that CLIP showed "almost opposite preferences" to the generative models was not found. That phrase compared BLIP-2 OPT with its Flan-T5 variants. Brightness and tint had minimal effects on every model, and CLIP varied most under rotation and prompt changes (V3 problem 1, rows 19 and 20).
- Pezzelle (2023) found that CLIPScore fell about 16 points when the subject became "they" and about 40 points for a fully underspecified caption. For 82 of 100 images, at least one unrelated caption outscored the valid underspecified one (Sec 3; one annotator, "for illustrative purposes"). There was no generative arm.

**Generative scoring wins on adversarial text and loses on retrieval.**
- *Lin et al. (2024) scored BLIP by caption likelihood.* On ARO it beat ITC and ITM (Relation 89.1 against 63.1 and 58.7, Table 1).
- *Much of that win is a language prior.* BLIP's decoder fed noise images already reached 87.6 on Relation, and text-only language models reached 95.0 to 98.6 on the two order sets (V1 row 4).
- *On retrieval it lost in one direction only.* COCO 5k image-to-text raw likelihood gave R@1 19.7 against ITM's 71.9, and 48.0 after a validation-tuned prior correction (Table 3a). Text-to-image it was on par: 55.6 against 54.8 (Table 3b; V1 row 8).
- *CapPa's blind decoder row* reached 83.7 on ARO VG-Attribution and 86.2 on VG-Relation (V1 row 29).
- *Masked Diffusion Captioning* beat CLIP on seven compositional columns (77.9 to 98.8, against 27.2 to 72.8), but its autoregressive counterpart scored about the same (76.0 to 98.4). The advantage belongs to generative scoring, not to masking (V2 row 23).
- *MASS's per-token PMI depended on the backbone.* It beat ITM with OFA (Winoground group 9.5 → 20.3) but barely with BLIP-2 (15.5 against 18.0) (Table 3; V1 rows 13, 14).

**Mechanisms on the contrastive side point both ways.**
- CLIP's text encoder favoured the first-mentioned object and its image encoder the largest object (CLIP Under the Microscope, CVPR 2025; abstract only). That is one route by which a pooled point keeps a dominant reading.
- Agate found both readings in a contrastive tower's patch features.

**Our COCO evidence concerns H-a, the encoder** [project].
- The 80% arm's gain sits in the image tower.
- It did not change VWSD: 52.92, +0.36 over 15% masking (p = 0.760).
- It did not grow with the number of valid matches: image-to-text +0.85 → +0.19 across R tertiles.
- Nothing has yet compared our decoder with our dual encoder.

**Fairness** [reader-inferred].
- "One point per item" does not mean one reading. A dual encoder's similarity profile over candidate readings (a prompt softmax) is also a distribution.
- Geometry limits it only when valid readings lie far apart in the embedding space; this is the PVSE and PCME argument covered by the 2026-10-03 review.
- S3 must therefore compare distributions with distributions. The contrastive model's prompt softmax and a soft-label probe on its embedding are the baselines.

**Strength.** Weak; the sub-claim is open.

### 1.4 S4: the decoder or its distribution used at retrieval time

**Cost and practical form.**
- *Per-pair costs.* Lin's caption likelihood costs one decoder pass per pair, as ITM does (Sec 3; V1 row 9). MASS doubles that for the prior: 0.55 s against 0.27 s per sample (App. D; V1 row 16). A diffusion score took 68 min, against about 4 min for OpenCLIP, on Flickr30K text-retrieval validation. Its authors called such models "slow retrievers applied to a small set of candidates" (V1 rows 20, 21).
- *The published generative retrieval numbers are re-rankings.* MASS re-ranked CLIP's top 20 (V1 problem 5). DiffusionITM's Flickr task re-ranked 10 candidates: the naive classifier scored 11.1 (chance is 10%), the corrected score 61.5, and CLIP RN50x64 71.9 (Table 1a; V1 problem 6).
- *UniDisc*, a masked joint image-text diffusion model, retrieved by likelihood among 16 random candidates. Its advantage over an autoregressive model was in joint retrieval only (0.64 against 0.17). Single-direction retrieval tied (0.85 against 0.85; 0.95 against 0.96), and there was no contrastive baseline (Fig 7; V3 problem 4).

**Scoring a masked model.**
- Salazar et al. (2020) masked each token in turn, one pass per token. They noted that a masked model can be fine-tuned to score without masking, in one pass (abstract; V1 row 26).
- Kauf and Ivanova (2023) showed that the original PLL inflates multi-token words. Masking a word's later subtokens too raised BLiMP accuracy by 0.2 to 1.3 points (Table 1; V1 rows 23, 24).
- Holtzman et al.'s (2021) prior-corrected score failed on HellaSwag, ARC-Easy and BoolQ (digest-level), which G1 read as prior correction hurting when the prior is informative.

**Sampling for coverage.**
- IC3 sampled 10 captions and summarised them with GPT-3. CLIP text-to-image MRR on COCO rose from 0.501 to 0.734 for BLIP (Table 5; V3 row 27), and noun recall against all references from 0.266 to 0.353 (Table 7; N's raw read).
- The summary merged readings into one caption.
- Candidate-set diversity correlated with MRR at only r = 0.059 to 0.094 (App. C.5; N's raw read).
- IC3's Flickr claim of "up to 84.46%" does not match its Table 6 (+47.5%), whose R@1 and R@5 columns are identical in the HTML (V3 row 28).

**Several outputs per input.**
- Winner-take-all training made ensemble members specialise, but beam search over one model matched it at M ≥ 4 (above).
- Set embeddings collapse without help. MaxMatch (ACL 2025) gained rSum 442.6 → 469.9 over SetDiv on Flickr30k with ResNet + Bi-GRU, and 514.8 → 520.8 with Faster R-CNN + BERT (V2 rows 33 to 35).
- The known fixes:
  - soft winners (Rupprecht et al. 2017, ε = 0.05; oracle LSP accuracy 59.7 → 62.8 with 10 hypotheses; V2 rows 27 to 30);
  - learned winner scores (Resilient MCL: best EMD, slightly worse oracle error; V2 row 31);
  - annealing (Annealed MCL: matched permutation-invariant training on WSJ0-mix, 16.85 against 16.88; V2 row 32).
- AMER generated several query vectors for text retrieval (abstract only).

**Both absences hold.**
- No ECCV Caption mAP@R, CxC or other multi-positive metric exists for a generative likelihood against a contrastive score from the same model (V1, claim G1-1). IG Captioner (ECCV 2024) compares a captioner's likelihood with CLIP on COCO and Flickr, but has no multi-positive metric and was not read in full.
- No retrieval method lets a reconstruction or generative loss choose which embedding slot a target belongs to (V2, G4 absence claim).

**For our decoder, costs differ from Lin's and Salazar's** [reader-inferred from [code]].
- At a training-like mask, scoring costs one pass per (pair, mask).
- At 100% text masking the logits depend only on (image view, caption length). One pass per (image, view, length) then scores every caption of that length by indexing, so full-gallery decoder scoring in both directions costs a few dozen gallery encodings.
- By Lin's and DiffusionITM's results, the prior term matters only when ranking the generated modality, which for us is image-to-text (Section 3, row 2).

**Strength.** Moderate for re-ranking feasibility and prior correction. Weak that decoder scores improve one-to-many metrics. None for per-reading sets from a masked decoder.

### 1.5 S5: masking teaches under-specification

**No direct evidence.**
- ProLIP's masked inclusion (B4) remained the only training-side use of masked inputs as under-specified views (N, closest list).
- Pezzelle (above) and FOCUS (ACL 2025; abstract only) found that dual encoders and VLMs mishandle underspecified text.
- D-MLM (MICCAI workshop 2025) predicted Gaussian parameters for masked tokens and patches with a KL loss and claimed ambiguity awareness. It measured only classification, grading and segmentation (V3 row 30).
- No source measured whether the masking ratio changes a decoder's calibration against human readings (not found in N's Q14, Q22, Q34, Q35).

**Two points narrow S5 for us** [reader-inferred].
- *Spread is learned anyway.* Cross-entropy is a proper scoring rule, so any decoder trained on several captions per image learns the mixture of readings for whatever contexts it is trained on. Masking changes which contexts those are (any subset of the caption, any 25% view of the image), not whether a spread is learned.
- *A captioner learns the same unconditional spread.* An autoregressive captioner learns P(first word | image), which is a fully under-specified context. What is specific to masking is arbitrary-subset conditioning and image-view under-specification.
- *Only the MLM head can show readings.* The pixel decoder is trained with mean-squared error on normalised pixels and predicts a conditional mean, so it cannot show several readings (N design note; [lead]).

**Our COCO results speak to the encoder, not the decoder** [project]. The gain needs high text masking and an MLM that reads the masked-pass image tokens. It behaves like vision supervision and shows no polysemy signature.

**Strength.** None direct. Exposure to "trivial" is high unless the masking-specific readouts (D6, D7) come out positive.

### 1.6 S6: mixing combined with masked reconstruction

**MixGen is the nearest precedent.**
- *Design.* MixGen (WACV 2023 workshop) blended two images at λ = 0.5 and concatenated their captions for a quarter of each batch.
- *Gain.* On its 3M-image ALBEF retrain, COCO fine-tuned RSUM rose 485.6 → 491.8 and zero-shot Flickr30K RSUM 552.8 → 558.1 (Tables 1, 2; V2 rows 4, 5).
- *The mixed pairs also went through ALBEF's MLM.* The batch was forwarded "without modifying any of their original training settings" (Sec 3; V3 row 33).
- *Its ablation does not support "the concatenated caption matters"* (Table 7, 1M images, one run per cell; V2 problem 8). Keeping a random half of the concatenated tokens (variant e) retained almost all of the gain (COCO-Ft 477.1 against 477.7). Beta-sampled λ alone fell below baseline. Differences of about 5 RSUM are within what one-run cells show.
- *The variant closest to "either one" (b) is not a proper test.* It gave one randomly chosen caption as the only target, a noisy single target rather than two valid ones (N10).

**Other mixing methods.**
- *PCME++* mixed images only (Mixup or CutMix on 25% of the batch, λ ~ Beta(2,2)). It set the matching label of a mixed image and its source caption to λ.
  - Alone it moved ECCV mAP@R 38.9 → 39.0 (RSUM 535.9 → 535.5); inside the full bundle it added +0.5 (39.6 → 40.1) (Table 3; V2 row 13).
  - Assigning 1 − λ to the other source's caption is our reading, not the paper's (V2 row 14).
- *TiMix's region mixing* had no effect without its patch-text alignment task: VQA 77.89 against 77.83, NLVR 82.59 against 82.66 (V2 row 21). Its retrieval gain over its own mPLUG baseline was +1.7 / +1.3 R@1 (V2 row 20).
- *Geodesic multimodal mixup* used mixed embeddings as hard negatives. At matched temperature 0.05 its COCO gain was +0.2 / +0.4 R@1 (V2 row 17).
- *Unmixing with reconstruction exists for images only.* MixMAE's dual reconstruction gained +0.6 ImageNet top-1 and +2.6 ADE20K mIoU (V2 row 24); MixedAE gained +0.3 to +0.4 over MAE (V2 row 26). Mix-up SSL predicted the blending ratio (ICME 2022; abstract only).

**Absences after V2.**
- The broad claim fails. M3ixup (Pattern Recognition 2025) mixes visual features, sentence embeddings and losses for captioning (CIDEr 133.8 → 135.3 offline, per its abstract). An IBM patent interpolates images and captions.
- The narrowed claims hold (V2 section 4): no found work combines pixel blends with an either-one caption target or a masked cross-modal reconstruction objective for polysemy, and none uses blended images as ambiguity probes.
- Takatsuki's Visual Anagrams show that synthetic ambiguous inputs with exactly known readings make usable decoder probes.

**Strength.** Weak that mixing helps one-to-many retrieval: PCME++'s isolated row is null.

### 1.7 Verdicts after verification

| Sub-claim | Status | Closest work | Novelty risk [reader-inferred] | What changed in verification |
|---|---|---|---|---|
| S1 distribution holds several readings | partly established, for autoregressive MLLMs on illusions; open for masked decoders and natural data | Takatsuki 2026, Agate 2026 | medium | Raven-Bear: both-at-once reports exist (V3 row 4) |
| S2 measured against human readings | protocols established, direction negative; open for masked decoders and art | Lan 2025, Nie 2020, Baan 2022, MultiEmo 2026 | low to medium | absence narrowed to sampled meaning clusters on still images (V1 G2-1); AL-28 votes are about 100, not 140 |
| S3 generative against contrastive on ambiguity | open | Panagopoulou 2024, Pezzelle 2023 | low | the CLIP-versus-generative "opposite preferences" claim was not found (V3 row 19) |
| S4 decoder at retrieval | components established, coverage use open | Lin 2024, MASS, UniDisc, IC3 | medium | MASS and DiffusionITM are re-rankers; UniDisc wins only joint retrieval |
| S5 masking teaches under-specification | open beyond ProLIP | ProLIP (B4), Pezzelle | low, but high "trivial" exposure | none |
| S6 mixing with masked reconstruction | open in narrowed form | MixGen inside ALBEF, PCME++ MSDA | low | MixGen ablation reading unsupported; M3ixup and the IBM patent break the broad absence |

## 2. Evidence table

| # | Claim | Sources | Setting | Numbers (locator) | Strength (reason) |
|---|---|---|---|---|---|
| 1 | A slot distribution splits between two readings, tilted by context | Takatsuki 2026 | LLaVA family, inference, 50 anagrams | reversal in 37 to 44 of 50 (Sec 3.2); about 10% beam mass on both vs at least 98% for controls (Sec 3.3); Raven-Bear 43 to 51% (App. B.4.1) | moderate: one family, illusions |
| 2 | Encoder features hold both readings, output commits | Agate 2026 | LLaVA-1.6 + SAE on CLIP ViT-L layer 22 | 50/69 co-active (Sec 4.3); dominance 0.558 (Sec 4.1); steering flips 33.3% / 0 of 7 (Sec 4.4) | weak: one model, workshop |
| 3 | Most VLMs prefer one reading of a bistable image | Panagopoulou 2024 | 12 VLMs, 3,364 images | figures only; five of six families prefer the same reading (Sec 5) | weak to moderate |
| 4 | Sample entropy under-reads ambiguity; image perturbation helps | VSE 2026 | 5 VLMs, VQA | +0.009 to +0.073 AUROC over the strongest baseline (Table 2) | weak for H-b: no human labels |
| 5 | Generators are too peaked over senses | Cekinmez 2026 | generators, text models, humans | 0.10 / 0.25 / 0.47; 0.73 when asked to predict humans (Fig 2, Tables 1, 2) | weak to moderate: workshop |
| 6 | Model vs vote distribution on NLI | Nie 2020 | fine-tuned encoders, 100 votes | SNLI JSD 0.23 vs chance 0.383 vs human 0.0614; MNLI KL 0.67 to 0.88 vs chance 0.36 (Table 5) | strong as protocol |
| 7 | ECE misleads; use instance-level measures | Baan 2022 | RoBERTa, ChaosNLI | oracle ECE 0.25; temperature: ECE 0.14 → 0.03, DistCE 0.26 → 0.22 (Table 1) | strong |
| 8 | A VQA model is too peaked; one temperature fixes much of it | Lan 2025 | BEiT3, LXMERT | KL 1.701 → 0.647 at t = 2.0; 3.232 when tuned for accuracy (Tables 1, 2) | strong as protocol, moderate as direction |
| 9 | Human ceiling for image emotion distributions | MultiEmo-Bench 2026 | 10,344 images, about 23 votes each | split-half PLCC 0.700 (Table 4) or 0.716 (Sec 3); GPT-4o 0.634 | moderate: preprint |
| 10 | Paintings carry several emotions | ArtEmis 2021 | at least 5 annotators, 81,446 artworks | 45.6% strong majority; 61% mixed valence (Sec 3.2) | strong, descriptive |
| 11 | Agreement is low even on a filtered set | ArtECulture 2026 | 6,792 artworks, about 4.5 votes per culture | α 0.306 to 0.324; cross-culture κ 0.088 | weak as an outside check (filtered) |
| 12 | Dense per-painting votes exist | AL-28 2024; local CSV | 1,658 local paintings | about 100 per painting (paper says 140); local median 108 [lead] | moderate |
| 13 | Raw generative scores fail only when ranking the generated modality | Lin 2024; DiffusionITM 2023 | BLIP zero-shot; Stable Diffusion, 10-way | i2t R@1 19.7 / 48.0 tuned vs ITM 71.9; t2i 55.6 vs 54.8 (Table 3); naive image retrieval 11.1 = chance (Table 1a) | strong for one model each |
| 14 | Generative wins on adversarial text are partly a language prior | Lin 2024; CapPa 2023 | BLIP; Cap | blind BLIP 87.6 on ARO Relation; text LMs 95.0 to 98.6 on the order sets; CapPa blind row 83.7 (VG-Attribution) | moderate |
| 15 | Masked and autoregressive captioners score alike | MDC 2025 | ViT-B/16, 10M pairs | MDC 77.9 to 98.8, ARC 76.0 to 98.4, CLIP 27.2 to 72.8 (tab:vlc); linear probe 83.1 vs 83.3 | moderate |
| 16 | Per-token PMI with a null image | MASS 2025 | OFA, BLIP-2 | Winoground group OFA 9.5 / 15.8 / 20.3 (ITM / TL / MASS); BLIP-2 15.5 / 11.0 / 18.0 (Table 3); 2 passes | moderate, backbone-dependent |
| 17 | Likelihood retrieval from a masked joint model | UniDisc 2025 | from scratch, 16 random candidates | joint 0.64 vs 0.17; single-direction tied (Fig 7) | weak |
| 18 | Sample-and-pool helps retrieval but merges readings | IC3 2023 | BLIP, OFA, BLIP-2 | COCO MRR 0.501 → 0.734 (Table 5) | moderate |
| 19 | Winner-take-all vs one model with beam search | sMCL 2016 | COCO captioning | oracle CIDEr-D 0.922 (sMCL) vs 0.943 (beam) at M = 5 | moderate |
| 20 | Set embeddings collapse; assignment fixes help | MaxMatch 2025; Rupprecht 2017 | Flickr, COCO; pose | rSum 442.6 → 469.9 over SetDiv; oracle LSP 59.7 → 62.8 | moderate |
| 21 | CLIP penalises valid underspecified captions | Pezzelle 2023 | 100 COCO pairs | about −16 and −40 CLIPScore; 82 of 100 images (Sec 3) | weak: illustrative |
| 22 | Blend + concatenation (through MLM) helps retrieval | MixGen 2023 | ALBEF, 3M and 1M images | COCO fine-tuned RSUM +6.2 (3M); 1M ablation within one-run noise (Table 7) | weak to moderate |
| 23 | Image mixing with a soft label leaves ECCV unchanged | PCME++ 2024 | CLIP B/32 fine-tune | mAP@R 38.9 → 39.0 alone; +0.5 in the bundle (Table 3) | moderate (a null) |
| 24 | Unmixing with reconstruction helps image SSL | MixMAE, MixedAE 2023 | ImageNet | +0.6 top-1 / +2.6 mIoU; +0.3 to +0.4 | moderate, image only |
| 25 | Our 80% arm shows no polysemy signature on COCO | [project] | B/32, COCO | mAP@R 37.94 vs 37.24 (15%) vs 37.11 (contrastive); VWSD +0.36 (p = 0.760); no growth with R | moderate: 3 to 5 seeds |

## 3. Contradictions and resolutions

| # | Claim A | Claim B | Resolution |
|---|---|---|---|
| 1 | G5: MixGen's variants (b) and (c) show the concatenated caption matters | V2: variant (e), half of the concatenated tokens, keeps the gain (477.1 vs 477.7); one run per cell | **Resolved against A.** MixGen shows a gain for the whole recipe, not for any one ingredient. "Either one" was never tested properly: (b) used a single random caption as the only target. |
| 2 | G1 closest list: raw likelihood fails on COCO and Flickr retrieval | Lin Table 3b: text-to-image raw 55.6 vs ITM 54.8. DiffusionITM: the naive score is at chance for image retrieval | **Resolved by direction** [reader-inferred mechanism, consistent with both papers]. When ranking candidates of the conditioning modality, the prior term is constant per query and cancels; when ranking the generated modality, it varies and dominates. Our MLM generates text, so image-to-text needs the correction and text-to-image does not. |
| 3 | N: Takatsuki's reports are exclusive (about 10% beam mass on both) | V3: Raven-Bear 43 to 51% on three of five LLaVA models | **Resolved conditionally.** Exclusivity held for object-identity bistability, not for this figure-ground stimulus, and varied by model. S1's "both at once" form is not ruled out. |
| 4 | G3: ArtECulture is an outside check on ArtELingo agreement (α 0.31; 70 to 76% majority) | V1: it keeps only images with a majority in all three cultures | **Resolved against A.** Its figures describe a filtered, more agreeable set. Use ArtEmis's 45.6% majority for unfiltered data. |
| 5 | AL-28 text: about 140 votes per painting | The paper's own totals: 200K on 2,000 = 100; local CSV median 108 | **Resolved: about 100.** Whether the 2,000 equal ArtELingo's highest-disagreement set D is not stated (V1 problem 2). |
| 6 | G2: ChaosNLI models "barely beat or lose to chance" | V1: SNLI JSD 0.23 vs chance 0.383 | **Resolved by dataset.** SNLI JSD is clearly better than chance and its KL is near chance; αNLI and MNLI are at or worse than chance. |
| 7 | G2: Lan Table 1 vs Table 2 LXMERT KL (1.886 vs 1.771) | V1: Table 2 is after accuracy-oriented scaling | **No contradiction.** |
| 8 | N: CLIP showed opposite preferences to generative VLMs and was most sensitive to brightness and tint | V3: the phrase concerned BLIP-2 variants; brightness and tint had minimal effects on all models | **Resolved against A.** The only supported S3 statement: most models of both kinds prefer one reading. |
| 9 | Generative scoring "beats CLIP" on compositionality (CapPa, MDC, Lin ARO) | Blind decoders score 80.7 to 87.6 on ARO's attribution and relation sets; text LMs 95.0 to 98.6 on the order sets; MDC's autoregressive twin scores the same | **Resolved conditionally.** The win is generative scoring of adversarial text with a language prior, not masking and not image understanding. |
| 10 | G1: MASS and DiffusionITM give COCO and Flickr retrieval numbers | V1: CLIP top-20 and 10-way re-rankings | **Resolved.** They are not comparable with Lin's full-gallery figures; they show that re-ranking is feasible. |
| 11 | N: UniDisc retrieves better than an autoregressive model | V3: only joint retrieval (0.64 vs 0.17); single-direction tied | **Resolved: the advantage is narrow.** |
| 12 | sMCL as a precedent for several captions per image (0.922 vs 0.831) | Beam search over one model: 0.943 at M = 5 | **Resolved conditionally.** Winner-take-all beat beam search only at M ≤ 3, or with a fine-tuned CNN (1.184). |
| 13 | VSE: sample entropy under-reads ambiguity | Wang and Nalisnick (abstract): sample diversity tracks ambiguity | **Unresolved.** VSE's account was tested with a proxy on misanswered adversarial items; Wang and Nalisnick were read only as an abstract. Practical rule: perturb the input (view sampling) rather than rely on token sampling alone. |
| 14 | MultiEmo human ceiling 0.700 (Table 4) | 0.716 (Sec 3) | **Different protocols.** 0.700 compares halves of about 10 votes; quote the protocol with the number. |
| 15 | G5: PCME++ is "either one" (a soft label for each source caption) | V2: the paper sets m = λ for the mixed image and its caption and is silent on 1 − λ | **Our reading, not theirs.** |
| 16 | IC3 text: up to 84.46% Flickr gain for BLIP | Table 6: +47.5%; R@1 and R@5 columns identical | **Unresolved.** Suspected table error; cite the Table 5 COCO cells only. |

## 4. Gap analysis

Ranked by how central each gap is to H-b, with its exposure to "so what?" [reader-inferred].

1. **A masked decoder's reading distribution against dense human distributions on natural multi-reading images**, with a captioner baseline and a contrastive baseline (S1, S2, S3).
   - *Centrality:* highest.
   - *Exposure:* medium to high. Predicting an image's emotion histogram is established label distribution learning (SAMNet, DDES; MultiEmo for MLLMs), so a decoder's emotion slot is a new way to produce a known quantity unless it adds something a soft-label probe cannot.
   - *What would fill it:* D4, D5.
2. **Masking-specific abilities: readings selected by partial caption context, and spread across image views** (S1, S5).
   - *Centrality:* high. This is the only place where "masking adds over a captioner" can show by construction.
   - *Exposure:* low to medium if positive. The nearest precedent is Takatsuki's textual cues for an autoregressive MLLM on illusions.
   - *What would fill it:* D6, D7, and D2 for synthetic inputs.
3. **The decoder used at retrieval to cover several readings** (S4).
   - *Centrality:* high for the "usable" half of H-b.
   - *Exposure:* medium. Decoder re-ranking is known, and coverage@k on ArtELingo is new, but no third-party ArtELingo retrieval numbers exist at all (V1, claim G3-1), so there is nothing to compare against.
   - *What would fill it:* D1, D8.
4. **Synthetic ambiguity with exact readings (blends) as an instrument for decoders against dual encoders** (S1, S3, S6).
   - *Centrality:* medium; it is cheap and controlled.
   - *Exposure:* medium to high, since blends are not natural ambiguity. Its value is as an instrument.
   - *What would fill it:* D2.
5. **Masked-decoder scoring on ECCV mAP@R against the same model's contrastive score** (S4; the G1-1 absence holds).
   - *Centrality:* medium.
   - *Exposure:* medium, since COCO multiplicity is paraphrase-level.
   - *What would fill it:* D1.
6. **Either-one mixing with masked reconstruction** (S6).
   - *Centrality:* medium.
   - *Exposure:* high. Augmentation papers rarely survive "is it just regularisation?", and PCME++'s isolated mixing row is null.
   - *What would fill it:* D10, D11.
7. **Set embeddings whose slots are chosen by a generative loss** (S4; the absence holds).
   - *Centrality:* medium.
   - *Exposure:* medium; collapse risk is high.
   - *What would fill it:* D9.
8. **A first ArtELingo retrieval benchmark, contrastive and masked** (evaluation).
   - *Centrality:* low for H-b, but it comes free with D4 and also gives H-a evidence on data built for several readings.

## 5. Measurement toolkit for H-b

### 5.0 What counts as one reading on ArtELingo (the author's open question)

**Recommendation** [reader-inferred]: the annotator's emotion label is the primary reading unit; content clusters of the explanations are a secondary unit; each caption is the retrieval positive, not a reading.

**Why the emotion label.**
- It is assigned by a human, not by a model.
- It is countable and gives a distribution per painting.
- The distribution-level metrics and their human-ceiling protocols already exist (Nie, Baan, Lan, MultiEmo).
- A dense reference exists: AL-28, median 108 votes per painting [lead].

**Its weaknesses.**
- "Something else" is a catch-all, 11.7% of ArtEmis labels (V1 row 52).
- The prior is skewed toward positive emotions: ArtEmis v2 found 62% positive against 26% negative (V1 row 83); locally, contentment dominates [lead].
- With five votes, the distinct-label count mostly reflects sampling (Section 1.2).
- It is an affective reading of the viewer, so two annotators with the same emotion can read different content.

**Why not the alternatives as primary.**
- *Content clusters* show readings beyond emotion, but they are partly model-defined: the entailment model decides the clusters (Kuhn et al. 2023), and Yao et al.'s clusters came from MLLM predictions. Use them as a secondary unit.
- *Each caption as a reading* turns paraphrase into "readings". Much answer disagreement is wording (Bhattacharya et al. 2019, digest-level).

**Keeping MultiMAE distinct from CoSiR.** Emotion is read out as a latent; it is never a query condition.

### 5.1 Human side

| Measure | Data | Pitfalls (source) |
|---|---|---|
| Emotion histogram per painting, 5 or more votes | ArtELingo English test, retrieval split 4,975 paintings (up to 12 captions in the full test JSON) [lead] | Too few votes for distribution metrics (Uma 2021; SAMNet's 8 to 11 votes had no ceiling, digest-level). Compare models with the noise floor: the expected JSD between two 5-vote samples of one distribution (computed per painting by resampling). |
| Dense emotion histogram | AL-28 local CSV: 1,658 paintings, median 108 votes, 10 labels including "other" [lead] | 1,160 of these paintings are in ArtELingo train: hold them out of training (1.9% of the 61,402 train paintings; computed), or use only the 439 test-or-unseen ones. The 10-to-9 label mapping is not established here. The votes pool 28 languages, and per-language distributions cluster (AL-28 Fig 4; ArtECulture cross-culture κ 0.088 on a filtered set). How the 2,000 were chosen is not stated. |
| Human ceiling | split-half reliability of the dense votes, MultiEmo protocol (10 random splits) | Half-sample ceilings understate a full sample (V1 row 51). A second human reference is how well the English 5-vote histogram predicts the remaining multilingual votes, the bar a model trained on English should approach [reader-inferred]. |
| Several captions per painting | 5 per painting in the retrieval split | Paraphrase versus reading (Bhattacharya; Self-CIDEr is lexical and correlates about 0.6 with human diversity judgements, V2 row 19). |
| COCO readings | 5 captions, ECCV Caption positives; blends with known readings | Mostly paraphrase-level multiplicity (brief); blends are out of distribution for models never trained on them. |

### 5.2 Model side

| Readout | How | Pitfalls (source) |
|---|---|---|
| Emotion-slot marginal | Emotion word as the first token of the MLM target; mask the whole explanation; average P(e) over K image views and over caption lengths | **Length leak:** the decoder knows the caption length from padding [lead]; average over the training length distribution, never over the annotator's own length. **Train-test shift:** at 80% training, a fully masked caption is rare. Emotion words must be single tokens (Kauf: multi-token inflation). |
| Caption score under K masks (PLL-style) | mean log p of masked tokens over K random masks at the training ratio; or the 100%-masking one-pass parallel score | **Prior:** correct image-to-text only, with a null image (MASS's black image; Lin's noise image) and α tuned on validation (Lin's α* = 0.819 on COCO, V1 row 1). PMI hurts when the prior is informative (Holtzman, digest-level). Normalise by the number of scored tokens. **Cost:** K passes per pair, so re-rank only, except at 100% masking. |
| Iterative-unmasking samples | MaskGIT-style: predict all, keep the most confident, re-mask the rest, under a schedule and temperature | Greedy confidence unmasking is nearly deterministic (MDC); parallel marginals mix readings and repeat words (DDCap); temperature matters (Kuhn: best 0.5). |
| View-sampling spread | redraw the 13-patch view K times; split total entropy into mean within-view entropy plus between-view disagreement (mutual information between view and reading) [reader-inferred] | The spread may track which salient region was dropped rather than readings. Control: give the dual encoder the same views (`vision.encode` accepts a patch subset [code]) and give Par-cap views it never trained on. |
| Semantic clustering of samples | semantic entropy (Kuhn 2023) with an entailment model, or an emotion classifier on the samples | Semantic entropy was built to flag errors, which is the opposite reading from H-b's benign spread. VSE: confident visual features suppress sample diversity. The classifier's accuracy (ArtEmis: BERT 65.7% vs expert humans 60.3% on 250 utterances, V1 row 54) adds noise. |
| Dual-encoder distributions (baselines) | softmax over 9 emotion prompts with a fitted temperature; a soft-label linear probe on the frozen image embedding (trained on every annotator's label) | Leaving these out turns "one point cannot" into a strawman. The soft-label probe is the strongest LDL-style baseline. |

### 5.3 Agreement metrics

| Metric | Use | Pitfalls |
|---|---|---|
| JSD, KL(H‖M), TVD per painting | main S2 comparison on the dense set | KL needs smoothing for zero mass. Always report uniform chance and a **prior-only** model (training emotion frequencies for every painting), which the skewed prior makes strong. Fit one temperature per model on validation first: Lan's single temperature cut KL 1.701 → 0.647. |
| Spearman of per-painting entropies; EntCE; RankCS | does the model know *which* paintings are ambiguous? | A global temperature cannot create a cross-painting ranking, so this is the metric least fixable by calibration [reader-inferred]. |
| Stratified agreement | thirds of dense human entropy (Lan's uncertainty bins; Nie's agreement bins) | Bins defined from noisy votes regress to the mean; use the dense set. |
| Coverage@k (image-to-text) | share of a painting's distinct emotions with at least one own caption in the top k | Confounded with recall: report it against a permutation baseline (expected coverage of the same number of own captions drawn at random) and at matched R@k [reader-inferred]. Text-to-image on ArtELingo has one positive painting, so polysemy shows only in image-to-text. |
| Minority-reading recall | for emotions with at least 2 of 5 votes (or at least 20% of dense votes) that are not the majority: model mass above the prior, or retrieval of such a caption in the top k | Few qualifying paintings at 5 votes; prefer the dense set. |
| Seeds and intervals | bootstrap over paintings × seeds | The Stages report's bootstrap covered queries only; include seed variance. |

**Digest-level caution.** Every pitfall above that rests on Holtzman, Peterson, Uma's per-dataset figures, SAMNet or Bhattacharya rests on digest-level reads. They are directions, not numbers to quote.

## 6. Candidate designs

GPU-hours use the lead's estimates: an ArtELingo masked run is about 4 h and a contrastive run about 2.3 h, both untested; eval-only probes take minutes to an hour. Kill thresholds are proposals [reader-inferred]. "Seed noise" means the 95% interval over seeds and paintings includes zero.

| ID | Family | Sub-claims | GPU-h | Code | Priority |
|---|---|---|---|---|---|
| D1 | (a) zero-training COCO: decoder scoring on ECCV | S4, S3 | about 6 | moderate (probe library) | 1 |
| D2 | (a) zero-training COCO: blend probe with exact readings | S1, S3, S6 | about 2 | small on top of D1 | 1 |
| D3 | (a) zero-training COCO: spread vs human caption diversity | S2 (validation only) | at most 1 | small | 5 |
| D4 | (b) ArtELingo triad: contrastive, ML-80, Par-cap, emotion slot | S1, S2, S3 | 31 | moderate (loader, collate) | 2 |
| D5 | (b) ArtELingo decomposition: AR-cap, ML-15, ML-100v | S5, fair baseline | 36 | moderate (AR decoder) | 3 |
| D6 | (c) view sampling, eval-only | S1, S2 | 1 to 2 | small | 2 |
| D7 | (c) context-selected readings, eval-only | S1 | under 1 | small | 2 |
| D8 | (d) decoder re-ranking and coverage on ArtELingo, eval-only | S4 | 2 to 4 | moderate | 3 |
| D9 | (d) set embeddings with decoder-chosen slots | S4 | about 30 | large | after 10-10 |
| D10 | (e) mixing: either-one vs both-at-once | S6 | 13.5 to 27 | moderate | 4 |
| D11 | (e) unmixing | S6 | about 14 | moderate | after 10-10 |

### D1. Decoder scoring on ECCV Caption (zero training, COCO)

- **Hypothesis (S4, S3).** Used as a prior-corrected re-ranker of the dual encoder's top k, the 80% decoder raises ECCV mAP@R in image-to-text, more on queries with many positives.
- **Data.** COCO 5k test; the ECCV Caption subset (1,261 images, 1,332 captions; Stages report).
- **Models.**
  - 80% (5 seeds), 15% (3) and contrastive (3).
  - M1 clean at 80% (3): a full-image decoder, the closest existing thing to a captioner [lead].
  - `fusion_none` at 80% (3): its MLM reads text alone, so it is a learned language prior, the analogue of Lin's blind decoder.
- **Convincing baseline.** The same model's dual-encoder ranking. M1 clean's decoder answers "is it masking or just captioning?".
- **Scores.**
  - The 100%-masking parallel score: one pass per (image, view, length), K = 8 views.
  - The training-ratio score: K = 8 random 80% masks over the top 50.
  - PMI against a black or noise image for image-to-text, with α tuned on COCO val. Text-to-image uses the raw score.
- **Metrics.** ECCV mAP@R, R-Precision, R@1 by direction; COCO 5k R@K; R tertiles via the existing `stratified_eccv.py`.
- **Cost.** About 0.5 GPU-h per checkpoint × 12 (seeds 42 to 44 of four arms) ≈ 6 GPU-h. Code: a probe library (scoring, prior, re-rank, metrics hook) of roughly 250 lines plus tests.
- **Under H-b.** The prior-corrected re-rank is at or above the dual encoder on mAP@R, with the gain growing with R, and the 80% decoder beats M1 clean's despite M1 clean's lower MLM loss (2.898 vs 3.053).
- **Under "masking adds nothing".** The re-rank lowers mAP@R, as Lin's image-to-text 48.0 vs 71.9 suggests; M1 clean is at least as good; any gain is flat in R.
- **Kill.** If every α lowers image-to-text mAP@R beyond seed noise, with no relative gain in the high-R tertile, drop decoder re-ranking as a COCO retrieval use. Keep the scorer for ArtELingo coverage.
- **Novelty risk.** Medium for scoring itself (Lin, MASS, UniDisc, MDC, IG Captioner); low for a same-model ECCV comparison (the G1-1 absence holds).
- **Dependencies.** Access to the checkpoints; the probe library. The causal text tower means visible tokens after a masked slot reach it only through the fusion learners and the decoder [code].

### D2. Blend probe with exactly known readings (zero training, COCO)

- **Hypothesis (S1, S3, S6).** On a 50/50 pixel blend of images A and B, the decoder keeps mass on both sources' captions, and one or two visible tokens of A's caption select A. The dual encoder's single embedding retrieves one source more often than both.
- **Data.**
  - 1,000 COCO test pairs chosen for low cross-caption similarity.
  - A λ sweep {0.3, 0.4, 0.5, 0.6, 0.7}, for a psychometric curve like Takatsuki's rotation.
  - A patch-mix variant: each of the 49 patches drawn from A with probability s ∈ {0.25, 0.5, 0.75}, so the 13-patch view's composition varies.
- **Readouts.**
  - **Balance:** mean score of A's captions minus B's, at 100% masking over K views.
  - **Both-covered@k:** the share of blends whose top k over the 25k-caption gallery holds at least one caption of each source, for the dual encoder and for decoder re-ranking.
  - **Selection:** the shift in balance after revealing j = 1, 2, 3 content tokens of an A caption.
  - **View composition:** balance regressed on A's share of the visible patches.
  - Unblended A and B serve as controls.
- **Models.** Zero-shot CLIP B/32, contrastive, 15%, 80%, M1 clean. `fusion_none` is a sanity check: its balance should not depend on λ.
- **Cost.** About 2 GPU-h. Code small: about 60 lines for the blend set on top of D1.
- **Under H-b.** The decoder's balance is near zero at λ = 0.5, with both readings far above unrelated captions. The decoder's both-covered rate exceeds the dual encoder's. One or two tokens select. The 80% decoder's balance tracks view composition.
- **Under the null.** The dual encoder's both-covered rate matches or beats the decoder's, and both are one-sided, as Panagopoulou found for most models. M1 clean behaves like the 80% decoder.
- **Kill.** If the contrastive model's both-covered@10 is within 2 points of, or above, the best decoder's at every k, the "one point cannot" premise of S3 fails on synthetic two-reading inputs at B/32. Restate H-b without it.
- **Novelty risk.** Low: the absence of mixed images as ambiguity probes holds (V2).
- **Dependencies.** D1's library.
- **Caveat.** These checkpoints never saw blends.

### D3. Per-image spread against human caption diversity (zero training, COCO)

- **Hypothesis (S2, light).** Images whose five captions differ more get wider decoder distributions.
- **Data.** COCO 5k test. Human diversity is 1 minus the mean pairwise CLIP cosine of the five captions (already computed by the stratified analysis), and Self-CIDEr.
- **Model side.** The view-sampling spread; mean per-position entropy at 100% masking; the semantic entropy of 10 iterative-unmasking samples.
- **Baselines.** The entropy of the dual encoder's softmax over its top 50 captions, with a fitted temperature; M1 clean.
- **Metric.** Spearman per image, partial on caption length.
- **Cost.** At most 1 GPU-h; small code, since the sampler is needed later anyway.
- **Under H-b.** A positive correlation, larger for the decoder.
- **Under the null.** No difference.
- **Role.** Measurement validation only. COCO multiplicity is paraphrase-level, so a null says little about H-b; cap the effort.
- **Novelty risk.** Low: V1 narrowed the absence to meaning clusters on still images.

### D4. ArtELingo triad with an emotion slot (training)

- **Hypothesis (S1, S2, S3).** After temperature calibration on validation, ML-80's emotion-slot distribution matches the dense human distribution better than:
  - the contrastive model's prompt softmax and soft-label probe;
  - Par-cap's emotion slot.
- **Data.**
  - ArtELingo English train (308,723 captions on 61,402 paintings) minus the 1,160 AL-28 paintings.
  - Val: 15,531 captions on 3,119 paintings.
  - The retrieval test split (4,975 paintings × 5).
  - AL-28's 1,658 paintings as the held-out dense set [lead].
- **The emotion slot.** The MLM target is "<emotion> <explanation>" for every decoder, while InfoNCE sees the explanation alone. This keeps the retrieval task unchanged and keeps emotion out of the similarity, away from CoSiR's emotion-conditioned retrieval.
- **Models (3 seeds each).**
  - C: contrastive only.
  - ML-80.
  - Par-cap: M1 clean with 100% text masking and MAE weight 0, a CapPa-style parallel captioner. It is config work plus allowing a text ratio of 1.0, which `random_token_mask` currently rejects [code].
- **Baselines that make it convincing.**
  - prior-only;
  - C's prompt softmax;
  - a soft-label linear probe on C's frozen image embedding;
  - Par-cap;
  - AR-cap (D5) once it exists;
  - the AL-28 split-half ceiling and the English-5-vote reference (Section 5.1).
- **Metrics.** JSD, KL, TVD; entropy Spearman; RankCS; agreement by thirds of dense entropy. Secondary: R@K and coverage@k on the 4,975 split, which also give H-a evidence on data built for several readings.
- **Cost.** 3 × 2.3 + 6 × 4 = 30.9 GPU-h (computed).
- **Code.**
  - ArtELingo dataset class, data config and node copy (the WikiArt images take 26 GB): about 150 lines plus tests.
  - The emotion-prefix collate, which needs a second token sequence: small.
  - The emotion-slot readout and the probe baseline: small.
  - Measure the token-length distribution first. Explanations have a median of 14 words, and `data.max_text_len` = 32 may truncate a long tail; CLIP's limit is 77 [reader-inferred].
- **Under H-b.** After temperature calibration, JSD ranks ML-80 < Par-cap ≈ probe < prompt softmax < prior-only. ML-80 has the higher entropy correlation, with the largest gap in the most uncertain third.
- **Under the null.** ML-80, Par-cap and the probe are equal within seed noise, since all learn P(emotion | painting) by maximum likelihood from about five labels each. All sit well below the split-half ceiling.
- **Kill.** If ML-80 does not beat both Par-cap and the soft-label probe on JSD or entropy correlation, the masked decoder's part of S2 fails on the emotion readout. Continue only with D6 and D7.
- **Novelty risk.** Low to medium. LDL is established; a masked decoder against dense human distributions on art was not found.
- **Dependencies.** The ArtELingo loader; the AL-28 hold-out list; the label mapping; single-token emotion words.

### D5. ArtELingo decomposition: what in masking matters (training)

- **Hypothesis (S5, the fair baseline).** The decoder's spread and calibration depend on three things:
  - partial text context (ML-15 vs ML-80);
  - image under-specification (ML-100v, a 25% view with 100% text masking, vs Par-cap with the full image);
  - decoder type (AR-cap vs Par-cap).
- **Models (3 seeds each).**
  - AR-cap: a CoCa-style causal decoder over clean image tokens with InfoNCE. Its emotion is the first token, so its first step gives P(emotion | image).
  - ML-15.
  - ML-100v.
  - Optional: ML-50 as a dose point.
- **Metrics.** As D4, plus D6 and D7 on every arm.
- **Cost.** 9 × 4 = 36 GPU-h (computed); ML-50 adds 12. Code: the AR decoder, about 120 lines plus gradient and DDP tests and a `VARIANTS` entry; the rest is config.
- **Under H-b.** Calibration improves with text masking (15 → 80 → 100) and with image masking at fixed text masking (ML-100v vs Par-cap). AR-cap ≈ Par-cap.
- **Under the null.** All generative arms tie, or AR-cap leads.
- **Kill.** If AR-cap matches or beats every masked arm on D4 metrics, and D7 shows no image-conditioned benefit from partial context, report "masking adds nothing over a captioner" for S1 and S2.
- **Novelty risk.** Low: the controlled comparison is the contribution.
- **Dependencies.** D4's loader and readouts.

### D6. View sampling (eval-only, masking-specific)

- **Hypothesis (S1, S2).** Averaging the emotion slot over K redrawn 25% views gives a better-calibrated distribution than one full-image pass, and the between-view term is larger on paintings with high human entropy.
- **Models.**
  - ML-80 and ML-100v, for which 25% views are in distribution.
  - Controls: Par-cap on full images and on views it never trained on; C's prompt softmax and probe averaged over the same views, since any ViT can drop patches.
- **Metrics.** D4 metrics at K ∈ {1, 4, 16}; Spearman between the between-view term and human entropy on the dense set.
- **Cost.** 1 to 2 GPU-h; small code.
- **Under H-b.** K = 16 beats K = 1 and the full image for the masked arms. The between-view term correlates with human entropy beyond the prior-only model. View-averaging helps the dual encoder less.
- **Under the null.** View-averaging only adds noise, and the between-view term tracks dropped content, not readings.
- **Kill.** The correlation's interval includes zero on the 1,658 dense paintings.
- **Novelty risk.** Low to medium. VSE perturbed images in MLLMs for ambiguity detection; masked-view sampling for readings was not found.
- **Dependencies.** D4 (D5 for ML-100v).

### D7. Context-selected readings (eval-only, masking-specific)

- **Hypothesis (S1, the brief's own mechanism).** Given the painting and j visible tokens of one annotator's explanation, the emotion slot moves from the painting-level distribution to that annotator's emotion. Image and partial text combine.
- **Data.** ArtELingo test captions (up to 12 per painting). 98% of retrieval-split paintings have at least two distinct emotions (computed from [lead] counts).
- **Readout.** For j ∈ {0, 1, 2, 4, 8} visible content tokens (the M2b content-word list exists), the log-loss and accuracy of the annotator's emotion. Two visibility patterns: a prefix and a random subset.
- **Baselines.**
  - The same model with a null image, which isolates the image's contribution.
  - AR-cap via Bayes: P(e | image, first j words) ∝ P(e | image) · P(words | image, e), 9 passes per item. This works for prefixes only, which makes prefix contexts the fair comparison. Random-subset contexts are masking-only.
- **Cost.** Under 1 GPU-h; small code.
- **Under H-b.** ML-80 matches AR-cap on prefixes, beats its own null-image version at j ≥ 2, and gives well-calibrated curves on random subsets.
- **Under the null.** AR-cap beats ML-80 on prefixes, leaving only arbitrary-subset conditioning, which may not matter for retrieval.
- **Kill.** If ML-80 with the real image is no better than with the null image at j ≥ 2, the selection is text-only, not cross-modal.
- **Novelty risk.** Medium-low: Takatsuki's textual cues did this for an autoregressive MLLM on illusions.
- **Constraint.** Emotion is read out here, never supplied as a condition.

### D8. Decoder re-ranking and coverage on ArtELingo (eval-only)

- **Hypothesis (S4).** At matched R@k, re-ranking each painting's top 50 explanations raises coverage@k and minority-reading recall over the dual encoder, more with ML-80 than with Par-cap or AR-cap.
- **Variants.**
  - (a) Re-rank by the prior-corrected decoder score.
  - (b) Coverage-aware re-rank: allocate the top k across emotions in proportion to the decoder's P̂(e | painting), using each candidate's P(e | painting, candidate) from the emotion slot.
  - (c) Sample-then-embed, IC3-like but kept as a set: 10 iterative-unmasking samples are embedded by the text tower, and the painting scores a caption by max-similarity over {pooled image, samples}.
- **Baselines.** The dual encoder; the same re-rankers with Par-cap and AR-cap decoders; the dual encoder diversified by the soft-label probe's emotion posteriors, the contrastive version of (b).
- **Metrics.** Image-to-text R@K, coverage@k against the permutation baseline, minority-reading recall; text-to-image R@K for completeness.
- **Cost.** 2 to 4 GPU-h; moderate code (re-rank, coverage metrics, sampler).
- **Under H-b.** Coverage rises at matched recall, and ML-80 beats the captioners.
- **Under the null.** The whole coverage gain comes from emotion diversification, which the probe on the contrastive model gives equally.
- **Kill.** The probe-diversified dual encoder matches the decoder's coverage within seed noise.
- **Novelty risk.** Medium: decoder re-ranking and IC3 exist, and no third-party ArtELingo retrieval exists to compare against.
- **Constraint.** Emotion is a latent for diversification, never a query condition.

### D9. Set embeddings with decoder-chosen slots (training; after 2026-10-10)

- **Hypothesis (S4).** K = 4 image slot embeddings, each caption assigned to the slot whose slot-conditioned decoder reconstructs it best, raise coverage@k without lowering R@1. Assignment uses relaxed winner-take-all with ε = 0.05 (Rupprecht), InfoNCE on the assigned slot, and retrieval by max over slots.
- **Baseline.** The same slots assigned by embedding similarity (DivE- or MaxMatch-style), and K = 1.
- **Cost.** 2 arms × 3 seeds × about 5 h ≈ 30 GPU-h. Code large: a slot head, slot-conditioned decoding, assignment, DDP tests.
- **Under H-b.** Slots specialise by reading (emotion purity per slot is above chance) and coverage rises.
- **Under the null.** Slots collapse, or behave like similarity-assigned slots.
- **Kill.** One slot wins more than 90% of assignments despite the relaxation.
- **Novelty risk.** Low (the absence holds), but implementation risk is high.

### D10. Mixing: either-one vs both-at-once (training)

- **Hypothesis (S6).** Training on 50/50 blends whose MLM target is one source's caption at 80% masking, so visible tokens select the source, teaches the decoder to keep alternative readings. The effect transfers to natural paintings beyond both-at-once and beyond no mixing.
- **Arms (3 seeds each).**
  - X0 = ML-80, from D4.
  - X2 either-one: 25% of each batch blended (MixGen's B/4, PCME++'s 25%). Blends are kept out of InfoNCE, since a batch-softmax loss cannot take soft targets (PCME++).
  - X1 both-at-once: blend plus concatenated captions. On ArtELingo two explanations overflow 32 tokens, so run X1 with a longer `max_text_len` or on COCO.
- **Metrics.** D4 metrics on natural data; D2 on the trained models (now in distribution); D8 coverage.
- **Cost.** X2 alone is 13.5 GPU-h; with X1, 27 (mixing overhead untested). Code moderate: a blend collate, routing blends out of InfoNCE, the concatenation path.
- **Under H-b.** X2 beats X0 on dense entropy correlation and coverage. X1 matches or trails X0 on reading metrics, because a conjunction teaches "both", not alternatives.
- **Under the null.** X2 ties X0 on natural data and improves only the blend probe, trivially.
- **Kill.** X2 within seed noise of X0 on every natural metric: mixing is then a probe, not a lever.
- **Novelty risk.** Low; the neighbours are MixGen inside ALBEF, M3ixup and the IBM patent. The prior that mixing helps one-to-many retrieval is weak (PCME++ 38.9 → 39.0).
- **Dependencies.** D2's zero-shot result (if decoders already keep both readings on blends, X2's margin is smaller); D4.

### D11. Unmixing (training; after 2026-10-10)

- **Hypothesis (S6).** Reconstructing both captions from a blend makes the decoder attribute content to sources and keep readings apart. The decoder runs twice with a learned source-index query, optionally with a ratio-prediction head as in Mix-up SSL.
- **Baseline.** X2.
- **Evidence it might work.** For images only: MixMAE +0.6 top-1 and +2.6 mIoU; MixedAE +0.3 to +0.4.
- **Caveat.** MixMAE unmixes spatially disjoint patches, which is easy; pixel blends are not.
- **Cost.** About 14 GPU-h; moderate code.
- **Kill.** As D10.
- **Novelty risk.** Low.

## 7. Recommended sequence and decision points [reader-inferred]

**Constraints.**
- Six A6000s on node405 and node411 until 2026-10-10 01:30 [lead].
- All GPUs are busy until about 02:45 on 10-07, and three until about 07:00.
- About 400 GPU-h are free after the queue [lead].
- From about 12:00 on 10-07, six GPUs give about 369 GPU-h to the deadline (computed).
- Before any local GPU use, check the shared RTX 3090 and take the lock, per the shared-resources rule.

**Phase 0 (10-06 evening → 10-07 07:00): code, then the COCO probes.**
- Write and test:
  - the probe library (D1, D2);
  - the ArtELingo loader with the emotion-prefix collate and the AL-28 hold-out list;
  - the token-length check;
  - the change allowing a text ratio of 1.0.
- Copy the WikiArt subset to the nodes.
- Run D1 and D2 on the first free GPU (about 8 GPU-h).

**Decision point 0 (10-07 morning).**
- If decoder re-ranking hurts image-to-text mAP@R and the dual encoder covers both blend readings at least as often as the decoder, S3 and S4 are weak on COCO. Still run D4, since S2 does not depend on them, but move D10 down.
- If the decoder keeps both readings clearly better, move D10 up.

**Phase 1 (10-07 morning).** An ArtELingo smoke run (`train=debug`) and a one-epoch timing pilot (about 2 to 4 GPU-h), to replace the untested 4 h and 2.3 h estimates.

**Phase 2 (10-07 about 12:00 → 20:00).**
- D4's nine runs: 30.9 GPU-h in two waves.
- Meanwhile, write AR-cap and the blend collate.

**Decision point 1 (10-07 about 17:00, after seed 42 of each D4 arm).**
- Sanity-check the emotion slot, fit temperatures, take a first JSD against AL-28.
- If every arm's slot collapses to the prior-only distribution, fix the readout before launching more runs.

**Phase 3 (10-07 20:00 → 10-08 morning).** D5 (36 GPU-h) and D10's X2 (13.5 GPU-h) in two waves.

**Phase 4 (10-08).** Eval-only D6, D7 and D8 on every checkpoint (about 6 to 10 GPU-h); D3 on COCO (1 GPU-h).

**Decision point 2 (10-08 evening).** Read the deciding contrast: ML-80 against the better captioner (Par-cap or AR-cap) on D4, D6, D7 and D8 metrics.
- If any masking-specific metric shows an effect beyond seed noise, run seeds 45 and 46 for ML-80, the better captioner and (if positive) X2: 4 to 6 runs, 16 to 27 GPU-h.
- If everything is null, write up the null and stop launching.
- Optional either way: ML-50 (12 GPU-h); COCO mirrors of Par-cap and AR-cap so that D1 and D2 get their fair baseline (3 × about 7 h each).

**Buffer (10-09 evening → 10-10 01:30).** No new 4 h launches after about 18:00 on 10-09; final evaluations and pulling results only.

**Total.** About 8 (D1, D2) + 3 (pilots) + 31 (D4) + 36 (D5) + 13.5 to 27 (D10, without or with X1) + 7 to 11 (D3, D6 to D8) + 16 to 27 (confirmation) + 12 (optional ML-50) ≈ 126 to 155 GPU-h (computed). The two optional COCO captioner mirrors add about 42 more. That is well inside about 400. The binding constraints are code correctness and review time.

**Leave for after 10-10.**
- D9 and D11.
- Cross-lingual readings.
- A second backbone.
- Any use of emotion as a retrieval condition, which belongs to CoSiR.

## 8. Strongest counter-arguments

1. **"S1 is a property of maximum likelihood, not of masking."**
   - *The case:* any captioner trained by cross-entropy on several captions per image learns the mixture of readings (Section 1.5).
   - *What settles it:* D4 and D5. ML-80 against Par-cap and AR-cap on dense-distribution metrics, and the masking-only readouts D6 and D7. If the captioners tie on everything, H-b's "masked" qualifier is unsupported.
2. **"Label distribution learning already does this."**
   - *The case:* a soft-label probe on CLIP features predicts emotion histograms; the decoder's emotion slot is an expensive classifier, and LDL is established (SAMNet, DDES, MultiEmo).
   - *What settles it:* the probe baseline in D4. If the probe matches the decoder, S2's novelty shrinks to readings as joint (emotion, explanation) structures (D7) and to retrieval coverage (D8).
3. **"The readings are annotator noise."**
   - *The case:* five votes per painting; ArtEmis majority only 45.6%; α about 0.31 even on an agreement-filtered set; five-vote distinct counts look like sampling (Section 1.2).
   - *What settles it:* dense AL-28 votes, the split-half ceiling, beating the prior-only model, and stratifying by dense entropy. Agreement that rises with dense entropy and approaches the ceiling would rule noise out.
4. **"Dual encoders are not one reading."**
   - *The case:* a pooled point has a similarity profile over readings, and Agate found both readings in contrastive patch features.
   - *What settles it:* D2's both-covered rates, and D4's prompt-softmax and probe baselines.
5. **"Generative retrieval loses."**
   - *The case:* Lin's image-to-text 19.7 raw and 48.0 tuned against ITM's 71.9; the 2026-10-03 review left open whether re-ranking helps ECCV mAP@R.
   - *What settles it:* D1 and D8, reported as coverage at matched recall rather than R@1.
6. **"Your own data say masking is vision supervision."**
   - *The case:* the COCO 80% gain lives in the image tower, with no VWSD gain and no growth with R [project].
   - *Answer:* that is H-a. H-b is about the decoder, and it is falsified separately if D4 to D7 are null.
7. **"Half-answered already."**
   - *The case:* Takatsuki shows autoregressive slot distributions splitting between readings; UniDisc and MDC show masked likelihood scoring.
   - *Answer:* the remaining contribution must be the dense human comparison on natural images plus the masking-specific abilities.
8. **"Too close to CoSiR."**
   - *The case:* the emotion slot puts emotion into text.
   - *Answer:* emotion is only ever predicted, never supplied as a retrieval condition, and InfoNCE never sees it (D4, D7, D8).
9. **"Unpublishable."**
   - *The case:* one dataset, B/32, and a prior direction (models too peaked) that predicts a null.
   - *Answer:* a well-measured null on a new comparison is reportable at workshop level, where several of the closest works appeared: Panagopoulou (CMCL), Agate (ICML workshop), Cekinmez (COLM workshop) [reader-inferred].
10. **"Per-position marginals are not readings."**
    - *The case:* the parallel decoder knows the length and gives independent marginals, so a slot readout is a classifier and samples can be incoherent.
    - *What settles it:* D7's joint behaviour (partial text moves the slot coherently), and D8(c) or D3 sample coherence via semantic clustering.

## 9. Limitations of this review

- **Digest-level numbers.** These were not re-read from raw text: VQAScore, SelfEval, Holtzman, Jaini, Pearson and Fathi table values; Plank; Uma per dataset; Peterson; Bhattacharya; Stengel-Eskin; Xie; SAMNet; PercepT's tables; SEVLM; SemArt; Personality-Captions; V-FLUTE; i-Mix, Un-Mix, MixCo, LeMDA, VLMixer, MixReorg and Mix-up SSL. Of these, I relied only on directions from Peterson, Uma, Holtzman and Bhattacharya, and on no numbers from the rest.
- **N's raw reads not re-checked by V3.** Yao et al.'s cluster counts, IC3's Table 7 and App. C.5, and UniDisc's Fig 5b.
- **Abstract-only items.** Wang and Nalisnick, AMER, FOCUS, CLIP Under the Microscope, M3ixup, VLMixer. AmbiBench's paper was unverifiable; only its dataset card exists.
- **Not opened.** "Retrieving Emotional Stimuli in Artworks" (ICMR 2024), the most likely holder of an ArtEmis retrieval number. IG Captioner in full. ArtELingo's per-language κ values.
- **Search skew.** arXiv HTML and a US web index; OpenReview and ACL PDFs largely unreadable; 2026 preprints poorly indexed; no Semantic Scholar citation chains for N. Absence claims are bounded by the listed queries.
- **Project and local facts.** The lead's local counts (ArtELingo splits, AL-28 overlap, compute) were taken as given. The code facts come from my reading of four model files. The costs of every new arm are estimates, and none of the ArtELingo timings has been measured.
- **Designs.** Kill thresholds and expected outcomes are proposals, not predictions. Seed noise on ArtELingo metrics is unknown until D4's first seeds exist.

<details><summary>Source key</summary>

Agate 2026, ICML 2026 Workshop on Philosophy of ML, arXiv 2606.08031. AMER: Chen et al. 2025, arXiv 2511.02770. ArtECulture: Chen et al. 2026, arXiv 2608.03358. ArtELingo: Mohamed et al. 2022, EMNLP. ArtELingo-28: Mohamed et al. 2024, EMNLP. ArtEmis: Achlioptas et al. 2021, CVPR. ArtEmis v2: Mohamed et al. 2022, CVPR. Baan et al. 2022, EMNLP. Bhattacharya et al. 2019, ICCV. CapPa: Tschannen et al. 2023, NeurIPS. Cekinmez et al. 2026, COLM 2026 Sci-FM workshop, arXiv 2608.00410. CLIP Under the Microscope: Abbasi et al. 2025, CVPR. D-MLM: Gowda et al. 2025, MICCAI workshop. DDCap: Zhu et al. 2022, arXiv. DDES: Bergeron et al. 2026, arXiv. DiffCap: He et al. 2023, arXiv. DiffusionITM: Krojer et al. 2023, NeurIPS. Diverse Beam Search: Vijayakumar et al., AAAI 2018. FOCUS: Zhou et al. 2025, ACL. Geodesic Multi-Modal Mixup: Oh et al. 2023, NeurIPS. Holtzman et al. 2021, EMNLP. IC3: Chan et al. 2023, EMNLP. IG Captioner: Yang et al., ECCV 2024, arXiv 2311.17072. Kauf and Ivanova 2023, ACL. Kuhn et al. 2023, ICLR. Lan et al. 2025, AAAI. Lin et al. 2024, ICML. M3ixup: Li et al. 2025, Pattern Recognition 158. MASS: Chung et al. 2025, AAAI. MaskGIT: Chang et al. 2022, CVPR. MaxMatch: Alomari et al. 2025, ACL. MDC (Masked Diffusion Captioning): Feng et al. 2025, Findings of EMNLP. MixedAE: Chen et al. 2023, CVPR. MixGen: Hao et al. 2023, WACV Workshops. MixMAE: Liu et al. 2023, CVPR. Mix-up SSL: Zhang et al. 2022, ICME. MultiEmo-Bench: Chen et al. 2026, arXiv 2605.14635. Nie et al. 2020, EMNLP. Panagopoulou et al. 2024, CMCL. PCME++: Chun 2024, ICLR. Peterson et al. 2019, ICCV. Pezzelle 2023, ACL. ProLIP: covered by the 2026-10-03 review (B4). Resilient MCL: Letzelter et al. 2023, NeurIPS. Annealed MCL: Perera et al. 2024, NeurIPS. Rupprecht et al. 2017, ICCV. Salazar et al. 2020, ACL. SAMNet: Yang et al. 2022, IEEE TIP. sMCL: Lee et al. 2016, NeurIPS. Takatsuki et al. 2026, EMNLP 2026 (author-stated), arXiv 2609.13254. TiMix: Jiang et al. 2024, AAAI. Uma et al. 2021, JAIR. UniDisc: Swerdlow et al. 2025, arXiv. VEGAS: arXiv 2607.08489. VSE (Visual Semantic Entropy): Huy et al. 2026, ECCV 2026 (author-stated). Wang and Chan 2019, CVPR. Wang and Nalisnick 2025, arXiv. Yao et al. 2026, Findings of EACL.
</details>
