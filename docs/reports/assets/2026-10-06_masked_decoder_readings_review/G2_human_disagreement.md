> Supporting file for docs/reports/literature/2026-10-06_masked_decoder_readings_review.md. Phase 2 bibliography as written; V1 lists the corrections, and the report uses corrected values.

# G2: Does a model's predictive distribution match human disagreement, and how is that measured?

Agent: bibliography agent, theme G2. Last searched 2026-10-06 19:54.

**Read-scope note (applies to every entry).** The fetch tool returns a small model's answer to my extraction prompt, not raw page text. After the coordinator's method note I downloaded the raw arxiv HTML (own dir `scratchpad/hb_review/dl_G2/`) and re-read the load-bearing tables with grep: entries are tagged **[raw-verified: Table N]** for numbers checked against raw text, and the rest are "as extracted by the fetch tool" (not re-read; spot-check before quoting). Numbers marked (secondary) came from a search snippet. Raw-verified: Lan [8] Tables 1-2; Baan [4] Table 1; Nie [3] Table 5; Wang & Chan [9] Table 2; MultiEmo [15] Table 4 and Sec 3; ArtEmis [16] Sec 3.2.

## 1. Search strategy

- Sources: general web search (arXiv, ACL Anthology, CVF, Nature/OATML pages, AAAI OJS), then arxiv.org/html and ar5iv for full text. Date range: 2015-2026, with deliberate push on 2024-2026.
- Queries:
  - Q1 ChaosNLI (Nie 2020); Q2 Baan 2022 calibration; Q3 Plank 2022; Q4 Stengel-Eskin 2023 ambiguous VQA
  - Q5 semantic entropy image captioning VLM uncertainty 2025; Q6 visual emotion LDL (Flickr_LDL, Twitter_LDL, Emotion6)
  - Q7 Bhattacharya 2019; Q8 Peterson CIFAR-10H; Q9 Uma 2021 survey; Q10 Kuhn semantic uncertainty
  - Q11 captioning diversity metrics validity (Self-CIDEr, mBLEU); Q12 VLM calibration to human annotator disagreement 2024-2025
  - Q13 Farquhar Nature 2024; Q14 captioner distribution vs human caption variability; Q15 ArtELingo/ArtEmis emotion distribution agreement
  - Q16 semantic entropy VLM captioning ambiguity; Q17 Semantic Entropy Probes / AUROC; Q18 LDL visual emotion 2024-2025 new datasets
  - Q19 multi-valid-caption evaluation 2024-2025
- Inclusion: the paper (i) compares a model's output distribution (or a proxy such as entropy, diversity, cluster count) with a human-derived distribution or with known multiplicity, or (ii) defines the metrics or sampling-and-clustering measure used for that comparison, or (iii) supplies per-item human label distributions for images or art.
- Exclusion: hallucination-only uncertainty papers with no human-multiplicity signal (Seifi et al. 2026 "Dissecting Training-Free Uncertainty Estimation in MLLMs", arXiv 2609.22206, VL-Uncertainty 2411.11919, radiology semantic entropy 2510.09256): read, none uses ambiguity or disagreement labels, evaluate against a single correctness label. Text-only emotion LLM-vs-human distribution paper (Inoshita et al. 2026, arXiv 2604.27345, JSD 0.299-0.587, entropy rho 0.20-0.47 on GoEmotions) and the soft-label digits paper (Pavlovic et al., arXiv 2605.18648): informative but text or toy-image, left out for count. IC3 (arXiv 2302.01328) treats captioner as a distribution but does not compare it with human variance (its text says it does not quantify this). ArtELingo's own paper (EMNLP 2022) could not be read (PDF only), so ArtEmis, its direct predecessor, is used for agreement figures.
- Counts: identified about 70 candidates, screened about 40, full-text (HTML extraction) read 24, included 16.
- Coverage skew: strong on NLP and VQA disagreement; weak on captioning (no paper found that matches a captioner's sampled distribution to the spread of human captions on real images); weak on art emotion LDL with painting data; semantic-entropy-for-captioning applications are missing.

## 2. Annotated bibliography

### (a) Human label variation and learning from disagreement

**[1] Plank, B. (2022). The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation. EMNLP 2022. arXiv 2211.02570.**
- Tier: peer-reviewed (EMNLP 2022).
- Read scope: full text (arxiv html), sections on sources of variation, evaluation, gaps.
- Setting: position paper plus survey; no model trained.
- Change tested / baseline: none; argues against the ground-truth assumption.
- Effect: not applicable (no experiment).
- Multiplicity measured? Defines human label variation (HLV) as variation from ambiguity, subjectivity, annotator uncertainty, or multiple correct answers, kept separate from error.
- How disagreement is measured (human side): per-item annotation distribution; recommends stratifying by agreement or label-entropy level, and evaluating against individual annotators.
- How the model's distribution is measured: soft evaluation: cross-entropy, KL or JSD to the human distribution, entropy correlation (Pearson between instance-level entropies).
- Agreement metric and value: none reported. Gap named: telling error from signal.
- Relevance to MultiMAE / H-b: gives the vocabulary and the menu of soft metrics (JSD/KL, entropy correlation) that an H-b test would reuse. [reader-inferred]
- Method weaknesses: position paper, so no empirical weaknesses to assess; its list of "sources" mixes ambiguity with subjectivity, so a model's spread cannot be assigned to one source from this paper alone. [reader-inferred]

**[2] Uma, A. N., et al. (2021). Learning from Disagreement: A Survey. JAIR 72:1385-1470.** (J. Artif. Intell. Res. 72). URL jair.org/index.php/jair/article/view/12752.
- Tier: peer-reviewed (JAIR).
- Read scope: partial full text (first 100k characters of the text version): datasets, metrics, conclusions.
- Setting: benchmark of training strategies (gold, aggregated, soft labels, gold+soft, others) on six datasets: POS tagging (14,000 items), Phrase Detectives information status (96,305), medical relation extraction (975), RTE (800), LabelMe images (10,000), CIFAR-10H (10,000 images, per-item about 50 judgements).
- Change tested and baseline: soft-label training vs training on aggregated or gold labels.
- Effect: conclusion quoted by the extractor: with many judgements by high-quality coders per item, soft-label training beats aggregated or gold labels under both hard and soft evaluation; with few or noisy judgements, gold+soft is best on hard evaluation. Per-dataset numbers not extracted (UNVERIFIED beyond the conclusion).
- Multiplicity measured? Yes, as distribution similarity.
- How disagreement is measured (human side): per-item label distribution from crowd judgements.
- How the model's distribution is measured: soft metrics: cross-entropy, JSD, entropy correlation; hard metrics: accuracy, F1, crowd-truth-weighted F1.
- Agreement metric and value: not extracted.
- Relevance to MultiMAE / H-b: the finding that distributional supervision helps only with enough judgements per item matters for ArtELingo (5+ captions, 1-5+ distinct emotions per painting). [reader-inferred]
- Method weaknesses: results conditional on annotator quality and count; with 5 judgements per item the human distribution is a coarse estimate, which can distort soft-metric ranking. [reader-inferred]

### (b) Calibration to human distributions

**[3] Nie, Y., Zhou, X., Bansal, M. (2020). What Can We Learn from Collective Human Opinions on Natural Language Inference Data? (ChaosNLI). EMNLP 2020. arXiv 2010.03532.**
- Tier: peer-reviewed (EMNLP 2020).
- Read scope: full text (ar5iv), data collection, Table 5 and agreement-split analysis.
- Setting: fine-tuned pretrained encoders (BERT, RoBERTa, XLNet, ALBERT, BART) on NLI; ChaosNLI = 100 annotations per item for 3,113 SNLI+MNLI items and 1,532 abductive NLI items (464,500 labels).
- Change tested / baseline: compare model softmax to the human distribution; human baseline estimated from annotator splits.
- Effect [raw-verified: Table 5]: ChaosNLI-SNLI: chance JSD 0.383, KL 0.5457; models roughly JSD 0.23, KL 0.48-0.53 (e.g. BERT-base 0.2345/0.481, RoBERTa-base 0.2294/0.5045); estimated human JSD 0.0614, KL 0.0411. ChaosNLI-alpha: chance 0.3205/0.406; BERT-base 0.3209/3.7981 (worse than chance on KL), RoBERTa-base 0.2394/0.8272; human 0.0421/0.0373. ChaosNLI-MNLI: chance 0.3023/0.3559; models about 0.31/0.72-0.79 (worse than chance on KL); human 0.0695/0.0381. So models barely beat or lose to uniform chance on divergence. The accuracy split (about 0.9 high-agreement vs about 0.5 low-agreement) and the 'accuracy and divergence uncorrelated' claim are as extracted, not re-read.
- Multiplicity measured? Yes: full 100-vote label distributions.
- How disagreement is measured (human side): normalised vote counts over 100 annotators; items binned by distribution entropy.
- How the model's distribution is measured: softmax output over labels.
- Agreement metric and value: Jensen-Shannon distance (primary), KL (secondary), accuracy; values above.
- Relevance to MultiMAE / H-b: the canonical template for "spread vs annotator distribution" and for the agreement-bin accuracy split that H-b would mirror on ArtELingo. [reader-inferred]
- Method weaknesses: softmax of a fine-tuned classifier is trained on one hard label, so its spread reflects training, not a designed reading distribution; and 100 labels per item exist only for about 4.6k items, so the design is small. [reader-inferred]

**[4] Baan, J., Aziz, W., Plank, B., Fernandez, R. (2022). Stop Measuring Calibration When Humans Disagree. EMNLP 2022. arXiv 2210.16133.**
- Tier: peer-reviewed (EMNLP 2022).
- Read scope: full text (ar5iv), measures and ChaosNLI-SNLI table.
- Setting: RoBERTa fine-tuned for NLI, evaluated on ChaosNLI-SNLI; oracle that outputs the human distribution.
- Change tested / baseline: replace ECE (against majority label) by instance-level measures; baseline is ECE and temperature scaling (TS).
- Effect [raw-verified: Table 1]: RoBERTa accuracy 0.74, ECE 0.14, E[DistCE] 0.26; RoBERTa+TS ECE 0.03, DistCE 0.22; oracle accuracy 1.00, ECE 0.25, DistCE 0.00; also RankCS 0.62 (RoBERTa, with or without TS) vs 1.00 (oracle), E[|EntCE|] 0.30 -> 0.21 with TS. So ECE punishes the perfect model; TS cuts ECE 0.14 to 0.03 while DistCE moves only 0.26 to 0.22.
- Multiplicity measured? Yes, three instance-level measures: EntCE (model entropy vs human entropy), RankCS (class ranking vs vote ranking), DistCE (total variation distance to the human distribution; strictest).
- How disagreement is measured (human side): vote distribution per item from ChaosNLI (100 votes).
- How the model's distribution is measured: softmax probabilities; compared per instance.
- Agreement metric and value: as above.
- Relevance to MultiMAE / H-b: supplies exactly the three tests H-b needs once the decoder yields a distribution over readings: entropy match, ranking match, distance match; and a warning that accuracy-type calibration can hide a mismatch. [reader-inferred]
- Method weaknesses: needs many annotations per item; cannot split true disagreement from noise; human distribution is the pooled population, not subgroups. [author-acknowledged, limitations section as extracted]

**[5] Peterson, J. C., Battleday, R. M., Griffiths, T. L., Russakovsky, O. (2019). Human uncertainty makes classification more robust. ICCV 2019. arXiv 1908.07086.**
- Tier: peer-reviewed (ICCV 2019).
- Read scope: full text (arxiv html): collection, Sec 5.2, Table 1-2.
- Setting: CIFAR-10H = 511,400 judgements on the 10,000 CIFAR-10 test images (about 50 per image, 47-63, 2,571 workers); fine-tuning CNNs with human soft labels.
- Change tested / baseline: train on human soft labels vs hard labels, knowledge distillation from ensembles, mixup, class-level penalty vectors.
- Effect: CIFAR-10 accuracy 83.5 -> 84.5; ImageNet-Far accuracy 49.4 -> 51.4, cross-entropy 2.9 -> 1.8; FGSM cross-entropy about halved (Sec 5.2, Table 2, as extracted). Human soft labels beat all listed baselines (Table 1).
- Multiplicity measured? Yes: per-image human label distributions; uncertainty concentrated in about 30% of images.
- How disagreement is measured (human side): vote histogram per image.
- How the model's distribution is measured: cross-entropy to the human distribution; standard CNNs "fail to exhibit human-like uncertainty on their own".
- Agreement metric and value: cross-entropy (values above).
- Relevance to MultiMAE / H-b: oldest image-side evidence that a model trained on one point per item does not carry the human spread; a distribution-level target changes that. [reader-inferred]
- Method weaknesses: single-label images (10 object classes), so disagreement is perceptual, not interpretive; the collection does not scale to large training sets. [author-acknowledged, as extracted]

### (c) Vision-language: VQA disagreement, captioning diversity

**[6] Bhattacharya, N., Li, Q., Gurari, D. (2019). Why Does a Visual Question Have Different Answers? ICCV 2019. arXiv 1908.04342.**
- Tier: peer-reviewed (ICCV 2019).
- Read scope: full text (arxiv html): data, taxonomy, results.
- Setting: about 45,000 questions: VizWiz 29,921 and VQA 2.0 15,034, 10 crowd answers each; a Q+I+A model (Faster R-CNN features, GloVe+GRU, pre-trained VQA answer predictions) predicts reasons.
- Change tested / baseline: predict which of 9 reasons (low-quality image, insufficient evidence, invalid, difficult, ambiguous, subjective, synonyms, granularity, spam) cause answer differences, vs related baselines.
- Effect: average precision 44.55 (VQA 2.0) and 50.02 (VizWiz), above baselines by 12+ points; poor on subjective (<26) and difficult (<32) reasons.
- Multiplicity measured? Yes, but as a reason label, not a model-vs-human distribution.
- How disagreement is measured (human side): disagreement = the 10 answers are not all identical (exact string match); reasons labelled by crowd. A 2-person validity threshold was reported; an "ambiguity 81.3%" figure appeared in my extraction without a clear denominator: UNVERIFIED, do not quote.
- How the model's distribution is measured: not measured; the model predicts reasons from the question-image pair.
- Agreement metric and value: average precision per reason.
- Relevance to MultiMAE / H-b: shows that a large share of answer disagreement is about wording (synonyms, granularity) rather than meaning, so any H-b spread metric must cluster by meaning first. [reader-inferred]
- Method weaknesses: exact-string disagreement counts paraphrases as disagreement and inflates it. [reader-inferred]

**[7] Stengel-Eskin, E., Guallar-Blasco, J., Zhou, Y., Van Durme, B. (2023). Why Did the Chicken Cross the Road? Rephrasing and Analyzing Ambiguous Questions in VQA. ACL 2023. arXiv 2211.07516.**
- Tier: peer-reviewed (ACL 2023).
- Read scope: full text (ar5iv extraction): data, ontology, results.
- Setting: 1,820 annotated VQAv2 triples; 241 ambiguous image-question pairs with 629 rewritten questions (1-99 answers each, mean 2.9); ViLT-T5 question generator.
- Change tested / baseline: group answers by the underlying question they answer, rewrite the question per group; clustering baseline GloVe.
- Effect: answer-clustering F1 79.0 (VT5+KMeans) vs 72.4 (GloVe); generation BLEU-4 0.26, CIDEr 2.21 (as extracted); human annotation agreement 73.5% (ambiguity) and F1 88.4 (clustering) in an MTurk pilot.
- Multiplicity measured? Yes: answers partitioned into readings (answer groups), each paired with a disambiguating question; ontology (property, dynamic, pragmatic).
- How disagreement is measured (human side): grouping of the crowd answers by the question each addresses.
- How the model's distribution is measured: the generator is scored on whether its rewritten question matches target group vs distractor answers (human 3-way ratings); no distribution match.
- Agreement metric and value: as above.
- Relevance to MultiMAE / H-b: nearest vision-language example of "several readings of one input, each with its own set of valid outputs", and a way to define reading clusters from human outputs. [reader-inferred]
- Method weaknesses: small (211 test questions), high-precision subset of ambiguity only, built on VQAv2's noisy answers. [author-acknowledged, limitations as extracted]

**[8] Lan, J., Frassinelli, D., Plank, B. (2025). Mind the Uncertainty in Human Disagreement: Evaluating Discrepancies between Model Predictions and Human Responses in VQA. AAAI 2025. arXiv 2410.02773.**
- Tier: peer-reviewed (AAAI 2025).
- Read scope: full text (arxiv html): HUD definition, Table 1, calibration section.
- Setting: BEiT3 (VQA-acc 84.03) and LXMERT (72.5) on VQA 2.0 validation subsets with several distinct human answers (BEiT3: 3,248 items; LXMERT: 15,408).
- Change tested / baseline: measure model distribution against human response distribution; compare accuracy-oriented temperature scaling with human-distribution-oriented temperature scaling.
- Effect [raw-verified: Table 1 and Table 2]: KL is KL(H||M). BEiT3 all: TVD 0.436, KL 1.701, EntCE 0.541; LXMERT all: VQA-acc 0.671, TVD 0.448, KL 1.886, EntCE 0.555 (Table 1; Table 2's LXMERT 'before' row shows 0.442/1.771/0.542, a different number set in the paper, so quote Table 1 or Table 2 explicitly). BEiT3 by HUD level, low/medium/high: accuracy 0.921/0.845/0.759, KL 2.150/1.544/1.408. Accuracy-oriented TS (T=0.6) raised BEiT3 KL 1.701 -> 3.232; human-distribution TS (T=2.0) lowered it to 0.647 (TVD 0.436 -> 0.340) (Table 2); LXMERT KL 1.771 -> 0.731 in Table 2 (1.886 in Table 1).
- Multiplicity measured? Yes: "human uncertainty in disagreement" (HUD), combining label variation with annotator confidence (yes/maybe/no scored 1.0/0.5/0.01), in three equal-size levels (about 3.1, 4.0, 4.8 distinct answers).
- How disagreement is measured (human side): distribution of the 10 crowd answers, confidence-weighted, binned into HUD levels.
- How the model's distribution is measured: softmax over the answer vocabulary (classification VQA).
- Agreement metric and value: TVD, KL, human entropy calibration error (EntCE); values above.
- Relevance to MultiMAE / H-b: the closest published recipe for "model spread vs human spread, binned by human uncertainty" in vision-language; its metric set carries over if the decoder is made to emit an answer or caption distribution. [reader-inferred]
- Method weaknesses: answer strings are treated as distinct classes (no meaning-level merging), so synonym variation counts as uncertainty; only VQA 2.0 and only temperature scaling tried. [author-acknowledged for the second, as extracted; the first is reader-inferred]

**[9] Wang, Q., Chan, A. B. (2019). Describing like Humans: on Diversity in Image Captioning. CVPR 2019. arXiv 1903.12020.** (Extended in TPAMI 2020, "On Diversity in Image Captioning: Metrics and Methods", not read separately.)
- Tier: peer-reviewed (CVPR 2019).
- Read scope: full text (ar5iv extraction): metric definitions, Table 2, model comparison.
- Setting: eight captioners (NIC, SoftAtt, AdapAtt, Att2in, FC, CVAE, GMMCVAE, CGAN), COCO, several generation strategies (random sampling, cropping, noise, synonym switching).
- Change tested / baseline: new set-level diversity metrics, LSA-based diversity and Self-CIDEr (kernelised LSA on CIDEr similarity), vs mBLEU-mix.
- Effect [raw-verified: Table 2]: agreement with human diversity judgements (Self-CIDEr / LSA / mBLEU-mix): overall Pearson 0.616 / 0.601 / 0.585; overall Spearman 0.617 / 0.602 / 0.575; average per-image Spearman 0.674 / 0.678 / 0.644. So Self-CIDEr's edge over mBLEU-mix is small, and LSA is slightly higher on the per-image measure.
- Multiplicity measured? Yes: diversity of a caption set (human or model) via topic structure.
- How disagreement is measured (human side): the 5 human captions per COCO image are the reference set; human diversity scored with the same metric.
- How the model's distribution is measured: sample a caption set from the model and score its Self-CIDEr or LSA diversity; humans' captions score markedly higher than accuracy-optimised models.
- Agreement metric and value: correlation of metric with human diversity ratings (above); no distribution distance.
- Relevance to MultiMAE / H-b: a ready diversity metric for sets of decoded captions, validated against human diversity judgements, but it counts surface and topic variety, not distinct readings. [reader-inferred]
- Method weaknesses: CIDEr similarity rewards lexical difference, so paraphrases score as diverse; validation correlation of about 0.6 leaves much variance unexplained. [reader-inferred]

**[10] Xie, H., Sherborne, T., Kuhnle, A., Copestake, A. (2019). Going Beneath the Surface: Evaluating Image Captioning for Grammaticality, Truthfulness and Diversity. arXiv preprint 1912.08960.**
- Tier: preprint (pre-2024).
- Read scope: partial full text (ar5iv extraction), framework and diversity definition.
- Setting: ShapeWorld synthetic images (about 200k training instances, 4,096 validation/test per variant); captioners trained from scratch.
- Change tested / baseline: grammaticality-truthfulness-diversity (GTD) framework; diversity = number of distinct language constructions produced / number in the generator's sampled reference captions.
- Effect: truthfulness on Quant-Count 0.50 and Quant-Ratio 0.46 (Fig 5); diversity varies widely across dataset types (Fig 6, no numbers extracted).
- Multiplicity measured? Yes: because ShapeWorld generates captions with known true variety, the reference distribution of valid captions is exact.
- How disagreement is measured (human side): no humans; the generator's own sampled captions are "a proxy for optimal caption diversity".
- How the model's distribution is measured: count of distinct constructions in sampled captions vs the generator's count.
- Agreement metric and value: ratio observed/optimal construction count.
- Relevance to MultiMAE / H-b: the only design found in which the set of valid readings is known exactly; the same logic would apply to synthetic mixed images in theme G5. [reader-inferred]
- Method weaknesses: only validated on synthetic data. [author-acknowledged, as extracted]

### (d) Meaning-level uncertainty from generators

**[11] Kuhn, L., Gal, Y., Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. ICLR 2023 (spotlight). arXiv 2302.09664.**
- Tier: peer-reviewed (ICLR 2023).
- Read scope: full text (arxiv html): method, Table 2, ablations.
- Setting: no training; OPT 2.7B-30B, CoQA (about 8,000 questions) and TriviaQA (8,000 subset); DeBERTa-large (MNLI) for entailment.
- Change tested / baseline: semantic entropy (sample about 10 answers, cluster by bidirectional entailment, sum sequence probabilities in each cluster, entropy over clusters) vs predictive entropy, length-normalised entropy, lexical similarity.
- Effect: TriviaQA, OPT-30B AUROC 0.83 vs 0.80 (length-normalised entropy); wrong answers average 3.89 semantic clusters vs 1.89 for right answers (Table 2). Best temperature 0.5; multinomial sampling beat beam search.
- Multiplicity measured? Yes in the sense of number and mass of meaning clusters, but the target is correctness, not human disagreement.
- How disagreement is measured (human side): none.
- How the model's distribution is measured: cluster-level entropy over sampled outputs.
- Agreement metric and value: AUROC for predicting wrong answers (above).
- Relevance to MultiMAE / H-b: the standard recipe for a meaning-level spread of a generator; applying it to decoded captions would give a "number of readings" score to correlate with annotator emotion spread. [reader-inferred]
- Method weaknesses: needs an entailment model that is itself imperfect; for open captions, "bidirectional entailment" will split or merge readings in ways the NLI model decides. [reader-inferred]

**[12] Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. Nature 630:625-630.**
- Tier: peer-reviewed (Nature).
- Read scope: abstract only plus a secondary search snippet; the Nature page redirected to a login and was not read.
- Setting: six LLMs (including LLaMA 2 and GPT-4 per press summary), five datasets; sampled answers clustered by entailment.
- Change tested / baseline: semantic entropy vs naive entropy and P(True).
- Effect (secondary): mean AUROC 0.790 vs 0.691 (naive entropy) and 0.698 (P(True)) across five datasets and six models. UNVERIFIED against the paper.
- Multiplicity measured? As in [11]: meaning-level spread; target is confabulation, not human disagreement.
- How disagreement is measured (human side): none.
- How the model's distribution is measured: discrete semantic entropy over entailment clusters.
- Agreement metric and value: AUROC (secondary).
- Relevance to MultiMAE / H-b: confirms the method's current standing; says nothing about benign spread (several valid readings). [reader-inferred]
- Method weaknesses: not assessed (read scope: abstract_only). One caution applies to H-b in general: semantic entropy treats spread as a sign of error, which is the opposite reading from H-b where spread over valid readings is wanted. [reader-inferred]

**[13] Huy, T. D., Nguyen, T., Chowdhury, T., et al. (2026). Visual Semantic Entropy: Do Vision Language Models Recognize Visual Ambiguity? arXiv preprint 2606.31407 (also listed as a Springer chapter in search results; venue not verified). preprint>=2024.**
- Tier: preprint (>=2024).
- Read scope: full text (arxiv html v1), Tables 2 and 5, as extracted.
- Setting: no training; Qwen2.5-VL-7B, Gemma3-4B, Intern3.5-VL-8B, LLaVA-NeXT-8B, Qwen3-VL-8B on A-OKVQA, OK-VQA, MM-Vet, plus adversarial VILP (900) and VLM-are-biased (458).
- Change tested / baseline: perturb only the image (Gaussian noise sigma=20), cluster the answers into prototypes, score mass-weighted pairwise dispersion; vs semantic entropy, SNNE, KLE, VL-Uncertainty, token entropy, verbalised confidence.
- Effect: A-OKVQA AUROC 0.783 (Qwen2.5), 0.778, 0.724, 0.798, 0.792 across models; up to +9.2% over VL-Uncertainty (Table 2); VILP Qwen2.5 0.650 vs 0.535, VLM-are-biased 0.826 vs 0.783 (Table 5).
- Multiplicity measured? Visual ambiguity, but operationalised through correctness of the answer (AUROC against right/wrong), not through human disagreement.
- How disagreement is measured (human side): none; ambiguity is proxied by adversarial dataset splits.
- How the model's distribution is measured: prototype-cluster dispersion over answers sampled under image perturbation.
- Agreement metric and value: AUROC (above).
- Relevance to MultiMAE / H-b: states that overconfident visual embeddings suppress sampling diversity so semantic entropy under-reports ambiguity: a warning that decoder spread may collapse unless the masked decoder is trained to keep it. [reader-inferred]
- Method weaknesses: ambiguity ground truth is correctness, so benign multi-reading items are not separated from wrong ones. [reader-inferred] Author-acknowledged: perturbation cost, temperature sensitivity, dependence on the similarity model, limited model families.

### (e) Emotion label distributions for images

**[14] Yang, J., Li, J., Li, L., Wang, X., Ding, Y., Gao, X. (2022). Seeking Subjectivity in Visual Emotion Distribution Learning (SAMNet). IEEE TIP 2022. arXiv 2207.11875.**
- Tier: peer-reviewed (IEEE TIP).
- Read scope: full text (ar5iv extraction), Tables I-II.
- Setting: Flickr_LDL (11,150 images, votes from 11 annotators) and Twitter_LDL (10,045 images, votes from 8), Mikels' eight emotions; 80/20 split; CNN trained from the standard backbone, 448x448.
- Change tested / baseline: multi-branch network where each branch simulates one individual annotator; 14 baselines (PT-Bayes, PT-SVM, AA-kNN, AA-BP, SA-IIS/BFGS/CPNN, CNNR, DLDL, ACPNN, JCDL, SSDL, E-GCN, CSR).
- Effect: best mean rank 1.3 (Flickr_LDL) and 1.5 (Twitter_LDL); Flickr_LDL KL 0.40, top-1 accuracy 0.74 (as extracted; strongest-baseline values not extracted).
- Multiplicity measured? Yes: per-image vote distributions over 8 emotions.
- How disagreement is measured (human side): summed votes of 8-11 annotators per image, normalised.
- How the model's distribution is measured: predicted 8-way distribution scored by Chebyshev, Clark, Canberra, KL (lower better), cosine, intersection (higher better), plus top-1.
- Agreement metric and value: KL 0.40 for best model on Flickr_LDL; no human upper bound reported.
- Relevance to MultiMAE / H-b: standard metric suite for image emotion distributions, and the point that a model should keep individual-level diversity; ArtELingo's 5+ annotators per painting are in the same range of annotators per item. [reader-inferred]
- Method weaknesses: with only 8-11 votes, the target distribution is itself noisy and no human split-half ceiling is given, so a KL of 0.40 cannot be judged against an achievable floor. [reader-inferred] Author-acknowledged: no individual-level annotations or personalisation data.

**[15] Chen, T., Furusawa, T., Hirakawa, Y., Shimizu, R., Fan, M., Wada, T. (2026). MultiEmo-Bench: Multi-label Visual Emotion Analysis for Multi-modal Large Language Models. arXiv preprint 2605.14635. preprint>=2024.**
- Tier: preprint (>=2024).
- Read scope: full text (arxiv html), dataset section and Tables 3-4, as extracted.
- Setting: 10,344 images (from EmoSet test 17,716 and FI test 3,474), 20 annotators each, 236,998 votes (22.91 per image, multi-select), eight Mikels emotions; 11 MLLMs (GPT-4o, GPT-5 family, Gemini 2.5/3 flash, Claude Sonnet 4, Qwen3-VL 2B-32B), zero-shot.
- Change tested / baseline: re-annotate with many votes per image; compare MLLM predicted distributions with the crowd distribution, with a split-half human baseline.
- Effect: dominant-emotion accuracy best 51.05% (GPT-5, Table 3, as extracted). Distribution task [raw-verified: Table 4]: PLCC GPT-4o 0.634, GPT-5 0.631, GPT-5.1 0.629, Qwen-VL-32B 0.592, Gemini-2.5-flash 0.541, vs human split-half 0.700; Chebyshev GPT-5 0.578 vs human half-half 0.559; KL GPT-4o 1.294 (human KL not applicable); human SRCC 0.656 vs GPT-4o 0.615. Only 4,907 images (47.44%) have more than 10 votes for the dominant emotion; 9.3% (962) have several dominant emotions; split-half reliability PLCC 0.716, SRCC 0.666.
- Multiplicity measured? Yes: full vote distribution and multiple dominant emotions.
- How disagreement is measured (human side): normalised vote counts over about 23 votes; split-half correlation as reliability.
- How the model's distribution is measured: MLLM outputs converted to distributions; eight metrics: Chebyshev, Canberra, Wasserstein, KL, JS, cosine, PLCC, SRCC.
- Agreement metric and value: human split-half PLCC about 0.70-0.72 as the ceiling.
- Relevance to MultiMAE / H-b: gives a numeric ceiling (about 0.70 PLCC) for "typical agreement" in image emotion and a human-baseline protocol that ArtELingo could copy. [reader-inferred]
- Method weaknesses: "emotion distribution" from MLLM prompts is a verbal output, not a model's predictive probabilities; English only, Mikels taxonomy. [reader-inferred / author-acknowledged for the last two, as extracted]

**[16] Achlioptas, P., Ovsjanikov, M., Haydarov, K., Elhoseiny, M., Guibas, L. (2021). ArtEmis: Affective Language for Visual Art. CVPR 2021. arXiv 2101.07396.** (Read for the ArtELingo link: ArtELingo extends it with more languages.)
- Tier: peer-reviewed (CVPR 2021).
- Read scope: partial full text (arxiv html), Sec 3 and Sec 6.
- Setting: WikiArt paintings, at least 5 annotators per painting, each giving one of 9 options (8 emotions plus "something else") and a free-text explanation; neural speakers trained to produce explanations.
- Change tested / baseline: dataset paper; speaker models vs human captions in a Turing-style test (500 artworks).
- Effect (Sec 6, as extracted): emotion-grounded speaker tied with human ground truth in 50.3% and rated more human-like in 20.6%; basic speaker 40% (16.3% wins, 23.7% ties); text-to-emotion accuracy: BERT 65.7%, LSTM 63.1%, human experts 60.3% (250 utterances, Sec 6).
- Multiplicity measured? Yes, descriptively: 45.6% of paintings (37,145) have majority agreement on the same fine-grained emotion; 61% mix positive and negative emotions; 79% have mixed polarity if "something else" is a third class (Sec 3.2).
- How disagreement is measured (human side): per-painting emotion histogram from 5+ annotators; polarity mixture.
- How the model's distribution is measured: not measured; speakers are judged for plausibility, not for spread.
- Agreement metric and value: majority-agreement share 45.6%.
- Relevance to MultiMAE / H-b: states directly that most paintings have several valid emotions; supports using the emotion histogram as the human-side spread for H-b on ArtELingo. [reader-inferred]
- Method weaknesses: spread is confounded with annotator-pool differences and with the "something else" option; a captioner's samples were never compared with the annotator spread. [reader-inferred]

## 3. Closest-to-H-b list

1. **Lan et al. 2025 [8]**: bins items by human uncertainty and scores a vision-language model's answer distribution with TVD, KL and EntCE, then shows human-distribution temperature scaling fixes the gap (KL 1.701 -> 0.647). Does not establish anything for generators, since answers are class labels and no meaning-level merging is done.
2. **Baan et al. 2022 [4] with Nie et al. 2020 [3]**: the metric set (EntCE, RankCS, DistCE; JSD/KL; accuracy by agreement bin) that an H-b test would reuse, with a worked case (ECE 0.25 for a perfect oracle). Text-only classification; softmax is not a decoder distribution over free-form readings.
3. **Kuhn et al. 2023 [11] / Visual Semantic Entropy [13]**: the only available way to turn a generator's samples into a meaning-level spread (cluster count, entropy over clusters). Both validate against correctness, not against human disagreement, and no paper in Q5/Q16 applies it to image captioning against annotator spread.
4. **Wang & Chan 2019 [9] with Xie et al. 2019 [10]**: caption-set diversity metrics (Self-CIDEr, valid-reading counts with an exact reference on synthetic data). Self-CIDEr is lexical and correlates only about 0.6 with human diversity judgements; the exact-reference design is synthetic only.
5. **ArtEmis [16] with MultiEmo-Bench [15]**: the human-side numbers for emotion spread (45.6% majority agreement; human split-half PLCC about 0.70), but no model distribution is compared with them for art.

## 4. Search limitations

- Direct PDF fetches returned binary; all numbers come from html/ar5iv extractions by the fetch tool (see read-scope note). Several numbers are the extractor's reading of tables and should be re-checked.
- Farquhar et al. (Nature) was not readable (paywall redirect); its entry rests on the abstract and a secondary snippet.
- ArtELingo's own paper could not be read (Q15, one PDF attempt returned binary); no per-painting disagreement figure for ArtELingo itself was obtained, only for ArtEmis. Whether ArtELingo reports agreement across languages is not established here.
- Not found in Q5, Q14, Q16, Q19: a paper that matches a captioner's sampled caption distribution (or its semantic-entropy cluster count) against the spread of human captions or emotions on real images. This is an absence claim bounded by those queries.
- Not found in Q6, Q18: label-distribution-learning work on painting emotion with per-painting votes beyond ArtEmis-derived models (Dimensional Distribution Emotion State, arXiv 2605.26262, uses ArtEmis but was only skimmed: DDES-Net 39.8% accuracy, tau 0.34 on ArtEmis; excluded).
- Emotion6 and the earlier Flickr_LDL/Twitter_LDL dataset papers (Yang et al. 2017 AAAI/IJCAI) were not read; dataset sizes come from [14].
- Search engine is US-only and favours arXiv; ACL/AAAI venue pages were not read in full. Several 2026 arXiv ids (2604-2609) appear in results; I could confirm they resolve, not that they are peer-reviewed.
