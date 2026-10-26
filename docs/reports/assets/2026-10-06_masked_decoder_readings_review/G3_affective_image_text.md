> Supporting file for docs/reports/literature/2026-10-06_masked_decoder_readings_review.md. Phase 2 bibliography as written; V1 lists the corrections, and the report uses corrected values.

# G3: Subjective and affective image-text data as a testbed for several readings per image

Phase 2 bibliography, lit-review mode. Written 2026-10-06 19:55.

Read-scope caveat that applies to everything below: WebFetch returns a small-model summary of a page, not raw text. Numbers were first extracted that way; the ones marked [raw-checked] were then re-read in the raw arXiv HTML (curl to dl_G3/) with the locator given. Several PDFs (ArtEmis CVPR PDF, ArtELingo PDF, ArtELingo-28 ACL PDF, SemArt PDF) came back as binary and could not be parsed (no pdftotext/pypdf here, and installs were not allowed). Where only a summary was possible, locators are given as the summary reported them and numbers should be re-checked against the paper before they go into a report. Items I could not confirm are marked UNVERIFIED.

## 1. Search strategy

- Sources: general web search (US), arXiv abs/html/ar5iv, ACL Anthology, CVF open access. Semantic Scholar and OpenReview were not queried directly.
- Date range: 2018 to 2026-10-06, with weight on 2024-2026. Last searched 2026-10-06.
- Queries:
  - Q1 ArtEmis annotators per image emotion agreement
  - Q2 PercepT arXiv 2606.03345
  - Q3 ArtELingo-28 multilingual affective captioning shared task
  - Q4 Affection ICCV 2023 (the brief said ICCV; the venue is CVPR 2023, see entry 5)
  - Q5 No Culture Left Behind ArtELingo-28 annotations per image
  - Q6 ArtEmis image-text retrieval CLIP fine-tuned (two phrasings, standard)
  - Q7 ArtEmis / ArtELingo retrieval Recall@K CLIP BLIP (extended)
  - Q8 ArtEmis cross-modal retrieval emotion-aware 2025 (extended)
  - Q9 affective captioning diversity of human explanations 2024-2025
  - Q10 emotion-conditioned captioning, multiple emotions, Self-BLEU (extended)
  - Q11 datasets with several human interpretations per image, perspectivist, ambiguous images
  - Q12 visual metaphor / figurative / IRFL / memes
  - Q13 SemArt / Text2Art retrieval
  - Q14 Personality-Captions
  - Q15 Sherlock visual abductive reasoning (fetched directly by arXiv id 2202.04800)
  - Plus direct fetches of arXiv pages for every included paper.
- Inclusion: (i) a dataset where one image carries several human readings (emotion plus explanation, or comments), or (ii) a paper reporting agreement, noise or validity statistics on such data, or (iii) a paper reporting image-text retrieval or matching on such data, or (iv) a model that predicts or generates a distribution over affective readings. Exclusion: papers using ArtEmis only as a classification source without a statement about annotator multiplicity; generic MLLM emotion benchmarks without multi-annotator data.
- Counts: about 45 candidates identified, about 25 screened, 13 read at the summary level (listed below), 12 beyond the 13 excluded (listed in section 5).
- Coverage skew: the affective-art lineage is almost entirely from one group (Achlioptas, Mohamed, Elhoseiny, Church) plus one external 2026 re-annotation (ArtECulture). No ArtEmis or ArtELingo image-text retrieval number from a third party was found; this is a finding, not a gap I could close.

## 2. Annotated bibliography

### 2a. The ArtEmis lineage

**1. Achlioptas et al., 2021. ArtEmis: Affective Language for Visual Art. CVPR 2021 (oral). arXiv 2101.07396.**
- Tier: peer-reviewed (CVPR 2021).
- Read scope: ar5iv HTML summary (annotation, agreement, validity paragraphs); the PDF could not be parsed. Locators below are not precise.
- Setting: dataset paper plus neural speakers (SAT, M2-style) and emotion classifiers (BERT, ResNet). WikiArt paintings, English.
- Change tested and strongest baseline: new corpus versus COCO captions on affect, abstraction and similes (COCO is the baseline).
- Effect: not a retrieval paper. Emotion prediction from text: BERT 65.7%, humans 60.3%; binary positive/negative 91% (human). Emotion alignment of the grounded neural speaker with the majority emotion: 52.2%.
- Multiplicity measured: yes. [raw-checked: 439,121 explanations, 6,377 annotators, 36,347 words, 45.6% (37,145) strong majority, 94.7% in 1K trials, 97.5% appropriate] The paper reports 439,121 explanations on 81,446 artworks from 6,377 workers, at least 5 annotators per artwork. Several secondary sources say "455K on 80K" (not found in the v1 HTML), so cite 439,121.
- What counts as a reading or disagreement: one annotator's (emotion label out of 9 including "something else", free-text explanation). Disagreement is differing emotion labels across the same painting's annotators.
- Agreement statistics: strong majority on one fine-grained emotion for 45.6% of paintings (37,145); 61% of paintings carry at least one positive and one negative emotion; "something else" is 11.7% of labels. By genre, landscapes reach majority agreement most often (60.0%), nudes least (32%).
- Retrieval protocol and numbers: none.
- Noise / validity analysis: 97.5% of sampled utterances judged reasonable; Turkers picked the target artwork from the explanation in 94.7% of 1K trials (a discriminativeness check, a small retrieval-like test); 76.6% of utterances need more than 4-year-old emotional maturity (COCO: 34.5%); 20.5% of utterances use similes or metaphors. Neural-speaker Turing test: 50.3% judged on par with human.
- Relevance to MultiMAE / H-b: [reader-inferred] the base testbed definition. The 45.6% majority figure and 61% mixed-valence figure bound how much true multi-reading structure exists per painting. The 94.7% discriminativeness result shows explanations carry painting-specific content, so a decoder can be scored against them.
- Method weaknesses: the majority-emotion statistic depends on the 9-way label granularity and on the number of annotators (about 5 to 6); with so few votes, "no majority" partly reflects sampling noise rather than real ambiguity [reader-inferred]. Annotators write one explanation each, so the set of explanations per painting is a small sample of the reading distribution [reader-inferred].

**2. Mohamed et al., 2022. It is Okay to Not Be Okay: Overcoming Emotional Bias in Affective Image Captioning by Contrastive Data Collection (ArtEmis v2.0). CVPR 2022. arXiv 2204.07660.**
- Tier: peer-reviewed (CVPR 2022).
- Read scope: arXiv HTML summary (method and results). Annotator counts per image not found in the summary.
- Setting: dataset extension plus captioning models trained on ArtEmis versus v2 ("Combined").
- Change tested: contrastive collection. For 52,933 emotionally biased paintings (absolute emotion score above 0.30), workers pick, among 24 visually near paintings (VGG16 neighbours), the one that evokes the opposite emotion, then label and explain it. Baseline: original ArtEmis.
- Effect: v1 emotion mix 62% positive versus 26% negative; Combined 47% versus 45%. Local entropy of emotions among 20 visual neighbours rose from 0.805 to 0.855 (+6%). Captioning on Combined versus ArtEmis: CIDEr +20%, METEOR +7%; human preference for Combined outputs 73% of 100 samples. The summary also gave 260,533 new instances; the combined size (~455K) is UNVERIFIED.
- Multiplicity measured: yes, indirectly. [raw-checked: entropy 0.805 to 0.855 at K = 20, 62% / 26% split, 260,533 instances, at least five submissions per painting, 3% no-image option (7,752)] Local emotion entropy over visually similar paintings is the measure; it is a property of neighbourhoods, not of annotators of one painting.
- What counts as a reading or disagreement: opposite-valence labels on near-duplicate-looking paintings. Disagreement is sampled across images rather than across annotators.
- Agreement statistics: none reported at painting level in the summary.
- Retrieval protocol and numbers: none. (The nearest-neighbour step uses VGG16 features only to build candidates.)
- Noise / validity analysis: human preference test above; the "No image available" option was used in about 3% of cases (7,752 instances).
- Relevance to MultiMAE / H-b: [reader-inferred] shows that readings of near-identical-looking images diverge in valence, so an image-conditioned decoder must output a spread, and a per-image distribution is not the same as a neighbourhood distribution. Also a reminder that the ArtEmis emotion prior is skewed (positive 62%).
- Method weaknesses: the entropy gain is measured on the collection's own neighbourhood definition (VGG16, 20 neighbours), so it could partly reflect the way candidates were constructed [reader-inferred]. Gains in CIDEr/METEOR are on references from the same collection protocol, so the metric cannot separate diversity gain from distribution shift [reader-inferred].

**3. Mohamed et al., 2022. ArtELingo: A Million Emotion Annotations of WikiArt with Emphasis on Diversity over Language and Culture. EMNLP 2022. arXiv 2211.10780.**
- Tier: peer-reviewed (EMNLP 2022 main, 2022.emnlp-main.600).
- Read scope: arXiv v1 HTML summary (Tables 3-5, 7; Figs 3, 6). Exact table locators are as the summary reported them.
- Setting: dataset plus baselines. Adds Arabic and Chinese (and a small Spanish set) to ArtEmis English, same protocol.
- Change tested and baseline: cross-lingual and cross-cultural annotations versus ArtEmis English; baselines XLM-based emotion classifiers (XLM-ACE, 3-Headed-E) and captioners (SAT, M2-transformer).
- Effect: English emotion accuracy 0.663 (XLM-ACE), Arabic 0.558, Chinese 0.927 (Table 5); English captioning BLEU-4 6.2 to 8.7, Chinese 8.3 to 11.9 (Table 7).
- Multiplicity measured: yes. Annotation counts (Table 3): English 429K (6,377 workers), Chinese 426K (745), Arabic 369K (656), Spanish 4.8K (31); total 1.2M on about 80K paintings, five annotators per painting and language; more than 51K paintings have 5 or more annotations in all three languages.
- What counts as a reading or disagreement: one annotator's (emotion, caption); disagreement is between annotators within a language and between languages.
- Agreement statistics: Cohen's kappa within and across languages (Fig 6: same-language pairs agree more than cross-language pairs; values not captured). Genre agreement as a log-likelihood ratio A (Table 4): landscapes -1.08 (most agreement), sketches 0.85 (least).
- Retrieval protocol and numbers: none reported (the summary and the paper's focus are emotion classification and captioning). Not found in Q6-Q8 either.
- Noise / validity analysis: rejection of too-short or too-similar captions plus manual review. Authors acknowledge WikiArt skew: Western works 78.7%, Chinese 0.5%, Arabic 0.6% of the collection.
- Relevance to MultiMAE / H-b: [reader-inferred] this is the dataset the project plans to use. It provides about 5 readings per painting per language, and cross-language readings of the same painting give a second, independent reading sample.
- Method weaknesses: kappa over a 9-way label with 5 annotators who differ per painting is a pairwise statistic over non-fixed rater pairs, so its magnitude is hard to interpret [reader-inferred]. Chinese emotion accuracy of 0.927 versus English 0.663 suggests label-distribution differences (a more peaked Chinese label prior) rather than model quality [reader-inferred].

**4. Mohamed et al., 2024. No Culture Left Behind: ArtELingo-28, a Benchmark of WikiArt with Captions in 28 Languages. EMNLP 2024 (2024.emnlp-main.1165, pp. 20939-20962). arXiv 2411.03769.**
- Tier: peer-reviewed (EMNLP 2024 main).
- Read scope: arXiv HTML summary (data collection, Fig 4, baselines tables); ACL PDF not parsable. No shared task found (see section 4).
- Setting: dataset plus zero-shot, few-shot and one-vs-all zero-shot baselines; XLM-RoBERTa for emotion prediction; MiniGPT-4, InstructBLIP, ClipCap for captioning.
- Effect: caption-to-emotion accuracy 0.37 (base) and 0.354 (ArtELingo) to 0.664 (ArtELingo-28), F1 0.342 / 0.313 to 0.638 [raw-checked, Table 4; the summary's "ArtELingo only = 0.37" was the untuned base row]. Zero-shot captioning on 25 unseen languages: MiniGPT-4 BLEU-4 1.09, CIDEr 0.33; InstructBLIP 0.54 / 0.05; ClipCap 0.18 / 0.02. Few-shot MiniGPT-4 BLEU-4 13.5 (20% of data) versus 13.1 (100%).
- Multiplicity measured: yes. 2,000 WikiArt images, about 140 annotations per image over 28 languages (Table 1 of the paper lists 5.68 annotations per image for ArtEmis, 15.3 for ArtELingo's 3 languages and 140 for the 25 new languages; the 5.68 is not an English-in-ArtELingo-28 figure, as an earlier summary suggested) [raw-checked, Table 1], 200K new annotations over 25 languages, 220 annotators from 23 countries (8.8 per language).
- What counts as a reading or disagreement: each language's annotators are a "culture"; disagreement is the divergence between per-language emotion distributions.
- Agreement statistics: pairwise KL divergence between per-language emotion distributions; hierarchical clustering (Fig 4) gives two large clusters, African languages in the larger; Darija and Urdu far apart despite shared script.
- Retrieval protocol and numbers: none.
- Noise / validity analysis: 1 to 5% rejection per language; 500 random annotations per language translated and validated; authors note tokenizer-induced metric scale differences across languages and that annotators are mostly educated bilingual internet users.
- Relevance to MultiMAE / H-b: [reader-inferred] the densest per-image reading sample found: about 140 annotations on 2,000 paintings. This allows estimating a per-painting emotion distribution with low sampling noise, i.e. a reference distribution to test a decoder against.
- Method weaknesses: only 2,000 images, so any per-image distribution test has a small number of items [reader-inferred]. Translation to English for validation can hide culture-specific wording [reader-inferred].

**5. Achlioptas et al., 2023. Affection: Learning Affective Explanations for Real-World Visual Data. CVPR 2023 (not ICCV; the brief's venue is wrong). arXiv 2210.01946.**
- Tier: peer-reviewed (CVPR 2023).
- Read scope: arXiv HTML summary (dataset statistics, Table 3, Fig 7, user studies).
- Setting: dataset plus neural speakers (Show-Attend-Tell backbone, with and without pragmatic reranking) and CLIP-based identification.
- Change tested and baseline: real-world photos (not art) versus ArtEmis and COCO; pragmatic speakers versus default speaker.
- Effect: speaker Table 3: CLIPScore 66.7 to 69.2 (pragmatic about 2.5 higher), RefCLIPScore 75.0 to 76.3, unique productions 78.7 to 83.7%, emotion alignment 48.1 to 55.9%. Text-to-emotion BERT accuracy 72.5% (9-way), 95.5% (binary); image-to-emotion ResNet-101 59.1% on 5,672 test images with a strong majority.
- Multiplicity measured: yes. 526,749 explanations on 85,007 images from 6,283 workers, at least 6 annotators per image.
- What counts as a reading or disagreement: same as ArtEmis (label plus explanation); explanations must reference visible content.
- Agreement statistics: [raw-checked: 67.5% (57,381), 526,749 / 85,007 / 6,283, at least 6 annotators, listening study 99.5/97.9 and 96.5/89.7] strong majority emotion on 67.5% (57,381) of images versus 45.6% in ArtEmis; 50.0% of images have at least one positive and one negative label; labels 71.3% positive, 21.1% negative.
- Retrieval protocol and numbers: a "listening study" with CLIP. Each ground-truth image-explanation pair is paired with k uniformly random distractor images from the same dataset (Fig 7). Accuracy picking the true image: k = 1 distractor 97.9% (Affection) vs 99.5% (COCO); k = 10 distractors 89.7% vs 96.5%. It is not full-gallery retrieval (random distractors, small k), and the CLIP variant is not stated in the summary (UNVERIFIED).
- Noise / validity analysis: emotional Turing test, both utterances judged human-made 41.4% to 46.2% of the time. Concreteness score 2.82 (COCO 3.55), subjectivity 0.53 (ArtEmis 0.47).
- Relevance to MultiMAE / H-b: [reader-inferred] the only paper in this lineage with a CLIP matching number on affective explanations; it shows explanations remain image-discriminative even though emotions disagree. A second affective testbed, with a higher majority share (67.5%) than art.
- Method weaknesses: random distractors make the task easy and cannot reveal whether other valid explanations of the same image get ranked above the true one [reader-inferred]. Majority share rises with six or more annotators per image versus five in ArtEmis, so the 67.5% versus 45.6% comparison mixes domain with annotator count [reader-inferred].

**6. Mohamed, Church, Elhoseiny, 2026. Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data (PercepT). arXiv preprint 2606.03345, submitted 2026-06-02.** Title confirmed; the short name is "PercepT (Perception topic Transformer)" and the task is "P-Topics modeling".
- Tier: preprint, preprint>=2024.
- Read scope: arXiv HTML summary (Tables 1-2, human evaluation, limitations). Paper not read in full.
- Setting: unsupervised clustering of image-caption pairs into perception topics (stage 1) plus attention-pooling mapper from image to topics (stage 2). Data: ArtELingo (~1.2M art pairs, 28 languages) and Affection (~500K). 
- Change tested and baseline: PercepT versus BERTopic (multimodal), CEMTM, VLM2VEC (v1, v2), k-means on CLIP embeddings, k-means on fused embeddings.
- Effect (Table 1, 2): silhouette 0.97 vs BERTopic 0.37; V-measure 0.19 vs 0.07; AMI 0.18 vs 0.07; mapping AUC 0.94 vs 0.77; F1 0.61 vs 0.38. Human evaluation: 4 CS PhD students, 113 samples, PercepT preferred 58.4% vs BERTopic 19.5%, ties 22.1%.
- Multiplicity measured: partly. A P-Topic is a distribution over image-caption pairs; an image can be mapped to several topics. No annotator-level agreement is reported.
- What counts as a reading or disagreement: a topic cluster that combines a factual and an affective aspect.
- Agreement statistics: none; authors list no quantitative image-relevance measure as a limitation.
- Retrieval protocol and numbers: none (the paper reports clustering and topic-mapping classification only).
- Noise / validity analysis: only the small human study above (no inter-rater agreement given). Machine translation of multilingual captions may reduce cultural signal (author-acknowledged).
- Relevance to MultiMAE / H-b: [reader-inferred] a topic-assignment task rather than retrieval; it defines "several perceptions per image" as multiple topic memberships, which is the nearest published notion to the project's readings and the baseline named in the author's other project. Not a decoder-distribution measure.
- Method weaknesses: silhouette 0.97 is computed in the model's own learned embedding space, while the baseline's silhouette is computed in a different space, so the gap is not like-for-like [reader-inferred]. Topics are derived from the same captions used to evaluate them, so the multi-label ground truth is self-generated [reader-inferred]; the author's internal Stage 2 report notes the same circularity in the mapper targets (read for context only, not cited).

**7. Chen et al., 2026. ArtECulture: Benchmarking Culture-Conditioned Visual Emotion Understanding in Multimodal Large Language Models. arXiv preprint 2608.03358.**
- Tier: preprint, preprint>=2024.
- Read scope: arXiv HTML summary (dataset and agreement paragraphs only).
- Setting: new benchmark of 6,792 artworks with English, Chinese and Arabic annotations from native speakers (Prolific); MLLMs evaluated zero-shot with a retrieved "cultural emotion knowledge base" of 8,439 concept-emotion tuples.
- Change tested: culture-conditioned prompting versus unconditioned. Strongest baseline numbers not captured; best model below 50% overall zero-shot accuracy.
- Multiplicity measured: yes. 92,062 emotion-explanation pairs, about 13.55 per image (30,932 English, 31,559 Chinese, 29,571 Arabic).
- What counts as a reading or disagreement: an annotator's emotion in a given culture; disagreement is across annotators and across cultures.
- Agreement statistics: within-culture Krippendorff's alpha 0.309 (English), 0.324 (Chinese), 0.306 (Arabic); the majority emotion receives on average 70.81% (English), 76.32% (Chinese) and 70.13% (Arabic) of votes per image [raw-checked, Agreement Analysis paragraph; alpha, 13.60%, 20.33% (924 and 1,381 images) and kappa 0.088 also raw-checked]; only 13.60% of images get the same emotion in all three cultures, 20.33% get three distinct emotions; cross-culture Fleiss kappa 0.088.
- Retrieval protocol and numbers: none (the "retrieval" is of knowledge-base tuples for prompting).
- Noise / validity analysis: agreement coefficients above; no explanation-level validity study captured.
- Relevance to MultiMAE / H-b: [reader-inferred] the only external re-collection found with chance-corrected agreement on ArtEmis-style data. Alpha around 0.31 is a quantitative upper bound on how much of the emotion label is stable across annotators.
- Method weaknesses: the summary reports majority-vote shares but not how many annotators vote per image per culture (about 4.5 each on average), so the 70 to 76% majority vote share is not comparable with the 45.6% in ArtEmis [reader-inferred]. Not assessed further (read scope: summary of dataset paragraphs only).

### 2b. Retrieval and matching, and distributional modelling of the annotators

No paper in Q6-Q8 reports image-to-text or text-to-image retrieval (R@K) on ArtEmis, ArtELingo or ArtELingo-28 with CLIP zero-shot or fine-tuned. The closest numbers are the Affection listening study (entry 5), and the art-adjacent retrieval benchmarks below. This absence is bounded by Q6-Q8 and the direct dataset-paper reads; it does not exclude unindexed workshop papers or theses.

**8. Bergeron et al., 2026. Dimensional Distribution Emotion State: Leveraging Valence and Arousal as a Common Embedding Space for Visual Emotion Analysis (DDES). arXiv preprint 2605.26262.**
- Tier: preprint, preprint>=2024.
- Read scope: arXiv HTML summary, first 100,000 of 102,258 characters (Tables 1 and 3).
- Setting: image-to-emotion models trained on an enhanced ArtEmis ("ArtEmis Enhanced") plus other emotion datasets.
- Change tested and baseline: distributional target (kernel density over valence-arousal built from all annotators' labels and explanation sentences, via the NRC VAD lexicon and a pretrained language model) versus a categorical network (CES-Net) and a dimensional point network (DES-Net); Qwen-2.5-7B as an MLLM reference.
- Effect: ArtEmis accuracy DDES-Net 39.8% versus CES-Net 39.7% (Table 1, essentially tied). Zero-shot transfer (Table 3): EEmo-Bench 21.2%, WikiArt Emotions 10.5% for DDES-Net.
- Multiplicity measured: the target keeps all annotator readings as a distribution; the evaluation is still accuracy, i.e. not a distribution-matching score in the figures captured. UNVERIFIED whether a divergence metric is also reported.
- What counts as a reading or disagreement: a point in valence-arousal space for each annotation; disagreement is the spread of the density.
- Agreement statistics: none reported beyond the distributional representation itself.
- Retrieval protocol and numbers: none.
- Noise / validity analysis: none captured.
- Relevance to MultiMAE / H-b: [reader-inferred] a precedent for using the full set of per-annotator readings as a target distribution on ArtEmis; the 39.8% versus 39.7% tie shows that distributional training did not help accuracy on this data.
- Method weaknesses: valence-arousal coordinates of explanation sentences come from a lexicon plus an LM, so the target depends on that proxy [reader-inferred]. Accuracy rewards the mode, so a distributional benefit would not show up in that metric [reader-inferred].

**9. Garcia and Vogiatzis, 2018. How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval (SemArt, Text2Art). ECCV 2018 Workshops. arXiv 1810.09617.**
- Tier: peer-reviewed (ECCV 2018 Workshops).
- Read scope: ar5iv HTML summary; PDF not parsable.
- Setting: from-scratch cross-modal retrieval models (CML, with ResNet50 image features and bag-of-words text encodings); 21,384 triplets (image, one artistic comment, attributes) from the Web Gallery of Art, split 19,244 / 1,069 / 1,069.
- Change tested: multi-modal retrieval models against other encoders; metrics median rank (MR) and R@K (K = 1, 5, 10).
- Effect: best CML + ResNet50 + BoW: text-to-image R@10 45.4%, MR 14; image-to-text R@10 45.7%, MR 14. Human check on the easier variant: humans 88.9%, best model 75%.
- Multiplicity measured: no. One comment per painting.
- What counts as a reading or disagreement: a single expert comment per painting, so no disagreement is represented.
- Agreement statistics: none.
- Retrieval protocol and numbers: Text2Art, full test gallery of 1,069 items, numbers above. This is the standard retrieval protocol on art images and is the oldest.
- Noise / validity analysis: none captured.
- Relevance to MultiMAE / H-b: [reader-inferred] shows the retrieval protocol used for art-text matching has a single positive per query, so it cannot see several readings; useful as the contrast to ArtELingo's 5 to 6 captions per painting.
- Method weaknesses: single comment includes context such as author and technique, so retrieval can partly rely on metadata-like text rather than visible content [reader-inferred].

### 2c. Affective captioning and generation

**10. Zhang et al., 2024. Training A Small Emotional Vision Language Model for Visual Art Comprehension (SEVLM). arXiv preprint 2403.11150 (March 2024).**
- Tier: preprint, preprint>=2024.
- Read scope: arXiv v1 HTML summary (abstract-level plus results lines).
- Setting: small VLM trained on ArtEmis v1 and v2 to predict an emotion label and generate an explanation. Components: valence-arousal-dominance (VAD) lexicon features, a VAD head, and a contrastive head that aligns image, emotion label and explanation features.
- Change tested and baseline: M2, SAT, NLX-GPT2, fine-tuned LLaVA 7B, GPT-4V.
- Effect: ArtEmis v1 emotion accuracy 65.6% versus baseline 63.5%; emotion alignment 62.1%.
- Multiplicity measured: no (the summary shows accuracy against the single majority label plus n-gram metrics).
- What counts as a reading or disagreement: not modelled; the target is one label and one reference.
- Agreement statistics: none.
- Retrieval protocol and numbers: none (the contrastive head is an auxiliary loss).
- Noise / validity analysis: none captured.
- Relevance to MultiMAE / H-b: [reader-inferred] an example of ArtEmis generation work that scores against one reference and one label, so coverage of the human explanation set is not measured; Q9 and Q10 found no paper that does measure coverage of the full per-image explanation set.
- Method weaknesses: not assessed beyond summary (read scope: summary of abstract and results).

**Affection (entry 5)** and **ArtEmis v2 (entry 2)** also belong here: Affection reports unique-production rates (78.7 to 83.7%) and a pragmatic variant that raises CLIPScore; ArtEmis v2 reports human preference and entropy of the data, not of the generations.

### 2d. Other datasets with several human interpretations per image

**11. Hessel et al., 2022. The Abduction of Sherlock Holmes: A Dataset for Visual Abductive Reasoning. arXiv 2202.04800 (v2, 2022-07-25; ECCV 2022 per common citation, venue UNVERIFIED from the page read).**
- Tier: treat as peer-reviewed ECCV 2022 [UNVERIFIED venue]; the page read lists arXiv only.
- Read scope: arXiv HTML summary, first 100,000 of 109,796 characters (Table 2, task definitions).
- Setting: 363K commonsense inferences on 103K images (81K Visual Genome, 22K VCR); split 90K / 6.6K / 6.6K images; at least 3 clue-inference pairs per image.
- Change tested and baseline: CLIP RN50x64 fine-tuned with multitask clue + inference training versus single-task.
- Effect (Table 2): retrieval P@1 33.4% among about 1K candidates [raw-checked, Table 2, last model row; human comparison 42.3]; localization 87.2% (human upper bound 92.3%); comparison pairwise accuracy 27.1% (multitask; 26.0% single-task, human agreement 42.3%).
- Multiplicity measured: yes. Several inferences per image; each inference carries a Likert confidence (definitely 31%, likely 51%, possibly 18%); the comparison task has 19K human plausibility judgments over about 10 inferences.
- What counts as a reading or disagreement: each distinct inference is a reading; plausibility is graded, not binary.
- Agreement statistics: human agreement on the comparison task 42.3% (model 27.1%).
- Retrieval protocol and numbers: image-to-inference retrieval, about 1K candidates, mean rank and P@1; 33.4% best.
- Noise / validity analysis: 17K pairs validated, 97.5% acceptable, 98.3% bounding-box accuracy, 71.9% rated interesting.
- Relevance to MultiMAE / H-b: [reader-inferred] the best non-affective testbed found: graded plausibility labels for several valid readings per image, with a public retrieval protocol, and non-art images so a COCO-initialised model is closer to in-domain than WikiArt.
- Method weaknesses: the retrieval positives are the annotators' own inferences, but the candidate pool contains other inferences of the same image, which may be equally valid; retrieval P@1 therefore undercounts plausible answers [reader-inferred].

**12. Shuster et al., 2019. Engaging Image Captioning via Personality (Personality-Captions). CVPR 2019. arXiv 1810.10665.**
- Tier: peer-reviewed (CVPR 2019).
- Read scope: ar5iv HTML summary (Tables 4, 5, 7, 8).
- Setting: 241,858 captions on YFCC100M images, 215 personality traits; train 186,858 images, validation 5,000, test 10,000 (5 captions each per the summary). Models: TransResNet retrieval, UpDown generation.
- Effect: retrieval R@1 77.5% (ResNeXt-IG-3.5B image encoder, Transformer text pretraining) over 500 candidates (495 random plus the correct one); generation CIDEr 16.5; human preference 64.5% win rate over traditional captions, and 49.5% for the best retrieval model against human captions.
- Multiplicity measured: partly. The same image gets different personalities across the dataset, not several traits for one image in training; test images have 5 captions.
- What counts as a reading or disagreement: a personality-conditioned caption; the style is given as input, so the "reading" is controlled rather than inferred.
- Agreement statistics: none.
- Retrieval protocol and numbers: described above (500 candidates, R@1 77.5%).
- Noise / validity analysis: personality fit 87.7% and image fit 90.2% in human evaluation (Table 8).
- Relevance to MultiMAE / H-b: [reader-inferred] a conditional-style dataset, so closer to the author's conditional-retrieval line than to H-b; included as a size and protocol reference.
- Method weaknesses: 495 random distractors make retrieval easy, as with Affection's listening study [reader-inferred].

**13. Saakyan et al., 2024. V-FLUTE: Visual Figurative Language Understanding with Textual Explanations. arXiv 2405.01474 (title as returned by the search: "Understanding Figurative Meaning through Explainable Visual Entailment" for v2).**
- Tier: preprint, preprint>=2024. Author list not confirmed (UNVERIFIED; omitted here).
- Read scope: search-result snippet only (abstract level).
- Setting: 6,027 (image, caption, label, explanation) instances covering metaphors, similes, idioms, sarcasm and humor, built by human-LLM collaboration with expert verification.
- Multiplicity measured: no evidence of several readings per image; one label and one explanation per instance.
- Relevance to MultiMAE / H-b: [reader-inferred] a figurative testbed where the reading is a label (entailment or contradiction) rather than a set; weak fit as a second testbed.
- Method weaknesses: not assessed (read scope: snippet only).

## 3. Closest-to-H-b list

1. **ArtELingo-28 (entry 4)**: about 140 annotations per painting over 28 languages on 2,000 paintings; the only data found dense enough to estimate a per-painting reading distribution, with KL divergence between language-level emotion distributions. It does not report any retrieval or decoder-side measure.
2. **Affection (entry 5)**: the only affective-explanation paper with a CLIP matching number (listening study, 97.9% / 89.7% at 1 / 10 random distractors) and a 67.5% majority-emotion share; it shows explanations remain discriminative while emotions vary, and it does not test whether other annotators' explanations are retrieved.
3. **Sherlock (entry 11)**: several inferences per image with graded plausibility, a public retrieval protocol (P@1 33.4%) and a comparison task with human agreement 42.3%; it provides graded multi-reading labels outside art but not a decoder or distribution measure.
4. ArtEmis (entry 1) for the agreement statistics (45.6% majority, 61% mixed valence) and DDES (entry 8) as the one precedent for training on the full annotator distribution (tie with a categorical baseline on accuracy).
5. ArtECulture (entry 7) for chance-corrected agreement (alpha 0.31, Fleiss kappa 0.088 across cultures) as an outside check on the stability of the emotion label.

## 4. Search limitations

- Not found in Q6-Q8, Q13: any paper reporting image-text retrieval Recall@K on ArtEmis, ArtELingo or ArtELingo-28 (CLIP zero-shot or fine-tuned). The Affection listening study and SemArt Text2Art are the nearest. Workshop and thesis work indexed outside the searched sources could still exist.
- Not found in Q9, Q10: affective captioning work that measures coverage or diversity of the full per-image set of human explanations (Self-BLEU or similar against the annotators' set). Affection's unique-production rate and ArtEmis v2's entropy of the data are the only diversity-flavoured numbers found.
- Not found: a shared task built on ArtELingo-28 (Q3, Q5). The ArtELingo-28 paper presents baselines and a benchmark on the project site (artelingo.org); no separate shared-task paper was found.
- Not read from the PDF: ArtEmis CVPR PDF, ArtELingo PDF, ArtELingo-28 ACL PDF, SemArt PDF, because the PDFs came back binary and no PDF text tool was available. All numbers for those papers come from HTML-derived summaries and should be re-derived from the papers before they go into a report. Specific unresolved items: ArtEmis size (439,121 on 81,446 versus "455K on 80K" in the abstract and secondary pages), ArtEmis v2 combined size, CLIP variant in Affection's listening study, ArtELingo Cohen kappa values.
- The brief says Affection is ICCV 2023; the CVF page and arXiv list it as CVPR 2023 (arXiv 2210.01946). The brief's list also did not give an arXiv id for ArtELingo-28; it is 2411.03769.
- Datasets I looked at and did not include as sources: IRFL (Yosef et al., arXiv 2303.15445; read scope snippet only; figurative-vs-literal image selection, no multi-reading per image), A-MAR (arXiv 2604.19689; QA over 227 SemArt paintings; mentions multiple plausible reasoning formulations but does not collect them), "Attribute-Grounded Selective Reasoning" (arXiv 2605.15755; 1,400-image salience benchmark with 15 annotators, a different kind of disagreement, formal attributes), SEVLM's siblings EmoSEM (2504.14658), EEmo-Logic (2602.01173), EmoVerse (2511.12554) and EmoScene (2604.00933) (not opened beyond titles), perspectivist NLP resources (SemEval-2021 Le-Wi-Di, text mostly). Memes and visual-metaphor datasets with several human readings per image were searched in Q11-Q12 only; no suitable one with annotator-level multiplicity was confirmed.
- Counts: about 45 identified, about 25 screened, 13 included; raw HTML re-check done for entries 1 to 5, 7 and 11 (key numbers only); entries 6, 8, 9, 10, 12, 13 remain summary-level.
