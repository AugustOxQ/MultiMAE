# Cross-modal masking and polysemy: deep-think note

Date: 2026-10-05, written under a hard time limit, so shorter than the brief allowed and with less re-derivation than planned.

Marks: **[R]** = a mean we re-derived in this session from `res/coco/multimae/*/run.json` (baseline seed 42 from its `eval` block). **[U]** = from the brief, unverified (every p-value, every ArtELingo count), or from the cited report and not re-derived (Stage 0 and literature numbers).

## 1. What the results show about the mechanism, and what they do not

**Shown.**

1. The gain followed the MLM's share of the masked objective. ECCV mAP@R rose with the text-masking ratio: 36.95 (15%), 37.39 (25%), 37.58 (40%), 37.66 (60%), 37.81 (80%, 3 seeds), 37.89 (90%) [R]. MAE off gave 37.54 and MAE off plus 40% gave 37.53, so the two levers did not add [R]. In Stage 2, 80% masking reached 37.87 against 37.24 (multilearner) and 37.11 (contrastive) [R]; Welch p 0.044 and 0.027, paired p 0.077 against multilearner [U].
2. At 80% the MLM was no longer a cloze task. Its test loss per masked token about doubled, from 1.57 to 3.05 nats [R]: with four of five tokens hidden, the decoder predicts the caption's words almost from the image alone. The working objective was a parallel captioning loss read from 25% of the patches (close to CapPa's parallel prediction as bibliography B4 describes it [U]), and pixel MAE contributed nothing.
3. A better MLM was not a better retriever. M1 clean lowered the MLM loss (1.49 against 1.57) and left mAP@R at 37.05; M2b made each masked token harder (2.13) at the same count and gave 37.25 [R].
4. The gain was general. Against multilearner, 80% masking moved mAP@R +0.63, rsum +4.94, COCO 1K R@1 +0.85, t2i R@1 +1.26, i2t R@1 +0.67 and PMRP +0.22; ECCV mAP@R rose in both directions (i2t about +0.5, t2i +0.72) [R].

**Not shown.**

- That the vision-tower gradient carries the 80% gain. M1 detached ran at 15%, where there was no mAP@R gain to remove (36.95 against 36.94 [R]); it lost 2.26 rsum and 0.29 PMRP on 2 seeds [R]. The detach test at 80% does not exist.
- That the image matters at 80%. `fusion_none` at 80% (text-only MLM) was never run, so "cross-modal" is an assumption at the dose where the effect lives.
- Any tie to polysemy. Every metric moved together, which is what a generally better encoder looks like. PMRP is not evidence: a learning-rate change on the contrastive model alone (R2) gave +0.73 PMRP with mAP@R flat [R], more than any masked arm. H1 (object grounding) found no support in Stage 0 or in M2b. VWSD showed no effect at 15% [U] and has not been read at 80%.
- Robustness. Three seeds, one recipe, one backbone, one dataset. The Holm result against multilearner is borderline once `s2_mae0` joins the family [U], and the R2 recipe was never crossed with masking.

## 2. Could cross-modal masking improve polysemy understanding?

**Strongest argument for (one-to-many).** InfoNCE with one positive per step treats every other caption in the batch as wrong, including captions that fit the image. Over training, the image embedding is pulled toward the mean of its five captions and pushed away from valid alternatives. High-ratio image-conditioned MLM is a likelihood model of words given the image. Across annotators it must spread probability over everything any of them says, and it never punishes a valid alternative. Its gradient asks the visible patches to keep whatever any caption might mention, which is the content one-to-many matching needs. A gain on human-verified ECCV positives in both directions is what this predicts.

**Strongest argument against.** Retrieval uses one pooled vector per item. The distribution over readings lives in the decoder, which is discarded, and a point cannot hold several readings. The lever review points the same way: at this scale probabilistic and mean-only inference scored the same, one-to-many gains came from the loss (about +1 mAP@R for PCME++), and higher text-masking rates improved ordinary retrieval in ITM models (Verma et al., MaskVLM) [U]. Our +0.63 arrived with matching recall gains, so the simpler reading is extra supervision of the vision tower, not multiplicity. For lexical ambiguity the case is weaker still: masking removes context while VWSD rewards adding it, and COCO captions rarely use one word in two senses.

**Our judgment (a guess).** The one-to-many argument is coherent and untested. No current measurement separates it from "a generally better encoder". The deciding quantity is an interaction: does the gain grow with the number of valid matches per query?

## 3. What a real polysemy evaluation should be

**ArtELingo, assessed.** Its strength is that multiplicity is designed in: several annotators give different readings of one painting, and the emotion label says when two readings differ and are not paraphrases (82% of test paintings with at least 3 emotions, median entropy 1.52 bits [U]). Its weaknesses are real. (a) Disagreement mixes genuine alternative readings with annotator noise, and nothing in the data separates them. (b) Affective explanations are weakly grounded ("it feels calm") and fit many paintings, so recall will be low for every model and the ceiling is unknown. (c) Paintings are far from COCO photographs; a COCO fine-tune will probably lose to zero-shot CLIP there (guess), so models must be trained on ArtELingo itself. (d) It tests readings of images, not word senses.

**Measurements that would make it a polysemy test.**

1. Stratify test paintings by annotator emotion entropy. The claim predicts a gain over contrastive that grows with entropy; low-entropy paintings are the built-in control. Test the interaction with a paired bootstrap over the 6,246 test paintings [U], which has far more power than seeds.
2. Reading coverage at k: of a painting's distinct emotion readings, the share with at least one caption in the top k, reported beside plain recall. A model that collapses to the majority reading keeps R@1 and loses coverage.
3. Minority-reading recall: t2i recall for captions carrying the painting's minority emotion against its majority emotion.
4. Noise controls: an emotion-only baseline (how much retrieval a caption's emotion alone achieves), and a judged sample of about 200 caption-painting pairs to estimate how many "alternative" captions are valid.

**Cheaper first.** The same stratification runs on COCO today with no training: per-query ECCV AP by the query's number of ECCV positives and by the diversity of an image's five captions, and mAP@R split into original and added positives. We are confident that ECCV Caption and CxC support this, and that CUB captions (class-level positives, used by PCME) and ArtEmis (ArtELingo's English parent) exist. We know of no retrieval benchmark with labelled alternative readings beyond these.

## 4. Candidate paper directions

Costs use 4.2 h (contrastive) and 7.2 h (masked) per run on one A6000 [R].

**A. A caption-likelihood auxiliary loss helps dual encoders where matches are many.** Claim: the gain of high-ratio image-conditioned MLM concentrates on high-multiplicity queries (COCO strata, ArtELingo entropy strata) and survives a one-to-many loss. Key experiment: 2 x 2 (InfoNCE or a PCME++-style loss, with or without 80% MLM), 5 seeds, stratified. Baseline to beat: contrastive on the R2 recipe with the one-to-many loss. Kill: an interaction whose interval includes zero, or no gain on top of the one-to-many loss. Compute: about 115 GPU-hours on COCO (about 20 h on 6 GPUs), about 60 more for ArtELingo (guess). Novelty risk: medium to high; MACCO, MaskVLM and Verma et al. are close [U], and only the one-to-many measurement is ours.

**B. Embeddings that keep the readings.** Claim: a small set of image embeddings, each trained through the MLM decoder to explain a different caption, raises reading coverage at equal R@1. Key experiment: set size 1, 2, 4 on ArtELingo and ECCV Caption. Baseline: single-vector 80% MLM and a PVSE or DivE-style set. Kill: coverage no better than the single vector. Compute: 150 to 200 GPU-hours plus moderate code (guess). Novelty risk: medium; Llip and DivE occupy nearby ground [U].

**C. A controlled null.** Claim: masked objectives improve a fine-tuned CLIP dual encoder as generic vision supervision, with no polysemy-specific effect, shown by dose, detach, tower-swap and stratified analyses. Baseline: the same contrastive fine-tune on two recipes. Kill: the gain vanishes on the stronger recipe. Compute: 60 to 100 GPU-hours. Novelty risk: low risk and a low ceiling (a findings or workshop paper).

## 5. Recommendation for the next two weeks

**Week 1.** (1) Finish Stage 2 with VWSD and the diagnostics as planned. (2) Run the COCO stratified analysis on the existing Stage 2 checkpoints; it decides between A and C at no training cost. (3) Run the two missing controls at 80%, `fusion_none` and M1 detached, 3 seeds each (about 43 GPU-hours). (4) Run 80% masking on the R2 recipe, 3 seeds, plus one more R2 contrastive seed (about 26 GPU-hours).

**Week 2.** An ArtELingo pilot: zero-shot, contrastive and 80% MLM trained on its train split, 3 seeds, scored with the stratified and coverage metrics. Go on to A only if the entropy interaction is positive; otherwise write C.

**Stop.** Reading PMRP as evidence for anything. VWSD beyond the planned Stage 2 reading. Pixel MAE (drop the image decoder and save its compute). M1 clean, M3, M6, mean pooling and further dose points. Comparing fusion variants before the 80% `fusion_none` control says whether fusion matters at all.
