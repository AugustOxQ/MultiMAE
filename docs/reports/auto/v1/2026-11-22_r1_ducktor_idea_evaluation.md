# Idea evaluation, loop r1: both ideas get Accept with Revisions, idea 2 (masked sub-space PVSE) worth pursuing pending a one-seed mechanism screen, idea 1 (masked captions under blends) only after its claim moves from retrieval to the decoder on blends

Date: 2026-11-22. Skill: Supervisor-Skills idea-evaluator via Ducktor duck 0.2.3, run by an Opus worker at xhigh effort, once per idea, then a trade-off. Inputs: the two idea figures in `docs/reports/assets/2026-10-26_r1_ideas/` (confirmed by the user), the direction log ("Now" and the entry of 2026-10-26), the synthesis of the [step-3 note](2026-10-26_r1_candidate_ideas.md), the two literature checks of 2026-10-26 ([idea 1](../../literature/2026-10-26_masking_under_augmentation.md), [idea 2](../../literature/2026-10-26_masked_subspace_pvse.md)), the results brief of the [emotion-readout report](2026-10-08_hb_emotion_readout.md) (2026-10-08), `GLOSSARY.md`, and the user's facts on goal, results and resources from the controller's brief.

Conventions:
- Scores are on a 1 to 10 scale (x/10). Each starts at 5 and moves only on reported data or a stated mechanism.
- ECCV Caption mAP@R is given at one decimal, as in the published tables (our contrastive model 37.1, the masked model at 80% 37.9).
- Severity labels: CRITICAL (rejects the idea by itself, cannot be fixed within its lifecycle), MAJOR (needs 2 to 4 weeks of dedicated work to defend), MINOR (under a week of writing or literature work).
- Flaw labels: F1 (no novelty over the closest prior work), F3 (the baseline is not the real baseline), F6 (a claim the planned experiments cannot verify).
- The figure calls (a) and (b) two "readings" of "masked sub-space". Here they are variant (a) and variant (b), so "reading" keeps its glossary meaning: one valid interpretation of an image.
- Verdicts, scores and rationales are our reading through the skill, not results.

## Idea 1: a masked caption loss under a designed blend augmentation

### 1. First impression

- Paper type: Novel Method (a training objective under a designed augmentation). Paradigm: STEM (experiments, ablations), so the five dimensions and the flaw list apply as written.
- One-sentence story: when one picture holds two scenes, a masked caption decoder whose few visible words name one scene must take the rest of that scene's words from the image, so the words select a reading and the image must keep both.

The story is clear on blends. The step from blends to natural one-to-many retrieval, the main metric, is not in the sentence. The audit turns on that step.

### 2. Fatal-flaws audit (early gate)

No CRITICAL flaw. Two MAJOR flaws, so the verdict path is Accept with Revisions.

| # | Flaw | Severity | Defense |
|---|---|---|---|
| 1 | F6. The main claim (more polysemic retrieval, ECCV mAP@R) is not tied to the masked mechanism. The pooled retrieval embedding meets the masked loss only as an auxiliary gradient through the image tower. The blend's contrastive term, in its α-weighted form, is PCME++'s mixing (step-3 note). So a gain on mAP@R cannot be credited to text-selected unmixing, and a null would not refute it. On this metric the design falls back to "a masked loss as an auxiliary loss", the framing the user set aside on 2026-10-26. The selection task also exists only because the augmentation creates it. | MAJOR | Make the decoder claim on blends primary: the visible words select the source, and the image supplies its hidden words. Pre-register the must-have control from the check (A's visible words with image B alone) and the prefix captioner on blends without masking. Keep mAP@R as a no-harm check. If a retrieval claim stays, it must exceed MixGen on our fine-tune at the same mix rate and the masked model at 80% without blends, and grow with the number of valid matches. |
| 2 | F3. The real baseline is PCME++ with its mixing, and the recipe gap is larger than the effects in play. PCME++'s mixing row moved mAP@R from 38.9 to 39.0. MixGen's single-caption variant gave no gain (COCO RSUM 470.0 against 470.7). Our contrastive model is at 37.1 against 39.0 for the published InfoNCE row at CLIP B/32: a 1.9-point recipe gap, larger than our 0.8-point masking gain. Shared with idea 2. | MAJOR | Reproduce the published InfoNCE row (39.0) in our code, or report the within-recipe test with the external bar (VSE∞ and PCME++, 40.0 to 40.2) beside it. The second recipe of the lever review raised PMRP and rsum but not mAP@R, so plan 2 to 4 weeks. |

**Novelty (F1): MINOR.** The idea 1 check (40 searches, 2026-10-26) retrieved no directly overlapping work under its keywords. No source blends two images, keeps each source caption as a separate target, and lets the visible words of a masked caption choose the source. One axis differs from every close work: the target structure. That rules out duplication, but the axis is thin.

| Closest work (from the idea 1 check) | Axis that differs |
|---|---|
| Detector-Free Weakly Supervised Grounding by Separation, Arbelle et al., ICCV 2021 | Target: an alpha map, not caption tokens. Setting: grounding, not retrieval. |
| VLMixer: Unpaired VLP via Cross-Modal CutMix, Wang et al., ICML 2022 | Object: words swapped for patches in one pair. No two-image blend, no either-one target, unpaired data. |
| MixGen, Hao et al., 2023 (WACV workshop) | Mechanism: fixed text for the mixed sample (the single-caption variant takes one random source caption). No selection by visible words. |
| Improved Probabilistic Image-Text Representations (PCME++), Chun, ICLR 2024 | Mechanism: mixing with weighted positives, as idea 1's contrastive half, with no masked selector. |
| CLIPSep, Dong et al., ICLR 2023, and The Sound of Pixels, Zhao et al., ECCV 2018 | Modality: query-conditioned mix-and-separate in audio. |

RobustMixGen (2025) and ARMADA were seen only in reference lists, and a winner-takes-all masked-token loss in vision-language got one search. Those, and anything after 2026-10-26: unverified; literature check required.

**Data-refuted check: not refuted, untested.** None of our runs trained on blends with either-one targets. The boundary per result:

| Our result | What it tested | Bearing on idea 1 |
|---|---|---|
| Masked model at 80%: 37.9 against 37.1 mAP@R (5 seeds, Holm p 0.001). Gain in the image tower, no growth with the number of valid matches (direction log, 2026-10-06) | masking as an auxiliary loss on natural pairs | A different claim. A blend-trained model carries this gain too: the attribution risk of flaw 1. |
| Emotion readout on ArtELingo-28: JSD 0.277 against 0.267 (parallel captioner) and 0.265 (linear probe) | the decoder's distribution with the whole caption hidden | A different claim. |
| Visible words steer the decoder: log-loss 1.925 to 1.450 with 8 words | that the selector exists | Supports the selector. Also the source of the language-prior risk. |
| Decoder reranking: mAP@R down 15 to 19 points | the decoder score as a retrieval score | Refutes any variant that ranks by the decoder. Idea 1 ranks by pooled embeddings, so it is outside. |
| Blend probe: small effect of visible words | models never trained on blends | The state before training. The mechanism is untested. |
| VWSD down about 3 points in masked arms | a cost of masking | A cost to report. |

### 3. Lifecycle and capability match

| Aspect | User's input | Assessment |
|---|---|---|
| Idea category | a designed objective for vision-language retrieval | Frontier exploration (3 to 9 months). The polysemy framing is young (Lenses, CVPR 2026). |
| Lifecycle | not stated | About 6 to 9 months. The pieces date from 2018 to 2024 and the check found no direct competitor. |
| Weekly effective hours | unknown | Assumption: 20 hours a week. Agents write the code. |
| Realistic execution | one PhD student, agent support | Our estimate: code 1 to 2 weeks (the blend probe code exists), screen and 3-seed confirmation 3 to 5 weeks, writing 3 to 6, revision 2 to 4. About 2.5 to 4 months. No lifecycle-short flag. |
| Infrastructure | one RTX 3090 (nights, weekends), DAS6 nodes on request | Enough. |
| Fit | | Green |

The lifecycle is reasoned from the pace of the checks' sources. The user should sanity-check it against their view of the subfield.

### 4. Five-dimension radar

| Dimension | Score | Evidence | Lift suggestion |
|---|---|---|---|
| Higher | 4/10 | Mechanism-based, not yet confirmed by data. A natural-metric gain must pass through the pooled embedding, which the blend's contrastive term pulls toward two captions at once. The check's theory says contrastive learning keeps only shared content (Daunhawer et al., ICLR 2023). Mixing moved mAP@R by 0.1 in PCME++ (38.9 to 39.0). Our masking gain did not grow with the number of valid matches. | Error-driven analysis: headline the blend metrics (balance, both covered) and stratify mAP@R by the number of valid matches. |
| Faster | 5/10 | No grounds given. Cost neutral: blends replace part of the batch and the decoder pass stays as today. | None needed. |
| Stronger | 6/10 | Mechanism-based. It targets ambiguity directly, with a concrete selector and a test bed of exact readings. Held down by the language prior (the check's bistable-image study, our steering result) and by the VWSD drop of about 3 points in masked arms. | Noise tolerance: the must-have control, and a spatial blend (one source per region) beside the alpha blend. |
| Cheaper | 6/10 | Mechanism-based. Blends make two-reading samples with exact labels at no annotation cost. Lenses (CVPR 2026, per the idea 2 check) annotated 732k captions in 5 lens categories. | Count training pairs with known readings per GPU-hour against an annotated set. |
| Broader | 6/10 | Transplants query-conditioned mix-and-separate (The Sound of Pixels, CLIPSep) and text-selected unmixing (Grounding by Separation) into masked caption reconstruction. Part of this transplant exists. | Cross-domain transplantation: the same selector on ArtELingo, where readings are natural. |

Highest ceilings: Stronger and Cheaper, 6/10 each. No dimension reaches 7, so the rubric calls the idea vague and asks the user to name the dominant axis before a re-run of this evaluation. Our suggestion: Stronger, framed as "keep both readings of an image whose readings are known exactly".

### 5. Paradigm-shift probe

| Probe | Answer | Rationale |
|---|---|---|
| First Principles | Partial | Challenges "a mixed sample has one fixed target" (MixGen, PCME++'s weighted positives) with an either-one target chosen by context. |
| Elephant in the Room | No | The known elephant here, the language prior of vision-language models, is a risk for idea 1. Idea 1 does not solve it. |
| Technology Cycle | No | Blends and masked decoders were both available by 2018 to 2022. |
| Hamming's Rule | Partial | A model that keeps several readings would matter. Unmixing blends alone would not change the field's priorities. |

Probe score 2 of 8 (Yes 2, Partial 1, No 0). Disruptive potential: none. Frame it as incremental work.

### 6. Feasibility

| Risk | Level | Mitigation |
|---|---|---|
| Compute | Low | About 5 arms × 3 seeds × 3 to 7 GPU-hours, so 45 to 105 GPU-hours: one to two weeks of nights and weekends on the 3090, or one to two days on a DAS6 node. Screen at one seed first. |
| Data | Low | COCO with ECCV Caption, ArtELingo and VWSD are on disk. Blends are made on the fly. |
| Engineering | Low | The blend probe code exists. The either-one target and the multi-positive loss are small changes to `mmae`. |
| Timeline | Low | A first decisive result (the must-have control on blends) in about 3 to 4 weeks. |

### 7. Verdict

**Accept with Revisions**

The flaws are fixable and our data do not refute the mechanism. As framed, the main metric does not test the mechanism, and no dimension reaches 7/10.

Top three actions to take first:
1. Move the primary claim to the decoder on blends. Pre-register the must-have control (A's visible words with image B alone) and the prefix captioner on blends without masking. Keep ECCV mAP@R as a no-harm check beside MixGen on our fine-tune at the same mix rate.
2. Run a one-seed screen: the masked model at 80% with and without blends, and the prefix captioner on blends. Read balance, both covered and the must-have control on blends from the COCO validation split. ECCV Caption sits on the 5k test images, so it must not decide the go. Go on only if the image, not the words, drives the selection.
3. Settle open points 1, 2 and 5 of the idea 1 check (a winner-takes-all masked-token loss in vision-language, Grounding by Separation's method, RobustMixGen and ARMADA). Write the one-sentence position against MixGen and PCME++'s mixing, and name the dominant axis.

## Idea 2: PVSE with masked sub-space training in place of the K attention heads

### 1. First impression

- Paper type: Novel Method. Paradigm: STEM.
- One-sentence story: let a masked caption decoder, not embedding similarity, decide which of K image sub-spaces each caption belongs to, so each sub-space learns one reading and retrieval takes the best sub-space.
- Variant (a): K learned dimension masks on the one pooled 512-d image embedding. Variant (b): K masked patch views through the shared encoder.

The story is compelling. The masked objective chooses which sub-space the contrastive loss trains, so masking shapes the retrieval embedding by construction. It also carries its own test: DML predicts that sub-space use rises with the masking ratio.

### 2. Fatal-flaws audit (early gate)

No CRITICAL flaw. Two MAJOR flaws, so the verdict path is Accept with Revisions.

| # | Flaw | Severity | Defense |
|---|---|---|---|
| 1 | F6. A gain from K sub-spaces need not come from the masked assignment. Multiple embeddings add about 1 mAP@R point at CLIP scale (earlier notes, per the idea 2 check). Doubling the dimensions matched multi-sense methods in NLP (Li and Jurafsky, EMNLP 2015). Plain GPO pooling (VSE∞, 40.0) matches PCME++ (40.1). Without a control that keeps the sub-spaces and swaps only the assignment signal, the paper cannot claim that masking did the work. | MAJOR | The decisive experiment at the same K: decoder-assigned against similarity-assigned sub-spaces (PVSE's best-of-K or Maximal Matching's Hungarian match), K=1 at the same total size, and random fixed masks for (a) or attentive views with multi-positive InfoNCE for (b). Contrast 80% with full caption masking. Three seeds for the confirmation. |
| 2 | F3, as in idea 1. The external bar at CLIP B/32 is 40.0 to 40.2, our contrastive model 37.1, the masked model at 80% 37.9. DivE reports 43.5 mean mAP@R on a backbone the check did not confirm. Shared with idea 1. | MAJOR | As in idea 1. Add DivE's backbone (its Table A1) to the check's open points. |

**Novelty (F1): MINOR, pending one paper.** The idea 2 check (35 searches, 2026-10-26) found no published design that assigns K image embeddings to readings through a masked decoder. For most close works the differing axis is the assignment signal.

| Closest work (from the idea 2 check) | Axis that differs |
|---|---|
| Learning Distinct and Representative Modes for Image Captioning (DML), Chen, Deng, Wu, NeurIPS 2022 | Object: caption-side codes for controllable captioning, not image-side retrieval embeddings. |
| Polysemous Visual-Semantic Embedding (PVSE), Song and Soleymani, CVPR 2019 | Assignment: best-of-K similarity, with a diversity loss. |
| Improving Cross-Modal Retrieval with Set of Diverse Embeddings (DivE), Kim, Kim, Kwak, CVPR 2023, and Maximal Matching Matters, Alomari et al., ACL 2025 | Assignment: similarity (smooth-Chamfer, Hungarian set match). |
| Conditional Image-Text Embedding Networks (CITE), Plummer et al., ECCV 2018, and Conditional Similarity Networks, Veit, Belongie, Karaletsos, CVPR 2017 | Assignment: a learned concept-weight branch, or one mask per known notion. Task: grounding or attribute similarity. Closest to variant (a). |
| Lenses: Toward Polysemous Vision-Language Understanding, Alomari, Asgarov, Thomas, CVPR 2026 | Supervision: 5 annotated lens categories, one of them Emotional, with an MLLM backbone. Its "lens-masking similarity" sits close to variant (a). Read from the abstract only. |

For variant (b), the check found no work where K masked views give K deliberately different embeddings. A-CLIP (ICCV 2023) and DreamLIP (ECCV 2024) keep one image embedding. How Lenses assigns captions to lenses, and anything after 2026-10-26: unverified; literature check required. If Lenses assigns without labels, F1 becomes MAJOR for variant (a).

**Data-refuted check: not refuted, untested.** None of our runs trained K sub-spaces. The boundary per result:

| Our result | What it tested | Bearing on idea 2 |
|---|---|---|
| Partial-view collapse (cosine 0.97) when the vision tower meets partial views only through pixel reconstruction | views without a discriminative or cross-modal loss | Refutes variant (b) without a per-view contrastive loss. The figure adds that loss, so the proposed design is outside. |
| Averaging several partial views added no signal to the emotion readout | random views of a model not trained to make them differ | A different claim. A warning: random scattered views of one image carry the same content. |
| A linear probe on any arm's pooled embedding: JSD 0.265, as low as any decoder | what one pooled vector holds | In (a) the sub-embeddings are masked copies of that vector, so a linear readout from them cannot exceed this probe. (a) adds only through its scoring (max over k) and its decoder, so its claim must be retrieval. |
| The masking gain needs the decoder to read the masked-pass image tokens, and three controls lose it (direction log, 2026-10-06) | where the 0.8-point gain comes from | If (a)'s decoder reads only a pooled sub-embedding, (a) drops this path and starts behind the 37.9 comparator. A cost, not a refutation. |
| Decoder reranking: mAP@R down 15 to 19 points | the decoder as a retrieval scorer | Idea 2 uses the decoder only to assign during training and ranks by max over k of embedding similarity, so it is outside. |

### 3. Lifecycle and capability match

| Aspect | User's input | Assessment |
|---|---|---|
| Idea category | a multi-embedding retrieval method | Frontier exploration (3 to 9 months). The closest polysemy work is recent (Lenses, CVPR 2026), so follow-ups are likely. |
| Lifecycle | not stated | About 6 to 9 months, at the short end if Lenses draws follow-ups. Set-embedding papers came every one to two years (PVSE 2019, DivE 2023, Llip 2024, Maximal Matching 2025). |
| Weekly effective hours | unknown | Assumption: 20 hours a week. Agents write the code. |
| Realistic execution | one PhD student, agent support | Our estimate: code 3 to 4 weeks (sub-space or view module, a decoder conditioned on sub-space k, per-image assignment, batches that hold all of an image's captions, max-over-k scoring in the retrieval and ECCV code), screen and confirmation 5 to 8 weeks, writing 3 to 6, revision 2 to 4. About 3.5 to 5.5 months. No lifecycle-short flag (1.3 × the 6-month midpoint is about 8 months). |
| Infrastructure | one RTX 3090 (nights, weekends), DAS6 nodes on request, none reserved | The 3-seed confirmation needs a DAS6 reservation or several weeks of nights. |
| Fit | | Yellow |

The lifecycle is reasoned from the pace of the checks' sources. The user should sanity-check it.

### 4. Five-dimension radar

| Dimension | Score | Evidence | Lift suggestion |
|---|---|---|---|
| Higher | 6/10 for (b), 5/10 for (a) | Mechanism-based, not yet confirmed by data. A max over K sub-space scores can return top-k sets that one dot product cannot (the dimension cap of single-vector retrieval, Weller et al., ICLR 2026, per the idea 1 check), and mAP@R rewards covering all valid matches. The field's effect is modest: about +1 mAP@R at CLIP scale, and DivE's RSUM went from 492.6 at K=1 to 500.8 at K=4 (Flickr, region features). (a) is lower because its decoder may lose the token path behind our 0.8-point gain. COCO's captions of one image often paraphrase one scene, so sub-spaces may split by style. | Error-driven analysis: stratify mAP@R and R-Precision by the number of valid matches. A one-to-many gain should grow with it. |
| Faster | 4/10 for (b), 5/10 for (a) | Training cost rises: the assignment scores every caption of an image against every sub-space. (b) also runs K encoder passes and stores K vectors per image. (a) keeps one 512-d vector per image, so retrieval cost stays close to today's if the masks do not overlap. | Early exit: assign without gradient, then one gradient pass on the assigned pairs only. |
| Stronger | 7/10 | Mechanism-based. It targets one-to-many ambiguity with a tested ingredient: DML kept 29 of 64 codes in use with a fully masked decoder and Hungarian matching, against 6 at 50% masking, 5 with no masking and 3 without Hungarian matching. Collapse fixes are known: Hungarian or annealed assignment, and shared parameters with no dropout in the assignment step (Shen et al., ICML 2019). Three test beds: COCO, ArtELingo, blends. Not 8: DML's evidence is caption-side, and our 80% masking lies between DML's reported levels. | Decoupled representations: report sub-space use and its change with the masking ratio as the mechanism's signature. |
| Cheaper | 6/10 | Mechanism-based. No reading labels, since the decoder assigns. Lenses needs 5 annotated lens categories over 732k captions. | Once Lenses' method is read, state what its labels buy over the decoder's assignment. |
| Broader | 7/10 | Mechanism-based. Transplants DML's masked-decoder assignment from captioning codes to image-side retrieval sub-spaces, and joins two families: masked reconstruction and set embeddings (PVSE, DivE, Maximal Matching). The NLP lesson cuts against fixed K: one context-conditioned vector won (Li and Jurafsky in NLP, Llip and FLAIR in vision-language). | Unification: add Llip-style text-conditioned pooling as a comparator, so the "K sub-spaces or one conditioned vector" question gets an answer. |

Highest ceilings: Stronger and Broader, 7/10 each, both mechanism-based. Lead with them. No dimension reaches 8, so Strong Accept is out of reach before data.

### 5. Paradigm-shift probe

| Probe | Answer | Rationale |
|---|---|---|
| First Principles | Partial | Challenges "set elements are assigned to captions by the same similarity they are trained to raise". DivE and Maximal Matching report the collapse this loop produces. Reconstruction gives an independent assignment signal. |
| Elephant in the Room | Partial | Set collapse is known and unresolved: PVSE never measured it, and Maximal Matching says smooth-Chamfer still collapses. Idea 2 offers a path. |
| Technology Cycle | No | PVSE (2019) and DML (2022) supply every part. |
| Hamming's Rule | Partial | Label-free readings per image would matter for polysemous retrieval (Lenses needed labels). It is not a field-wide top-three problem. |

Probe score 3 of 8. Disruptive potential: possible (seeds only, no Yes). Name the independent assignment signal in the introduction. Handbook 2.3 is not needed below a score of 5.

### 6. Feasibility

| Risk | Level | Mitigation |
|---|---|---|
| Compute | Medium | One variant: about 5 arms × 3 seeds, each above the 3 to 7 GPU-hours of a plain run (measure the factor in a smoke run). Several weeks of nights on the 3090, or a few days on a reserved DAS6 node. Screen at one seed first. |
| Data | Low | COCO (about 5 captions per image) and ArtELingo (about 5 per painting) are on disk. Blends come from the blend probe code. Alpha blends do not suit (b): every patch holds both sources, so (b) needs spatial blends. |
| Engineering | Medium | No slot or multi-embedding module exists, retrieval bypasses the fusion module, and the decoder reads image tokens today (a new conditioning path for (a)). A one-to-one match needs K at least the caption count, so a smaller K needs a balanced many-to-one match. DDP needs a gradient for every parameter in every step, so an unassigned sub-space breaks training unless every sub-space gets a caption. |
| Timeline | Medium | A first decisive result (the one-seed screen) in about 6 to 8 weeks. |

### 7. Verdict

**Accept with Revisions, worth pursuing, pending the validation experiment**

The validation experiment is the screen in action 1. It is a check on the development split, not on held data.

Top three actions to take first:
1. Run the decisive experiment as a one-seed screen on the COCO validation split: decoder-assigned against similarity-assigned sub-spaces at the same K, plus K=1, at 80% and at full caption masking. Read sub-space use, blend separation (one sub-space retrieves source A's captions, another source B's) and validation mAP@R. The masking ratio is a sweep, so ECCV Caption (5k test images) must not choose it.
2. Fix one variant before code, by the geometry of the readings. (a) suits readings that are not spatial (emotions, alpha blends) but must choose its decoder input: sub-embedding k alone (k stays decisive, the token path behind our gain is lost) or tokens plus k (the path stays, but the decoder may ignore k, as DML found under partial masking). (b) suits spatial readings and needs views of different content: block or attentive masks, since A-CLIP's attentive views beat random ones (Flickr I2T R@1 59.3 against 54.6).
3. Read Lenses' method (CVPR 2026) before code. Plan the recipe gap: reproduce the published InfoNCE row (39.0) or report the within-recipe test with the external bar beside it. Ask for a DAS6 reservation for the 3-seed confirmation.

## Trade-off between the two ideas

| | Idea 1: masked captions under blends | Idea 2: masked sub-space PVSE |
|---|---|---|
| Where masking meets the claim | in the decoder, on blends only. On mAP@R it is an auxiliary loss | in the retrieval embedding: the decoder picks which sub-space each caption trains |
| Language prior | the main failure mode: A's visible words may rebuild A without the image | cancels across k (same words for every sub-space). The risk is a decoder that ignores k |
| Mechanism test | the must-have control (A's words, image B) | decoder- against similarity-assigned, 80% against full masking |
| Refuted by our data | no | no ((b) without a per-view contrastive loss would be) |
| Novelty (per the checks) | one thin axis, the either-one target | the assignment signal. Lenses (CVPR 2026) not yet read |
| Top scores | Stronger 6/10, Cheaper 6/10 | Stronger 7/10, Broader 7/10 |
| First decisive result | about 3 to 4 weeks, small code | about 6 to 8 weeks, a new module |
| Verdict | Accept with Revisions | Accept with Revisions, pending the screen |

Our reading: the two ideas fit together. Blends give readings known exactly, which idea 2 needs to show separation. Alpha blends suit variant (a), spatial blends suit (b).

## What I did

- **Read:** the skill's SKILL.md in full, and its references fatal-flaws, five-dimensions, lifecycle-capability-matching, paradigm-shift-probe and worked-examples. The report-writing rule and the plain-English guide. The project CLAUDE.md, both idea figures, the direction log, the step-3 note, both literature checks in full, lines 1 to 46 of the emotion-readout report, and `GLOSSARY.md`. The header of `mmae/engine/eccv.py`, to confirm that ECCV Caption runs on the COCO 5k test images.
- **Skill steps, once per idea:** step 1 (first impression and paradigm), step 2 (fatal-flaws audit, with F1 grounded in the two checks and the data-refuted rule applied per result), steps 3 to 6 (lifecycle, five dimensions, paradigm probe, feasibility), step 7 (integrity gate, run silently) and step 8 (verdict). Then the trade-off, which the skill allows for candidate ideas.
- **Skipped, and why:** no web search was available, so F1 rests on the checks of 2026-10-26, and later work is marked unverified. `domain-evaluation-frameworks.md` was not read, since both ideas are STEM. The paradigm sub-references and handbook 2.3 were not read, since both probe scores are below 5. No other code was read: engineering risks come from CLAUDE.md and the brief.
- **Deviations:** the Read tool showed the whole direction log and the whole step-3 note. I used two facts from the direction log's entry of 2026-10-06 (no growth with valid matches, three controls lose the gain), cited where used. The step-3 note's collapsed recommendation was on screen but not used. The blend test-bed point in the trade-off overlaps it, and it follows from the note's visible text and the variant analysis here.
- **A conflict in the skill:** the fatal-flaws reference says that two MAJOR flaws make a CRITICAL one, and also that two MAJOR flaws lead to Accept with Revisions. I followed the verdict list and the worked examples, which treat MAJOR items as below the fatal level.
- **Assumption:** 20 effective hours a week for the student.

## Key judgment

Our reading: the deciding trade-off is where masking acts. In idea 2 the masked decoder chooses which sub-space each caption trains. Masking thus shapes the retrieval embedding by construction, and its signature is testable (DML: 29 against 6 codes in use). Idea 1 is cheaper, but on mAP@R it falls back to the auxiliary-loss framing the user set aside. Start with idea 2's one-seed screen, with idea 1's blends as its test bed.
