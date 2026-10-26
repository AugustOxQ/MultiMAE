# The masked decoder did not model how people disagree about a painting's emotion better than a plain captioner or a linear probe

> 2026-10-25, about the experiment of 2026-10-07 and 2026-10-08. Full report: [the emotion readout test](../reports/auto/v1/2026-10-08_hb_emotion_readout.md).

## Summary

- We tested whether the masked model's caption decoder holds several readings of one painting, measured as the spread of emotions that people report.
- On both measures fixed in advance, it did not beat a plain captioner or a linear probe.
- A linear probe on any model's frozen image embedding did as well as every decoder or better.
- Visible caption words steered the decoder's emotion guess, but that test favours masking by its design.
- On COCO, the decoder made retrieval worse when we used its score alone to rerank captions.
- The main caveats: the decoders were lightly trained, and the test showed the masked model an input it never met in training.

## How we got here

The project asks whether cross-modal masked reconstruction helps with polysemy: one image, several valid readings. Up to 2026-10-06 we tested this on COCO retrieval. Hiding 80% of the caption during training raised the main COCO metric by about 0.7 points. The gain showed no sign of polysemy.

On 2026-10-06 you split the goal into two claims:

- H-a: masked training makes the retrieval embeddings keep more readings.
- H-b: the masked decoder's own predicted distribution holds the readings.

You chose H-b first, on the ArtELingo paintings. A literature review found no study that compares a masked decoder's distribution with human readings. It named the main risk: any captioner trained on several captions per image also learns a mix of readings. This step is H-b's first test (spec 2026-10-07, results 2026-10-08).

## 1. Does the decoder's emotion spread match people's?

**Problem.** H-b says the masked decoder predicts several readings of one image. We needed a reading with real human spread to compare against.

**Idea.** Each ArtELingo annotator gives a painting one of 9 emotion labels. A held set of 1,600 paintings has about 100 votes each, in the 27 languages other than English. We never trained on these paintings. We gave the decoder one extra output that predicts the emotion; the decoder never sees the emotion as input. Then we compared its predicted distribution per painting with the votes, on two measures:

- the Jensen-Shannon distance: how far apart the two distributions are;
- the entropy Spearman: does the model's uncertainty rank the paintings the way human disagreement does?

The comparators matter more than the plain CLIP model here. A parallel captioner trains with the full image and every caption token hidden, so it never learns to use visible words or a partial image. A linear probe on the frozen image embedding learns the vote spread directly.

The floor gives every painting the same training label frequencies. The ceiling compares two random halves of the votes. The pass rule, fixed in advance: on one of the two measures, the masked model beats both the captioner and the probe, with Holm p < 0.05 over the four tests.

**Did it work.** No. The masked model was behind both comparators on both measures. Over paintings the gaps are clear. Over the 3 seeds per model they are not significant (seed-level test, Holm p = 0.105). So the masked model is not better; that it is worse is only suggested.

Two caveats could change the order of the masked model and the captioner. The decoders were lightly trained, because early stopping on retrieval picked the second epoch. And at test time the masked model saw the whole caption hidden, which it never met in training; the captioner trained that way.

**Key evidence.** Mean over 3 seeds, 1,600 paintings.

| Model | Jensen-Shannon distance (lower is better) | Entropy Spearman (higher is better) |
|---|---|---|
| Masked model | 0.277 | 0.41 |
| Parallel captioner | 0.267 | 0.51 |
| Best linear probe | 0.265 | 0.56 |
| Floor (label frequencies) | 0.318 | |
| Ceiling (half the votes against the other half) | 0.235 | 0.77 |

A linear probe reached 0.265 on any model's embedding, the plain CLIP model's included. The masked model's prediction stayed closer to the floor, and its shortfall sat on the paintings where people agree. On the third of paintings where people disagree most, no model got much beyond the floor. Adding the pixel decoder changed nothing.

> Full report: sections 3 and 4.

## 2. Readouts with partial views and partial captions

**Problem.** If a plain captioner models the spread as well, H-b needs something that masking adds.

**Idea.** We tried two readouts that use what masked training teaches:

- several partial views of the painting: does their disagreement track human disagreement?
- a few visible caption words: does the emotion guess move toward that annotator's own emotion, and does the painting still matter?

**Did it work.** Partly, and the positive part is weak.

- Views: more views were better than one, but the full image was better still. View disagreement tracked human disagreement only through the model's overall uncertainty, which section 1 already measured.
- Words: visible words moved the guess toward the annotator's emotion. The captioner never saw a word in training, so this test favours masking by its design.
- The "no painting" control, a plain grey image, scored worse than the floor. So we cannot tell whether the painting still adds anything once words are visible.

**Key evidence.** Error of the emotion guess on the annotator's own label (log-loss, lower is better). The 8-word row covers only the 16,041 of 31,282 captions with at least 8 content words, so the two rows use different captions.

| Condition | Masked model | Parallel captioner | Floor |
|---|---|---|---|
| No words visible | 1.925 | 1.874 | 2.041 |
| First 8 content words visible | 1.450 | 1.873 | 2.050 |

> Full report: sections 5 and 6.

## 3. Probes on the earlier COCO models

**Problem.** We wanted two more checks with no new training: can the decoder help retrieval directly, and can it hold two readings of one blended image?

**Idea.** First, we used each model's decoder to rerank its own top 50 captions per image. The score was the caption's likelihood, corrected by its likelihood under a blank image. Second, we blended two unrelated images. We asked whether the model kept both readings, and whether a few visible caption words pulled it toward one source.

**Did it work.** No for reranking, and only a small effect for blends.

- Reranking by the decoder score alone cut mAP@R by 15 to 19 points against the model's own embedding order. This is image-to-text retrieval, scored on human-checked caption matches. Added to the embedding score, the decoder changed mAP@R by at most 0.14.
- The decoder mostly scored the caption itself, whatever the image.
- The blend pass rule could not tell the models apart, because every value sat near zero.
- Three visible words pulled the masked decoder toward their own source image. A model with the same 80% caption masking but a full-image decoder showed no such pull. The two decoders also read different inputs (partial views against the full blend), so the comparison is not clean.

**Key evidence.** One by-product may matter for the COCO gain. In three models the image encoder meets partial views only through pixel reconstruction. In these three, the embeddings of partial views collapsed: views of different images became almost the same (mean cosine 0.97 to 0.98, against 0.43 to 0.50 in the other models). The same three models lost the COCO gain of 80% caption masking. This is a correlate, and it does not explain the gain: retrieval uses the full-image embedding, which these models keep intact.

> Full report: section 7.

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our view.** On this readout the result is a clear negative: the masked decoder is not better. The probe is the strongest finding. Every model's image embedding already carries the emotion spread as far as any model here learned it, and no decoder adds to it.

The two caveats of section 1 could change the order of the masked model and the captioner. We doubt that either closes the gap to the probe on the entropy Spearman (0.41 against 0.56), but these runs cannot tell.

We advise a cheap check first: one seed each of the masked model and the captioner, with checkpoints picked on the emotion loss (2 runs, about 6 GPU-hours). If the masked model stays clearly behind the probe, close this line. The collapse analysis from section 3 costs almost nothing and can run beside it.

</details>

## Next

No MultiMAE experiment ran after 2026-10-08. The 12 ArtELingo models were deleted after their readouts, so any follow-up trains new ones. The cluster reservation ended, so new runs need a new one.

The direction is your decision. The options:

1. Rerun the masked model and the captioner with checkpoints picked on the emotion loss (6 runs of about 3 hours). It tests whether this selection narrows the gap.
2. The captioner comparisons. First, cross partial or full image with 80% or 100% caption masking (6 runs). Second, add an autoregressive captioner, which writes the caption word by word (new code, 3 runs), and rerun the words test with a no-image control that the decoder does not penalise. They tell which factor costs the masked model, and whether the steering by words is specific to masking.
3. Close the emotion-spread line. Go back to the COCO masking gain and study the collapse correlate from data on disk. Or start a new idea with a literature check.

## Glossary

- **Reading**: one valid interpretation of an image; here, one annotator's emotion.
- **H-a**: masked training makes the retrieval embeddings keep more readings.
- **H-b**: the masked decoder's own predicted distribution holds several readings.
- **Masked model** (ML-80): CLIP fine-tuned with 80% of caption tokens hidden, a 25% image view, and an emotion output.
- **Masked model with the pixel decoder** (ML-80+MAE): the same, plus image reconstruction.
- **Plain CLIP model** (C): CLIP fine-tuned with the contrastive loss alone.
- **Parallel captioner** (Par-cap): the same decoder, reading the full image with every caption token hidden.
- **Linear probe** (soft-label probe): a linear classifier on the frozen image embedding, trained on each painting's label histogram.
- **Floor** (prior-only) and **ceiling** (split-half ceiling).
- **Held set of paintings** (AL-28, ArtELingo-28): WikiArt paintings with emotion votes in 28 languages, English included; we used the 27 others.
- **mAP@R**: the main COCO metric: precision over the human-checked caption matches of each image (ECCV Caption).
- **The tests' labels**: section 1 is D4; section 2 is D6 (views) and D7 (words); section 3 is D1 (reranking) and D2 (blends).
