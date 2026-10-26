# MultiMAE direction log

One entry per loop of the research loop: what we read, what the user decided, why, and where the files are. Newest
first. The wrap-up step adds the next entry. Details live in the linked reports and handoffs, not here.

## Now (2026-10-26)

The project's first loop on the research loop (r1) is at step 3. The last result (2026-10-08): on ArtELingo the masked decoder's emotion distribution was no closer to dense human votes (JSD 0.277) than a parallel captioner's (0.267) or a linear probe on frozen embeddings (0.265); the only masking-specific effect is that visible caption words steer the decoder. The user reads the framing "can a masked loss help polysemy" as ill-defined, since a masked loss is a common auxiliary loss, and wants masking used in a designed way. Two deep literature checks run on the user's two ideas (masked training under a designed multimodal augmentation; a PVSE setting with masked sub-space training); step 4 picks the idea. No compute is reserved and no run is planned.

## Parked ideas

<!-- Candidates that a loop proposed but did not choose, and that were never tried. One line each: the idea in plain
words, "never tried" or "deferred by the user (YYYY-MM-DD)", and a link to where it was proposed. Step 3 reads this
list first; an idea leaves it when a loop picks it (or the user drops it). Ideas tried and killed stay in the entries. -->

- Rerun the masked model and the captioner with checkpoints picked on the emotion loss, to test whether light training caused the D4 gap (never tried; [H-b report, section 9](../reports/auto/v1/2026-10-08_hb_emotion_readout.md))
- The 2×2 on ArtELingo: partial or full image against 80% or 100% text masking, to find which factor costs the masked model its D4 gap (never tried; [H-b report, section 9](../reports/auto/v1/2026-10-08_hb_emotion_readout.md); D5 of the [H-b review](../reports/auto/v1/2026-10-06_masked_decoder_readings_review.md))
- An autoregressive captioner as the fair comparator for the visible-words readout (D7), with a no-image control that the decoder does not penalise (never tried; [H-b report, section 9](../reports/auto/v1/2026-10-08_hb_emotion_readout.md))
- The partial-view collapse as a correlate of the COCO masking gain: study it from the checkpoints on disk (never tried; [H-b report, section 7.2](../reports/auto/v1/2026-10-08_hb_emotion_readout.md); [10-08 handoff, option 5](handoffs/2026-10-08-after-hb-first-handoff.md))
- Blend training with either-one caption targets against both-at-once (D10), unmixing (D11), and designs D3, D8, D9 (never tried; [H-b review, section 6](../reports/auto/v1/2026-10-06_masked_decoder_readings_review.md))
- A caption-likelihood auxiliary loss helps dual encoders where matches are many: a 2×2 of InfoNCE or a one-to-many loss with or without 80% MLM, stratified by multiplicity (never tried; [deep-think note, candidate A](../reports/auto/v1/2026-10-05_cross_masking_polysemy.md))
- Embeddings that keep the readings: a small set of image embeddings, each trained through the MLM decoder to explain a different caption, against a PVSE or DivE-style set (never tried; [deep-think note, candidate B](../reports/auto/v1/2026-10-05_cross_masking_polysemy.md); close to the user's idea 2 of 2026-10-26)
- A controlled-null paper: masked objectives as generic vision supervision for a fine-tuned CLIP, shown by dose, detach, tower-swap and stratified analyses (never tried; [deep-think note, candidate C](../reports/auto/v1/2026-10-05_cross_masking_polysemy.md))
- The remaining levers of the lever review (never tried; [lever review, section 8](../reports/auto/v1/2026-10-03_lever_review.md))

## 2026-10-26 · H-b's first readout not supported; the user reframes the question and asks for principled designs

- **Read:** D4, the pre-registered test, negative. The masked model's (ML-80) emotion distribution against dense ArtELingo-28 votes: JSD 0.277 against 0.267 for the parallel captioner and 0.265 for the strongest linear probe; entropy Spearman 0.41 against 0.51 and 0.56 (prior only 0.318; human ceiling 0.235 and 0.77). Over paintings all four gaps go against ML-80 (Holm p ≤ 0.0034); over the 3 seeds Holm p = 0.105, so ML-80 is not better, worse only suggested. A linear probe on any arm's frozen embedding reached 0.265. Visible caption words steer the decoder (D7: log-loss 1.925 to 1.450 with 8 words), with no fair captioner comparator. On COCO, decoder reranking cut mAP@R by 15 to 19 points (D1); the blend probe (D2) showed a small effect of visible words and the collapse of partial-view embeddings in models whose vision tower meets partial views only through pixel reconstruction. Caveats: lightly trained decoders (epoch 2), the fully hidden caption at readout, 3 seeds, one backbone.
- **Meaning (C):** "The current way of training cannot reveal a critical point. A masked reconstruction loss exists in many models' training as an additional loss, so framing the problem as 'can a masked loss help polysemy' is possibly ill-defined. Use MAE in a designed way: under a multimodal augmentation, train with the masked loss in a specific way that helps polysemy; or in a PVSE setting, use masked sub-space training instead of the K sub-attentions. Do more principled research." (user, 2026-10-26, before seeing ours; our reading, in the report's results brief, read D4 inside the frame as a clear negative with the probe as the strongest finding and proposed a cheap rerun with a new selection rule)
- **Decided (A):** a full loop on a new idea; step 3 is a deep literature check of the user's two ideas and a brainstorm from it (user, 2026-10-26). No rerun of the H-b arms.
- **Idea (B):** open, awaiting step 4.
- **Why:** the H-b readout gave no masking-specific positive beyond word steering, and the probe showed that the pooled embedding already carries the emotion spread. The user sees the question itself, a generic auxiliary loss, as the problem, so the next design must tie the masked objective to the polysemy claim by construction.
- **Links:** [H-b report with results brief](../reports/auto/v1/2026-10-08_hb_emotion_readout.md) · [user-read report](../user_read/2026-10-25_hb_emotion_readout.md) · [spec of the first H-b experiment](specs/2026-10-07-hb-first-artelingo-design.md) · [handoff into this chat](handoffs/2026-10-25-r1-decide-handoff.md)

## Before the research loop (reconstructed on 2026-10-26 from reports, specs, handoffs and memory)

### 2026-10-08 · The first H-b spec ran: D4 not supported, D1 killed, D7 steering only; the direction left open

- **Read:** as in the r1 entry above (the same report).
- **Meaning (C):** not recorded as a reading. The user's words at the close: "wrap up and commit; remove checkpoints if not useful; leave a handoff" (user, 2026-10-08 17:40).
- **Decided (A):** open; the direction discussion was deferred to the next chat, which became r1 decide on 2026-10-25. The 12 ArtELingo checkpoints were deleted after their readouts (7.7 GB); readouts kept.
- **Links:** [report](../reports/auto/v1/2026-10-08_hb_emotion_readout.md) · [spec](specs/2026-10-07-hb-first-artelingo-design.md) · [handoff](handoffs/2026-10-08-after-hb-first-handoff.md)

### 2026-10-07 · The first H-b spec decided

- **Idea (B):** the core of the H-b review's designs: D1, D2, D4, D6, D7 (D5 and D10 deferred to a second spec); the reading unit is the annotator's emotion, 9 classes as dedicated decoder outputs; the emotion slot always hidden; two co-primary metrics (JSD, entropy Spearman) with ML-80 to beat both Par-cap and the strongest probe on one of them (hierarchical bootstrap, Holm over four); both ML-80 versions (MAE off primary); launch-first staging (user, 2026-10-07). Amended before launch: the emotion loss as a separate term with weight 0.07 in every decoder arm, since the joint MLM mean had favoured ML-80 (user, 2026-10-07 18:10). "If smooth then go for it": spec, plan, build and launch without stopping (user).
- **Links:** [spec](specs/2026-10-07-hb-first-artelingo-design.md) · [handoff into the spec chat](handoffs/2026-10-07-hb-design-handoff.md) · registry `tests/20261007_hb/runs.md`

### 2026-10-06 · Stage 2 confirmed the 80% masking gain without a polysemy signature; the goal restated and H-b chosen first

- **Read:** 80% text masking reached ECCV mAP@R 37.94 over 5 seeds against 37.11 (contrastive) and 37.24 (multilearner), Holm p 0.001 and 0.003; MAE off null; the gain sits in the image tower (tower swap +0.90) and needs the caption decoder to read the masked-pass image tokens (three controls lose it); VWSD down about 3 points in every masked arm; no growth with the number of valid matches.
- **Meaning (C):** the gain is "a bit small" (user, 2026-10-05 17:30); the goal restated as "enable a more polysemic way of understanding through cross-modal masked reconstructions" and split into H-a (the encoder claim) and H-b (the model claim) (user, 2026-10-06 about 19:45).
- **Decided (A):** H-b first, ArtELingo as the main testbed, the COCO checkpoints as the development bed for decoder probes; an integrative literature review before any experiment; the fair baseline for H-b is a plain image-conditioned captioner (user, 2026-10-06). The user added cross-modal mixing as an idea: blend two images, keep each source caption as a separate target ("either one").
- **Links:** [Stages 1 and 2 report](../reports/auto/v1/2026-10-06_masking_stages_1_2.md) · [user-read report](../user_read/2026-10-06_masking_and_polysemy.md) · [deep-think note](../reports/auto/v1/2026-10-05_cross_masking_polysemy.md) · [direction-discussion handoff](handoffs/2026-10-06-direction-discussion-handoff.md) · [H-b review](../reports/auto/v1/2026-10-06_masked_decoder_readings_review.md)

### 2026-10-05 · Stage 1 closed; the Stage 2 arms decided

- **Read:** the text-masking dose peaked at 80% (3 seeds 37.81, +0.86 over multilearner); MAE off +0.59 at 2 seeds; M1 clean, M1 detached, M3, M6 and the R2 recipe failed the screen or sat at the threshold.
- **Decided (A):** Stage 2 arms: contrastive, multilearner, 80% text masking, MAE off (the user chose MAE off over 90%), 3 seeds with the seeded sampler (user, 2026-10-05 about 10:55); later the 80% controls `fusion_none`, M1 clean and M1 detached at 3 seeds (user).
- **Links:** registry `tests/20261003_ml_improve/runs.md` · [Stages 1 and 2 report](../reports/auto/v1/2026-10-06_masking_stages_1_2.md)

### 2026-10-03 · The improve-multilearner line decided

- **Read:** the v1 baselines: fusion raised PMRP over contrastive (56.88 against 56.49), rsum +2.4, ECCV mAP@R unchanged. The lever review found no source that ablates a masked objective on ECCV mAP@R, PMRP, CxC or VWSD, and PMRP a weak instrument.
- **Decided (A):** ECCV Caption mAP@R as the primary metric with every other metric tracked; the staged plan (Stage 0 diagnostics, Stage 1 screen, Stage 2 confirmation at 3+ seeds); recipe changes go to every arm (user, 2026-10-03). Compute: DAS6 nodes reserved by the user for five days.
- **Links:** [spec](specs/2026-10-03-improve-multilearner-design.md) · [handoff](handoffs/2026-10-03-improve-fusion-multilearner-handoff.md) · [lever review](../reports/auto/v1/2026-10-03_lever_review.md) · [baselines report](../reports/auto/v1/2026-10-03_baselines.md) · [Stage 0 diagnostics](../reports/auto/v1/2026-10-03_stage0_diagnostics.md)

### 2026-10-02 · The research question set

- **Decided (A):** the project's main question is "can multimodal-fusion-masked models help polysemic retrieval?"; the COCO 5k R@K comparison of the fusion variants is a sanity step (user, 2026-10-02). The user later judged VWSD "just word disambiguation, not real polysemy" and proposed ArtELingo as a real-polysemy setting (user, 2026-10-05).
- **Links:** this log (the decision lived in project memory until 2026-10-26)

### 2026-10-01 · The v1 refactor to the `mmae` package

- **Decided:** rewrite the pre-refactor code (tag `legacy-v0`) as the `mmae` package with Hydra and Accelerate (who decided: not recorded).
- **Links:** [refactor spec](specs/2026-10-01-mmae-refactor-design.md) · [refactor report](../reports/auto/v1/2026-10-01_refactor.md)
