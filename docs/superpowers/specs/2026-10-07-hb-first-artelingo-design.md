# H-b, first spec: the masked decoder's emotion distribution on ArtELingo, plus COCO decoder probes

Date: 2026-10-07. Branch `hb-artelingo` (from `ml-improve-stages-1-2` at b31ea05). Background: the H-b literature
review `docs/reports/auto/v1/2026-10-06_masked_decoder_readings_review.md` (designs D1 to D11 in its section 6 and in
`docs/reports/assets/2026-10-06_masked_decoder_readings_review/S_synthesis.md`), and the user-read summary
`docs/user_read/2026-10-06_masking_and_polysemy.md`.

## 1. Goal

H-b claims that a cross-modal masked model represents several readings of one image in its decoder's predictive
distribution, which a contrastive dual encoder (one pooled vector per item) cannot. The COCO line tested only the
retrieval embedding (H-a). This spec tests the decoder for the first time:

- **D4 (main):** on ArtELingo, does the masked caption decoder's distribution over the annotators' emotion labels
  match a dense human vote distribution better than a plain parallel captioner and a soft-label probe on CLIP
  features?
- **D6, D7 (masking-specific readouts):** does averaging over masked image views track human ambiguity, and does a
  partial caption select one annotator's reading?
- **D1, D2 (COCO dev bed, existing checkpoints):** does the decoder help as a prior-corrected re-ranker, and does it
  keep both readings of a 50/50 image blend?

## 2. Decisions (user, 2026-10-07)

| Decision | Choice |
|---|---|
| Scope | Core: D1, D2, D4, D6, D7. D5 (AR-cap, ML-15, ML-100v) and D10 (blend training) go in a second spec after the first D4 seed is read. |
| Reading unit | The annotator's emotion label, 9 ArtEmis classes. Each caption is a retrieval positive, not a reading. |
| Emotion representation | A class, not words: 3 of the 9 labels are two CLIP tokens ("content ment", "disgu st", "something else"), which would bias a per-position readout. One extra decoder output with a 9-way head. |
| AL-28 tenth label | "other" (0.6% of votes) is merged into "something else"; a sensitivity check drops it. |
| "something else" | Kept as the ninth class in the primary metric; a sensitivity check renormalises over the 8 named emotions. |
| Emotion slot in training | Always hidden: the emotion is only ever a prediction target, never an input; the text tower never sees it. |
| Emotion loss weight (amended 2026-10-07) | A separate term with the same fixed weight, 0.07, in every decoder arm (section 5.1), instead of one more target inside the MLM mean, whose share differed between ML-80 and Par-cap. |
| D4 decision rule | Two co-primary metrics (JSD, entropy Spearman), pre-registered in section 7. |
| ML-80 in D4 | Both versions: image decoder off (ML-80, primary) and on (ML-80+MAE, secondary). On COCO at 80% the MAE loss changed nothing (ECCV mAP@R 38.00 vs 37.87, 3 seeds each). |
| Staging | Launch first: the training path is built, tested and launched before the readouts; the readouts are written while wave 1 trains; a gate on wave 1's checkpoints precedes wave 2. |
| Compute | node404, node405, node411 (9 A6000), reserved to 2026-10-10 01:30 (node411 01:47). |
| Defaults (proposed 2026-10-07 07:30, not objected to) | COCO recipe (10 epochs, batch 128, current learning rates, seeded sampler), 3 seeds (42 to 44); model selection on validation rsum; all AL-28 paintings out of training and validation; temperature fitted on ArtELingo validation labels; English captions only. |

## 3. Scope

In: D4, D6, D7 on ArtELingo; D1, D2 on COCO with existing checkpoints.

Out (second spec or later): D5 (AR-cap, ML-15 and ML-100v on ArtELingo), D10 and D11 (blend training), D3, D8
(re-ranking and coverage on ArtELingo), D9 (set embeddings), the R2 recipe, any use of emotion as a retrieval
condition (that belongs to CoSiR), readings across languages as a target of their own.

## 4. Data

**Files.** ArtELingo English JSONs at `/data/PDD/artelingo` (node `/local/wding/Dataset/artelingo`), images at
`/data/PDD/wikiart_proj/wikiart` (node `/local/wding/Dataset/wikiart_proj/wikiart`); both already in the cluster
`DATA_MAP`. AL-28: `/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv`
(173,745 votes on 1,658 paintings, 28 languages; all 1,658 images present locally).

**Labels.** `EMOTIONS = (amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something else)`,
in this order everywhere. Train label counts (before the hold-out): contentment 87,260, awe 50,289, something else 34,372, sadness 32,656,
amusement 30,888, fear 28,225, excitement 25,678, disgust 14,880, anger 4,475.

**Hold-out.** Every AL-28 painting (by the `painting` key) is removed from every train and validation file the
runs read: 1,160 train paintings and 59 validation paintings. The 103 AL-28 paintings in the test file stay there
(nothing trains on test). A test asserts that no AL-28 painting reaches a train or validation dataset.

**Splits.**
- Train pairs: `artelingo_train.json` minus held-out paintings, one (image, caption, emotion) item per caption,
  shuffled by the seeded sampler.
- Validation: retrieval on `artelingo_val_retrieval.json` (5 captions per painting) minus held-out paintings, for
  rsum model selection; loss pairs flattened caption-major from the same file (as COCO, so an unshuffled batch holds
  distinct paintings); each caption's emotion is looked up in `artelingo_val.json` by (painting, caption).
- Test: retrieval on `artelingo_test_retrieval.json` (4,975 paintings x 5 captions), emotions looked up in
  `artelingo_test.json`. Captions whose (painting, caption) lookup is ambiguous take the first match; the loader
  reports how many.
- Temperature fitting (section 6.2): the individual emotion labels of the validation pairs.

**Caption length.** `data.max_text_len = 40`. With BOS and EOS, 2.6% of train captions exceed 32 tokens and 0.02%
exceed 40 (40,000 sampled). The emotion class does not take a caption position.

**Dense human target (AL-28).** Per painting, the histogram of its **non-English** votes over the 9 classes
(median 104 votes). English votes (6,515, about 5 per painting) are excluded from the target so that the English
reference below never scores against itself. Paintings with fewer than 20 non-English votes are dropped: 1,600 of
1,658 remain.

## 5. Models and arms

### 5.1 Emotion head

`model.emotion_head: bool` (default false, read with `cfg.get` so old run configs build). When true:
- the text decoder gets one extra learned query, placed ahead of the caption queries and never padded; the caption
  queries and their position embeddings are unchanged;
- its output goes through a new linear head to 9 logits (the caption head keeps the CLIP vocabulary);
- the emotion has no input token and no text-tower position: it is always a target (decision: always hidden);
- its cross-entropy over the 9 classes is a separate loss term with one fixed weight in every arm:
  `loss = w_c loss_contrastive + w_mae loss_mae + w_mlm loss_mlm + w_e loss_emotion`, with `loss_mlm` the mean
  cross-entropy over the masked caption tokens (unchanged) and `loss_emotion` the mean emotion cross-entropy over
  the captions, `w_e = model.loss.weights.emotion = 0.07`.
  *Amended 2026-10-07 (user decision after the final code review).* The first version folded the emotion into the
  MLM mean as one more target per caption. With 16.4 real tokens per train caption, that gave the emotion about 7.1%
  of the MLM term in ML-80 (13 hidden tokens) but 5.75% in Par-cap (all 16 hidden), so ML-80's emotion slot trained
  about 23% harder than the captioner it is compared against, tilting the D4 test toward H-b. 0.07 keeps ML-80's
  former effective share (one target among about 14) and is the same for every decoder arm.

Batches carry `emotion` (B,) long; a model with `emotion_head` raises if it is missing. COCO batches carry none.

### 5.2 Text masking ratio 1.0

`random_token_mask` accepts a ratio in (0, 1]. At 1.0 every real, non-special token is masked (BOS, EOS and padding
never). Par-cap needs it.

### 5.3 Arms (D4)

All on `data=artelingo` (`artelingo_cluster` on DAS6), seeds 42, 43, 44, `train.seeded_sampler=true`, the current
recipe, `wandb.group=hb_d4`:

| Arm | Overrides | Role |
|---|---|---|
| C | `model=contrastive` | contrastive dual encoder |
| ML-80 | `model=fusion_multilearner model.masking.text_ratio=0.8 model.loss.weights.mae=0 model.emotion_head=true` | primary masked model: 25% image view, 80% of caption tokens hidden |
| ML-80+MAE | as ML-80 with `model.loss.weights.mae=1` | secondary: does the pixel task change the distribution |
| Par-cap | `model=fusion_multilearner model.masking.text_ratio=1.0 model.mlm_image_source=clean model.loss.weights.mae=0 model.emotion_head=true` | parallel captioner: full clean image, every caption token hidden |

(`emotion_head` gets a default of false in `configs/model/base.yaml`.) ML-80
and Par-cap differ only in the image the caption decoder reads (25% view vs full image) and in the share of caption
tokens hidden (80% vs 100%). Extended COCO metrics and VWSD are off for ArtELingo runs.

## 6. Readouts and baselines (D4)

### 6.1 Emotion distribution of a decoder arm

For a painting, every caption position is hidden (no annotator text), and the emotion query's softmax is read.
- **Caption length.** The decoder sees the caption length through padding, so the readout averages over the
  training length distribution, never over a particular annotator's length: lengths at the 10th, 20th, ..., 90th
  percentiles of real non-special token counts in train, equal weights.
- **Image.** ML-80 and ML-80+MAE: K = 16 random 25% views (fixed seeds), the input they train on. Par-cap: the full
  clean image, its training input.
- Probabilities are averaged over lengths and views (mixture, not logit average).

Caveat (stated in the report): ML-80 trains with 80% of tokens hidden, so a fully hidden caption is somewhat out of
distribution for it but in distribution for Par-cap. This biases D4 against ML-80.

### 6.2 Temperature

One scalar temperature per (arm, seed, readout), fitted by minimising the mean NLL of the individual validation
labels (held-out paintings excluded), over log T in [-3, 3]. Applied to the logits before averaging over lengths and
views. Every baseline below gets the same fit.

### 6.3 Baselines

- **Prior-only:** the train label frequencies, the same distribution for every painting.
- **Prompt softmax:** each arm's dual encoder; cosine between the painting's pooled embedding and the 9 prompts
  "a painting that evokes {label}." times the logit scale, then the fitted temperature. C's is the H-b baseline as
  phrased; the other arms' are reported.
- **Soft-label probe:** multinomial logistic regression on the frozen, L2-normalised pooled image embedding of each
  arm and seed, trained on the train paintings' per-painting label histograms (soft targets), weight decay chosen on
  validation NLL from {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}. The primary comparison uses, per seed, the probe with the
  lowest validation NLL among the four arms' embeddings (the strongest probe, chosen without test data).

### 6.4 Human references

- **Split-half ceiling:** each painting's non-English votes split into two random halves, 10 times; JSD and entropy
  Spearman between halves (the MultiEmo protocol). It understates a full sample.
- **English reference:** each painting's English labels from the ArtELingo files (about 5), scored against the
  non-English target: the bar a model trained on English labels can realistically approach. Paintings without
  English labels are left out of this reference only.

## 7. Metrics and the pre-registered decision rule (D4)

**Target set:** the 1,600 AL-28 paintings with at least 20 non-English votes (section 4).

**Co-primary metrics:**
1. **JSD:** mean over paintings of the Jensen-Shannon distance (base 2, in [0, 1]) between the model's and the
   human distribution. Lower is better.
2. **Entropy Spearman:** Spearman correlation over paintings between the model's entropy and the human entropy.
   Higher is better. A global temperature cannot change this ranking, so calibration cannot fake it.

**Comparisons:** ML-80 against Par-cap and against the strongest probe, on both metrics: four tests.

**Test:** hierarchical bootstrap, B = 10,000. In each replicate, resample each arm's 3 seeds with replacement
(independently per arm), then resample the paintings with replacement (the same painting sample for every arm, so
the comparison is paired by painting); compute the difference of the metric. Two-sided p = 2 min(P(diff <= 0),
P(diff >= 0)). Holm over the four tests.

**Support for H-b on D4:** for at least one metric, ML-80 beats both Par-cap and the probe with Holm-adjusted
p < 0.05. **Kill (from the review):** otherwise, the masked decoder's emotion readout shows no advantage over a
captioner and a probe; the second spec then weighs D5's captioner comparisons and the masking-specific readouts
(D6, D7) before any further ArtELingo training.

**Secondary (reported, not tested for the claim):** KL(human || model) with the model side as is and the human side
untouched (models give no zero mass), TVD, RankCS (rank agreement of the class orderings), both primary metrics by
thirds of human entropy, the 8-named-emotion version, the "other dropped" version, ML-80+MAE against ML-80, prompt
softmax and prior-only for every arm, each arm against the split-half ceiling and the English reference, and
ArtELingo retrieval (R@1/5/10 both directions, rsum) on the 4,975-painting test split as H-a evidence.

## 8. D6: view sampling (secondary, evaluation only)

For ML-80 and ML-80+MAE, the section 6.1 readout at K in {1, 4, 16} views. The between-view term is the mutual
information between view and emotion, H(mean over views) minus the mean per-view entropy. Reported:
- JSD and entropy Spearman at each K; whether K = 16 beats K = 1;
- Spearman between the between-view term and the human entropy, with a bootstrap interval over paintings and seeds
  (the review's kill: the interval includes zero);
- controls on the same views: Par-cap on K views it never trained on, and C's prompt softmax and the strongest probe
  averaged over the same views (the vision tower encodes a 13-patch subset through `ids_keep`).

## 9. D7: readings selected by a partial caption (secondary, evaluation only)

For every caption in `artelingo_test.json` (31,282) and j in {0, 1, 2, 4, 8} visible content tokens (content words
as in `mmae/data/stopwords.py`), two patterns: a **prefix** (the first j content tokens visible) and a **random
subset** (j random content tokens, fixed seed); every other caption token is hidden. Readout: the emotion query's
distribution (ML-80 arms averaged over K = 4 views, Par-cap on the full image, temperatures from 6.2), scored by the
log-loss and accuracy of that caption's annotator emotion. Captions with fewer than j content tokens are left out at
that j (counts reported).

Control: the same model with a **null image** (the all-zero tensor after CLIP normalisation, i.e. the mean colour).
The review's kill: ML-80 with the real image is no better than with the null image at j >= 2 (bootstrap interval of
the log-loss difference over captions and seeds includes zero). Par-cap never sees visible tokens in training, so
D7 favours masking by construction; the AR-cap comparison on prefixes comes with D5.

## 10. D1: decoder scoring on ECCV Caption (COCO, existing checkpoints)

Checkpoints (seeds 42 to 44, all at 9919d8d): `s2_txt80_mae0` (ML-80), `s2_txt80` (ML-80+MAE), `s2_multilearner`
(ML-15), `s2_m1clean_txt80` (full-image decoder at 80%, the nearest existing captioner) and `s2_none_txt80` (text-only
decoder: a learned language prior). 15 checkpoints.

Task: image-to-text re-ranking of each COCO 5k test image's top 50 captions from the same model's dual encoder;
ECCV Caption mAP@R (and R-Precision) on its query subset, overall and by thirds of R.

Decoder scores of a caption c for image x (the caption decoder predicts every hidden position in parallel):
- **parallel score:** every caption token hidden, mean log p of c's tokens; masked-image models averaged over K = 8
  views of 25%, the M1-clean model on the full image;
- **training-ratio score:** K = 8 random masks at the model's own text ratio (80% or 15%), mean log p of the hidden
  tokens, averaged over masks (and over views, one view per mask).

Prior correction (image-to-text only): PMI = s(c | x) - alpha s(c | null image). Two re-rankers, each tuned on the
COCO 5k val split: (a) the PMI score alone within the top 50 (alpha tuned); (b) z-scored dual-encoder similarity plus
beta times the z-scored PMI (alpha, beta tuned). Under H-b the re-rank matches or beats the dual encoder and the gain
grows with R; the review's kill: every alpha lowers mAP@R beyond seed noise with no relative gain in the top third of
R.

## 11. D2: blend probe with exactly known readings (COCO, existing checkpoints)

- **Pairs:** 1,000 pairs (A, B) of COCO 5k test images (fixed seed), each with a mean cosine between A's and B's
  captions (zero-shot CLIP B/32 text embeddings) in the lowest decile of 100,000 random pairs.
- **Blends:** pixel blends lambda A + (1 - lambda) B for lambda in {0.3, 0.4, 0.5, 0.6, 0.7}; a patch-mix variant
  where the 13-patch view holds m patches of A and 13 - m of B (m in {3, 5, 7, 9, 11}) at their own grid positions.
- **Models:** zero-shot CLIP, `s2_contrastive` (C), and the five D1 checkpoint families.
- **Readouts:**
  - dual encoder: **both-covered@k** (k in {5, 10, 20}): share of blends with at least one of A's and one of B's
    captions in the top k of the 25,000 test captions; and **balance**, A's share of the top 10 drawn from A's and
    B's captions;
  - decoder: balance as the mean parallel score of A's 5 captions minus B's; both-covered@k after re-ranking the
    dual encoder's top 50 with D1's PMI scorer (D1's alpha);
  - **selection:** with j in {1, 2, 3} content tokens of one of A's captions visible, the shift of the decoder balance
    toward A (masking-specific);
  - balance against lambda (or m) as a psychometric curve. `s2_none_txt80`'s decoder ignores the image, so its
    balance should not move with lambda (control).
- The review's kill: C's both-covered@10 is within 2 points of, or above, the best decoder's at every k. The
  checkpoints never saw blends; the report says so.

## 12. Engineering

**Files (new unless marked).**
- `mmae/data/artelingo.py`: `EMOTIONS`, `heldout_paintings(csv)`, `al28_targets(csv)` (non-English 9-class
  histograms, "other" merged, min-votes filter), `ArtelingoPairs` (image, caption, emotion), `ArtelingoRetrieval`
  (image, 5 captions, 5 emotions).
- `configs/data/artelingo.yaml`, `configs/data/artelingo_cluster.yaml` (`name: artelingo`, paths, `al28_csv`,
  `max_text_len: 40`, the `limit_*` keys); `configs/data/coco*.yaml` get `name: coco`; code reads `data.get("name",
  "coco")`. The cluster config is adopted into the cluster-run skill (`cluster configs --adopt`).
- `mmae/data/collate.py` (edit): items may carry an emotion (pairs) or 5 emotions (retrieval); the batch gets
  `emotion` when they do.
- `mmae/engine/trainer.py`, `evaluate.py` (edit): dataset dispatch on `data.name`; extended COCO metrics only for
  COCO.
- `mmae/models/masking.py` (edit): ratio 1.0. `mmae/models/decoders.py`, `mmae/models/model.py`, `mmae/losses.py`
  (edit): emotion query, head and loss; `configs/model/base.yaml` (edit): `emotion_head: false`.
- `mmae/engine/hb/` (new package): `emotion.py` (6.1 readout, D6 views, D7 partial captions), `calibrate.py` (6.2),
  `baselines.py` (6.3), `metrics.py` (JSD, entropy, Spearman, KL, TVD, RankCS), `bootstrap.py` (section 7),
  `coco_probes.py` (D1, D2).
- `scripts/hb_eval.py` (D4, D6, D7 over a folder of runs, writes JSON), `scripts/hb_coco_probes.py` (D1, D2),
  `scripts/hb_table.py` (section 7's decision rule and the secondary tables).
- Registry: `tests/20261007_hb/runs.md`, the dated debug folder for this line.

**Tests (fast, CPU, tiny CLIP and a fake ArtELingo `make_fake_artelingo` in `tests/helpers.py`).**
- Loader: emotion lookup, held-out paintings never in train or validation, caption-major validation order, the
  ambiguous-lookup count, "other" merged in AL-28 targets, the 20-vote filter.
- Model: `emotion_head` and text ratio 1.0 added to `VARIANTS` (gradient and leak-guard tests); the emotion loss is
  part of the MLM mean as specified (recomputed from captured logits); ratio 1.0 hides every real non-special token
  and nothing else; a model with `emotion_head` raises without `batch["emotion"]`; old configs without the key build.
- Readouts: the 9-way readout sums to 1; length averaging uses the specified percentiles; the temperature fit
  recovers a known temperature on synthetic data; JSD, entropy Spearman, KL and TVD match scipy on toy data; the
  bootstrap's p-value and Holm on a synthetic case with a known answer; the decision rule on synthetic arms.
- Every new guard is checked to fail when the guarded code is broken on purpose.
- Smoke: `train.py data=artelingo train=debug` with the tiny CLIP on CPU; then a real-data smoke on the local RTX
  3090 under the GPU lock (real B/32, `data.limit_train=4096`, 1 epoch): the emotion loss falls below ln 9 = 2.197
  and validation retrieval runs. Then `cluster check` on a node for each arm's command.

## 13. Launch, gate and timing

- **Wave 1 (9 GPUs):** seeds 42 and 43 of all four arms, and seed 44 of ML-80 (the primary arm).
- **Gate before wave 2** (on the first finished wave-1 checkpoints of each decoder arm): the 9-way readout sums to
  1; the fitted temperature lies in [0.25, 4]; the validation NLL is below prior-only's; the mean JSD between the
  model's distributions and the prior-only distribution exceeds 0.02 (not collapsed to the prior). If any check
  fails: stop, debug, and do not launch wave 2.
- **Wave 2:** seed 44 of C, ML-80+MAE and Par-cap.
- **D1, D2** run when their code passes review, on free GPUs (the COCO checkpoints are copied to the node as for the
  Stage 2 diagnostics).
- **Cost (estimates, untimed on ArtELingo):** masked runs about 4 h, contrastive about 2.3 h: 9 x 4 + 3 x 2.3 =
  about 43 GPU-h for D4's training; D1 and D2 about 8 GPU-h; D4, D6 and D7 readouts a few GPU-h.
- **Deadlines:** no new 4-hour launch after 2026-10-09 20:00; results pulled before the reservation ends.

## 14. Outputs

Run folders under `res/<wandb.project>/hb_d4/` (pulled from the nodes); readout JSONs under `res/hb/`; registry
`tests/20261007_hb/runs.md`; the report `docs/reports/auto/v1/<date>_hb_first.md` with a `reports_sum.md` row,
followed by the final whole-branch review on the most capable model (the user's rule).

## 15. Risks and caveats

- A fully hidden caption is out of distribution for ML-80 (section 6.1); the bias works against H-b.
- The models learn from about 5 English labels per painting; the target pools 27 other languages, whose
  distributions differ by culture (AL-28 Fig. 4). The English reference measures how much of the gap is cultural.
- "something else" hides readings inside one bucket; the 8-named version checks whether it drives a result.
- One dataset and one backbone (B/32); D5's captioner comparisons decide what is specific to masking.
- ArtELingo run times are untimed; the first epoch's wall time is checked against the estimate.
