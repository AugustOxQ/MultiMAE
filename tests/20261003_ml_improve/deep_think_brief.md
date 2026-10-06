# Deep-thinking brief: cross-modal masking and polysemy in MultiMAE

You are asked to think deeply, as a research collaborator, about the "cross-masking" line of MultiMAE: CLIP ViT-B/32
towers fine-tuned on COCO with InfoNCE plus a masked pass (image patches dropped for MAE, caption tokens masked for
image-conditioned MLM) and fusion modules feeding the reconstruction decoders. Retrieval always uses the clean pooled
CLIP embeddings (a dual encoder), so the masked pass can only act through training gradients.

## Ground rules
- Read-only on the repo and on the cluster. Do not edit, commit, launch, kill or sync anything.
- The only file you write is `tests/20261003_ml_improve/2026-10-05_cross_masking_deep_think.md`.
- Re-derive any number you lean on from the files below rather than trusting this brief. Say when you are guessing.
- Write like a paper draft (we, past tense for what was done); no em or en dashes.

## The research question (the user's words)
"The very broad research question is to see if masked model can somehow improve polysemy understanding. Not yet sure
how this can lead to a proper topic now, but we need to work towards that." The user considers COCO R@K only a sanity
check. Polysemy here has two readings: one-to-many matching (an image fits many captions) and lexical ambiguity.

## What the user has said since (2026-10-05)
- The Stage 2 gain (about +0.5 to +0.6 ECCV mAP@R) is "a bit small".
- SemEval-2023 VWSD is "just word disambiguation, not real polysemy" (a homonym plus a context word that resolves it).
- ArtELingo (WikiArt paintings, about 5 annotators each giving an emotion and an explanation; used in the user's other
  project, CoSiR) is a candidate "real polysemy" setting: one image, several valid readings. Decision on hold. Local
  copy `/data/PDD/artelingo/` (English JSON splits: train 308,723 captions over 61,402 paintings; test 31,282 over
  6,246; 82% of test paintings get at least 3 distinct emotions; median emotion entropy 1.52 bits).

## Read these (in this order)
1. `CLAUDE.md` (architecture, switches, metrics).
2. `docs/superpowers/specs/2026-10-03-improve-multilearner-design.md` (staged plan, metrics, H1/H2, arms, success bar).
3. `docs/reports/auto/v1/2026-10-03_lever_review.md` (integrative lit review; supporting files in
   `docs/reports/assets/2026-10-03_lever_review/`, B4 covers masking and polysemy).
4. `docs/reports/auto/v1/2026-10-03_stage0_diagnostics.md` (where the fusion PMRP gain sits; H1 vs H2).
5. `tests/20261003_ml_improve/runs.md` (every Stage 1 and Stage 2 result and ruling, in order) and `caveats.md`.
6. Code as needed: `mmae/models/model.py`, `masking.py`, `fusion.py`, `decoders.py`, `losses.py`.

## Results so far (re-derive from res/coco/multimae/*/run.json if you rely on them)
Stage 1 (2 seeds per arm, current recipe, vs the 3-seed baselines; * Welch p < 0.05):
```
arm                           n           mAP@R            PMRP            rsum             CxC   ref
contrastive (baseline)        3     36.94±0.17      56.49±0.05     441.94±1.30      55.66±0.42 
multilearner (baseline)       3     36.95±0.24      56.88±0.04     444.30±1.49      56.34±0.27 
contrastive_ep15              1    36.43 -0.51     55.99 -0.50    436.32 -5.62     54.47 -1.19    contrastive
contrastive_meanpool          2    36.36 -0.58     56.39 -0.10   419.65 -22.29*    50.63 -5.03*   contrastive
contrastive_r2lr              2    36.90 -0.04     57.21 +0.73*   446.08 +4.14*    56.56 +0.90*   contrastive
multilearner_m1clean          2    37.05 +0.10     56.61 -0.27    443.49 -0.80     55.97 -0.37    multilearner
multilearner_m1detached       2    36.84 -0.12     56.59 -0.29*   442.04 -2.26     55.84 -0.50    multilearner
multilearner_m2bcontent       2    37.25 +0.30     56.92 +0.04    444.90 +0.60     56.27 -0.07    multilearner
multilearner_m3pooled         2    36.85 -0.11     56.64 -0.24    442.15 -2.15     55.81 -0.54    multilearner
multilearner_m6maskedview     2    37.26 +0.30     56.81 -0.06    446.26 +1.96     56.72 +0.38    multilearner
multilearner_mae0             2    37.54 +0.58*    56.98 +0.10*   446.30 +2.01     56.75 +0.41    multilearner
multilearner_mae0_txt40       2    37.53 +0.58     57.03 +0.15    447.75 +3.45*    56.85 +0.51    multilearner
multilearner_txt25            2    37.39 +0.44     56.92 +0.04    444.94 +0.64     56.15 -0.19    multilearner
multilearner_txt40            2    37.58 +0.63     56.97 +0.09    446.55 +2.25     56.68 +0.33    multilearner
multilearner_txt60            2    37.66 +0.70     56.97 +0.10    448.46 +4.16*    57.09 +0.75    multilearner
multilearner_txt80            3    37.81 +0.86*    56.99 +0.11    449.23 +4.93*    57.25 +0.91*   multilearner
multilearner_txt90            2    37.89 +0.94*    56.90 +0.02    448.92 +4.62*    57.23 +0.89*   multilearner
s2_contrastive                3    37.11 +0.18     56.51 +0.03    441.48 -0.46     55.31 -0.35    contrastive
s2_multilearner               3    37.24 +0.29     56.86 -0.01    444.89 +0.59     56.30 -0.04    multilearner
s2_txt80                      3    37.87 +0.91*    57.08 +0.21*   449.84 +5.54*    57.23 +0.89    multilearner
(diff vs the reference baseline's 3-seed mean; * Welch p<0.05 when >=2 seeds)
```
Stage 2 (seeded sampler, seeds 42-44; s2_mae0 and seeds 45/46 of s2_txt80 still running):
```
arm                 n          mAP@R           PMRP           rsum        CxC R@1       ECCV R-P       ECCV R@1         1K R@1
s2_contrastive      3    37.11±0.14     56.51±0.03    441.48±0.72     55.31±0.11     46.73±0.17     80.82±0.67     73.23±0.22 
s2_multilearner     3    37.24±0.13     56.86±0.08    444.89±0.53     56.30±0.20     46.89±0.10     81.07±1.00     73.76±0.15 
s2_txt80            3    37.87±0.28     57.08±0.04    449.84±0.32     57.23±0.44     47.47±0.23     81.91±1.00     74.61±0.38 

vs s2_contrastive: diff, Welch p, paired-by-seed p (Holm over variants on mAP@R)
  s2_txt80       Holm mAP@R p = 0.027; mAP@R +0.75 (W 0.027, P 0.012); PMRP +0.57 (W 0.000, P 0.001); rsum +8.36 (W 0.001, P 0.005); CxC R@1 +1.92 (W 0.013, P 0.013); ECCV R-P +0.74 (W 0.014, P 0.003); ECCV R@1 +1.09 (W 0.201, P 0.221); 1K R@1 +1.37 (W 0.010, P 0.036)

vs s2_multilearner: diff, Welch p, paired-by-seed p (Holm over variants on mAP@R)
  s2_txt80       Holm mAP@R p = 0.044; mAP@R +0.63 (W 0.044, P 0.077); PMRP +0.22 (W 0.025, P 0.022); rsum +4.94 (W 0.000, P 0.005); CxC R@1 +0.93 (W 0.051, P 0.046); ECCV R-P +0.58 (W 0.035, P 0.073); ECCV R@1 +0.84 (W 0.360, P 0.497); 1K R@1 +0.85 (W 0.047, P 0.026)

success bar (seeds 42-44):
  s2_txt80 (n=3): beats contrastive True (Holm p 0.027), beats multilearner True (Holm p 0.044), rsum guard True, PMRP guard True -> SUCCESS (PROVISIONAL: an arm has fewer than 3 seeds, so the Holm family or n is incomplete)
  s2_mae0: not enough seeds yet

success bar (every completed seed of each variant):
  s2_txt80 (n=3): beats contrastive True (Holm p 0.027), beats multilearner True (Holm p 0.044), rsum guard True, PMRP guard True -> SUCCESS (PROVISIONAL: an arm has fewer than 3 seeds, so the Holm family or n is incomplete)
  s2_mae0: not enough seeds yet
```
Key readings so far: the gain grows with the MLM's share of the masked objective (text-masking ratio 15% to 80%, or
MAE off); cutting the MLM gradient into the vision tower (M1 detached) removes it; giving the MLM the clean image
(M1 clean) or the other modality's pooled token (M3) does not help; content-word masking (M2b) gives half the gain of
a higher random ratio. Stage 0: the fusion-specific PMRP gain sits in the image tower; H2 (softer similarity) held in
relative form; no VWSD effect.

## What we want from you
Think as long as you need. Then write the file with these sections:
1. What the results so far actually show about the mechanism (and what they do not), with the evidence for each claim.
2. Whether and how cross-modal masking could plausibly improve polysemy understanding, in either sense. Give the
   strongest argument for and the strongest argument against. Be concrete about the mechanism (for example: InfoNCE
   with one positive pulls an image toward the mean of its captions; image-conditioned MLM over several annotators'
   texts must model a distribution of words given the image). Check this against the literature in the lever review.
3. What a "real polysemy" evaluation should be. Assess ArtELingo honestly (including annotator noise vs genuine
   alternative readings, affective vs descriptive language, domain shift), and propose the measurements that would make
   it a polysemy test (for example, stratifying paintings by annotator disagreement as a built-in control). Name other
   candidate datasets only if you are confident they exist, and say how sure you are.
4. Two or three candidate paper directions, each with: the claim, the key experiment, the baseline it must beat, what
   result would kill it, compute estimate on 6 A6000 GPUs, and novelty risk.
5. Your recommendation for the next two weeks, and what you would stop doing.
Keep it under about 2,500 words. When done, reply in the chat with only: DONE <path>.
