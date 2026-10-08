# Final review: docs/reports/auto/v1/2026-10-08_hb_emotion_readout.md

Reviewer: final whole-report review (opus), 2026-10-08 11:35. Read-only on the repo; scratch scripts in the session
scratchpad (`rev/d4_rederive.py`, `d4_analyse.py`, `d6_check.py`, `d1_rederive.py`, `d1_analyse.py`, `d2_rederive.py`,
`collapse.py`, `al28.py`, `train_tab.py`).

## Verdict: Approved with fixes

Every pre-registered verdict re-derives exactly (D4 not supported, D6 no kill, D7 no kill, D1 kill, D2 kill floor-bound),
and so do nearly all numbers. Four claims are stronger than the evidence:
- the D4 "significantly worse" wording, which a seed-level check does not support;
- D7's image effect, measured against a null image that scores worse than the prior;
- D6's view-disagreement positive, which mostly restates the readout's entropy ranking, and whose control the report
  says it lacks even though the encode files hold it;
- two sentences in sections 5 and 6 that call computable baselines unavailable.

The fixes are wording, plus three numbers to add, all computed below. No new runs are needed.

Counts: Critical 0, Important 3, Minor 10.

## Important

### I1. D4: "both comparators were significantly better" does not survive a seed-level check

- **Where.** Summary item 1 (line 9: "so both comparators were significantly better than ML-80"), section 4.1
  (line 159: "The result goes beyond the review's null ... all four intervals exclude zero"), and the closing paragraph
  (line 18).
- **What is wrong.** The pre-registered bootstrap re-derives exactly. The tail probabilities, though, come from painting
  resampling, not from seeds. With 3 seeds per arm, resampling draws among only 10 multisets. ML-80's best per-seed
  entropy Spearman (0.452) lies below Par-cap's worst (0.486), so no seed draw can cross zero. The report's caveat in
  section 8 ("intervals rest on little seed information") does not reach the Summary.
- **Re-derived.**
  - My own code: temperature fit, probe by scipy L-BFGS-B to convergence, my own PCG64 RNG, my own vectorised
    hierarchical bootstrap with B = 10,000. Results: JSD vs Par-cap +0.0103 [+0.0032, +0.0169], p 0.0032; JSD vs probe
    +0.0125 [+0.0080, +0.0171]; rho vs Par-cap −0.1025 [−0.1631, −0.0455]; rho vs probe −0.1473 [−0.2012, −0.0980]. The
    last three have p < 1e-4. All match Table 5.
  - Seed-level Welch t-tests on the per-seed metrics, 3 against 3:
    - ML-80 JSD 0.2736 / 0.2809 / 0.2769 against Par-cap 0.2655 / 0.2735 / 0.2617: p 0.078.
    - JSD against the probe: p 0.026.
    - rho, ML-80 0.452 / 0.362 / 0.417 against Par-cap 0.547 / 0.486 / 0.506: p 0.039.
    - rho against the probe: p 0.030.
    - Holm over the four: 0.106 each.
- **Fix.** Keep the verdict: support needs ML-80 to win, and it lost every comparison. Replace "significantly better"
  and "goes beyond the review's null" with wording like this: "every pre-registered difference went against ML-80;
  under the pre-registered hierarchical bootstrap all four intervals exclude zero, but with 3 seeds a seed-level Welch
  check gives Holm p = 0.11 for each, so the data show that ML-80 is not better, and only suggest that it is worse."
  Put the Welch line in section 4.1 next to Table 5.

### I2. D7: the null image is worse than the prior, and the prior-only baseline is computable

- **Where.** Section 6, line 304 ("A per-caption prior-only log-loss on these test captions is not in the result
  files"). Also Summary item 5 (line 13), line 304 ("image and partial text combined"), and section 9 (line 519, "What
  remains of H-b ... the joint behaviour D7 shows").
- **What is wrong.** The sentence is false. Every encode.npz holds `train_counts` (the prior) and `d7_label`. The
  user's rule asks for a baseline beside every headline number, and without this one the real-minus-null gap reads as
  the painting's contribution.
- **Re-derived.**
  - Prior-only log-loss on the 31,282 test captions: 2.041 nats (2.039 with the prior's own fitted T = 1.089; 2.050 on
    the 16,041 captions with j = 8).
  - ML-80 at j = 0: real 1.925, null 2.302. The null is +0.261 above the prior, and the real image is only 0.116 below
    it. About 69% of the −0.377 real-minus-null gap is the null image's own penalty.
  - Other nulls: ML-80+MAE 2.219 (+0.178); Par-cap 2.182 (+0.141, with real 1.874, 0.167 below the prior).
  - At j ≥ 2 the real-minus-null gaps (−0.132 to −0.218) are smaller than ML-80's j = 0 null penalty. The null control
    therefore cannot show that the painting adds anything once words are visible.
  - Every Table 12 value re-derives (all 9 patterns, n, log-loss, accuracy and intervals within bootstrap noise).
- **Fix.**
  - Add a prior-only column or row to Table 12 (2.041; 2.050 at j = 8).
  - Replace the false sentence with the j = 0 decomposition: the painting is 0.116 nats better than the prior, while
    the null image is 0.261 worse.
  - State that the pre-registered kill verdict ("no kill") stands as defined, but that the null control cannot
    establish an image contribution at j ≥ 2.
  - In Summary item 5, line 304 and section 9, reduce the "joint behaviour" claim to what the data show: visible words
    steer ML-80's emotion slot, and the image's added value under partial text is untested.

### I3. D6: the control the report calls missing is in the encode files, and the MI correlation runs through the readout's entropy

- **Where.** Section 5, line 278 ("we did not compute the between-view term for Par-cap or the dual encoders on the
  same views"). Also Summary item 4 (line 12), section 8 (line 513), and section 9 (line 519, "a weak positive from view
  disagreement (D6)").
- **Re-derived.**
  - ML-80's MI-vs-human-entropy Spearman is 0.222 / 0.136 / 0.173 (mean 0.177), and ML-80+MAE's is 0.138. Both
    match Table 11.
  - Par-cap's `al28_dec_views` (16 views x 9 lengths) are in its encode files. On the same views, with its own
    views16 T, its Spearman is −0.017 / −0.061 / −0.022 (mean −0.033). The raw correlation is therefore specific to the
    view-trained arms.
  - ML-80's MI correlates 0.35 to 0.39 with the entropy of its own 16-view readout. That entropy correlates 0.36 to
    0.45 with human entropy, which is exactly D4's entropy Spearman.
  - Partial Spearman of MI with human entropy, controlling for the readout entropy: ML-80 0.058 / 0.011 / 0.022
    (mean 0.030); ML-80+MAE 0.014.
  - Mean Spearman within quintiles of readout entropy: ML-80 0.105 / 0.029 / 0.050.
  - So "the views disagreed more on paintings people disagree on" mostly restates that ML-80's readout is flatter on
    those paintings. D4 already measured that ranking, and ML-80 does it worse than Par-cap.
- **Fix.**
  - Report the Par-cap control (−0.03) and the partial correlation (0.03) in section 5, and add them to Table 11.
  - Remove "it has no control" from lines 12, 278 and 513.
  - In section 9, replace "a weak positive from view disagreement (D6)" with "view disagreement tracked human entropy
    in the view-trained arms only, and almost entirely through the readout's overall entropy (partial Spearman 0.03)".

## Minor

- **M1. Registry timestamps.**
  - The registry entry labelled "2026-10-08 10:45" was written by 10:25:54, the mtime of `tests/20261007_hb/runs.md`.
    The `.superpowers/sdd/hb-report/progress.md` entry labelled "11:15" was likewise written by 10:45.
  - The report cites "(registry, 10:45)" at lines 474 and 489, and "entries of 04:33 and 10:45" at line 505. Correct
    the labels with `TZ=Europe/Amsterdam date` in both files.
  - The ordering claim at line 491 (M1 detached scored after the interpretation) is supported. `d2-brief.md` (10:25)
    states the interpretation for M1 clean and the control only, and the build script (10:36) is the first file that
    scores M1 detached.
- **M2. Line 53, "The first real readouts started at 21:31".** The gate encode started at 21:23:54
  (`res/hb/gate/encode_ml.log`). It is still after 9fd0bba (20:44), so the pre-result claim holds. Write 21:24.
- **M3. Line 427, "the text-only control was taken out of the 'best decoder' at the same time".** This was a separate
  ruling (Task 4a follow-up, 87cb368, 00:19), after the Task 3 floor ruling. Both came before D2 started (04:33).
  Reword.
- **M4. Line 394, "Spearman correlation of 0.685 ... (parallel score; r² 0.533)".** The r² is the mean per-query
  Pearson r². I re-derived it: ML-80 0.533 / 0.734, range 0.533 to 0.911, M1 clean 0.339 / 0.495. It is not the square
  of the Spearman (0.685² = 0.47). Label it "Pearson r²".
- **M5. Line 242, "Among the decoders ML-80's readout hedged most and moved least from the prior".** ML-80+MAE hedged
  more (2.730 bits, JSD to the prior 0.193, against 2.723 and 0.200). Say "the two ML-80 readouts".
- **M6. Line 129, "fell by 28.2 to 29.5 points by the last epoch".** These are arm means (C 28.2, ML-80 28.8,
  ML-80+MAE 28.9, Par-cap 29.5). Per run the drops ran from 25.2 to 31.1. Say so.
- **M7. Summary item 3 and line 244, "narrowed the gap ... but did not close it".** ML-80 on the full image against
  Par-cap:
  - JSD 0.271 ± 0.007 against 0.267 ± 0.006 (per seed 0.263 / 0.276 / 0.273 against 0.266 / 0.274 / 0.262; Welch p
    0.50).
  - rho 0.461 ± 0.036 against 0.513 ± 0.032 (Welch p 0.14).
  - The JSD gap is within seed noise. Say "narrowed the JSD gap to within seed noise; rho still 0.05 lower".
- **M8. Table 3 against Table 6 prior.** The gate used the raw prior (validation NLL 2.0265, JSD 0.3167). Table 6's
  prior-only has its own fitted temperature (T 1.089, NLL 2.0253, JSD 0.3181). State in 2.4 that prior-only gets the
  same temperature fit, so 0.317 and 0.318 do not look inconsistent.
- **M9. Figure 3 (d4_primary.png).** It shows bars on a JSD axis that starts at 0.20, which inflates the visual gaps.
  Use dots with whiskers, or start the axis at 0.
- **M10. Figure 2 (training.png).** C's epoch-7 point averages 2 seeds (C seed 42 stopped at epoch 6). Note it in the
  caption.

## Checked and honest (no change needed)

- **D1 kill-rule timing (line 324).** The Plan 3 ledger shows the definition was ruled in the final-review fix wave
  (`final-fix-brief.md`, 23:37), after the first ML-80 checkpoint was logged (23:15/23:16). It was coded at d8ba081
  (00:12), when all three ML-80 checkpoints had finished. The report states the 00:12 version and that results existed
  by then, which is honest. At −15 to −19 points with seed SDs of 0.05 to 0.38, any reasonable definition fires.
- **D2 floor-bound kill.** The ruling to report the ceiling and read D2 from balance and selection was made in Task 3's
  review, before D2 ran. The selection index itself is defined in plan 3 (ad3a80c, 17:21), before any result.
- **D4 rulings M3 (strongest probe on post-temperature NLL) and M5 (D7 overall kill).** Both are in 9fd0bba (20:44),
  before the first real readout (21:24).
- **ML-80 OOD caveat.** It is stated in 2.3, 4.5 item 2 and section 8, as the spec requires.
- **D7 "favours masking by construction".** This is stated wherever D7 is used.

## Re-derived and matched

- **Data (my own CSV parse).**
  - 173,745 votes, 1,658 paintings, 28 languages.
  - 6,515 English votes; "other" 1,089 (0.6%).
  - 1,600 paintings with ≥ 20 non-English votes; median 104, range 83 to 125.
  - Mean human entropy 2.484 bits; thirds cuts 2.358 / 2.686 (533 / 533 / 534).
  - AL-28 names and counts identical to every encode file.
  - Train: 60,242 paintings, 302,841 labels; validation: 15,233 labels on 3,060 paintings.
  - Train marginals and the AL-28 mean target as in 2.1.
- **Training (run.json, metrics.jsonl, config.yaml).**
  - All 12 runs at 6f77ad4, clean, completed.
  - Best epoch 2 in 11 runs and 1 for C seed 42; every run stopped at epoch 7, C seed 42 at epoch 6.
  - 31.24 GPU-h in total; masked runs 2.88 to 3.06 h, contrastive 1.41 to 1.65 h.
  - ML-80 peak 238.01 ± 1.16, epoch 7 209.21 ± 2.68.
  - Validation emotion loss: ML-80 1.714 → 1.654, Par-cap 1.871 → 1.867. C's validation loss 2.65 → 4.34.
  - Table 4 in full, including Welch / paired p 0.291 / 0.132, 0.282 / 0.298 and 0.605 / 0.507.
  - Configs match Table 2: ratios, image source, MAE weight, emotion weight 0.07, lr 1e-4 / 1e-5, warmup 500, wd 0.05,
    patience 5.
- **Gate.** gate_all.json matches Table 3 (T, NLL, JSD to the prior and to the humans, rho); it passed at 21:52:09. The
  run start and end times match 2.5 (last run ended 00:52).
- **D4 (independent implementation).**
  - Every row of Table 6: JSD, rho, validation NLL and T for all 12 runs and all readouts.
  - Strongest probe per seed: C / ML-80+MAE / C (post-T NLL 1.8471 / 1.8497 / 1.8516). At T = 1 seed 44 would pick
    ML-80 (1.8552 against 1.8556), whose probe gives 0.265 / 0.544.
  - Probe 0.265 / 0.558; the four tests and the secondary ML-80+MAE test (+0.0021 [−0.0043, +0.0083], −0.020).
  - Gap closed 49.4 / 57.0 / 61.8 / 64.5%.
  - Table 9 in full; human JSD to the prior 0.317.
  - Table 8 thirds, including the tempered prior 0.412 / 0.298 / 0.245.
  - Table 7 variants.
  - Split-half with my own RNG: 0.235 / 0.767 (report 0.770).
- **D6.** Table 10 (all K, full image, controls) and Table 11 per seed.
- **D7.** Table 12 for ML-80, ML-80+MAE (−0.290 at j = 0, −0.104 at prefix 8) and Par-cap (flat at 1.873 to 1.874).
- **D1 (my own AP@R and re-rankers from scores.npz; ECCV positives via eccv_i2t).**
  - 1,261 queries, R 6 to 48, thirds 486 / 430 / 345.
  - Dual mAP@R against run.json within 0.0004.
  - Random 8.27 / 6.19 / 8.40 / 11.02 and oracle 62.09 / 67.39 / 62.22 / 54.46 (exact expectation); 10.76 positives
    (62%), 8.23 / 11.15 / 13.86.
  - Every Table 13 cell (best α, differences, SDs, top-third, kill verdicts, (b) differences, collapsed counts).
  - Table 14 (the text-only decoder-only middle third is 6.74 against 6.75, rounding).
  - Table 15 Welch p 0.021 / 0.004 / 0.004 / 0.0001 / 0.006; ML-80 minus ML-80+MAE −0.00 / −0.03 (p 0.98 / 0.72);
    ML-80 minus ML-15 +2.27 / +1.67 (p 0.0005 / 0.0002).
  - Deficit by third −26.42 / −15.73 / −9.40; lift over random 5.23 / 5.06 / 3.39.
  - Real-vs-null Spearman 0.685 / 0.826, range 0.513 to 0.930; Pearson r² as M4; PMI-vs-dual 0.10 to 0.21.
  - Decoder-only lifts 1.56 / 3.65 / 0.50 / −0.55.
  - Control: α = 1 equals the dual order exactly; unsnapped it reorders 33 to 196 queries, max |dec − null| 1.9e-6.
  - The scorer fix moved image-reading tuned results by at most 0.0044.
  - Test MLM loss 3.050 / 2.898. Timing 20.7 to 23.4 min per checkpoint, 5.62 h in total.
- **D2 (my own balance, selection and cluster bootstrap with a different RNG).**
  - Pairs: 1,324 distinct images, 928 used once, at most 8 uses; 324 components, the largest 47.
  - Table 16 (both-covered and ceilings 0.37 to 3.50) and Table 17 in full; largest seed SD 0.024.
  - Decoder rises 0.711 / 0.601 / 0.603 / 0.489; dual 0.80 to 0.86. Control 0.499 at every λ, score gap −0.017.
  - Table 18 in full: ratios, medians, near-ties 19.0 / 20.7 / 55.0 / 7.7, own-minus-other 2.85 → 1.55.
  - Rise CIs [+0.025, +0.045], [+0.024, +0.042], [+0.009, +0.031], [−0.025, −0.006]; ML-80 minus M1 clean
    [+0.043, +0.058].
  - Timing 0.3 to 30.8 min per model, 5.78 h.
- **Collapse (CPU, my own code, 200 test images, seed-42 checkpoints, view seeds 1234 and 999).**
  - View-view cosine: M1 clean 0.980, text-only 0.968, M1 detached 0.975, against C 0.435, ML-80 0.432, ML-15 0.498.
  - View-to-full R@1: 0.005 / 0.010 / 0.015 (chance 0.005), against 0.87 to 0.97 for the others.
  - Unchanged under the second view seed. Matches Table 19.
  - The configs confirm the "partial views reach the tower through" column (MAE weights, image sources, fusion).
- **Chain and timings.**
  - Spec 7e250bf 17:01 and amendment 0beb174 18:10.
  - Readout rulings at 9fd0bba 20:44.
  - Provisional table 22:44, final 01:53, D1 tables 04:35, D2 finished 10:21.
  - Ledgers as cited.
  - Emotion-weight arithmetic (1/14 = 7.1%, 1/17.4 = 5.75%).
  - Review quotes (48.0 / 71.9, α* 0.819 in section 5.3, D4 null "tie within seed noise", counter-arguments 1 to 3).
  - Stages 1 and 2 numbers (37.94, 37.24, 37.11, 37.09, 36.98).
- **Form.** No dashes. Times are written without offsets. `reports_sum.md` row present and `check_reports_sum.py` prints
  OK. All 9 figures were opened: axes labelled, legends present, and captions match the content (style notes in M9 and
  M10).

## Fix wave

Fixer: opus, 2026-10-08 11:54. Every new number below is computed and printed by
`docs/reports/assets/build_2026-10-08_hb_emotion_readout.py`, which was re-run end to end on the local GPU (section K
included; `nvidia-smi` showed no compute process; run under `flock -n -o -E 75 /tmp/gpu0.lock`, 203 s, exit 0). No
existing number changed: the script's checks still print max |re-derived JSD minus hb_d4.json| 0, D1 dual vs run.json
0.0004, D2 balances and selection 1.1e-16, and Table 19 reproduced to the last digit. `scripts/check_reports_sum.py`
prints `reports_sum.md: OK`. Nothing committed.

### I1. D4 wording
- Build script section E now prints a seed-level check: Welch t-test (scipy, unequal variances) on the 3 per-seed
  metrics of each side, Holm (`mmae.engine.hb.bootstrap.holm`) over the four. ML-80 JSD 0.2736 / 0.2809 / 0.2769 vs
  Par-cap 0.2655 / 0.2735 / 0.2617, p 0.078; vs probe p 0.026; rho vs Par-cap p 0.039; rho vs probe p 0.030; Holm
  0.105 each. ML-80's best rho seed 0.452 vs Par-cap's worst 0.485; best JSD seed 0.2736 vs Par-cap's worst 0.2735.
  These match the review.
- Report: Table 5 gained two columns (seed-level Welch p, its Holm p). Section 4.1 now says the bootstrap tails come
  mostly from painting resampling (10 seed multisets per arm), that the data show ML-80 is not better than either
  comparator and only suggest it is worse; "goes beyond the review's null" replaced. The pre-registered bootstrap numbers
  are unchanged. Title ("did not match dense human votes better than"), Summary item 1, closing paragraph, section 8
  (Seeds), section 9 and the reports_sum row (both the Latest line and the table row) say the same.

### I2. D7 prior-only baseline
- Build script section H computes the prior-only log-loss on the D7 test captions: prior = column sums of
  `train_counts` from the ML-80 seed-42 encode.npz (checked identical `d7_label`/`d7_valid` across the three arms),
  log-loss = mean of -log prior[d7_label] over each pattern's valid captions; also with the prior's own fitted
  T = 1.089 (from hb_d4.json `readouts.prior.T`). Results: 2.041 at j = 0, prefix/random 1 and 2; 2.040 at 4; 2.050 at
  8 (16,041 captions); with T: 2.039 / 2.038 / 2.047. Matches the review's 2.041 / 2.050 / 2.039.
- Decomposition printed at j = 0: ML-80 real 1.925 (-0.116 vs the prior), null 2.302 (+0.261), 69% of the -0.377
  gap is the null penalty; ML-80+MAE null +0.178 (gap -0.290, 61%); Par-cap real -0.167, null +0.141 (46%). ML-80's
  j >= 2 real-minus-null gaps -0.218 to -0.132, all smaller in size than its j = 0 null penalty (printed as a check:
  True).
- Report: Table 12 gained a prior-only column (caption gives the fitted-T offset, 0.002 to 0.003); Figure 6 gained the
  prior-only line, legend moved below the panels. The false sentence ("not in the result files") was replaced by the
  j = 0 decomposition as a list. D7 is reframed: words steer the emotion slot (log-loss falls with j with the null image
  too, 2.302 to 1.582 / 1.566); the cross-modal claim is not established. The pre-registered "no kill" stays as computed,
  with the caveat. Summary item 5, closing paragraph, D2's comparison paragraph and its "What D2 says" (D2 noted as the
  COCO counterpart with its own controls, needing no null image), section 8, section 9 (the "joint behaviour" claim
  removed), the AR-cap option (a no-image condition the decoder does not penalise) and the reports_sum row updated.

### I3. D6 controls
- Build script section G, new `d6_controls`: MI from `tables.between_view_mi` with each arm's views16 T, readout entropy
  = entropy of `calibrate.mixture` of the views16 logits; Spearman via `bootstrap._spearman_rows`; partial Spearman =
  partial correlation of ranks; quintiles of readout entropy. Asserts: readout-entropy vs human Spearman equals the D4
  views16 entropy Spearman exactly, and ML-80/ML-80+MAE MI Spearman equals hb_d4.json per seed. Intervals: raw with
  `tables.boot_spearman` (Table 11's bootstrap, seed 0, B = 10,000), partial with the same scheme (seed 1, B = 10,000).
- Results: Par-cap MI vs human entropy -0.017 / -0.061 / -0.022, mean -0.033 [-0.086, 0.019]; ML-80 MI vs readout
  entropy 0.385 / 0.350 / 0.371; readout entropy vs human 0.452 / 0.362 / 0.417; partial ML-80 0.058 / 0.011 / 0.022,
  mean 0.030 [-0.020, 0.082]; ML-80+MAE 0.014 [-0.032, 0.062]; within-quintile mean ML-80 0.105 / 0.029 / 0.050
  (0.061). All match the review. New: Par-cap's partial is -0.257 [-0.306, -0.206] (reported in Table 11, not
  interpreted; Par-cap never trained on views).
- Report: Table 11 gained the Par-cap row and a partial column; section 5 rewritten (two controls as a list; view
  disagreement adds no view-specific signal beyond D4's entropy ranking; pre-registered no-kill stands on the raw
  correlation). "No control" removed from Summary item 4, section 5 and section 8; section 9's "weak positive from view
  disagreement" replaced with the reviewer's wording (partial Spearman 0.03).

### Minor
- M1. Registry `tests/20261007_hb/runs.md`: entry label 2026-10-08 10:45 -> 10:25 (D2 finished 10:21; file mtime
  10:25:54). Ledger `progress.md`: 11:15 -> 10:45 and, by the same check, 11:55 -> 11:37 (the final-review entry; file
  mtime 11:37:05, it was 11:38 when I checked). Report citations "(registry, 10:45)" x2 and "entries of 04:33 and 10:45"
  -> 10:25.
- M2. Section 1.3: "The first real readout, the gate's encode of ML-80 seed 42, started at 21:24"; source
  `res/hb/gate/encode_ml.log` (21:23:54) added.
- M3. Section 7.2: the control's exclusion is now a separate, later ruling (Task 4a follow-up, `87cb368`, 00:19); both
  rulings before D2 started (04:33).
- M4. Section 7.1: "mean per-query Pearson r² 0.533" and "Pearson r² 0.533 to 0.911".
- M5. Section 4.5 item 1: "the two ML-80 readouts hedged most and moved least" with both arms' entropy and JSD to the
  prior (2.723 / 0.200 and 2.730 / 0.193).
- M6. Section 3: "28.2 to 29.5 points on average per arm, 25.2 to 31.1 per run"; the build script prints per-run drops
  and the all-run range.
- M7. Summary item 3 and section 4.5 item 3: JSD gap within seed noise (Welch p 0.50), rho 0.05 lower (p 0.14); both
  p values printed in section E.
- M8. Section 2.4: prior-only gets the same temperature fit (T 1.089, NLL 2.0253, JSD 0.318), the gate used the raw
  prior (2.0265, 0.317); section E prints both.
- M9. Figure 3 redrawn as dots (mean, diamond) with SD whiskers and open seed circles; caption updated.
- M10. In the script C's curve uses the epochs common to all seeds, so it ends at epoch 6 rather than averaging 2
  seeds at epoch 7 (section B now prints seeds per epoch: C [3, 3, 3, 3, 3, 3, 2], drawn 6). Caption says C's curve
  ends at epoch 6, where seed 42 stopped.

### Humanizer pass
No em or en dashes. Section 4.1 split into two paragraphs, the self-referential "which the final review added"
replaced by "(not pre-registered)", the closer of section 4.3 merged into its paragraph, Summary Holm value made 0.105
throughout. Terms gained "Seed-level Welch check" and "Partial Spearman". Section 10 timing updated to about 3.5 min.

### Left for the re-review
- D6 Par-cap partial (-0.257) is reported without interpretation.
- D7's "about 69%" decomposition assumes the null image's penalty is additive; the report states it as a share of the
  gap at j = 0 only.

## Scoped re-review (2026-10-08 11:56)

All findings addressed, no new problems: prior-only D7 log-loss 2.041, the null image's 69% share of the j = 0 gap, Par-cap view disagreement -0.033 and the partial Spearman values were re-derived. Report final.
