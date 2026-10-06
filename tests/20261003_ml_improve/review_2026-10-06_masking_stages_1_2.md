# Final whole-branch review: `docs/reports/auto/v1/2026-10-06_masking_stages_1_2.md`

Reviewer: Claude (Fable 5.1), 2026-10-06 17:20. Read-only review; nothing in the repo was edited, committed or launched.

Method. Every table and every number in the Summary was re-derived with the reviewer's own code (not the build script) from `res/coco/multimae/{default,ml_improve}/*/{run.json,config.yaml}` (status `completed` only, seed from `config.yaml`), `res/coco/diagnostics/{stage0,stage2,stage2_controls}/diagnostics.json`, and, for the stratified analysis, from the raw Stage 2 embeddings in `res/coco/diagnostics/stage2/embeddings/` with an independent per-query AP@R implementation (own sort, own bootstrap RNG). The spec's git history, the run log and the model code (`mmae/models/model.py`, `mmae/models/fusion.py`) were read to check the pre-registration account and what each control changes. The six PNGs were inspected. `scripts/check_reports_sum.py` prints OK. Scripts and output: the session scratchpad (`rederive.py`, `rederive.out`, `strat_check.py`), not kept in the repo.

## Verdict

No wrong number and no wrong conclusion found. Every load-bearing number matches to the printed precision. The conclusions follow from the numbers, and the pre-registration account (3-seed vs 5-seed) is honest and matches the run log and the run timestamps. The findings below are one major wording-precision issue in the mechanism section (a control is described as changing less than it does), and minor points of style, completeness and one inaccurate caption note.

## Re-derived and matched (load-bearing numbers)

- Inventory: 53 completed runs (Stage 1 30, Stage 2 arms 14, controls 9); GPU-hours 204.0 / 91.4 / 65.9, total 361.3; five dead Stage 1 folders (status `running`: M6 s43, 80% s43, 90% s42, 90% s43, 80% s44), none counted; Stage 1 commits {eb85340, 9203048, 0ca9b05, a0f946f, 7a12d9f, 1b23184}, every Stage 2 run at 9919d8d; contrastive 4.21 h, masked arms 6.87 to 7.58 h, 80% / contrastive = 7.17 / 4.21 = 1.70.
- Table 2 (every cell: means, stds, five deltas, Welch p, pass/fail) and the exact margins M2b +0.2996, M6 +0.3008; ranking 90% (+0.94) then 80% (+0.86); 80% at seeds 42+44 = 37.90; R2 vs contrastive on six metrics incl. i2t R@1 +2.03 (p 0.080); R2 PMRP minus multilearner +0.34.
- Table 3 (dose curve, all cells); 25% share 0.44 / 0.86 = 51%; MLM ratio 80% / 15% = 1.95.
- Table 4 (all cells); MAE off Stage 1 mean minus Stage 2 multilearner +0.30; MAE off S1 vs S2 -0.34 (p 0.018); 80% S1 vs S2 +0.06 (p 0.791).
- Table 5 (all 10 metric columns, 3-seed and 5-seed rows); per-seed mAP@R 37.76 / 37.66 / 38.19 / 37.96 / 38.13; every 80% seed above every baseline seed (37.02 to 37.37); multilearner vs contrastive +0.13 (0.314), PMRP +0.35 (0.011), rsum +3.41 (0.004); all section 3.2 deltas and p-values (Welch and paired).
- Table 6: 3 seeds +0.75 [+0.16, +1.34] raw 0.0275 Holm 0.0549; +0.63 [+0.03, +1.22] raw 0.0440 Holm 0.0880; MAE off Holm 0.436 / 0.696; 5 seeds +0.82 [+0.51, +1.14] raw 7.4e-4 Holm 0.0015; +0.70 [+0.38, +1.01] raw 0.0017 Holm 0.0033; guards +4.94 / +0.22 and +4.82 / +0.24. Mean seed std 0.164. 6.8% of 10.22; 1.06 below 39.0 (26.72 and 39.0 confirmed in the baselines and lever-review reports).
- Table 7 (all losses); perplexities 4.8 / 21.2 / 38.9 / 18.1; 0.61 and 0.16 nats.
- Table 8 (all cells) from `stage2_controls/diagnostics.json`; diag rsum vs run.json max |diff| 0.020; the 13 runs and 36 swaps shared with `stage2/` agree exactly; 80% image swap rsum 449.30 (+7.82, p < 0.001), PMRP 56.83 (+0.32, p < 0.001); multilearner image swap rsum +3.60 (p 0.007); text-swap range 36.63 to 36.94; 80% text swap vs multilearner +0.13 (p 0.283).
- Table 9 (all cells incl. paired p); Holm over the three controls 0.0608 each; M1 detached vs clean -0.11 (0.568), rsum -1.48 (0.131); best epochs 7, 8, 8 for M1 clean vs 9 or 10 elsewhere.
- Table 10 (all cells); fine-tuning cost -2.16; 16 of 18 masked runs below 55.08; Stage 0 -0.22 (0.851); Stage 2 vs Stage 0 multilearner -2.23 (0.135); pooled -1.69 (0.077); 0.216 points per item; binomial SE 2.32.
- Table 11 and section 5.2: my own per-query AP@R from the embeddings reproduces every run's run.json i2t and t2i mAP@R within 0.02 and gives identical stratum means (i2t +0.85 / +0.46 / +0.19, high minus low -0.66; t2i +0.64 / +0.80 / +0.74, +0.11), per-seed high-minus-low -0.80 / -0.36 / -0.82, and bootstrap CIs within 0.05 of the report's with a different RNG; tertile counts 486 / 430 / 345 and 524 / 372 / 436; relative gains +2.3% / +1.6% / +0.8%; CI half-widths 0.55 to 1.00. As an extra check, Spearman correlation between R and the seed-averaged per-query gain is -0.03 (i2t) and +0.02 (t2i).
- Pre-registration: the spec has exactly one commit, `dba32b0`, 2026-10-03T18:26:51+02:00 (18:26 Amsterdam), no uncommitted change; the first Stage 1 jobs were launched 18:21 to 18:23 that day and take 4 to 7 h, so no Stage 1 result preceded it. Spec section 3 says "over at least 3 seeds (5 for the final candidate if the budget allows)" and section 7 "The final candidate gets seeds 45 and 46 if the budget allows", as quoted. Run timestamps: s2_txt80 seed 42 ended 16:27, seed 45 created 16:31, seeds 43 and 44 ended 20:14 and 20:28, seed 46 created 20:21, s2_mae0 seed 42 ended 20:47, seed 45 ended 23:42, seed 46 ended 03:29 on 10-06. The run log's 16:30 entry ("Primary Stage 2 tests stay on seeds 42-44") and 20:33 correction are as the report describes, so section 3.3's account ("three 80% seeds in, borderline vs multilearner known, no MAE-off result yet, seeds 45 and 46 running") is accurate and complete.
- Configs: Table 1's switches (R2 lr_text 5e-05, lr_vision 5e-06, layer decay 0.7, freeze 2; M6 weight 0.25; M3 pooled_conditioning; M2b text_mode content; R3 mean pooling; R5 15 epochs) and section 1.3's recipe (10 epochs, batch 128, lr 1e-4, lr_backbone 1e-5, 500 warmup, wd 0.05, monitor val rsum) match every `config.yaml`; every Stage 2 run has `seeded_sampler: true`, the controls are `fusion_none` + text_ratio 0.8 and `fusion_multilearner` + text_ratio 0.8 + `mlm_image_source` clean / clean_detached.
- `tests/20261003_ml_improve/stage2_table.py` prints the same tests. `reports_sum.md` row: every number in it matches the report; `check_reports_sum.py`: OK. No em or en dashes in the report; no hyphen used as a dash; all times are plain Amsterdam local time.

## Findings, by severity

### Critical

None.

### Major

1. **Section 4.3 (and Summary item 4, Table 1, Terms): `fusion_none` at 80% changes more than "the MLM reads no image".** From `mmae/models/fusion.py`, `NoFusion.forward` returns the raw projected tokens of each modality, so compared with the 80% arm the control also (a) removes the three learner transformers of `MultiLearnerFusion` (the MLM decoder reads raw projected text tokens instead of the text and joint learners' output) and (b) stops the MAE decoder reading the masked caption. The report's Terms entry ("each decoder reading only its own modality") is accurate, but 4.3 attributes the result to the image alone: "The image matters: `fusion_none` at 80% masks the same tokens, but its MLM reads text alone, and it gained nothing", and 4.1 attributes the whole 0.61-nat MLM-loss gap to the missing image ("The image carried much of that prediction: without it ... 0.61 nats higher"), although part of it may be the missing learner capacity. The mechanism conclusion itself survives, because M1 clean and M1 detached keep the learners and the MAE's text access and still lose the gain, so the thing all three controls share with each other and not with the 80% arm is the MLM's memory not being the masked-pass image tokens. Suggested wording for 4.3: "`fusion_none` at 80% removes the fusion altogether (no learners, the MLM reads text alone and the MAE reads image alone) and gained nothing; M1 clean keeps the fusion and the MAE's text access and gives the MLM the clean image instead, and also lost the gain. Together they say the gain needs the masked-pass image tokens in the MLM's memory; `fusion_none` alone cannot separate the image from the fusion learners." And in 4.1: "without the fusion (`fusion_none`) the loss was 0.61 nats higher; most of that gap is the image, since the fusion with the clean image lowers it a further 0.16 below the 80% arm, but the learners' own contribution is not separated."

### Minor

2. **Figure 2 caption and the 2-of-3 note are inaccurate.** The seed-42 Stage 0 multilearner run's `run.json` does hold `test/loss_mlm` (1.5646) in `results.test`; only its re-evaluated `eval.test` block lacks the losses, and the build script's `load()` takes `eval.test` *instead of* `results.test` when it exists, dropping the losses. With all three runs the 15% MLM loss is 1.568 (the report's 1.569 uses seeds 43 and 44), the ratio stays 1.95, nothing else changes. Fix: read losses from `results.test` in `load()` (merge the two blocks) and drop the "re-evaluated without losses" sentence from the Figure 2 caption; Table 3's 15% cell becomes 1.568.

3. **Section 1.3: "died when the node reservations were renewed on 2026-10-05 around 01:30".** The run log's 01:36 entry said "renewed" and its 01:47 entry corrected it: "reservations were killed (user), not renewed on purpose". Say "when the node reservations ended around 01:30" (the five dead folders are the ones listed in the 01:36 entry; the log's "all 6 running jobs" lists five, and five folders carry status `running`, so "five" is right).

4. **Section 3.3 / Table 6: say explicitly that the five-seed test is unpaired and compares 5 candidate seeds with 3-seed baselines (seeds 45 and 46 have no baseline runs; Welch df about 6), and that the paired-by-seed analysis exists only for seeds 42 to 44.** Figure 3 and Table 5 imply it; one sentence would close it. Worth adding in the same place: the Holm family. The report reads the spec's "Holm-corrected over the variants" as two tests per baseline; under the stricter reading (one family of four tests, 2 variants x 2 baselines) the three-seed Holm p-values become 0.110 and 0.132 and the five-seed ones 0.003 and 0.005, so the verdict does not depend on the family definition. Stating this removes a reviewer's objection at no cost.

5. **Summary item 5 and section 5.2 state the stratified null without its power caveat; section 5.3 supplies it.** "the 80% gain did not grow" is correct as a description of the point estimates, but the t2i high-minus-low interval [-0.80, +1.01] admits growth of a full point. Suggest "did not grow (i2t fell, t2i flat; intervals of about plus or minus 0.7 to 1.0 points)" in the Summary, matching section 5.3. The i2t decline, the per-seed consistency and the Spearman check (rho -0.03 and +0.02 between R and the per-query gain) all support the reading.

6. **Section 2.1: "Re-deriving the means corrected two entries of the run log."** Only M2b's status changes (the log's 00:14 entry called it a formal pass; it is +0.2996). The log's 09:14 entry already recorded M6 as a formal pass (+0.30), so only one entry was corrected; say so.

7. **Section 3.2 and Table 9 test about ten tracked metrics per comparison with uncorrected p-values.** The spec pre-registers them as tracked (Welch on every tracked metric), so this is per spec, but one clause ("secondary metrics, uncorrected") in section 1.3's statistics paragraph would make the hygiene explicit. The same applies to the controls' PMRP and rsum tests that section 4.3 leans on (p <= 0.002: they would survive any correction over the table's 18 tests, which is worth saying).

8. **Table 7, MAE off row: the MAE loss 1.274 comes from a decoder trained at weight 0, so it measures nothing.** A footnote or a blank cell would avoid a reader comparing it with 0.710.

9. **Zero-shot VWSD: the report's 57.88 is the GPU diagnostics value; CLAUDE.md and the Stage 0 CPU dry run give 58.10 (one item).** The Stage 0 report documents the one-item gap; this report could cite it in a half sentence in Table 10's caption, since 57.88 is now quoted as the reference in Summary item 5.

10. **Definitions at first use.** "ECCV Caption" is used (Terms, section 1) without saying what it is (the human-verified extended COCO 5K positives of Chun et al., ECCV 2022). "Deep-think note" is explained only by its link. "Welch" is introduced in 1.3 as "two-sided Welch t-tests" without saying it allows unequal variances; fine for the audience, but one clause would match the report's own rule.

11. **Figure 1: the "guard -3" label sits on the R5 15 epochs bar in the right panel.** Move the label up (e.g. `len(STAGE1) - 1.3`) or to the top of the panel.

12. **Section 4.3, last paragraph, lists two readings of the M1 clean result; a third alternative, worth one sentence, is that the vision tower benefits from receiving MLM gradient through a forward pass on 25% of the patches (a partial-input forward, as in MAE encoder training), independently of what the caption task is.** This is close to the report's "information bottleneck" reading but is a statement about the vision tower's training signal rather than about the decoder; the planned "80% with MAE off" run would not separate it either, so it belongs in the limits.

13. **Caveats, "Compute is not matched".** Correct as stated; it could add that the three controls run at 6.87 to 7.58 h, so compute is matched among the masked arms and the controls, and only the contrastive baseline is cheaper.

## Alternative explanations considered and how the data bear on them

- MAE-loss interaction at 80%: untested, and the report says so (sections 4.3, 6, 7). The MAE loss column of Table 7 (0.719 for the 80% arm vs 0.657 to 0.664 for the controls) shows the MAE task changes when the MLM shares the masked view, so an interaction is possible; the proposed "80% with MAE off" run is the right next experiment.
- Token count and clean-pass sharing in M1 clean: M1 clean's MLM memory is the fusion over all clean-pass patch tokens (about 50) rather than the 25% kept tokens, computed in a second fusion pass, with its gradient entering the vision tower through the same clean forward that produces the retrieval embedding. The report names the second fusion pass, the different dropout masks, the gradient-conflict reading and the earlier validation peak (epochs 7, 8, 8). I confirmed from `model.py` lines 175 to 177 that the MAE path of M1 is unchanged (it reads `fused.image_memory` from the masked-pass fusion), so M1 isolates the MLM's image source as claimed, up to those confounds.
- Text-tower contribution: the swaps cannot measure it (mixing cost of 0.18 to 0.49 on every text swap), and the report says so. The 80% full model sits 0.15 below its own image tower paired with the contrastive text tower, consistent with the 80% text tower being no better than the contrastive one.
- Winner's curse: Stage 2 seeds 42 to 46 are fresh runs with a different sampler, so the Stage 2 estimate is not inflated by the Stage 1 selection; the report attributes MAE off's non-replication partly to it (section 3.4), which the numbers support (half of its Stage 1 margin was a low baseline draw: +0.30 against Stage 2's multilearner; the rest vanished between its own runs, -0.34, p 0.018).
- VWSD instability: Stage 0 (-0.22, p 0.851) and Stage 2 (-3.17, p 0.052) disagree on the masked-vs-contrastive gap; the report reports both, the pooled -1.69 (p 0.077) and the binomial SE. Its claim is correctly limited to "masked training did not help VWSD and 80% did not beat 15%".

## Storage

Nothing over 1 GB left behind; the scratchpad holds two small scripts and a 194-line output file.

## Re-review of the fix wave (2026-10-06 17:35, scoped to the 13 findings)

Verdict: approved. Every fix is correct and introduces no new error; one small residual inconsistency in section 1.3 (item 7 below) is worth a two-word edit but does not block.

New numbers re-derived with my own code and matched:
- 15% MLM test loss over all 3 Stage 0 multilearner runs: 1.5646 / 1.5555 / 1.5833, mean 1.5678 (report 1.568); ratio 80% / 15% still 1.95.
- Holm over one family of four (2 variants x 2 baselines), mAP@R: 3 seeds 0.1099 / 0.1320 (report 0.110 / 0.132); 5 seeds 0.0030 / 0.0050 (report 0.003 / 0.005).
- Welch degrees of freedom, 80% vs contrastive / multilearner: 3 seeds 2.89 / 2.84 (report "2.8 to 2.9"); 5 seeds 5.96 / 5.98 (report 6.0).
- Stratified high-minus-low CI half-widths, 80% minus multilearner: i2t 0.651, t2i 0.906, diversity 0.714 (report 0.65 / 0.91 / 0.71); over all nine comparisons 0.545 to 0.996 (report 0.55 to 1.00).
- Holm over the 15 control-vs-80% Welch tests (3 controls x mAP@R, PMRP, rsum, CxC R@1, 1K R@1): the six PMRP and rsum tests adjust to 0.0015, 0.0071, 0.0030, 0.0266, 0.0266, 0.0266 (report "0.002 to 0.027, all below 0.05"); the mAP@R, CxC and 1K tests adjust to 0.10 to 0.13, which the report does not claim as significant.
- Compute: s2_txt80 7.17 h / s2_contrastive 4.21 h = 1.703 (report 1.70); masked arms and controls 6.87 to 7.58 h.

Finding by finding:
1. (major, `fusion_none` attribution) Fixed. Summary item 4, 4.1, 4.3, the section 6 table and a new caveat now describe `fusion_none` as removing the fusion altogether (no learners, MLM reads text alone, MAE reads image alone), say it cannot separate the image from the learners, and rest the "masked-pass image tokens in the MLM's memory" claim on the three controls jointly. Checked against `mmae/models/fusion.py` (`NoFusion.forward` returns the raw projected tokens) and `mmae/models/model.py` lines 175 to 177 (M1 builds a second fusion over the clean tokens, detached for `clean_detached`; the MAE decoder still reads the first, masked-pass fusion): every statement is accurate, including "the MLM gradient still reaches the vision tower" for M1 clean and "no MLM gradient into the vision tower" for M1 detached. The 4.1 sentence on the 0.61-nat gap now says it mixes the missing image with the missing learners.
2. (Figure 2 caption, 15% loss) Fixed. `load()` now merges `results.test` with `eval.test` (eval wins on shared keys), the caption's "2 of 3 runs" sentence is gone, Table 3 says 1.568, and the redrawn dose.png shows three seed dots at 15% in the MLM panel.
3. (reservations) Fixed: "when the user ended the node reservations", matching the run log's 01:47 correction.
4. (unpaired 5-seed test, Holm family) Fixed in 3.3 and the Seeds caveat; numbers verified above.
5. (power caveat in the Summary) Fixed: Summary item 5, 5.3 and the section 6 table carry the half-widths and "modest growth is not excluded"; 5.3 quotes the t2i interval.
6. ("corrected two entries") Fixed: "corrected one entry", with the 00:14 and 09:14 entries cited correctly.
7. (uncorrected secondary tests) Fixed in 1.3 and 4.3, with one residual: 1.3 now says "Only the primary test of the success bar carries a multiple-test correction", but 4.3 also applies Holm over the three controls and over the 15 control tests. Suggest "Apart from the success bar (Holm, as the spec prescribes) and the control comparisons of section 4.3, we report the secondary tests uncorrected". Not blocking.
8. (MAE off's MAE loss) Fixed: parenthesised in Table 7 with a caption note.
9. (zero-shot VWSD 57.88 vs 58.10) Fixed in Table 10's caption, citing the Stage 0 report.
10. (definitions) Fixed: ECCV Caption (Chun et al., ECCV 2022), Welch t-test and the deep-think note are defined in Terms. Optional precision: ECCV Caption's queries are a subset (1,261 images and 1,332 captions, as the Table 11 counts show), which the Terms entry does not say.
11. (Figure 1 label overlap) Fixed: the redrawn stage1_arms.png puts "advance +0.3" and "guard -3" in an empty top row; nothing overlaps.
12. (third reading of M1 clean) Fixed: 4.3 lists the partial-input-forward reading and says an 80% MAE-off run would not separate the three readings.
13. (compute caveat) Fixed: matched among masked arms and controls (6.87 to 7.58 h), unmatched only against contrastive.

Also confirmed: `reports_sum.md` row updated to the new mechanism wording and consistent with the report; `scripts/check_reports_sum.py` prints OK; no em or en dashes and no hyphen used as a dash in the report or the row; all times remain plain Amsterdam local time.
