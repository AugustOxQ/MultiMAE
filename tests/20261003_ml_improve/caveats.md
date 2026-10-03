# Caveats and deferred items from the implementation of the multilearner-line switches (2026-10-03)

Source: the SDD ledger of docs/superpowers/plans/2026-10-03-improve-multilearner.md (task reviews and the final whole-branch review). Carry the caveats into the Stage 0 to 2 reports.

## Caveats for the write-ups
- PMRP positives (released ECCV Caption PM lists) are items whose image's COCO class vector is within two classes of the query image's (zeta <= 2; verified against instances_val2014.json on all 4,952^2 pairs). Class-set groups from identical PM neighbourhoods match identical class sets except 13 of 44,052 same-group pairs (2,160 groups vs 2,173 class sets; 48 test images have no PM entry).
- R2 freeze is lr = 0: vision gradients still enter the global clip norm (grad_clip 1.0), so text/head steps in the frozen epochs are smaller than with requires_grad=False when the total norm exceeds 1; vision Adam moments accumulate; no re-warmup at unfreeze (vision lr jumps to about 0.91 x 5e-6).
- R2 runs launched at 1b23184 log a mislabelled train/lr_backbone (lowest text layer group); per-tower lr logging (train/lr_text, train/lr_vision) exists from f7f1cee on.
- M3: the pooled token's norm at init is about a third of the memory tokens' (0.62/0.55 vs 1.69/1.79 on real B/32); read an M3 null with that in mind. M3 with mlm_image_source=clean_detached would still send MLM gradient into the vision tower through the pooled image token.
- M1 runs fusion twice and consumes extra dropout RNG, so its masks drift from the default arms' (Stage 2 pairs by data order only).
- E0e reports similarity means only (distributions can be computed from the saved embeddings). E0f probe: baseline is the normalised full caption from f7f1cee on, with full/t2i_R1 reported.
- Diagnostics job: check its log lists exactly 13 runs (12 baselines + zero-shot); cluster pull of a diag tag rsyncs all of node403's results root.

## Deferred minor items (triaged 'can wait' by the final reviewer)
- Task 1: minor (deferred): new pool test is near-tautological (plan-mandated); HF-pinned backbone tests carry the regression protection
- Task 1: minor (deferred): redundant function-local imports in the new test (plan-mandated form)
- Task 1: minor (deferred): commit trailer names the implementer's model (Haiku 4.5) instead of the reminder's Opus 5.5
- Task 2: minor (deferred): detach test's [clean-True] half passes without M1 too; only [clean_detached-False] discriminates
- Task 2: minor (deferred): old-config test checks only the attribute, no forward/state-dict load
- Task 2: minor (deferred): under M1 the first fusion call's text memory is computed and discarded (extra compute)
- Task 2: minor (deferred): change() hard-codes 224 px and the CLIP vocab range
- Task 2: minor (deferred, info): under M1 recorder's "fusion" capture is the second (text_fused) call; M1 consumes extra dropout RNG so masks drift vs default arms (matters only for paired-mask designs)
- Task 3: minor (deferred): unused tmp_path arg in test_collator_marks_content_tokens (plan-mandated)
- Task 3: minor (deferred): content_token_table loops the 49k vocab in Python per Collator; alphabetic word pieces without </w> count as content
- Task 3: minor (deferred): evaluate.py's Collator has no content_words (harmless: retrieval never runs the masked forward)
- Task 4: minor (deferred, design): M3 + mlm_image_source=clean_detached still sends MLM gradient into the vision tower via pooled_image_proj(image_emb); document or detach if ever combined
- Task 4: minor (deferred, design): M3 token is Linear(unit-norm embedding) -> small norm at init vs fusion memory; may weaken M3 early (interpretation caveat for an M3 null)
- Task 4: minor (deferred): no test pins "no pooled_* modules when off"; no repo leak test for M3 x concat or M3 x M1 (reviewer's scratch test passed both)
- Task 4: minor (deferred): M3 validation runs after build_backbone (plan-mandated placement)
- Task 4: minor (deferred): change log lacks entries for the two edited test files; model.py:166 is 127 chars
- Task 5: minor (deferred): weight-0 test only checks the key is absent (code path makes the total safe)
- Task 6: minor (deferred, fix in final wave): under split groups train/lr_backbone logs the first backbone group (vision layer 0: 0 while frozen, most-decayed after); log per-tower top-layer lr instead
- Task 6: minor (deferred): AcceleratorState reset in the frozen test not in try/finally; no blank line before MODALITIES; no check that lambdas stay aligned with param_groups after prepare
- Task 7: minor (deferred): test reads epoch 1 only; multi-epoch/multi-rank equality reasoned (one persistent generator advanced only by the sampler), not tested
- Task 8: minor (deferred): tiny-model VWSD test checks bounds only; no tie test; blank line mid-file would shift data/gold pairing; no per-item candidate-count check (real data clean: 463 x 10, all golds present)
- Task 9: minor (deferred): similarity_stats test cannot catch swapped same/other masks (all 0.0); no k < candidates check in neighbour_purity; docstrings do not state L2-normalised inputs; drop_words treats any kind != content as stop and skips words with attached punctuation other than "."
- Task 9: minor (deferred): "persons" not recognised; possessives/hyphens ("hot-dog") miscounted; "bat"/"ball" always baseball bat / sports ball
- Task 10: minor (deferred): diagnostics.json written only at the end (write once after the encode loop too); no PMRP cross-check print vs run.json (plan-mandated omission; reviewer showed per-query mean == package PMRP exactly on real PM data); per_query.pt lacks query ids; probe lacks unshortened R@1 baseline; duplicate-seed contrastive silently wins in swaps; bare KeyError without +diag keys; run list not logged; smoke test does not reach PM/extended branches; tuple-expression style nit
- Task 10: minor (deferred, fix in final wave): UserWarning from float(model.logit_scale.exp()) on a requires_grad tensor (use .detach() / .item())
- Task 11: minor (deferred): zero-shot VWSD numbers in docs have no cited result file (source: scratchpad diag_zs/out/diagnostics.json and Task 8 report)
