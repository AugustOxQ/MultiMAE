# Final review: masked-decoder readings review (H-b)

Reviewer: final whole-report review, 2026-10-06 21:05. Report: `docs/reports/literature/2026-10-06_masked_decoder_readings_review.md` (figures `hb_chain.png`, `design_map.png`; index row in `docs/reports/reports_sum.md`). Raw paper copies read with grep and `python3 -I` only (scratchpad `hb_review/dl_*`); a helper `fr_scripts/ctx.py` printed text windows. Local data re-derived with `/root/miniconda3/envs/MultiMAE/bin/python -I`.

**Verdict: APPROVED WITH FIXES.** Four MAJOR findings, all text edits with no new analysis. Two come from verifier errors that the report carried forward (V3 on Panagopoulou et al. and UniDisc). The other two are summary-level claims stated more broadly than the body's own evidence allows. The local data facts, project numbers, code facts, verifier recount (203 rows), sMCL correction, Cekinmez locator correction and GPU-hour arithmetic all reproduce.

## MAJOR

### M1. The Panagopoulou "correction" is wrong: the paper does say CLIP showed almost opposite preferences, and that CLIP was sensitive to brightness and tint

- **Where:** §2.3 bullet "Panagopoulou et al. (V3 problem 1): the statement that CLIP showed 'almost opposite preferences' to generative models was not found." §3.3: "N's statement that CLIP showed 'almost opposite preferences' to the generative models was not in the paper: that phrase compared BLIP-2 OPT with its Flan-T5 variants. Brightness and tint had minimal effects on every model, and CLIP varied most under rotation and prompt changes." Also §3.3 "scored CLIP by similarity and 11 generative VLMs by label likelihood", Table 5 row S3 ("the CLIP 'opposite preferences' claim was not found"), and Table 6 row 8 ("Against A").
- **Evidence:** `hb_review/dl_V2/nh/2405.19423.txt` (the copy V3 itself saved; `dl_N/` is identical). It contains **two** statements:
  - Sec 4.1 (char 16052): "The images with the highest variation across models where the 'Woman-Trumpeter', 'Schroeder Stairs', and 'Raven-Bear' with **CLIP variants showing almost consistently opposite preferences to the LLM based generative models**."
  - Sec 4.1 (char 17579): "although BLIP2 OPT was trained on the same image-text data as the Flan T5 variants, it exhibited almost opposite preferences in some image categories." V3 found only this one.
  - Sec 4.2: "We observed minimal effects from image manipulations... **The primary exception was the CLIP variants, which demonstrated sensitivity to variations in brightness and tint**, particularly in the 'Young-Old Woman,' 'Cat-Dog,' 'Grimace-Begger,' and 'Woman-Trumpeter' illusions." Rotation: "contrastive based CLIP-variants once again exhibit the most variation".
  - Sec 3.2 model list: three CLIP variants (clip-vit-base-patch32, -patch16, laion CLIP-ViT-B-32) plus nine generative models (Idefics ×2, LLaVA-1.5 ×2, mPLUG-Owl, InstructBLIP, BLIP-2 ×3), 12 in total. So the count is **9** generative VLMs, not 11. V3 row 18 has the same error.
  - V3 rows 19 (NOT FOUND) and 20 (CORRECTED) are therefore wrong, and so is V3 problem 1. N's original statement was right, scoped to three images.
- **Fix:**
  - §2.3: delete the Panagopoulou bullet from "Corrections that changed conclusions", or replace it with: "Panagopoulou et al.: V3 reported N's 'almost opposite preferences' statement as not found. It is in Sec 4.1, scoped to the three images with the highest cross-model variation (Woman-Trumpeter, Schroeder Stairs, Raven-Bear). V3 found only a second, BLIP-2 OPT against Flan-T5 statement. V3 rows 19 and 20 are overruled here."
  - §3.3, first paragraph, rewrite as: "Panagopoulou et al. scored three CLIP variants by similarity and nine generative VLMs by label likelihood. Most models of both kinds preferred one reading. On the three images with the highest cross-model variation (Woman-Trumpeter, Schroeder Stairs, Raven-Bear), the CLIP variants showed 'almost consistently opposite preferences to the LLM based generative models' (Sec 4.1). Brightness and tint had minimal effects except on CLIP, the 'primary exception' (Sec 4.2), and CLIP also varied most under rotation. The comparison is qualitative, on 29 images, with no human-distribution metric." Change the paragraph's lead sentence "weaker than first reported" accordingly.
  - Table 5, S3 "What verification changed": "none (V3's not-found was itself wrong; N's statement holds for three images)".
  - Table 6 row 8: Claim B → "V3: the phrase concerned BLIP-2 variants (overruled: Sec 4.1 has both statements; Sec 4.2 names CLIP as the brightness and tint exception)". Resolution → "For A, scoped to three images; most models of both kinds still prefer one reading."
  - Update the supporting-file header line of `assets/.../V3_verification_N.md` to say rows 18, 19, 20 and 24 are overruled by the report (see M3 for row 24).
  - S3's evidence strength ("weak") and novelty risk ("low") can stay. The point is that a qualitative contrastive-vs-generative difference on ambiguous images already exists in the closest work.

### M2. Summary item 5 restates the verifier's narrowed absence claim broadly, and the body contradicts it

- **Where:** Summary item 5: "Open: no source tests a masked cross-modal decoder's distribution against human readings, against a dual encoder, or at retrieval time."
- **Evidence:**
  - V3 §4 claims that no work tests **the whole chain**, and the retrieval clause is "used at retrieval time **to cover several readings**". It is bounded by N's Q1 to Q40 and three extra searches.
  - The report's own body cites counterexamples to the split, unqualified wording. UniDisc is "a masked joint image-text diffusion model" that "retrieved by likelihood among 16 random candidates" (§3.4; `dl_V2/nh/2503.20853.txt` Fig 7). Masked Diffusion Captioning's likelihood beat CLIP on seven compositional columns (§3.3; V2 G4 row 23). Table 12 row 7 also says "UniDisc and MDC show masked likelihood scoring".
  - So a masked cross-modal model's distribution has been used at retrieval time and compared with a dual encoder, just not on readings or polysemy.
- **Fix:** "Open (in N's 44 queries and V3's three extra searches): no source compares a masked cross-modal decoder's distribution with human readings, compares it with a dual encoder **on ambiguous or multi-reading items**, or uses it at retrieval **to cover several readings**. Masked likelihood has been used for retrieval and compositional matching (UniDisc, MDC), not for readings." Keep the second sentence about blends, and add "found" ("no found work trains with pixel blends...").

### M3. "UniDisc beat an autoregressive model only in joint retrieval" holds on DataComp only

- **Where:** §2.3 ("its advantage over an autoregressive model is in joint retrieval only"); §3.4 ("It beat an autoregressive model only in joint retrieval (0.64 against 0.17), tied in single-direction retrieval (0.85 against 0.85; 0.95 against 0.96)"); Table 5 row S4 ("UniDisc wins only joint retrieval"); Table 6 row 11.
- **Evidence:** `dl_V2/nh/2503.20853.txt`, Fig 7 table (char 28622), AR against UniDisc:

  | Retrieval | CLEVR-Ret | DataComp | Winoground |
  |---|---|---|---|
  | Text retrieval | 0.81 against 0.94 | 0.85 against 0.85 | 0.24 against 0.31 |
  | Image retrieval | 0.06 against 0.25 | 0.96 against 0.95 | 0.25 against 0.27 |
  | Joint retrieval | 0.06 against 0.5 | 0.17 against 0.64 | 0.06 against 0.20 |

  V3 row 24 read only the DataComp column.
- **Fix:**
  - §3.4: "On DataComp it beat the autoregressive baseline only in joint retrieval (0.64 against 0.17) and tied in single-direction retrieval (0.85 against 0.85; 0.95 against 0.96). On CLEVR-Ret it also led in text and image retrieval (0.94 against 0.81; 0.25 against 0.06), and on Winoground in text retrieval (0.31 against 0.24). There was no contrastive baseline (Fig 7)."
  - §2.3 and Table 6 row 11: "its clear advantage on DataComp is joint retrieval; single-direction results tie there but favour UniDisc on CLEVR-Ret and Winoground".
  - Table 5: "MASS and DiffusionITM are re-rankers; UniDisc ties AR in single-direction retrieval on DataComp".

### M4. "Every comparison we found had models more peaked than people" is contradicted by a result the report itself cites

- **Where:** Summary item 2 ("Established: ... every comparison we found had models more peaked than people"); §3.2 ("The direction was negative. Every comparison found models too peaked, or no better than chance on divergence"); the index row ("models are more peaked than human votes"). §3.7 and Table 12 row 9 rely on it as the prior direction.
- **Evidence:**
  - `dl_V2/nh/2608.00410.txt`, Sec 4.1 text and Table 2 (0.730 / 0.103 / 0.473): "Models consistently predicted substantially more diverse sense distributions (Hnorm≈0.73) than those observed in human judgments (Hnorm≈0.47)". Appendix F: models put 0.55 of mass on the dominant sense against 0.72 for humans.
  - The report quotes this in §3.1 ("models estimated about 0.73"). A verbalized estimate of the human distribution is a model-against-people comparison, and it went the other way.
  - The body also grades this "moderate", not "Established".
  - Nie et al. measure divergence, not peakedness. The "too peaked" direction rests on Lan et al. (temperature 2.0 helps), Cekinmez's generated outputs, Wang and Chan, and VEGAS.
- **Fix:**
  - Summary item 2: "... and the direction of the answer for sampled outputs and model probabilities: every such comparison we found had models more peaked than people. The exception is a verbalized estimate: asked to predict the human distribution, LLMs gave flatter ones than people (0.73 against 0.47; Cekinmez et al. 2026)."
  - Replace "Established" for the direction with "moderate", matching §3.2's evidence line.
  - Apply the same qualifier in §3.2's "The direction was negative" paragraph and in the index row.

## MINOR

1. **ChaosNLI αNLI is described wrongly for JSD.** §3.2 says "on αNLI models were at or worse than chance (Table 5)"; Table 6 row 6 says "αNLI and MNLI are at or worse than chance". `dl_V1/2010.03532.txt` Table 5, αNLI:
   - JSD: chance 0.3205. Six of nine models clearly beat it: RoBERTa-l 0.2128, ALBERT 0.2208, BART 0.2215, XLNet-l 0.2282, RoBERTa-b 0.2394, XLNet-b 0.2678. BERT-b (0.3209), BERT-l (0.3055) and DistilBert (0.3101) sit near chance.
   - KL: chance 0.406. Every model is worse, from 0.8272 to 3.7996.

   This comes from V1 problem 4. Under the strict definition it is a half-wrong conclusion, but it does not move the S2 verdict.
   - Fix for §3.2: "on αNLI every model's KL was 2 to 9 times chance (0.83 to 3.80 against 0.406) while six of nine beat chance on JSD; on MNLI both measures were at or worse than chance (JSD 0.3055 to 0.3165 against 0.3023)."
   - Same fix for Table 6 row 6.
2. **MixGen variant (e) changed two things.** Summary item 4 says "keeping a random half of its tokens kept almost all of the gain (COCO RSUM 477.1 against 477.7)"; §3.6 and Table 6 row 1 say the same. `dl_V2/2206.08358/mixgen.tex` lines 243 and 550: variant (e) keeps half the concatenated tokens **and** draws λ ~ Beta(0.1, 0.1), not λ = 0.5. So (e) differs from MixGen in both the text and the λ distribution. Against variant (a) (Beta λ, full concatenation, 467.4), halving the tokens gained 9.7. Separately, "Differences of about 5 RSUM are within what one-run cells show" is an inference: the table has no repeated runs.
   - Fix: "variant (e), which kept a random half of the concatenated tokens and also drew λ from Beta(0.1, 0.1), kept almost all of the gain (477.1 against 477.7; baseline 470.7)". In the Summary, add the 470.7 baseline.
   - Mark the noise sentence [reader-inferred]: "with one run per cell the table gives no noise estimate; variants expected not to help span 463.7 to 472.3 around a 470.7 baseline".
   - The conclusion ("cannot show the concatenated caption matters") stands, and if anything gets stronger.
3. **Summary item 1 says "Established"** where Table 5 says "partly established" and §3.1's evidence line says "moderate".
   - The slot-distribution evidence comes from Takatsuki et al. alone. Agate's evidence is SAE features in CLIP plus LLaVA's sampled captions.
   - Fix: "Partly established (moderate), in other models: ..."
4. **Agate's 0.558 is LLaVA's dominance, not CLIP's.** `dl_V2/nh/2606.08031.txt` Sec 4.1: 3,320 captions from LLaVA-1.6-Vicuna-7B, 40 per stimulus over 83 stimuli. Ten stimuli had P(neither) ≥ 0.95 under neutral prompts, so those captions committed to neither aspect. §3.1 runs from "CLIP ViT-L layer-22" straight to "Sampled captions committed" without naming the captioner.
   - Fix: "LLaVA-1.6's sampled captions mostly committed (mean dominance |P(A) − P(B)| = 0.558 over 83 stimuli; on 10 they named neither aspect)."
5. **The ArtELingo distinct-emotion counts use a different basis from the retrieval split.** §1.5 gives "86 have one distinct emotion, 687 two, ..." with no basis, and §3.2 says "at least 3.46 ... counted over up to 12 captions, so the comparison is approximate". `artelingo_test_retrieval.json` holds exactly 5 captions per painting, while the counts come from all of a painting's `artelingo_test.json` captions (5 to 12). I matched the five retrieval captions to their test-file emotions (all 24,875 matched, none ambiguous):
   - distinct emotions: 96 / 742 / 1,845 / 1,828 / 464 (1, 2, 3, 4, 5)
   - mean 3.37
   - 98.1% with at least two

   Fix:
   - §1.5: add "(counted over all of a painting's test captions, 5 to 12)".
   - §3.2: replace the approximate sentence with "Over the five retrieval captions the mean is 3.37 distinct emotions, between the 3.2 of a 50%-dominant painting and the 4.0 of a uniform one."
   - D7's 98% holds on both bases.
6. **Absences stated without their query bound** (the caller's check). In §3.4, "No ECCV mAP@R, CxC or other multi-positive metric exists for a generative likelihood against a contrastive score from the same model" was checked with only two V1 queries (V1 §4 G1-1, "HOLDS (bounded by my queries)"). The same goes for "No retrieval method lets a reconstruction or generative loss choose which embedding slot..." (V2 §4: three queries plus G4's) and, in §3.2, "No paper compares a captioner's sampled meaning clusters..." (V1 §4 G2-1: three queries). The report title is also unbounded, though it is qualified by "as a model of several readings".
   - Fix: "We found no ... (V1's two queries)", "(V2's three queries and G4's Q4, Q14 to Q16)", "(V1's three queries)". In the title, "No searched source" or "No source we found".
7. **"Only two readouts can show it by construction"** (Summary item 6; §6 D6: "where masking can beat a captioner by construction"). By D7's own text, an autoregressive captioner handles prefix selection through Bayes' rule; only random-subset selection is masking-only. D6's view averaging can also be given to a captioner (Par-cap on unseen views is a listed control).
   - Fix: "Two readouts favour masking by construction: averaging over masked image views, which a masked model trains on, and readings selected by an arbitrary subset of caption tokens. Prefix selection is also open to an autoregressive captioner."
8. **The phase total uses 3 GPU-hours for pilots** while Table 11 gives 2 to 4. That makes the range 125.5 to 156 rather than 126.5 to 155. It is copied from synthesis §7, line 611.
   - Fix: write "2 to 4 (pilots)" and "≈ 126 to 156", or keep 3 and say "(midpoint)".
   - Phase 4's "6 to 10" exceeds D6 + D7 + D8 from Table 10 (about 3 to 7) because it runs on every new checkpoint. Say so in one clause.
9. **M3ixup is described as mixing images.** §2.3 says "M3ixup and an IBM patent mix images and captions". M3ixup mixes visual features and sentence embeddings, not pixels (V2 §4: "It mixes features (not pixels)").
   - Fix: "M3ixup mixes visual features and sentence embeddings, and an IBM patent interpolates images and captions, for captioning".

## NIT

1. **The blind-decoder row is Cap's, not CapPa's.** §3.3 says "CapPa's blind-decoder row reached 83.7 on VG-Attribution and 86.2 on VG-Relation". The CapPa paper's Table 17 note: "'Blind dec.' corresponds to Cap without vision encoder" (`dl_V1/2306.07915.txt`). Fix: "the CapPa paper's blind autoregressive decoder (Cap with zeroed image features)".
2. **Table 4, V2 existence: 32 verified plus 1 plausible covers 33 of the 34 sources.** V2's own table lists 18 G4 sources but outcomes for 17. Fix: add "1 counted without an outcome in V2". Also, V3's metadata table marks UniDisc VERIFIED and AmbiBench PLAUSIBLE or UNVERIFIABLE, while its summary says 14/1/1; name which source is the "1 plausible" in V3's row.
3. **Terms used without definition:** RSUM (Summary item 4: the sum of the six COCO R@1/5/10 recalls), ARO, MRR, AUROC (the paper calls it AUC) and CIDEr-D. The user's rule asks for a definition at first use; these could go in a few words in the Terms paragraph.
4. **V2 row citations are ambiguous.** V2 numbers its G4 and G5 tables separately, so "V2 rows 13, 20, 21, 26" (§3.1) and "V2 rows 4 to 26, 36" (§3.6) need a G4 or G5 prefix.
5. **An optional supporting fact for Table 6 row 2 (DiffusionITM).** In Table 1b (text retrieval), the naive Diffusion Classifier scored 69.8 on Flickr against 65.8 for CLIP RN50x64 (`dl_V1/2305.16397.txt`). Ranking the conditioning modality needed no correction there, which supports the report's direction rule.
6. **Takatsuki's "37 to 44 of 50" locator:** it is Sec 3.2.1 with Appendix A.2.1 rather than "Sec 3.2". Harmless.

## Re-derived and confirmed

**Local data** (`al.py`, `al28.py`, `al5.py` in `scratchpad/fr_scripts/`):
- ArtELingo train: 308,723 captions on 61,402 paintings. Val: 15,531 on 3,119.
- Retrieval split: 4,975 paintings, exactly 5 captions each.
- Distinct emotions over the full test captions: 86 / 687 / 1,718 / 1,840 / 644, mean 3.474 (3.456 with 5+ capped).
- Train captions: median 14 words. 9 emotions, contentment dominant.
- AL-28: 173,745 rows on 1,658 paintings, median 108 votes (mean 104.8, range 2 to 131), 10 labels including "other", 28 languages. English: 6,515 rows on 1,303 paintings.
- AL-28 overlap with ArtELingo: 1,160 train, 59 val, 103 test, 336 none. 1,160 / 61,402 = 1.9%. 439 = 103 + 336.
- Sampling arithmetic: 9·(1 − (8/9)^5) = 4.01. The 50%-dominant case gives 3.17.

**Project facts:**
- From `2026-10-06_masking_stages_1_2.md`:
  - mAP@R: C 37.11 (3 seeds), ML-15 37.24 (3), ML-80 37.94 (5).
  - VWSD 52.92, +0.36 (p 0.760).
  - Gain over ML-15 by tertile of valid matches: i2t +0.85 falling to +0.19.
  - MLM loss: 1.569 / 3.053 / 2.898 (M1 clean). Perplexity 4.8 / 21.2 (exp checks).
  - A masked run took 6.87 to 7.58 h and a contrastive run 4.21 h. The 4 h and 2.3 h ArtELingo estimates match a 0.545 data ratio.
  - Training: 10 epochs, batch 128. ECCV subset: 1,261 images, 1,332 captions.
- From the code:
  - `random_patch_mask` keeps 49 − int(49·0.75) = 13 patches.
  - `random_token_mask` rejects ratio 1.0.
  - `QueryDecoder` uses learned per-position queries plus position embeddings, with `query_padding`. The MLM is called with `query_padding=text_padding`, so at 100% masking the logits depend only on (image view, length).
  - `max_text_len` = 32.

**Verifier recount:**
- V1: 91 rows = 68 confirmed + 14 corrected + 1 locator wrong + 3 not found + 3 not re-read + 1 plausible + 1 added.
- V2: G4 30/8/3 and G5 24/5/9, so 54 confirmed, 13 corrected, 12 not re-checked out of 79.
- V3: 26 confirmed, 6 corrected, 1 not found, out of 33.
- Total 203. The mismatches the report notes between V2's and V3's summaries and their rows are real.

**Corrections the report made itself:**
- sMCL (`dl_V2/1606.07839/captioning.tex`, `tab:caption`): sMCL 0.822 / 0.862 / 0.911 / 0.922 against beam search 0.754 / 0.833 / 0.888 / 0.943 for M = 2 to 5. sMCL leads up to M = 4. The paper's own text says "For M=5, beam search ... achieves greater oracle"; fine-tuned sMCL reaches 1.184.
- Cekinmez (`dl_V2/nh/2608.00410.txt`): 0.10 and 0.25 in Sec 4.1 and Fig 2; 0.47 in Sec 4.1 and Fig 2; Table 1 is the cross-lingual table; Table 2 holds 0.730, 0.103 and 0.473. The "image models" are text-to-image generators, so "image generators" in the Summary is right.

**Raw-text numbers confirmed:**
- Takatsuki: 37 to 44 of 50; logistic psychometric fit; cues "leave the cued interpretation essentially absent" on controls (Sec 3.2.2); about 10% against at least 98% (LLaVA-1.5-7B); "1" answered at 0.88 to 0.99; Raven-Bear 43 to 51% in three of five models (App. B.4.1).
- Agate: ViT-L/14-336 layer 22; 50 of 69; 12/12, 7/16, 1/4; 0.558.
- Panagopoulou: 29 × 116 = 3,364; most models comment on a single interpretation (Sec 4.7).
- VSE A-OKVQA gains over the strongest baseline: +0.009, +0.046, +0.073, +0.026, +0.011.
- Lin et al.:
  - Table 1 column order is Rel, Attr, COCO, Flickr: 89.1 against 63.1 and 58.7; blind BLIP 87.6 and 80.7; P_LLM 95.0 to 98.6.
  - Table 3a: 71.9 / 19.7 / 46.2 / 48.0 with α* 0.819. Table 3b: 54.8 against 55.6.
- CapPa blind-decoder row: Attr 83.7, Rel 86.2.
- ChaosNLI SNLI: JSD 0.2203 to 0.2439 against 0.383; KL 0.4682 to 0.5342 against 0.5457. MNLI KL: 0.6652 to 0.8845.
- Lan et al.: Table 1 bins, low = low human uncertainty (KL 2.150 / 1.544 / 1.408).
- MultiEmo human split-half 0.700. ArtECulture filtering sentence and κ 0.088. ArtEmis 45.6%.
- DiffusionITM: Table 1a image retrieval 71.9 / 11.1 / 61.5; 68 min against about 4 min.
- DDCap: 117.8 against an "AR Baseline" of 112.8.
- MixGen Table 7 cells: (a) 467.4, (b) 470.0 (one random caption, λ = 0.5), (c) 472.3, (d) 463.7, (e) 477.1; MixGen 477.7; baseline 470.7.
- PCME++ `tab:loss_abl`: 38.9 to 39.0 (RSUM 535.9 to 535.5); 39.6 to 40.1 (537.0).
- IC3: MRR 0.501 to 0.734 (Table 5); noun recall 0.266 to 0.353 (Table 7, exact); r = 0.059 to 0.094 (App. C.5); Table 6's R@1 and R@5 columns identical.
- Pezzelle: about 16 and about 40 points; 82 of 100 against 10 random captions.
- Yao et al.: 3.61 (Llama) and 4.44 (GPT) clusters per image.

**Searches and other checks:**
- Search counts in Table 3 match G1 to G5 and N (queries 20, 19, 15, 16, 14, 44).
- N's query bounds for S1 (Q1, Q11, Q17, Q29, Q33, Q37) and S5 (Q14, Q22, Q34, Q35) match.

**Design arithmetic and facts:**
- 3 × 2.3 + 6 × 4 = 30.9. D5: 36. ML-50: 12. Mirrors: 2 × 3 × 7 = 42.
- Phase total: 8 + 3 + 31 + 36 + 13.5 + 7 + 16 + 12 = 126.5, and 8 + 3 + 31 + 36 + 27 + 11 + 27 + 12 = 155.
- 61.5 h × 6 = 369. About 412 free after the queue, matching "about 400".
- Nodes 405 and 411 are in `running.txt`.
- Section 7 matches synthesis §7, lines 571 to 611.
- D-designs use only true data and code facts. Par-cap at MAE weight 0 is feasible; the Stages report's MAE-off arm ran at weight 0.

**Figures and index:**
- Figure 1 cards and Figure 2 markers, sub-claim dots and cost ranges match Table 10 and §3. Figure 1's S3 card stays true after M1.
- `check_reports_sum.py` prints OK. The index row is consistent with the report except the M4 wording.

**Presentation:**
- Section 7 is labelled a proposal, not a decision, in its heading, in Summary item 7 and in the index row.
- The report uses no em or en dashes, and no spaced hyphens as punctuation.

## Re-review of the fix wave

Scoped re-review, 2026-10-06 (after 21:13, when Figure 1 was rebuilt). Only the passages the author listed as changed were re-checked, against the findings above and the raw text.

**Verdict: APPROVED.** Every MAJOR and MINOR finding, and NITs 1, 2, 3, 4 and 6, are applied correctly. The numbers in the changed passages reproduce, and `scripts/check_reports_sum.py` prints "reports_sum.md: OK". One leftover inconsistency remains (R1 below). It is a one-line edit that changes no conclusion and does not block approval.

### Checked and correct

**M1. Panagopoulou et al.**
- §3.3 paragraph: three CLIP variants and nine generative VLMs. The Sec 4.1 quote is verbatim. CLIP is the "primary exception" for brightness and tint (Sec 4.2), and varied most under rotation. The paragraph says the comparison is qualitative.
- §2.3 "Made here" bullet, Table 5 row S3, Table 6 row 8 and §9 are consistent with this.
- Figure 1's S3 card reads "CLIP opposed the generative models on 3 of 29 images". The build script (lines 78 and 79) carries the same text.
- The V3 copy has a new header line marking problem 1, rows 18 to 20 and row 24 as superseded. The synthesis copy has the matching header line.

**M2. Summary item 5 and the index row.**
- The absence is now bounded: "within N's search queries (Q1 to Q40) and V3's three extra searches". N line 450 states the whole-chain absence over Q1 to Q40.
- It is scoped to "on ambiguous or multi-reading items" and "to cover several readings". UniDisc and MDC are named as masked-likelihood uses that did not measure readings.
- The title now reads "No source we found".

**M3. UniDisc.**
- §3.4 now gives DataComp 0.64 against 0.17 (joint), 0.85 against 0.85 and 0.95 against 0.96 (single direction).
- It adds CLEVR-Ret 0.94 against 0.81 and 0.25 against 0.06, and Winoground 0.31 against 0.24. All match the Fig 7 table in `dl_V2/nh/2503.20853.txt`.
- §2.3, Table 5 row S4 and Table 6 row 11 are consistent.

**M4. Direction of the S2 comparison.**
- Summary item 2, §3.2, §3.2's evidence line, §3.7, Table 12 row 9, the index row and Figure 1's S2 card are all scoped to sampled outputs and model probabilities, graded moderate.
- The verbalized exception is given as 0.73 against 0.47 (Table 2) and 0.55 against 0.72 of the mass on the dominant sense. The 0.55 and 0.72 appear in both Sec 4.1 and App. F of `2608.00410.txt`.

**MINOR 1. ChaosNLI per measure.**
- αNLI KL: 0.8272 to 3.7996 against 0.406, which is 2.04 to 9.36 times chance.
- αNLI JSD: six of nine models beat chance.
- MNLI: JSD 0.3055 to 0.3165 against 0.3023, and KL 0.6652 to 0.8845 against 0.3559.
- These match Table 5 in `dl_V1/2010.03532.txt`. §2.3 and Table 6 row 6 are consistent.

**MINOR 2. MixGen ablation.**
- §3.6, Summary item 4 and Table 6 row 1 now say variant (e) also drew λ from Beta(0.1, 0.1) (`mixgen.tex` line 550).
- Variant (a) is given as 467.4, and the 470.7 baseline is added.
- "The four other variants span 463.7 to 472.3" checks out: (a) 467.4, (b) 470.0, (c) 472.3, (d) 463.7.
- The noise sentence is marked [reader-inferred].

**MINOR 3 and 4. Summary item 1 and Agate.**
- Summary item 1 now says "Partly established (moderate evidence)".
- The 0.558 is now attributed to LLaVA-1.6's sampled captions over 83 stimuli. "On 10 they named neither aspect" matches the paper's "P(neither) ≥ 0.95" and "100% abstention ... on ten stimuli".

**MINOR 5. ArtELingo counts.**
- §1.5 now states its basis: all of a painting's test captions, 5 to 12.
- §3.2 now uses 3.37 over the five retrieval captions. My join of `artelingo_test_retrieval.json` to `artelingo_test.json` emotions (24,875 of 24,875 matched, none ambiguous) gives 96 / 742 / 1,845 / 1,828 / 464, mean 3.366, with 98.07% having at least two.
- D7's "98%" holds on both bases.

**MINOR 6. Bounded absences.** §3.2 now cites V1's three queries. §3.4 cites V1's two queries for G1-1, and V2's three queries plus G4's Q4 and Q14 to Q16 for the slot claim.

**MINOR 7. Masking-favoured readouts.** Summary item 6 and the D6 wording are fixed.

**MINOR 8. Costs.**
- Table 11 Phase 4 explains why it exceeds Table 10's D6 to D8 sum.
- The total states that 3 is the midpoint of 2 to 4. The arithmetic is unchanged: 126.5 to 155.

**MINOR 9.** M3ixup is now described as mixing "visual features and sentence embeddings".

**NITs.**
- NIT 1: §3.3 now attributes the blind-decoder row to Cap with zeroed image features.
- NIT 2: Table 4's V2 row now notes "1 without an outcome", and the V3 plausible discrepancy is explained in prose.
- NIT 3: RSUM, MRR, AUROC, CIDEr-D and ARO are defined in Terms. The definitions are accurate.
- NIT 4: V2 row citations now carry G4 or G5 prefixes.
- NIT 6: Takatsuki's locator is now "Sec 3.2.1, App. A.2.1". The heading "3.2.1 Bottom-up Modulation" exists in `2609.13254.txt`.

**Other changed passages.**
- The Summary's closing line ("six such points, four of them found by the final review") matches §2.3's list: sMCL, Cekinmez, Panagopoulou, UniDisc, ChaosNLI and MixGen (e).
- §9 and the AI-assistance statement describe the final review accurately.
- Table 6's caption lists rows 1, 6, 8, 11 and 12 as corrected here.

### Residual fixes

- **R1 (NIT, optional).** Table 12 row 1 still calls D6 and D7 "the masking-only readouts". That contradicts the corrected Summary item 6 and D6 text, and D7's own note that the prefix readout is open to AR-cap.
  - Current text: "plus the masking-only readouts D6 and D7."
  - Replace with: "plus the masking-favoured readouts D6 and D7 (D7's random-subset readout is masking-only)."
- **R2 (optional precision).** Figure 1's S3 card says "CLIP opposed the generative models on 3 of 29 images". The paper names these as the three images with the highest cross-model variation and says nothing about the other 26.
  - If the figure is rebuilt anyway, use: "CLIP opposed the generative models on the 3 most variable of 29 images".
  - Not required.
