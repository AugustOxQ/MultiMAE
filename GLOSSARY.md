# MultiMAE glossary

The project's terms, used by briefs, specs and reports. Created 2026-10-26 from the terms of the reports written so far; the grill (research loop step 5) adds to it, and a new name enters only after the user's yes.

| Term | Meaning | Code name or old name |
|---|---|---|
| polysemy (the project's sense) | one image, or one caption, with several valid readings; the project's goal is a more polysemic way of understanding through cross-modal masked reconstruction (user, 2026-10-06) | |
| reading | one valid interpretation of an image; on ArtELingo, one annotator's emotion | |
| H-a | the encoder claim: masked training makes the pooled retrieval embeddings keep more readings | |
| H-b | the model claim: the masked decoder's own predicted distribution holds several readings, which a contrastive dual encoder cannot | |
| one-to-many retrieval | retrieval where a query has several valid matches; measured on COCO with ECCV Caption | |
| ECCV Caption mAP@R | mean average precision at R over the human-verified many-to-many matches of ECCV Caption (Chun et al. 2022); the main COCO metric since 2026-10-03 | `eccv` |
| PMRP, CxC, rsum | the plausible-match R-precision of ECCV Caption (weak: class-level pseudo-positives); the CrissCrossed Captions metrics; the sum of R@1, R@5 and R@10 in both directions. All tracked beside mAP@R | |
| VWSD | SemEval-2023 visual word sense disambiguation; judged "just word disambiguation, not real polysemy" (user, 2026-10-05) | |
| ArtELingo | WikiArt paintings with about 5 English captions each, each caption with one of 9 emotion labels; the H-b testbed | `data=artelingo` |
| ArtELingo-28 | 1,600 WikiArt paintings with emotion votes in 28 languages (median 104 votes); the dense human distribution for D4; its paintings stay out of training and validation | AL-28, `al28_paintings.txt` |
| contrastive model | CLIP ViT-B/32 fine-tuned with InfoNCE alone | C, `model=contrastive` |
| multilearner | the fusion model: InfoNCE plus a pixel decoder (MAE) and a caption decoder (MLM) that reads the masked-pass image tokens through a fusion module; its 15% text masking version is the Stage 1 and 2 baseline | `fusion_multilearner`, ML-15 |
| masked model at 80% | the multilearner with 80% of caption tokens hidden and a 25% image view (13 of 49 patches) read by the caption decoder; on ArtELingo with the emotion slot and MAE off | ML-80; with the pixel decoder: ML-80+MAE |
| parallel captioner | the same model with the caption decoder reading the full clean image and every caption token hidden; the H-b comparator without masking of the image | Par-cap |
| M1 clean, M1 detached, text-only control | the COCO controls at 80% text masking: the caption decoder reads the clean full image; the same with no decoder gradient into the vision tower; no fusion, so the decoder reads only the masked caption | `fusion_none` for the text-only control |
| the R2 recipe | the second fine-tuning recipe of the lever review, tried in Stage 1 and on the 80% arm: raised PMRP and rsum, not mAP@R | |
| emotion slot | one extra learned decoder query whose output goes through a 9-way head; always a target, never an input | `model.emotion_head` |
| soft-label probe | multinomial logistic regression on the frozen, normalised pooled image embedding, trained on per-painting label histograms; the strongest probe is, per seed, the arm whose probe has the lowest validation NLL | linear probe |
| prior only, prompt softmax, split-half ceiling | the train label frequencies as one distribution for every painting; a dual encoder's cosine to 9 emotion prompts as a softmax; the agreement between two random halves of each painting's votes | floor, ceiling |
| JSD, entropy Spearman | Jensen-Shannon distance (base 2, in [0, 1]) between model and human distributions, averaged over paintings; the Spearman correlation over paintings between model entropy and human entropy | |
| null image | the all-zero tensor after CLIP normalisation (the mean colour); the no-image control of D7, which the decoder penalises | |
| selection index | in the blend probe, where the 0.5 blend's score of a caption falls between its scores on the two source images, with j of the caption's content tokens visible | |
| partial-view collapse | the pooled embeddings of 13-patch views of different images become almost identical (cosine 0.97) in models whose vision tower meets partial views only through pixel reconstruction | |
| D1 to D11 | the H-b review's candidate designs; D1 (decoder reranking), D2 (blend probe), D4 (emotion readout against AL-28 votes), D6 (view sampling) and D7 (partial captions) ran in the first H-b spec | |
| Stage 0, 1, 2 | the improve-multilearner line of 2026-10-03: evaluation-only diagnostics, the 15-arm screen, the confirmation at 3 and 5 seeds | |
| the lever review, the H-b review, the deep-think note | the two literature reviews (2026-10-03, 2026-10-06) and the brainstorm note of 2026-10-05, in `docs/reports/` | |
