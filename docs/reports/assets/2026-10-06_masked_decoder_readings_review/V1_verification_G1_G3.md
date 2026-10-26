> Supporting file for docs/reports/literature/2026-10-06_masked_decoder_readings_review.md. Verification of G1 to G3 as written; its corrections override those bibliographies, and the report uses corrected values.

# V1: Source verification of G1 (generative scoring), G2 (human disagreement) and G3 (affective image-text)

ARS deep-research Phase 2, lit-review mode, source verification agent. Written 2026-10-06. Its corrections override G1, G2 and G3 downstream.

## Method

- Metadata: arXiv API (https, one batched call for all 39 arXiv ids: title, authors, dates, comment, journal_ref) and the Semantic Scholar batch API (venue, DOI, ACL id). Venue gaps were settled by WebSearch (ECCV 2024 page, AAAI OJS, ACL Anthology, TMLR listing) and Crossref (Uma et al., JAIR). Farquhar et al. (Nature) was read from the open-access Nature page (curl).
- Numbers: read from the arXiv HTML of each paper, saved to `dl_V1/<id>.html` and converted with a local script (`dl_V1/scripts/h2t.py`, table cells joined per row) to `dl_V1/<id>.txt`; read with grep/sed/python -I only. No WebFetch digest was trusted for a table cell. Checked in full for: Lin 2306.01879 (Tables 1, 3), MASS 2501.11469 (Tables 1-4, App. D), DiffusionITM 2305.16397 (Table 1, Runtime), Kauf 2305.10588 (Table 1), McCarter 2603.10302 (App. A.1, Table 1), Tschannen 2306.07915 (ARO and SugarCrepe rows), Nie 2010.03532 (Table 5), Baan 2210.16133 (Table 1), Lan 2410.02773 (Tables 1-2, text), Kuhn 2302.09664 (Table 2), Farquhar (Nature page), MultiEmo 2605.14635 (Tables 3-4, Sec 3), ArtEmis 2101.07396, ArtEmis v2 2204.07660, ArtELingo 2211.10780, ArtELingo-28 2411.03769, Affection 2210.01946, ArtECulture 2608.03358, DDES 2605.26262, Sherlock 2202.04800.
- Not re-read from raw text (metadata only): VQAScore table values, SelfEval, Holtzman, Jaini, Pearson, Fathi, Plank, Uma per-dataset numbers, Peterson, Bhattacharya, Stengel-Eskin, Wang & Chan, Xie, SAMNet, Visual Semantic Entropy, PercepT table values, SEVLM numbers, SemArt, Personality-Captions, V-FLUTE. Their numbers stay "digest-level, unchecked" below.
- Absence claims: re-searched with WebSearch (extended) plus Semantic Scholar citation lists of ArtEmis, ArtELingo, ArtELingo-28, ArtEmis v2 and Affection (348 citing records, titles filtered by hand for retrieval, matching, CLIP, contrastive, diversity).
- Safety: no retrieved page contained text that tried to direct the agent.

## 1. Overall assessment

**Sources checked: 41 unique** (G1 13, G2 16, G3 13, ArtEmis counted once). All 39 arXiv ids resolve to the title and first author given. Uma et al. (JAIR) and Farquhar et al. (Nature) were checked outside arXiv.

| Existence outcome | Count | Sources |
|---|---|---|
| VERIFIED (arXiv API title and authors match; venue confirmed by S2, arXiv comment or a primary page) | 39 | all except the two below |
| VERIFIED (non-arXiv primary metadata) | 2 | Uma et al. (Crossref, JAIR 72:1385-1470, DOI 10.1613/jair.1.12752); Farquhar et al. (nature.com, 630:625-630, 19 June 2024) |
| PLAUSIBLE | 0 | |
| UNVERIFIABLE | 0 sources, 3 venue/tag claims | ArtEmis "oral" tag; Visual Semantic Entropy "Springer chapter" note; SelfEval had no arXiv venue comment (S2 and ML Anthology list TMLR 2025, which I accept) |
| FABRICATED | 0 | |

**Venue and author corrections** (agent text -> verified):
- McCarter et al. 2603.10302: "preprint" -> GEM Workshop at ICLR 2026 (arXiv comment); workshop tier. The v3 abstract is about stochastic-beam-search sampling for protein MLMs, not PLL cost (see row 9 below).
- Pearson et al. 2508.20783: "preprint" -> peer-reviewed. The arXiv comment says IWCS 2025; the ACL Anthology lists it in *SEM 2025 (2025.starsem-1.9). Use the Anthology: *SEM 2025.
- SelfEval 2311.10708: venue open -> TMLR 2025; authors Rambhatla and Misra confirmed.
- MASS 2501.11469: authors were missing -> Chung, Lim, Lee, Yu; AAAI 2025 confirmed (arXiv comment).
- Kauf and Ivanova 2305.10588: authors confirmed; ACL 2023 short (2023.acl-short.80).
- Jaini, Clark, Geirhos 2309.16779: author order confirmed (ICLR 2024 spotlight).
- VQAScore 2404.01291: the author string "Lin, Chen, Pathak, Ramanan, et al." is wrong. Authors: Lin, Pathak, Li, Li, Xia, Neubig, Zhang, Ramanan. There is no "Chen". ECCV 2024 confirmed by S2.
- Krojer et al. 2305.16397: NeurIPS 2023 confirmed (arXiv comment).
- Visual Semantic Entropy 2606.31407: "venue not verified" -> accepted at ECCV 2026 (arXiv comment). Authors Huy, Nguyen, Chowdhury, Yadav, To, Liao, Verjans, Phan.
- Lan et al. 2410.02773: AAAI 2025 confirmed (AAAI OJS 10.1609/AAAI.V39I4.32468). arXiv v1 date is 2024-09-17.
- SEVLM 2403.11150 (Zhang, Zheng, Wang, Guo): "preprint" -> ECCV 2024 (eccv2024.ecva.net poster 1499, ML Anthology).
- V-FLUTE 2405.01474: "preprint, author list unconfirmed" -> NAACL 2025 main, now titled "Understanding Figurative Meaning through Explainable Visual Entailment". Authors Saakyan, Kulkarni, Chakrabarty, Muresan.
- Sherlock 2202.04800: "ECCV 2022 UNVERIFIED" -> verified (arXiv journal_ref ECCV 2022; S2 DOI 10.1007/978-3-031-20059-5_32).
- Affection 2210.01946: CVPR 2023 confirmed (DOI 10.1109/CVPR52729.2023.00642); the brief's "ICCV" is wrong.
- ArtEmis v2 (CVPR 2022), ArtELingo (EMNLP 2022 main, 2022.emnlp-main.600), ArtELingo-28 (EMNLP 2024 main, 2024.emnlp-main.1165), SAMNet (IEEE TIP), SemArt (ECCV 2018 Workshops), Personality-Captions (CVPR 2019): confirmed.
- ArtEmis "CVPR 2021 (oral)": CVPR 2021 confirmed; the "oral" tag was not found on CVF, S2 or the arXiv page. Drop "oral" unless the author has a source.
- PercepT 2606.03345: title, authors (Mohamed, Church, Elhoseiny), date 2026-06-02 and "8 pages" confirmed; arXiv-only. ArtECulture 2608.03358 (Xiaolin Chen, Song, Shi, Han, Lee, Hsu) and DDES 2605.26262 (Bergeron, Dhossou, Tremblay, Lalonde): arXiv-only, no venue.

**Claims checked: 91 rows** (Section 3; a row may hold several numbers).

| Outcome | Count |
|---|---|
| CONFIRMED (incl. 1 reader inference and row 67, "except Western 78.7%") | 68 |
| CORRECTED | 14 |
| LOCATOR WRONG | 1 |
| NOT FOUND | 3 (rows 28, 56, 57) |
| PLAUSIBLE (gray zone, treat as fail) | 1 (row 87) |
| Not re-read (digest-level) | 3 rows covering about 20 sources |
| Added fact, no claim | 1 (row 69) |

**Load-bearing problems, in order of weight:**
1. **ArtECulture is not an independent re-collection (G3 entry 7, closest-to-H-b item 5).** It re-curates ArtELingo: "We retain an image only when majority agreement is achieved separately within all three cultures", leaving 4,928 ArtELingo images, plus 1,864 newly annotated non-Western artworks (2,783 collected). Its alpha 0.31, 70-76% majority shares and 13.60% all-culture agreement are computed on this agreement-filtered set. They cannot serve as an outside check on ArtELingo-wide stability and overstate agreement for the unfiltered data.
2. **ArtELingo-28's "about 140 annotations per painting on 2,000 paintings" is the paper's own number but does not add up (G3 entry 4, closest-to-H-b item 1).** The paper says "~140 annotations per image" (abstract, Sec 1, Table 1) and also "200K annotations" on "2000 images", which is 100 per image. Per-language counts run from 2,032 (Kazakh) to 10,493 (Urdu) and the collection aims at five annotators per image per language, so about 4 per language per image, about 100 in total. Our local public CSV (173,745 rows, 1,658 paintings, median 108, mean 105) matches the 200K/2,000 reading, not 140. Use "about 100 per painting (paper text says ~140; its own totals give 100)". The paper does not say how the 2,000 images were chosen beyond "representative of styles and genres". In the ArtELingo paper the 2,000 images with the most Cohen-kappa disagreement are a separate set D used for the genre analysis. Whether the ArtELingo-28 2,000 equal D is not stated and is not established here.
3. **G2 Lan et al. "KL 1.886 vs 1.771 inconsistency" is a misreading.** Table 1 reports LXMERT KL 1.886 before calibration. Table 2 reports LXMERT KL 1.771 after accuracy-oriented temperature scaling (T = 1.05) and 0.731 after human-distribution scaling (T = 2.0). The paper text says "from 1.886 to 0.731". There is no inconsistency. TS toward accuracy helped LXMERT slightly and hurt BEiT3 (KL 1.701 to 3.232).
4. **G2 Nie et al. summary "models barely beat or lose to chance on divergence" is too strong for JSD on SNLI.** On ChaosNLI-SNLI models are at JSD about 0.22 to 0.24 against chance 0.383 and humans 0.061 (a clear gap to chance), and KL 0.47 to 0.53 against chance 0.546 (near chance). On ChaosNLI-alpha and MNLI models are near or worse than chance. State the dataset with each number.
5. **MASS "COCO recall" is a CLIP top-20 re-ranking, not full-gallery retrieval (G1 entry 5).** Table 2 numbers (I2T R@1 57.3, T2I 55.6) come from "CLIP retrieves the top 20 candidates, followed by re-ranking". They cannot be set beside Lin's full-gallery 5k numbers (71.9 and 19.7). "Partly multiplicity" is overstated: recall counts a hit if any of the standard ground-truth pairs is retrieved, which is ordinary COCO recall.
6. **DiffusionITM Flickr numbers are a 10-way re-ranking set (G1 entry 10).** GDBench's Flickr30K task retrieves "among the 10 most similar examples based on the CLIP embedding space" (chance 10%). Hence 11.1 for the naive Diffusion Classifier is chance, and 71.9 for CLIP RN50x64 is a 10-way accuracy, not R@1 over 1,000 images. The 68 min vs about 4 min runtime is for Flickr30K text-retrieval validation under the same setup.
7. **Lin et al. omission: text-to-image.** Table 3b shows the raw generative score is competitive for T2I (COCO R@1 55.6 vs ITM 54.8; Flickr 76.8 vs 77.8). G1's closest-list statement "raw likelihood fails on COCO/Flickr retrieval (19.7 vs 71.9)" holds for image-to-text only.
8. **G2's absence claim needs a narrower wording** (Section 4, claim G2-1). Wang and Chan 2019, already in G2, compares set diversity of sampled captions with human captions on real COCO images. VEGAS (2607.08489) does a near version on video.
9. Smaller corrections: ArtEmis "landscapes 60.0% / nudes 32% majority agreement" is not in the paper (60.0% is a classifier accuracy); Sherlock localization 87.2 is the GT-box setting (Auto-Box 40.6); Sherlock comparison metric is "Human Acc"; MultiEmo human ceiling is 0.700 in Table 4 but 0.716 in Sec 3 (different protocol); McCarter cost note not in paper. See rows.

## 2. Per-source metadata table

Tier: P = peer-reviewed main venue, W = workshop, Pre = preprint. Existence: V = verified.

| File | Source (arXiv) | Title and first author match | Venue verified | Tier | Note |
|---|---|---|---|---|---|
| G1 1 | Lin 2306.01879 | V | ICML 2024 (arXiv comment "Published at ICML 2024"; S2) | P | 5 authors as listed |
| G1 2 | Lin VQAScore 2404.01291 | V | ECCV 2024 (S2) | P | author list wrong in G1 (no "Chen") |
| G1 3 | Tschannen 2306.07915 | V | NeurIPS 2023 (arXiv comment) | P | |
| G1 4 | SelfEval 2311.10708 | V | TMLR 2025 (S2, ML Anthology) | P | authors Rambhatla, Misra confirmed |
| G1 5 | MASS 2501.11469 | V | AAAI 2025 (arXiv comment) | P | authors Chung, Lim, Lee, Yu |
| G1 6 | Holtzman 2104.08315 | V | EMNLP 2021 (2021.emnlp-main.564) | P | |
| G1 7 | Salazar 1910.14659 | V | ACL 2020 (arXiv journal_ref; S2 has a mis-merged "Pseudolikelihood Reranking" record) | P | |
| G1 8 | Kauf and Ivanova 2305.10588 | V | ACL 2023 short (2023.acl-short.80) | P | authors confirmed |
| G1 9 | McCarter 2603.10302 | V | GEM Workshop, ICLR 2026 (arXiv comment) | W | G1 said preprint |
| G1 10 | Krojer 2305.16397 | V | NeurIPS 2023 (arXiv comment) | P | |
| G1 11 | Jaini 2309.16779 | V | ICLR 2024 spotlight (arXiv comment) | P | author order confirmed |
| G1 12 | Pearson 2508.20783 | V | *SEM 2025 (ACL Anthology; arXiv comment says IWCS 2025) | P | G1 said preprint |
| G1 13 | Fathi 2607.03831 | V | none (S2: arXiv) | Pre | |
| G2 1 | Plank 2211.02570 | V | EMNLP 2022 (2022.emnlp-main.731) | P | |
| G2 2 | Uma et al. JAIR | V (Crossref) | JAIR 72:1385-1470 | P | no arXiv id |
| G2 3 | Nie 2010.03532 | V | EMNLP 2020 (2020.emnlp-main.734) | P | |
| G2 4 | Baan 2210.16133 | V | EMNLP 2022 (2022.emnlp-main.124) | P | |
| G2 5 | Peterson 1908.07086 | V | ICCV 2019 (DOI 10.1109/ICCV.2019.00971) | P | |
| G2 6 | Bhattacharya 1908.04342 | V | ICCV 2019 (DOI 10.1109/ICCV.2019.00437) | P | |
| G2 7 | Stengel-Eskin 2211.07516 | V | ACL 2023 (2023.acl-long.569) | P | |
| G2 8 | Lan 2410.02773 | V | AAAI 2025 (AAAI OJS 32468) | P | |
| G2 9 | Wang and Chan 1903.12020 | V | CVPR 2019 | P | |
| G2 10 | Xie 1912.08960 | V | none | Pre | |
| G2 11 | Kuhn 2302.09664 | V | ICLR 2023 spotlight | P | |
| G2 12 | Farquhar (Nature) | V | Nature 630:625-630 | P | read in full (open access) |
| G2 13 | Huy 2606.31407 | V | ECCV 2026 accepted (arXiv comment) | P (accepted) | G2 said venue unverified |
| G2 14 | Yang SAMNet 2207.11875 | V | IEEE TIP | P | |
| G2 15 | MultiEmo-Bench 2605.14635 | V | none | Pre | |
| G2 16 | ArtEmis 2101.07396 | V | CVPR 2021 (DOI 10.1109/CVPR46437.2021.01140); "oral" not confirmed | P | |
| G3 2 | ArtEmis v2 2204.07660 | V | CVPR 2022 (arXiv comment) | P | |
| G3 3 | ArtELingo 2211.10780 | V | EMNLP 2022 (2022.emnlp-main.600) | P | |
| G3 4 | ArtELingo-28 2411.03769 | V | EMNLP 2024 (2024.emnlp-main.1165) | P | |
| G3 5 | Affection 2210.01946 | V | CVPR 2023 (DOI 10.1109/CVPR52729.2023.00642) | P | brief's ICCV is wrong |
| G3 6 | PercepT 2606.03345 | V | none | Pre | title and authors exact |
| G3 7 | ArtECulture 2608.03358 | V | none (S2: no venue) | Pre | |
| G3 8 | DDES 2605.26262 | V | none | Pre | |
| G3 9 | SemArt 1810.09617 | V | ECCV 2018 Workshops (DOI 10.1007/978-3-030-11012-3_52) | W | G3 listed "ECCV 2018 Workshops", correct |
| G3 10 | SEVLM 2403.11150 | V | ECCV 2024 (eccv2024.ecva.net poster 1499) | P | G3 said preprint |
| G3 11 | Sherlock 2202.04800 | V | ECCV 2022 (arXiv journal_ref, S2 DOI) | P | G3 said UNVERIFIED |
| G3 12 | Personality-Captions 1810.10665 | V | CVPR 2019 (DOI 10.1109/CVPR.2019.01280) | P | |
| G3 13 | V-FLUTE 2405.01474 | V | NAACL 2025 main (DOI 10.18653/v1/2025.naacl-long.1) | P | retitled; authors now known |

## 3. Claims checked

Outcome: CONFIRMED / CORRECTED / LOCATOR WRONG / NOT FOUND. Locators are the paper's own.

### G1

| # | Source | Claim as written | What the source says | Outcome |
|---|---|---|---|---|
| 1 | Lin | Table 3a COCO 5k i2t R@1: ITM 71.9, alpha=0 19.7, alpha=1 46.2, val-tuned 48.0 | Table 3a: 71.9 / 19.7 / 46.2 / 48.0 (R@5 90.6 / 40.6 / 73.1 / 74.2); alpha*_val 0.819 | CONFIRMED |
| 2 | Lin | Flickr30K i2t R@1 88.8 / 34.6 / 58.7 / 63.6 | Table 3a: 88.8 / 34.6 / 58.7 / 63.6 | CONFIRMED |
| 3 | Lin | Winoground text 27.5 / 33.7 / 36.6, ITM 35.5; EqBen 9.6 / 19.8 | Table 3a: 35.5 ITM, 27.5, 33.7, 36.6; EqBen 26.1 ITM, 9.6, 19.8, 19.8 | CONFIRMED (EqBen ITM 26.1 not stated in G1) |
| 4 | Lin | "blind text-only BLIP 87.6/80.7/98.6/99.1" on ARO | Table 1 row P_train(t) BLIP: 87.6 / 80.7 / 98.6 / 99.1. It is BLIP's own decoder fed Gaussian-noise images (a blind estimate), not a text-only LM. The true text-only LMs (P_LLM) get 95.0-98.0 / 95.2-98.6 on the order sets | CORRECTED (label) |
| 5 | Lin | ARO Table 1: ITC 63.1/81.6/34.3/41.7, ITM 58.7/90.3/45.1/51.3, VisualGPTScore 89.1/95.3/99.4/99.5 | Table 1: same | CONFIRMED |
| 6 | Lin | SugarCrepe ITC 85.8/73.8/85.7, ITM 88.7/81.3/87.6, alpha=0 93.3/91.0/91.0 | Table 1 rows match for ITC, ITM, alpha=0 (93.3 seen; 91.0/91.0 not re-read cell by cell) | CONFIRMED (partly cell-checked) |
| 7 | Lin | "generative scores still lag ITM on the two large retrieval sets" | Appendix text: "generative approaches still lag behind the ITMScore of BLIP for the two large-scale retrieval benchmarks" | CONFIRMED |
| 8 | Lin | Closest-list: raw likelihood fails on COCO/Flickr retrieval | True for I-to-T only. Table 3b T-to-I: COCO R@1 ITM 54.8 vs P(t given i) 55.6; Flickr 77.8 vs 76.8 | CORRECTED (direction qualifier) |
| 9 | Lin | "same cost as ITMScore" | Sec 3: "VisualGPTScore has the same computational cost as ITMScore" | CONFIRMED |
| 10 | Lin | weakness: ITM trained with hard negatives so comparison is not neutral | consistent with MASS's explanation for BLIP-2 ITM (reader inference stands) | CONFIRMED as reader inference |
| 11 | MASS | Table 1 VALSE counting Adv., OFA large: ITM 72.0, TL 66.8, MASS 76.7 | Table 1: 72.0 / 66.8 / 76.7 | CONFIRMED |
| 12 | MASS | Table 2 T2I R@1: CLIP ITC 36.5, OFA ITM 35.3, TL 38.0, MASS 55.6; I2T: CLIP 56.9, ITM 42.1, TL 33.4, MASS 57.3; Bias@1 TL 6.6 to MASS 3.5 | Table 2 identical | CONFIRMED (but re-ranking of CLIP top-20 candidates, Sec 4.3; G1 does not say so) |
| 13 | MASS | Table 3 Winoground group: OFA large ITM 9.5, TL 15.8, MASS 20.3 | Table 3: 9.5 / 15.8 / 20.3 | CONFIRMED |
| 14 | MASS | BLIP-2 T5 group: ITM 15.5, TL 11.0, MASS 18.0 | Table 3 row BLIP-2T5: 15.5 / 11.0 / 18.0 | CONFIRMED |
| 15 | MASS | Table 4 SVO-Probes OFA large group: ITM 39.7, TL 12.5, MASS 43.0 | Table 4: 39.7 / 12.5 / 43.0 (MASS - ITM = 3.3) | CONFIRMED |
| 16 | MASS | 0.55 s per sample vs 0.27 s (App. D) | "A forward pass of OFAlarge takes 0.27 seconds ... 0.55 seconds (1.8it/s) per sample" | CONFIRMED |
| 17 | MASS | "recall counts a hit when any ground-truth pair is retrieved (Sec 4.3)"; "multiplicity measured: partly" | Sec 4.3 says so, but it is the ordinary COCO recall over the 5 captions; the retrieval is CLIP top-20 re-ranked. No multi-positive metric | CORRECTED (overstated) |
| 18 | MASS | limitations: effect varies with base model | Broader Impact says the *social-bias* reduction "may vary with respect to the base VL model" | CORRECTED (scope: social-bias effect) |
| 19 | DiffusionITM | Flickr Img 11.1 (naive Diffusion Classifier) vs CLIP RN50x64 71.9; DiffusionITM 61.5; Winoground image 10.2 | Table 1a: CLIP RN50x64 71.9, OpenCLIP ViT-L/14 79.9, Diffusion Classifier 11.1, DiffusionITM 61.5, HardNeg 66.1; Winoground image 13.8 / 11.25 / 0.25 / 10.2 / 12.3. Task is a 10-candidate re-rank (footnote 6: random 10%) | CONFIRMED (add: 10-way) |
| 20 | DiffusionITM | 68 min vs about 4 min | Runtime paragraph: "With 10 noise samples per image-text-pair ... Flickr30K Text Retrieval validation takes 68 minutes on a single RTX A6000 (compared to around 4 minutes with OpenCLIP ViT-L/14)" | CONFIRMED |
| 21 | DiffusionITM | "slow retrievers applied to a small set of candidates" | quote present (Runtime paragraph) | CONFIRMED |
| 22 | DiffusionITM | main results with 250 noise samples | "keep samples constant at 250 for the main zero-shot experiments" | CONFIRMED |
| 23 | Kauf and Ivanova | BLiMP: BERT-base 84.2 to 84.7, BERT-large 84.8 to 85.0, RoBERTa-base 85.4 to 86.7, RoBERTa-large 86.5 to 87.5 | Table 1: PLL-original / PLL-word-l2r: 84.2 / 84.7, 84.8 / 85.0, 85.4 / 86.7, 86.5 / 87.5 | CONFIRMED |
| 24 | Kauf and Ivanova | PLL-original inflates scores of multi-token words | Sec 2, Fig 2: "inflates scores of multi-token words"; PLL-whole-word is lower than original (83.1, 82.6, 85.4, 85.9) | CONFIRMED (add: whole-word variant scores lower) |
| 25 | Salazar | WER down by up to 30% relative, +1.7 BLEU, +10 points BLiMP island effects and NPI | abstract: "reduces ... WER by 30% relative and adds up to +1.7 BLEU ... +10 points on island effects, NPI licensing in BLiMP" | CONFIRMED |
| 26 | Salazar | "maskless scoring via fine-tuning" UNVERIFIED | abstract: "One can finetune MLMs to give scores without masking, enabling computation in a single inference pass" | CONFIRMED (settles the open item) |
| 27 | McCarter | Table 1 Pearson/Spearman with exact PLL: ESM2-650M 0.964/0.940, AbLang2 0.869/0.869, ProteinMPNN 0.944/0.934 | Appendix A.1 Table 1 (190 one-edit CDR-H3 variants of Trastuzumab, tau = 1): same values | CONFIRMED (locator: App. A.1, Table 1) |
| 28 | McCarter | "cost O(L^4) to O(L^3) amortised" | No such statement. Paper: PLL costs O(L^3) per sequence; beam search with the approximation gives a "20EL x speedup" over mutation-centric sampling | NOT FOUND |
| 29 | Tschannen | ARO Cap 88.9/86.6/99.1/99.0; SugarCrepe CapPa B/16 89.95/88.71/84.35, OpenCLIP G/14 96.67/88.07/74.75 | match (Blind decoder row 83.7/86.2/98.8/98.7 also listed) | CONFIRMED |
| 30 | Tschannen | CapPa 85.7/86.7/99.2/98.8 | 85.7 and 86.7 seen; order-set cells not re-read | CONFIRMED (partly) |
| 31 | VQAScore, SelfEval, Holtzman, Jaini, Pearson, Fathi | table values | not re-read from raw text | Not re-read |

### G2

| # | Source | Claim as written | What the source says | Outcome |
|---|---|---|---|---|
| 32 | Nie | Table 5 ChaosNLI-SNLI: chance JSD 0.383 / KL 0.5457; models JSD about 0.23, KL 0.48-0.53; humans 0.0614 / 0.0411 | Table 5 columns are alpha, S, M. SNLI: chance 0.383 / 0.5457; BERT-b 0.2345 / 0.481, RoBERTa-b 0.2294 / 0.5045, others 0.21-0.31 JSD (DistilBert 0.2439 / 0.4682); Est. Human 0.0614 / 0.0411 | CONFIRMED |
| 33 | Nie | alpha: chance 0.3205 / 0.406; BERT-b 0.3209 / 3.7981; RoBERTa-b 0.2394 / 0.8272; human 0.0421 / 0.0373 | Table 5 same | CONFIRMED |
| 34 | Nie | MNLI: chance 0.3023 / 0.3559; models "about 0.31 / 0.72-0.79"; human 0.0695 / 0.0381 | chance and human match. Model KL spans 0.6652 (DistilBert) to 0.8845 (BART), JSD 0.3055-0.3165 | CORRECTED (range) |
| 35 | Nie | "models barely beat or lose to uniform chance on divergence" | SNLI JSD 0.23 vs chance 0.383 is a clear gap; SNLI KL is near chance; alpha and MNLI at or worse than chance | CORRECTED (see problem 4) |
| 36 | Nie | accuracy split about 0.9 vs about 0.5, "accuracy and divergence uncorrelated" | not re-read | Not re-read |
| 37 | Baan | Table 1: RoBERTa Acc 0.74, ECE 0.14, DistCE 0.26; +TS ECE 0.03, DistCE 0.22; oracle Acc 1.00, ECE 0.25, DistCE 0.00; RankCS 0.62 vs 1.00; |EntCE| 0.30 to 0.21 | Table 1 identical (3 seeds); text "even more so than RoBERTa (0.25 vs 0.14)" | CONFIRMED |
| 38 | Baan | DistCE = total variation distance | Eq. 7 and footnote: TVD = half the L1 distance | CONFIRMED |
| 39 | Lan | Table 1: LXMERT all VQA-acc 0.671, TVD 0.448, KL 1.886, EntCE 0.555; BEiT3 all TVD 0.436, KL 1.701, EntCE 0.541 | Table 1 same (BEiT3 all VQA-acc 0.842 in the subset; 84.03 is the test-set figure in the text) | CONFIRMED |
| 40 | Lan | BEiT3 by HUD low/med/high: accuracy 0.921/0.845/0.759; KL 2.150/1.544/1.408 | Table 1 same | CONFIRMED |
| 41 | Lan | "Table 2's LXMERT 'before' row shows 0.442/1.771/0.542, a different number set" | Table 2 values are after accuracy-oriented TS (t = 1.05), not "before" | CORRECTED |
| 42 | Lan | BEiT3 KL 1.701 to 3.232 under accuracy TS (T = 0.6), to 0.647 under human TS (T = 2.0), TVD 0.436 to 0.340 | text and Table 2: same | CONFIRMED |
| 43 | Lan | items: BEiT3 3,248, LXMERT 15,408; distinct answers about 3.1 / 4.0 / 4.8 | Sec 3: 3,248 (1,083 / 1,083 / 1,082) and 15,408 (5,136 each); 3.06 / 3.97 / 4.82 for BEiT3 | CONFIRMED |
| 44 | Kuhn | TriviaQA wrong answers average 3.89 clusters vs 1.89 right (Table 2); AUROC 0.83 | Table 2: 1.89 / 3.89; semantic-entropy AUROC 0.83, distinct-answer count 0.79 | CONFIRMED |
| 45 | Kuhn | "OPT-30B 0.83 vs 0.80 (length-normalised entropy)" | 0.83 in Table 2 (largest model, Fig 1a); the 0.80 is read from a figure, not a table | LOCATOR WRONG (0.80 has no table cell) |
| 46 | Kuhn | best temperature 0.5; multinomial beats beam search | Sec 4 (Fig 3b) and Table 4 | CONFIRMED |
| 47 | Kuhn | semantic entropy definition: sample about 10, cluster by bidirectional entailment, sum sequence probabilities per cluster, entropy over clusters | consistent with the method section as read | CONFIRMED |
| 48 | Farquhar | mean AUROC 0.790 vs naive entropy 0.691, P(True) 0.698 (marked unverified) | Nature page: "semantic entropy achieves the best AUROC value of 0.790 whereas naive entropy (0.691), P(True) (0.698) and the embedding regression baseline (0.687) lag behind"; "average over five datasets" | CONFIRMED (drop UNVERIFIED mark; add embedding-regression 0.687) |
| 49 | MultiEmo | Table 4: split-half PLCC 0.700 vs GPT-4o 0.634, GPT-5 0.631, GPT-5.1 0.629, Qwen-VL-32B 0.592, Gemini-2.5-flash 0.541 | Table 4 identical; human SRCC 0.656 vs GPT-4o 0.615; Cheb GPT-5 0.578 vs human 0.559; KL GPT-4o 1.294 (human n/a) | CONFIRMED |
| 50 | MultiEmo | split-half PLCC 0.716, SRCC 0.666; 4,907 (47.44%) images with more than 10 votes for dominant; 962 (9.3%) multiple dominant | Sec 3 same; but Table 4's human PLCC is 0.700 (10 random splits of the 20 votes), Sec 3 and App. say 0.716 | CORRECTED (two human values: 0.700 in Table 4, 0.716 in Sec 3; say which) |
| 51 | MultiEmo | 10,344 images, 236,998 votes (22.91 per image), 20 annotators, GPT-5 51.05% dominant-emotion precision (Table 3) | Sec 3 and Sec 4.2 same. Note the human split-half compares two halves of about 10 votes, so it understates a full-20-vote human | CONFIRMED (add caveat) |
| 52 | ArtEmis | 45.6% (37,145) strong majority; 61% positive and negative (79% with "something else"); 11.7% something-else | Sec 3.2: same | CONFIRMED |
| 53 | ArtEmis | 439,121 explanations, 6,377 annotators, 36,347 words, 81,446 artworks, at least 5 annotators per artwork | Sec 3: same. The abstract-level figure elsewhere is 455K; ArtEmis v2 also calls ArtEmis "455K" | CONFIRMED |
| 54 | ArtEmis | BERT 65.7%, LSTM 63.1%, expert humans 60.3% (250 utterances); Turing test 50.3% on par, 20.6% more human-like (500 artworks) | Sec 6: same (AMT study 53.0% in 1K trials also reported) | CONFIRMED |
| 55 | ArtEmis | 94.7% of 1K trials, 97.5% appropriate, 76.6% above 4-year-old maturity vs COCO 34.5% | Sec 3.3: same | CONFIRMED |
| 56 | ArtEmis | "20.5% use similes or metaphors" | not in the main text read (Table 4 reports Similes-percent of speakers, 0.197 etc.) | NOT FOUND |
| 57 | ArtEmis | "by genre landscapes reach majority agreement most often (60.0%), nudes least (32%)" | Not in the paper's main text. "60.0%" is a classifier's accuracy on 8,160 unimodal test images. Fig. 4d plots per-genre entropy without these numbers | NOT FOUND |
| 58 | ArtEmis | emotion alignment of grounded speaker 52.2% | Table 4 Emo-Align: 0.522 for the strongest grounded speaker column (others 0.326-0.451) | CONFIRMED |

### G3

| # | Source | Claim as written | What the source says | Outcome |
|---|---|---|---|---|
| 59 | ArtELingo-28 | "2,000 WikiArt images, about 140 annotations per image over 28 languages ... Table 1 lists 5.68 for ArtEmis, 15.3 for ArtELingo's 3 languages and 140 for the 25 new languages" | Table 1 row #Annot/Image: 5.68 / 15.3 / "15.3 (3) + 140 (25)"; #Images "80k, 80k, 80k (3), 2K (25)"; #Annotations "1.2M (3) + 200K (25)". Abstract and Sec 1: "~140 annotations per image". 200K / 2,000 = 100 | CORRECTED (140 contradicts the paper's own 200K / 2,000; local CSV agrees with about 100) |
| 60 | ArtELingo-28 | "200K new annotations over 25 languages, 220 annotators from 23 countries" | Sec 3: "220 native annotators spanning 23 countries"; 6.25K hours | CONFIRMED |
| 61 | ArtELingo-28 | Table 4 accuracy 0.37 base, 0.354 ArtELingo, 0.664 ArtELingo-28; F1 0.342 / 0.313 / 0.638 | Table 4 identical (also ArtELingo-28_O 0.636 / 0.606); evaluated on the ArtELingo-28 test set | CONFIRMED |
| 62 | ArtELingo-28 | zero-shot captioning MiniGPT-4 BLEU-4 1.09, CIDEr 0.33; InstructBLIP 0.54 / 0.05; ClipCap 0.18 / 0.02; few-shot MiniGPT-4 13.5 (20%) vs 13.1 (100%) | values 1.09, 0.33, 0.54, 13.5 found in the tables; remaining cells not individually located | CONFIRMED (partly) |
| 63 | ArtELingo-28 | annotations per language, "pairwise KL between per-language emotion distributions; two large clusters" | Sec 3.1: per-language counts 2,032 (Kazakh) to 10,493 (Urdu); KL formula D = sum p log p/q; Fig 4 | CONFIRMED |
| 64 | ArtELingo-28 | "annotators 8.8 per language" | arithmetic from 220 / 25; Fig 3 holds per-language counts | CONFIRMED (derived) |
| 65 | ArtELingo | Table 3: English 429K (6,377), Chinese 426K (745), Arabic 369K (656), Spanish 4.8K (31) | Table 3 same | CONFIRMED |
| 66 | ArtELingo | Table 5 English 0.663 (XLM-ACE), Arabic 0.558, Chinese 0.927; Table 7 BLEU-4 | 0.663, 0.558, 0.927 found in Table 5; Table 7 BLEU-4 cells not individually checked | CONFIRMED (partly) |
| 67 | ArtELingo | Table 4: landscape -1.08, sketch 0.85; Western 78.7%, Chinese 0.5%, Arabic 0.6% | genre table: landscape -1.08, sketch 0.85 (A = log ratio of the 2,000 highest-disagreement set D vs all U); origin table: Arabic 1.6k 0.6%, Chinese 1.4k 0.5%, total 250.0k. "78.7%" not located | CONFIRMED except Western 78.7% (NOT FOUND in text read) |
| 68 | ArtELingo | "more than 51K paintings have 5 or more annotations in all three languages" | abstract: "More than 51K artworks have 5 annotations or more in 3 languages" | CONFIRMED |
| 69 | ArtELingo | (not in G3) selection of the 2,000 | Sec 5: D = "the 2000 artworks with the most disagreement (based on Kappa)" among artworks with 5 annotations in each of the 3 languages | Added fact, see problem 2 |
| 70 | Affection | CVPR 2023 | arXiv + S2 DOI CVPR52729.2023.00642 | CONFIRMED |
| 71 | Affection | 526,749 explanations, 85,007 images, 6,283 workers, at least 6 annotators per image | Sec 1, Sec 3: same | CONFIRMED |
| 72 | Affection | 67.5% (57,381) strong majority vs 45.6% ArtEmis; 50.0% mixed valence; 71.3% positive, 21.1% negative, 7.6% something-else | Sec 3.2: same | CONFIRMED |
| 73 | Affection | listening study: k=1 97.9% vs COCO 99.5%; k=10 89.7% vs 96.5%; "CLIP variant not stated" | Sec 5: "pretrained CLIP ... (version ViT-B/32)", no fine-tuning, cosine similarity, uniformly random distractors; Fig 7. Variant is stated | CORRECTED (ViT-B/32, zero-shot; resolves UNVERIFIED) |
| 74 | Affection | Table 3: CLIPScore 66.7 to 69.2, RefCLIPScore 75.0 to 76.3, unique 78.7 to 83.7, emo-alignment 48.1 to 55.9 | Table 3 same | CONFIRMED |
| 75 | Affection | BERT 72.5% (9-way); ResNet-101 59.1%; concreteness 2.82 vs COCO 3.55; subjectivity 0.53 vs 0.47; Turing 41.4-46.2% | text same | CONFIRMED |
| 76 | Sherlock | retrieval P@1 33.4% about 1K candidates; human 42.3 | Table 2 last model row P@1 33.4 (random mean rank 495.4, so about 1K candidates); human 42.3 is the Comparison column ("Val/Test Human Acc"), not retrieval | CONFIRMED (the 42.3 is not a retrieval ceiling) |
| 77 | Sherlock | localization 87.2% (human 92.3%) | Table 2 column "GT-Box/Auto-Box": 87.2 / 40.6 for the best model; human 92.3 / (96.2) | CORRECTED (87.2 is the GT-box setting; Auto-Box 40.6) |
| 78 | Sherlock | comparison "pairwise accuracy 27.1% (multitask; 26.0% single-task)" | column is "Human Acc" (val/test): multitask 26.6 / 27.1; CLIP RN50x64 25.1 / 26.0; human 42.3 / 42.3. Text elsewhere calls it pairwise accuracy | CORRECTED (metric name; 26.0 is the CLIP baseline, not "single-task") |
| 79 | ArtECulture | alpha 0.309 / 0.324 / 0.306; majority share 70.81 / 76.32 / 70.13%; 924 images (13.60%) same emotion in all cultures; 1,381 (20.33%) three distinct; Fleiss kappa 0.088 | Agreement Analysis paragraph: same | CONFIRMED |
| 80 | ArtECulture | 6,792 artworks, 92,062 pairs, about 13.55 per image, 30,932 / 31,559 / 29,571 | dataset table: same; about 4.5 per image per culture | CONFIRMED |
| 81 | ArtECulture | "only external re-collection found with chance-corrected agreement on ArtEmis-style data" | Re-curation of ArtELingo: keeps images with majority agreement in all three cultures (4,928), adds 2,783 collected non-Western artworks (1,864 retained). Agreement figures are on this filtered set | CORRECTED (see problem 1) |
| 82 | ArtECulture | knowledge base 8,439 concept-emotion tuples | "8,439 tuples and 28,146 rationales" | CONFIRMED |
| 83 | ArtEmis v2 | entropy 0.805 to 0.855 (K = 20); 62% vs 26%; 260,533 instances; 52,933 biased paintings (score above 0.3); at least five submissions; 3% no-image (7,752) | Sec 3-4: same; candidates are 24 paintings (12 nearest of 100 VGG16 fc7 neighbours plus 12 with the strongest score) | CONFIRMED |
| 84 | ArtEmis v2 | CIDEr +20%, METEOR +7%, human preference 73% of 100 | abstract and Sec 5: same | CONFIRMED |
| 85 | ArtEmis v2 | "combined size (~455K) UNVERIFIED" | "Combined" was trimmed by 65K to match ArtEmis's size, "i.e. 455K"; the paper calls ArtEmis 455K | CORRECTED (resolves: 455K is the matched training size) |
| 86 | DDES | ArtEmis accuracy DDES-Net 39.8 vs CES-Net 39.7 (Table 1); tau 0.34 | Table 1: 39.8 / 39.7; tau 0.34 (DDES) vs 0.32 (CES); DES-Net 29.4 / 0.21 | CONFIRMED |
| 87 | DDES | zero-shot Table 3: EEmo-Bench 21.2%, WikiArt Emotions 10.5% | Table 3 DDES-Net unseen columns read "... 21.2 | 10.5 | 0.18" with a header that mixes Acc and tau; assignment of 21.2 and 10.5 to the two datasets is plausible but the parsed header is ambiguous | PLAUSIBLE (gray zone, treat as unverified) |
| 88 | PercepT | title, authors, date, tables | arXiv API title, authors exact; table values not re-read | CONFIRMED (metadata), table values not re-read |
| 89 | SemArt, Personality-Captions, SEVLM, V-FLUTE | numbers | not re-read | Not re-read |

### Extra rows from the priority list

| # | Source | Claim | Finding | Outcome |
|---|---|---|---|---|
| 90 | Kauf and Ivanova | "Kauf and Ivanova's PLL claim" | PLL-original inflates multi-token words; word-l2r adjustment improves BLiMP by 0.2 to 1.3 points; Table 1 above | CONFIRMED |
| 91 | Uma et al. | JAIR 72:1385-1470 | Crossref: same, authors Uma, Fornaciari, Hovy, Paun, Plank, Poesio | CONFIRMED |


## 4. Absence claims

Searches run 2026-10-06: WebSearch extended, four G1 and G2 and G3 queries each, plus Semantic Scholar citation lists.

**G1-1: no paper reports ECCV Caption mAP@R / CxC / multi-positive metrics for a generative likelihood versus a contrastive score from the same model.** HOLDS (bounded by my queries). Queries: "image captioner likelihood retrieval ECCV Caption mAP@R CxC multiple positives generative scoring versus contrastive same model"; "captioning model log-likelihood zero-shot image-text retrieval COCO Flickr30k compared with contrastive CLIP score same pretrained model CoCa generative retrieval". The only ECCV Caption hits were the ECCV Caption paper itself and PCME++. One relevant paper G1 missed: IG Captioner (Yang et al., arXiv 2311.17072, ECCV 2024). It compares a captioner's likelihood (and an information-gain score, a PMI relative) with CLIP classifiers on COCO/Flickr retrieval, and reports captioners beat CLIP for text-to-image. It trains its own models on LAION-5B, uses no multi-positive metric (no match for "ECCV", "CxC", "mAP" in the text) and was not read in full. It supports the claim's wording but belongs in G1's list of same-family comparisons.

**G1-2: no multimodal pseudo-log-likelihood work.** HOLDS (bounded). Query: "pseudo-log-likelihood image caption scoring vision-language masked language modeling head probing compositionality VisualBERT LXMERT FLAVA masked token probability image-text". All results were text-only PLL (Salazar, Kauf). The ECCV Caption paper's own MLM-pretrained ViLT and VinVL results (already noted in the 2026-10-03 review) are the nearest. Weak-form caveat: absence of a hit in a search index cannot exclude work that scores a VLM's MLM head without calling it PLL.

**G2-1: no paper compares a captioner's sampled captions or semantic-entropy clusters with the spread of human captions or emotions on real images or art.** HOLDS ONLY IN NARROW FORM. Queries: "semantic entropy image captioning compare sampled captions diversity with human annotator caption variability COCO five captions meaning clusters"; "human annotators captions semantic variation sentence-transformer pairwise distance AI-generated captions close to human consensus centroid ..."; "vision-language model sampled captions distribution versus human caption distribution annotator disagreement ... 2025". Counterexamples to the broad wording:
- Wang and Chan, CVPR 2019 (arXiv 1903.12020), already in G2 entry 9: they sample caption sets from eight captioners on COCO and compare their set diversity with human caption sets. A scalar diversity comparison on real images.
- VEGAS, "Human-Aligned Video Caption Evaluation via Gaze" (arXiv 2607.08489, 2026-07-09): on 332 egocentric video clips (Aria Everyday Activities), human captions have mean pairwise SBERT-PCA distance 1.53, naive Gemini captions lie 0.48 from the human centroid and "cover only a small portion of the semantic variation among human annotations" (Sec 4, Fig 2). Video, not images, and a single-caption comparison, not sampling or semantic-entropy clusters.
Narrowed claim that holds: no paper found that compares a captioner's *sampled meaning clusters or distribution* (semantic entropy, cluster count) with human caption or emotion distributions on still images or paintings.

**G3-1: no third-party image-text retrieval Recall@K on ArtEmis or ArtELingo.** HOLDS (bounded). Queries: "ArtEmis ArtELingo image-text retrieval Recall@K CLIP fine-tuning emotion explanation cross-modal retrieval paintings", plus title filtering of 348 citing records (ArtEmis 204, ArtELingo 24, ArtELingo-28 27, ArtEmis v2 59, Affection 34). No title indicated ArtEmis or ArtELingo retrieval. Near misses read: MMArt (arXiv 2608.10706) reports retrieval R@1 on its own VLM-generated perspective captions for 74,234 WikiArt paintings (narrative 44.0%); its emotional captions are rewritten from ArtEmis utterances, and ArtEmis itself is not the retrieval target. Abstract4D (2608.28339) reports retrieval on its own data and only lists ArtEmis in a comparison table. A-MAR (2604.19689) retrieves on SemArt and Artpedia. Three titles were not opened: "Retrieving Emotional Stimuli in Artworks" (ICMR 2024), "Bridging Subjectivity in Affective Explanation Captioning via Consensus-Prompted Emotion Reasoning" (TIP 2026), "Emotion Aggregation in Artistic Image Analysis: Effects of Label Distribution Learning" (PACLIC 2024). The first is the most likely to contain a retrieval number and should be opened before the claim is written as "none exist". S2 citation coverage is incomplete (ArtEmis shows 204 citing records).

## 5. Limitations

- Table values for VQAScore, SelfEval, Holtzman, Jaini, Pearson, Fathi, Plank, Uma (per dataset), Peterson, Bhattacharya, Stengel-Eskin, Wang and Chan, Xie, SAMNet, Visual Semantic Entropy, PercepT, SEVLM, SemArt, Personality-Captions and V-FLUTE were not re-derived; they remain digest-level as in the bibliographies.
- ArtEmis CVPR PDF and the supplement were not read; genre-level figures cited to ArtEmis could exist in the supplement. I checked the arXiv HTML only.
- Several HTML table headers (DDES Table 3, Sherlock Table 2) parse ambiguously; the rows flagged PLAUSIBLE or CORRECTED rely on column headers as parsed.
- Absence claims are bounded by WebSearch (US-only) and Semantic Scholar citation lists, which are incomplete. The "Retrieving Emotional Stimuli in Artworks" title was not opened.
- I did not check whether ArtELingo-28's 2,000 images equal the 2,000 highest-disagreement ArtELingo set; the local CSV was not opened (the statistics quoted are from the task brief).
- Venue tags rely on arXiv comments, S2 and one WebSearch per item; "oral" tags are unverified.
