# No published design assigns K image embeddings to readings through a masked decoder; the pieces exist apart (DML's masked decoder with Hungarian assignment of captions to codes, CITE's learned phrase-to-embedding assignment, DivE and MaxMatch's set collapse fixes), the NLP lesson warns that fixed K prototypes lose to context-conditioned vectors, and the bar at CLIP B/32 is 40.0 to 40.2 ECCV mAP@R

Date: 2026-10-26. Research-loop step 3 of loop r1, literature check of the user's idea 2 (a PVSE setting with masked sub-space training instead of the K attention heads). Run on the `deep-research` agent, Opus at `high`, deep depth (35 of 40 searches, 17 of 20 page reads), because the area is new to our notes. Earlier notes it builds on: the [lever review](../auto/v1/2026-10-03_lever_review.md) (sections 3.2, 3.4, 7), the [deep-think note](../auto/v1/2026-10-05_cross_masking_polysemy.md) (candidate B) and the [H-b review](../auto/v1/2026-10-06_masked_decoder_readings_review.md) (section 3.5). The brief gave the agent the project facts and two readings of "masked sub-space": (a) K sub-spaces of the embedding dimensions selected by masks; (b) K embeddings each from a different masked view, with the masked reconstruction loss assigning readings to sub-spaces.

## Literature check: "PVSE with masked sub-spaces" (idea 2), MultiMAE loop r1, 2026-10-26

Depth: deep. Budget used: 35 of 40 searches, 17 of 20 pages (3 pages were 403 errors on the CVPR 2026 Lenses PDF and its supplement). I read the three named notes first. Facts already in them (PCME, ProLIP, MERU, HyCoCLIP, FLAIR, LoopITR, ECCV Caption metric conflicts) are reused here and not searched again.

### Goal
1. Which methods give K embeddings or a distribution per image and caption? How do they make the K differ, what do they score on one-to-many metrics at CLIP scale, and how do they fail?
2. Sub-space and masked-dimension embeddings. Has anyone tied a sub-space to a reading? What does NLP's multi-sense history teach?
3. Masking as the source of diversity. Is our collapse of partial-view embeddings a known effect?
4. Assignment signals of the winner-takes-all family, their collapse modes and fixes. Has a masked reconstruction loss ever been the per-hypothesis loss?
5. What is the nearest prior design, which comparator is strongest at CLIP B/32, and which numbers must we beat?

### Answers

**Q1. The multi-embedding family. Confidence: high on the mechanisms, medium on the CLIP-scale numbers.**
- **PVSE** (CVPR 2019) uses K attention heads plus a residual, a MIL loss over the K×K pairs, a Gram-matrix diversity loss (pushes the K local features toward orthogonal) and an MMD loss. Its best K was 3 on COCO and 5 on MRW. It never measured head collapse; heads often attend to the same salient region.
- **DivE** (CVPR 2023) diagnoses two failures:
  - PVSE's max gives "sparse supervision": only one pair gets a gradient.
  - PCME's averaged match probability causes "set collapse".
  - Its fix: slot attention (slots compete for features), a smooth-Chamfer similarity (log-sum-exp), a diversity term and an MMD term.
  - K ablation on Flickr, region features (RSUM): K=1 492.6, K=4 500.8; flat from K=3 to 6.
- **MaxMatch** (ACL 2025) says smooth-Chamfer still collapses. It uses a one-to-one Hungarian set match plus an intra-set divergence loss. Its failure case: visually uniform images give repeated slots. It reports no K sweep and uses ResNet or BERT backbones, no CLIP. A Pith reviewer says the appendix pseudocode omits the matching step (not checked).
- **Llip** (ICML 2024) adds K mixture tokens and pools them by text-conditioned cross-attention. It has no diversity loss. ImageNet zero-shot at B/32 rises with K from 1 to 128. Learned-query averaging alone gave no clear gain over uniform averaging, so the gain comes from text conditioning. It reports no per-token usage and no ECCV or CxC numbers.
- **PCME++** (ICLR 2024): its probabilistic and mean-only inference score the same (from the notes).
- **Lenses** (CVPR 2026) is new since our notes:
  - Learned "lens tokens" give per-lens embeddings, with a "lens-masking similarity" and an intra-set diversity loss "to prevent semantic collapse".
  - The lenses are supervised: 5 annotated categories (Literal, Figurative, Abstract, Background, Emotional) over 105k images and 732k captions.
  - The backbone is an MLLM. Numbers were not checked (abstract only).
- **At CLIP scale, multiple embeddings add about +1 mAP@R** (from the notes). No paper we found reports PVSE or DivE with a CLIP B/32 backbone on ECCV Caption.

**Q2. Sub-spaces and the multi-sense lesson. Confidence: high for NLP, medium for "no cross-modal sense sub-spaces".**
- **Conditional Similarity Networks** (CVPR 2017) learn masks that select and reweight embedding dimensions, one mask per similarity notion. The condition is known at training time.
- **SCE-Net** (ICCV 2019) learns the conditions without labels. It finds that a dynamic condition-weight branch is essential.
- **CITE** (ECCV 2018) is the closest cross-modal case. It learns K image-text embeddings, and a concept-weight branch assigns each phrase to an embedding automatically. Grounding rose by 3 to 4 points over a single embedding.
- **Divide and Conquer** (CVPR 2019) splits the embedding into K neuron groups, each trained on one data cluster, with re-clustering every T epochs.
- **Matryoshka** (NeurIPS 2022, venue not re-checked) nests embedding prefixes for coarse-to-fine granularity, not for senses.
- **SLiCS** (preprint 2025) and **CLAY** (CVPR 2026, training-free text-conditioned similarity) work in CLIP space, without masks or senses.
- **Only Lenses ties sub-spaces to readings in vision-language, and it supervises them.**
- **The NLP lesson:**
  - Li and Jurafsky (EMNLP 2015): "Doubling the number of dimensions is sufficient to increase performance as much as using the complex multi-sense algorithm."
  - Arora et al. (TACL 2018): a word's senses sit in linear superposition inside one vector, and sparse coding recovers them.
  - Mu et al. (ICLR 2017): contexts form low-rank subspaces.
  - Wiedemann et al. (KONVENS 2019): BERT places senses in distinct regions.
  - Camacho-Collados and Pilehvar (survey, 2018): clustered sense vectors are hard to map to concepts and hard to use downstream.
  - Our reading: fixed K prototypes lost to one vector conditioned on context. Llip and FLAIR are the vision-language version of that answer.
  - Our ArtELingo result fits Arora: a linear probe on any frozen embedding already reads emotions as well as our decoder.

**Q3. Masking as the source of diversity. Confidence: high that no one does this on purpose; medium to low on the collapse.**
- In FLIP, A-CLIP, MSN and I-JEPA, masked views are trained to agree with the text or the full view. That is invariance, not diversity.
  - A-CLIP's 2×50% attention-selected views beat random views (Flickr I2T R@1 59.3 against 54.6).
  - DreamLIP (ECCV 2024) aligns sampled sub-captions with groups of image patches. This is the nearest case of "different parts explain different captions", but it keeps one image embedding.
- We found no work where K masked views give K deliberately different embeddings.
- **Our collapse is not reported in this exact form.** These effects are next to it:
  - MAE has dimensional collapse, and a uniformity term fixes it (Zhang et al., NeurIPS 2022, U-MAE).
  - I-JEPA found that pixel-space targets hurt linear probing.
  - MSN avoids the problem by training the masked-view embedding directly.
  - All three point the same way: a pooled partial-view embedding needs its own discriminative or cross-modal loss.

**Q4. Assignment signals. Confidence: high.**
- **MCL and sMCL** (Lee et al. 2016) train an ensemble by an oracle min loss, captioning included: oracle CIDEr about 1.18 at M=4, against about 1.07 to 1.10 for independent networks.
- **Collapse** means few heads win and the rest get no update. The fixes:
  - Relaxed WTA (Rupprecht et al. 2017, ε-gradient): pulls all heads toward the global mean, so it needs tuning.
  - Annealing (aMCL, preprint 2024; venue not checked).
  - Learned scoring heads (rMCL, NeurIPS 2023).
- **Theory** (Letzelter et al., ICML 2024; aMCL): each hypothesis converges to the conditional mean of its Voronoi cell. A head trained under winner-takes-all is the "average reading" of its cell.
- **Precedent: DML** (Chen et al., NeurIPS 2022).
  - Each caption maps to a codebook "mode".
  - A fully masked non-autoregressive decoder rebuilds the caption from mode plus image.
  - The image's captions are Hungarian-matched one-to-one to modes.
  - Without the masked objective, 5 of 64 modes stay in use; without Hungarian matching, 3; full DML, 29.
  - At mask probability 0.5, 6 modes stay in use; at 0.0, 5.
  - So full masking is required. Partial context lets the decoder ignore the code.
  - DML's codes are caption-side and serve controllable generation, not image-side retrieval embeddings.
- Diverse-captioning CVAEs suffer posterior collapse (Aliakbarian et al., preprint), and Wang et al. (NeurIPS 2017) used a mixture prior.

**Q5. Novelty and comparator. Confidence: medium.**
- **No published design matches "PVSE with masked sub-spaces".** The nearest pieces:
  - Assignment by masked reconstruction: DML.
  - Learned assignment of text to K embeddings: CITE and SCE-Net.
  - Set retrieval and its collapse fixes: DivE and MaxMatch.
  - Polysemy framing with an emotional lens: Lenses. This is the largest framing risk.
- **The strongest fair comparator at CLIP B/32**, from the PCME++ paper, Table 1 (COCO fine-tune, 3-run means):

| Method | ECCV mAP@R | CxC R@1 | 5K R@1 | RSUM |
|---|---|---|---|---|
| InfoNCE | 39.0 | 54.9 | 53.0 | 532.6 |
| VSE∞ (GPO pooling) | 40.0 | 57.1 | 55.2 | 536.5 |
| PCME++ | 40.1 | 56.8 | 55.1 | 537.0 |
| PCME++ (SWA) | 40.2 | 56.8 | 55.2 | 537.3 |

- So plain GPO pooling ties PCME++. Our best masked run is 37.94.
- DivE scores 43.5 mean ECCV mAP@R (I2T 36.0, T2I 51.0) and CxC R@1 72.3 / 55.5, on a non-CLIP backbone. Its exact backbone for that table was not checked.
- **Conflicts:**
  - Zero-shot CLIP B/32 scores 26.75 mAP@R in ECCV Caption's table and 26.8 in PCME++'s re-evaluation. ECCV Caption's own text says 28.0.
  - A SOTA2 leaderboard lists PCME++ B/32 at 32.3 with unlabeled columns. This is probably one direction only; not checked.

### Design directions for idea 2 (our reading)

**a. Decoder-assigned image slots** (the notes' D9, closest to reading (b)).
- K slot or learned-query image embeddings from the full image.
- Each caption goes to the slot whose slot-conditioned caption decoder rebuilds it best. Use high masking (DML shows low masking collapses the codes) and Hungarian assignment of an image's captions to slots, or annealed winner-takes-all.
- InfoNCE on the assigned slot; retrieval takes the max over slots.
- Nearest prior: DML, plus DivE and MaxMatch.
- Strongest comparator: the same slots assigned by embedding similarity (MaxMatch-style Hungarian, or DivE smooth-Chamfer), and K=1, all on our recipe. The external bar is VSE∞ or PCME++ at 40.0 to 40.2.

**b. Dimension-masked sub-spaces** (reading (a)).
- K learned masks on the 512-d embedding, in the style of CSN or CITE. The decoder reads the masked embedding, and its reconstruction score assigns the caption.
- Nearest prior: CITE, SCE-Net, Divide and Conquer, and Lenses (supervised).
- Comparators: one vector of the same total size, plus random fixed masks. Li and Jurafsky show that gains can come from capacity alone.
- On ArtELingo the comparator is the linear probe on frozen CLIP (Jensen-Shannon distance 0.265).

**c. Masked-view hypotheses** (reading (b) on the input side).
- K embeddings from K patch subsets; a reading belongs to the view whose decoder rebuilds the caption best.
- Each view must also get a contrastive loss, because our collapse result and U-MAE show pixel-only views degenerate.
- Nearest prior: A-CLIP multi-view, DreamLIP, MSN.
- Comparator: A-CLIP-style attentive views with multi-positive InfoNCE and PVSE's MIL.
- The risk: views tend to encode where in the image, not which reading.

### Sources

Earlier notes: the lever review, the deep-think note and the H-b review (links in the header).

| Title | Authors, year | Venue | URL | Shows (question) |
|---|---|---|---|---|
| Polysemous Visual-Semantic Embedding | Song and Soleymani, 2019 | CVPR | https://arxiv.org/abs/1906.04402 | losses; K best 3 to 5; no collapse check (Q1) |
| Improving Cross-Modal Retrieval with Set of Diverse Embeddings (DivE) | Kim, Kim, Kwak, 2023 | CVPR | https://arxiv.org/abs/2211.16761 | sparse supervision and set collapse; K ablation; ECCV and CxC numbers (Q1, Q5) |
| Maximal Matching Matters | Alomari et al., 2025 | ACL (arXiv 2506.21538) | https://arxiv.org/html/2506.21538 | Hungarian set match, intra-set divergence, failure cases (Q1) |
| Improved Probabilistic Image-Text Representations (PCME++) | Chun, 2024 | ICLR | https://arxiv.org/html/2305.18171v5 | B/32 comparator table, 3 runs (Q5) |
| Modeling Caption Diversity (Llip) | Lavoie et al., 2024 | ICML | https://arxiv.org/abs/2405.00740 | K up to 128, no diversity loss, text-conditioned gain (Q1, Q2) |
| Lenses: Toward Polysemous Vision-Language Understanding | Alomari, Asgarov, Thomas, 2026 | CVPR | https://openaccess.thecvf.com/content/CVPR2026/html/Alomari_Lenses_Toward_Polysemous_Vision-Language_Understanding_CVPR_2026_paper.html | supervised lens slots, emotional lens; abstract only (Q1, Q5) |
| Dynamic Visual Semantic Sub-Embeddings | (not checked), 2023 | preprint | https://arxiv.org/abs/2309.08154 | sub-embeddings, no ECCV (Q1) |
| ECCV Caption | Chun et al., 2022 | ECCV | https://arxiv.org/abs/2204.03359 | zero-shot CLIP 26.75 against 28.0 conflict (Q5) |
| Conditional Similarity Networks | Veit, Belongie, Karaletsos, 2017 | CVPR | https://arxiv.org/abs/1603.07810 | dimension masks per notion (Q2) |
| Learning Similarity Conditions Without Explicit Supervision | Tan et al., 2019 | ICCV | https://arxiv.org/abs/1908.08589 | label-free conditions (Q2) |
| Conditional Image-Text Embedding Networks | Plummer et al., 2018 | ECCV | https://arxiv.org/abs/1711.08389 | learned phrase-to-embedding assignment (Q2, Q5) |
| Divide and Conquer the Embedding Space | Sanakoyeu et al., 2019 | CVPR | https://arxiv.org/abs/1906.05990 | K neuron-group sub-spaces (Q2) |
| Matryoshka Representation Learning | Kusupati et al., 2022 | NeurIPS (not re-checked) | https://arxiv.org/abs/2205.13147 | nested granularity (Q2) |
| SLiCS | (not checked), 2025 | preprint | https://arxiv.org/abs/2508.20322 | concept sub-spaces of CLIP (Q2) |
| CLAY | Lim et al., 2026 | CVPR (per arXiv) | https://arxiv.org/abs/2604.11539 | text-conditional similarity on frozen VLM (Q2) |
| Do Multi-Sense Embeddings Improve NLU? | Li and Jurafsky, 2015 | EMNLP | https://arxiv.org/abs/1506.01070 | doubled dimensions match multi-sense (Q2) |
| Linear Algebraic Structure of Word Senses | Arora et al., 2018 | TACL | https://arxiv.org/abs/1601.03764 | senses superposed in one vector (Q2) |
| Geometry of Polysemy | Mu, Bhat, Viswanath, 2017 | ICLR | https://arxiv.org/abs/1610.07569 | context subspaces (Q2) |
| Does BERT Make Any Sense? | Wiedemann et al., 2019 | KONVENS | https://arxiv.org/abs/1909.10430 | contextual vectors separate senses (Q2) |
| From Word to Sense Embeddings | Camacho-Collados and Pilehvar, 2018 | survey (JAIR, not checked) | https://arxiv.org/abs/1805.04032 | limits of sense clusters (Q2) |
| Scaling Language-Image Pre-training via Masking (FLIP) | Li et al., 2023 | CVPR | https://arxiv.org/abs/2212.00794 | masked views kept invariant (Q3) |
| Attentive Mask CLIP | Yang et al., 2023 | ICCV | https://arxiv.org/abs/2212.08653 | 2×50% attentive views against random (Q3) |
| Masked Siamese Networks | Assran et al., 2022 | ECCV | https://arxiv.org/abs/2204.07141 | masked view matched to full view (Q3) |
| I-JEPA | Assran et al., 2023 | CVPR (not checked) | https://arxiv.org/abs/2301.08243 | pixel targets hurt linear probing (Q3) |
| How Mask Matters (U-MAE) | Zhang, Wang, Wang, 2022 | NeurIPS | https://arxiv.org/abs/2210.08344 | MAE dimensional collapse, uniformity fix (Q3) |
| DreamLIP | Zheng et al., 2024 | ECCV | https://arxiv.org/abs/2403.17007 | sub-caption to patch-group alignment (Q3) |
| Stochastic Multiple Choice Learning | Lee et al., 2016 | NeurIPS (not re-checked) | https://arxiv.org/abs/1606.07839 | oracle loss for captioning ensembles (Q4) |
| Resilient Multiple Choice Learning | Letzelter et al., 2023 | NeurIPS | https://arxiv.org/abs/2311.01052 | collapse definition, ε-WTA trade-off, scoring heads (Q4) |
| Annealed Multiple Choice Learning | Perera, Letzelter et al., 2024 | preprint (venue not checked) | https://arxiv.org/abs/2407.15580 | greedy minima; heads go to their cell means (Q4) |
| Winner-takes-all learners are geometry-aware... | Letzelter et al., 2024 | ICML | https://arxiv.org/abs/2406.04706 | heads as Voronoi quantizers (Q4) |
| Learning Distinct and Representative Modes (DML) | Chen, Deng, Wu, 2022 | NeurIPS | https://arxiv.org/html/2209.08231v2 | masked decoder plus Hungarian assignment; collapse ablation (Q4, Q5) |
| Diverse and Accurate Image Description (AG-CVAE) | Wang, Schwing, Lazebnik, 2017 | NeurIPS | https://arxiv.org/abs/1711.07068 | mixture-prior CVAE (Q4) |
| Posterior collapse in diverse captioning | Aliakbarian et al., 2019 | preprint | https://arxiv.org/abs/1912.08521 | posterior collapse under strong conditioning (Q4) |

### Open points
- **Lenses details** (how captions reach lenses, its baselines and numbers): the CVPR PDF returned 403. Fetch it from another mirror or look for an arXiv version.
- **PVSE or DivE at CLIP B/32 on ECCV Caption**: not found. Search the citing papers of DivE and MaxMatch for "ViT-B/32 ECCV Caption".
- **The ICLR 2025 "many-to-many" paper** that a search hit pointed to (probably ProLIP) was not identified. Open its proceedings PDF.
- **DivE's backbone for its ECCV table**, and whether the SOTA2 leaderboard's 32.3 for PCME++ is one direction only: read DivE Table A1 and the leaderboard's column headers.
- **The exact form of our partial-view collapse** (pooled embeddings of partial views converge under pixel-only loss): not found as such. Search "MAE pooled feature masked input cosine similarity anisotropy".
