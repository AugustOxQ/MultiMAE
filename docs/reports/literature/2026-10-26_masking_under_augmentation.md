# No source trains a masked cross-modal decoder on image blends with either-one caption targets; the pieces exist apart (text-conditioned unmixing, a masked loss on a cross-modal mix), and the field predicts two failure modes: hypothesis collapse and a language prior that resolves the blend without the image

Date: 2026-10-26. Research-loop step 3 of loop r1, literature check of the user's idea 1 (a masked objective under a designed multimodal augmentation). Run on the `deep-research` agent, Opus at `high`, deep depth (40 of 40 searches, 13 of 20 page reads), because the area is new to our notes. Earlier notes it builds on: the [lever review](../auto/v1/2026-10-03_lever_review.md) (sections 3.2, 3.4, 7) and the [H-b review](../auto/v1/2026-10-06_masked_decoder_readings_review.md) (sections 1.4, 3.5, 3.6, 6). The brief, the project facts given to the agent and the questions are in the r1 decide chat (handoff of 2026-10-26). Controller's note on the "conflict" below: the H-b review's "Takatsuki et al. 2026" is a separate probe paper that reads LLaVA-family probabilities on Visual Anagram stimuli, not the Visual Anagrams paper itself (Geng, Park and Owens, CVPR 2024), so the review's attribution stands.

## Literature check: designed masking under multimodal augmentation (MultiMAE loop r1, idea 1)

Depth: deep (opus). Budget used: 40 of 40 web searches, 13 of 20 page reads. I read the named sections of the two earlier notes first and did not repeat what they cover: MixGen, PCME++ mixing, TiMix, geodesic mixup, MixMAE, MixedAE, M3ixup, ProLIP, A-CLIP, COSMOS, PVSE, DivE, Llip and FLAIR.

**Bottom line.** No source I found does what idea 1 does:
- a pixel blend of two images,
- each source caption kept as a separate valid target ("either one"),
- the visible words of a masked caption choosing which source to rebuild,
- and the result scored on one-to-many retrieval.

Two pieces exist, each on its own:
- **Text-conditioned unmixing of alpha blends.** Grounding by Separation, ICCV 2021, used for grounding, not retrieval.
- **A masked language loss on a cross-modal mix on purpose.** VLMixer, ICML 2022: image patches replace words.

Query-conditioned "mix and separate" is standard in audio (Sound of Pixels, CLIPSep). The literature also predicts two failure modes for idea 1:
- the readings collapse into one hypothesis;
- the decoder resolves the blend from the visible words alone (a language prior), not from the image.

### Goal (restated)
1. Which image-text augmentations exist beyond our notes, and which put a masked loss on the augmented sample on purpose?
2. Who has designed a masked or multi-hypothesis loss that must pick one of several valid readings? What failure modes are known?
3. What mechanism arguments exist for or against masked cross-modal training keeping several readings? Is our partial-view collapse finding known?
4. Which datasets and metrics test "several known valid readings"?
5. What is the nearest prior design, and what is the strongest fair comparator?

### Answers

**Q1. Augmentations; masking on the augmented sample on purpose.** Confidence: medium. Each item is checked against its paper's abstract or a table, but the search did not cover every 2024 to 2026 variant.

Masking on the augmented sample, on purpose:
- **VLMixer (ICML 2022).** Cross-modal CutMix replaces grounded words with image patches. The masked language loss then rebuilds the original words from the mixed sequence. The paper calls this "reconstruct the original text from the two types of corruptions". COCO 5K text-to-image retrieval R@1 was 64.76 with full data. An MLM-only row reached 60.18, but the table does not isolate MLM plus CutMix. No one-to-many metric is reported. The data is unpaired, so this is not our setting.
- **Grounding by Separation (ICCV 2021).** Random alpha blends of two images. A segmentation network recovers the alpha map, conditioned on one source's caption. Grounding accuracy rose by up to 8.5 points over the previous detector-free methods. No retrieval or ambiguity metric is reported. This is the closest design to "a caption chooses a source in a blend", but its target is a mask, not caption tokens.
- **MixReorg (ICCV 2023).** Mixes patches across images and keeps each source's text. It adds a segmentation loss on the mixed image and contrastive losses on restored features. Zero-shot segmentation rose by 2.5 to 6.2 mIoU over GroupViT. No retrieval or ambiguity metric is reported.

Masking used by default, or as plain augmentation:
- **FLIP (CVPR 2023).** Masks patches with no reconstruction loss. Adding an MAE loss lowered zero-shot accuracy slightly (69.6 to 69.4 at 50% masking; 68.2 to 67.9 at 75%).
- **MaskCLIP (CVPR 2023).** Distills the full-image representation into the representation predicted from a masked image.
- **MixGen.** Its masked language loss runs on mixed pairs only because ALBEF's recipe was left unchanged (our earlier note).

Augmentations with no masked loss:
- **LaCLIP (NeurIPS 2023), LLM caption rewrites.** ImageNet zero-shot accuracy +8.2 on CC12M and +2.4 on LAION-400M. No one-to-many metric.
- **LeMDA (ICLR 2023).** Learned augmentation in feature space.
- **ITMix.** Fine-tunes CLIP for classification, not retrieval.

None of these reports ECCV Caption mAP@R, PMRP or CxC. The only mixing result on a one-to-many metric is still PCME++'s near-zero change (38.9 to 39.0, earlier note).

**Q2. Masked or multi-hypothesis losses that must pick one reading.** Confidence: medium-high for the precedents and failure modes; medium for "none in vision-language".

Mix-and-separate with a query:
- **Sound of Pixels (ECCV 2018).** Mixes two videos' audio; the visual stream conditions the separation.
- **CLIPSep (ICLR 2023).** The same with CLIP image or text queries. A later slide deck notes a weakness: performance drops when training uses only multi-source data.
- **Grounding by Separation** carries the idea to images with text queries (Q1).

Image-only unmixing exists in MixMAE and MixedAE (earlier note).

Multiple-hypothesis, or winner-takes-all, training:
- **Stochastic multiple choice learning (NeurIPS 2016).** An image-captioning ensemble trained this way produced captions that "capture ambiguity both in language and perception". Independently trained models instead gave near-identical captions. It is scored by oracle (best-of-k) error.
- **Rupprecht et al. (ICCV 2017).** Multiple-hypothesis models beat single-hypothesis ones on four tasks.

Known failure modes:
- **Hypothesis collapse.** Some heads never win and get no updates. The relaxed winner-takes-all loss fixes this but tends to level the heads (as described by Letzelter et al. 2023, a secondary source).
- **A poor local minimum.** Annealed multiple choice learning (NeurIPS 2024) says plain winner-takes-all can converge to an arbitrarily suboptimal local minimum, and fixes it with annealing.
- **Mixture degeneracy (Shen et al., ICML 2019, translation).** Two kinds were observed:
  - D1: independent parameters with online updates, where only one component trains.
  - D2: soft EM, or dropout in the step that assigns samples to components, where the latent variable is ignored.

  The robust recipe was hard EM, a uniform prior and shared parameters, with no dropout in that step.
- **Set collapse (Maximal Matching, ACL 2025).** In set-embedding retrieval, the set elements "become too similar". The fix is a one-to-one matching score plus a within-set divergence loss. COCO RSUM was 446.5 for a single model, below HREM's 451.0, and 452.6 for an ensemble. No ECCV Caption numbers were found.
- **Diverse captioning with latent codes (Seq-CVAE, ICCV 2019).** More diversity at matched quality, scored by best-of-k CIDEr. Pluralistic Image Completion (CVPR 2019) says a conditional VAE gives little diversity when there is only one ground truth per input.

I found no masked-token loss in vision-language that uses a winner-takes-all or minimum-over-captions target. One untargeted search found nothing.

**Q3. Mechanism arguments; is the collapse finding known?** Confidence: medium. The arguments come from theory papers; none was tested on our setting.

Arguments against the pooled embedding keeping several readings:
- **Shared content only.** Multimodal contrastive learning identifies only the latent content that the two modalities share (Daunhawer et al., ICLR 2023). The multi-view information bottleneck treats content that is not shared as superfluous (Federici et al., ICLR 2020). A reading that only some captions express is therefore what InfoNCE learns to drop. Readings are more likely to live in the decoder's output than in the pooled embedding.
- **Limits of one vector.** One vector scored by dot products cannot put high scores on every valid next word (softmax bottleneck, Chang and McCallum, ACL 2022). The number of top-k sets a single-vector retriever can return is capped by its dimension (Weller et al., ICLR 2026).
- **Incoherent conditionals.** Masked-language-model conditionals often fit no single joint distribution (Young et al., preprint 2023). Readings read off different visible subsets may therefore disagree with each other.

Arguments for, already in the earlier note: cross-entropy is a proper scoring rule, so the decoder learns a mixture over the contexts it sees. The pixel decoder's MSE loss predicts a mean.

Is the partial-view collapse known? Partly. Close results:
- **FLIP.** Inference on masked images loses zero-shot accuracy against full images: 66.4 vs 69.6 at 50% masking, 60.9 vs 68.2 at 75%. A short unmasked tuning phase recovers part of it (+0.5 and +1.3).
- **MAE theory (Zhang et al., NeurIPS 2022).** MAE features show dimensional collapse; a uniformity term fixes it.
- **The usual fix.** Train the masked view's embedding to match the full view's: Masked Siamese Networks (ECCV 2022), MaskCLIP, and A-CLIP (earlier note).

Our specific finding, that pooled partial-view embeddings converge (cosine 0.97 across different images) when the tower meets partial views only through pixel reconstruction, was not found as a measurement. It fits these results, so we should call it expected rather than new.

**Q4. Datasets and metrics.** Confidence: medium-high for the datasets I checked; low for the two named under "not checked".

Datasets with several known readings:
- **Constructed with exact readings:**
  - alpha blends (Grounding by Separation; our blend probe);
  - Visual Anagrams (Geng, Park and Owens, CVPR 2024): one image, two readings under flips, rotations or jigsaw permutations.
- **Natural ambiguity:**
  - bistable images (Panagopoulou et al., CMCL workshop at ACL 2024). Only 29 images. Most models strongly preferred one reading, and prompt wording moved their answers more than image changes did.
  - SQUID-E (NeurIPS 2022 Datasets and Benchmarks): human uncertainty judgments on a test set.
- **Label distributions from many annotators:**
  - CIFAR-10H (ICCV 2019);
  - ArtEmis (CVPR 2021) and ArtELingo (EMNLP 2022). ArtELingo has more than 51K artworks with 5 or more annotations in 3 languages.
- **Many-to-many caption sets:** ECCV Caption and CxC (earlier notes).
- **Not checked, search snippets only:** an ambiguous-image VLM benchmark titled "What Do VLMs See?" (OpenReview, ICLR 2026 submission); "Looking Beyond the One" (ACL 2026), on VQA questions with several valid answers.

Metrics:
- **MAP@R and R-Precision** (Musgrave et al., ECCV 2020). MAP@R rewards ranking all R positives first; R-Precision ignores their order.
- **Coverage, scored as oracle or best-of-k** (stochastic multiple choice learning, Seq-CVAE).
- **Distribution agreement with human votes:** JSD or cross-entropy against soft labels, as in our ArtELingo readout.
- **Calibration** (SQUID-E).

**Q5. Nearest design and comparator.** Confidence: medium. "Not found" rests on one search budget.

Nearest prior designs, with how each differs from ours:
- **Grounding by Separation:** same blend and text-selects-source logic, but the target is an alpha map, and it is for grounding.
- **VLMixer:** a masked language loss on a cross-modal mix on purpose, but no blend and no "either one" targets.
- **CLIPSep / Sound of Pixels:** the same principle with query-conditioned separation, in audio.
- **MixGen's single-caption variant:** one random source caption as the only target. It scored at baseline level (COCO RSUM 470.0 vs 470.7, earlier note). Our design differs because the visible words select the source.

Strongest fair comparators:
1. **PCME++ with its mixing.** The strongest one-to-many baseline at CLIP B/32 (earlier note).
2. **MixGen on our CLIP fine-tune**, with the same mix rate. This is the direct augmentation control.
3. **A plain captioner trained on blends with a source prefix.** Same selector, no masking. This control tests the user's point that masking must be tied to the claim, and the literature suggests prefix conditioning alone can steer captions (FlexCap, search snippet only).
4. **Our masked decoder without blends** (the 80% text-masking arm).

### Design directions the literature suggests
- **a. Text-selected unmixing through the masked caption (idea 1 as it stands).**
  - Nearest prior work: Grounding by Separation, VLMixer, CLIPSep.
  - Strongest comparators: comparators 3 and 2 above; PCME++ for retrieval.
  - Must-have control: the visible words of A with an image that has no A in it. This tests whether the decoder reads the image or only the words; the bistable-image study found that language priors dominate.
- **b. Multiple-hypothesis masked prediction.**
  - K decoder queries, trained with hard-EM winner-takes-all over an image's caption set (or over both sources of a blend), with shared parameters and a uniform prior (Shen et al.). Annealing if heads die.
  - Score it by coverage or oracle on ECCV Caption and by agreement with ArtELingo vote distributions.
  - Nearest prior work: stochastic multiple choice learning captioning, Shen et al.'s hard-EM mixture, set embeddings (PVSE, Maximal Matching).
  - Strongest comparators: PCME++; a set-embedding method (DivE or Maximal Matching); a latent-code captioner.
- **c. Make partial views carry readings: masked-to-full or masked-to-set alignment.**
  - Train the pooled embedding of a masked view (or of a blend) to stay close to every compatible caption. This is an inclusion-style "less specific view", and it directly targets our collapse.
  - Nearest prior work: Masked Siamese Networks, MaskCLIP, ProLIP, A-CLIP.
  - Strongest comparators: FLIP-style masking with unmasked tuning; PCME++.

### Conflict with an earlier note
The 2026-10-06 review credits Visual Anagrams to "Takatsuki et al." The paper is Geng, Park and Owens, CVPR 2024. No "Takatsuki" probe paper turned up. Either the note confuses the names or it means a separate paper I did not find. Please check that note's reference list. *(Controller: resolved in the header above; the review cites Takatsuki et al. as the probe paper on Visual Anagram stimuli, not as the stimuli's source.)*

### Sources
| Title | Authors, year | Venue | URL | Shows (question) |
|---|---|---|---|---|
| VLMixer: Unpaired VLP via Cross-Modal CutMix | Wang et al. 2022 | ICML (peer-reviewed) | https://arxiv.org/abs/2206.08919 ; https://ar5iv.labs.arxiv.org/html/2206.08919 | MLM on patch-for-word mixed text; COCO R@1 (Q1, Q5) |
| Detector-Free Weakly Supervised Grounding by Separation | Arbelle et al. 2021 | ICCV (peer-reviewed) | https://openaccess.thecvf.com/content/ICCV2021/html/Arbelle_Detector-Free_Weakly_Supervised_Grounding_by_Separation_ICCV_2021_paper.html | Alpha blend + caption-conditioned separation (Q1, Q2, Q5) |
| MixReorg | Cai et al. 2023 | ICCV (peer-reviewed) | https://arxiv.org/abs/2308.04829 | Cross-image patch mixing with text (Q1) |
| Improving CLIP Training with Language Rewrites (LaCLIP) | Fan et al. 2023 | NeurIPS (peer-reviewed) | https://arxiv.org/abs/2305.20088 | LLM caption rewrites (Q1) |
| Learning Multimodal Data Augmentation in Feature Space (LeMDA) | Liu et al. 2023 | ICLR (peer-reviewed) | https://arxiv.org/abs/2212.14453 | Learned multimodal augmentation (Q1) |
| Scaling Language-Image Pre-training via Masking (FLIP) | Li et al. 2023 | CVPR (peer-reviewed) | https://ar5iv.labs.arxiv.org/html/2212.00794 | Masked vs full inference gap; MAE loss no help; unmasked tuning (Q1, Q3) |
| MaskCLIP | Dong et al. 2023 | CVPR (peer-reviewed) | https://arxiv.org/abs/2208.12262 | Masked-to-full distillation in CLIP (Q3, direction c) |
| Masked Siamese Networks | Assran et al. 2022 | ECCV (peer-reviewed) | https://arxiv.org/abs/2204.07141 | Masked view matched to unmasked view (Q3) |
| How Mask Matters (U-MAE) | Zhang, Wang, Wang 2022 | NeurIPS (peer-reviewed) | https://arxiv.org/abs/2210.08344 | MAE dimensional collapse, uniformity fix (Q3) |
| The Sound of Pixels | Zhao et al. 2018 | ECCV (peer-reviewed) | https://openaccess.thecvf.com/content_ECCV_2018/html/Hang_Zhao_The_Sound_of_ECCV_2018_paper.html | Mix-and-separate (Q2) |
| CLIPSep | Dong et al. 2023 | ICLR (peer-reviewed) | https://arxiv.org/abs/2212.07065 | Text/image-queried separation; multi-source weakness from ICLR 2024 slides (Q2) |
| Stochastic Multiple Choice Learning | Lee et al. 2016 | NeurIPS (peer-reviewed) | https://arxiv.org/abs/1606.07839 | Captioning ensemble that captures ambiguity; oracle loss (Q2, Q4) |
| Learning in an Uncertain World | Rupprecht et al. 2017 | ICCV (peer-reviewed) | https://arxiv.org/abs/1612.00197 | Multiple-hypothesis meta loss; relaxed winner-takes-all (per a 2023 secondary source) (Q2) |
| Annealed Multiple Choice Learning | Perera, Letzelter et al. 2024 | NeurIPS (peer-reviewed) | https://arxiv.org/abs/2407.15580 | Winner-takes-all local minima; annealing fix (Q2) |
| Mixture Models for Diverse MT: Tricks of the Trade | Shen et al. 2019 | ICML (peer-reviewed) | https://ar5iv.labs.arxiv.org/html/1902.07816 | D1/D2 degeneracy and the robust recipe (Q2, direction b) |
| Maximal Matching Matters | Alomari et al. 2025 | ACL (peer-reviewed) | https://aclanthology.org/2025.acl-long.1533 ; https://www.alphaxiv.org/abs/2506.21538 | Set collapse in set-based retrieval; COCO RSUM (Q2, Q5) |
| Sequential Latent Spaces for Diverse Captioning (Seq-CVAE) | Aneja et al. 2019 | ICCV (peer-reviewed) | https://arxiv.org/abs/1908.08529 | Latent-code diverse captioning; best-of-k (Q2, Q4) |
| Pluralistic Image Completion | Zheng et al. 2019 | CVPR (peer-reviewed) | https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Pluralistic_Image_Completion_CVPR_2019_paper.html | Masked completion with many plausible fills; CVAE low diversity (Q2) |
| Identifiability Results for Multimodal Contrastive Learning | Daunhawer et al. 2023 | ICLR (peer-reviewed) | https://arxiv.org/abs/2303.09166 | Contrastive learning keeps shared content only (Q3) |
| Multi-View Information Bottleneck | Federici et al. 2020 | ICLR (peer-reviewed) | https://arxiv.org/abs/2002.07017 | Content that is not shared is superfluous (Q3) |
| Softmax Bottleneck ... Multi-mode Word Distributions | Chang, McCallum 2022 | ACL (peer-reviewed) | https://aclanthology.org/2022.acl-long.554 | One vector cannot score all modes (Q3) |
| On the Theoretical Limitations of Embedding-Based Retrieval | Weller et al. 2025/26 | ICLR 2026 (peer-reviewed) | https://arxiv.org/abs/2508.21038 | Dimension caps the top-k sets (Q3) |
| Inconsistencies in Masked Language Models | Young, Chen, You 2023 | preprint | https://arxiv.org/abs/2301.00068 | MLM conditionals fit no single joint (Q3) |
| Visual Anagrams | Geng, Park, Owens 2024 | CVPR (peer-reviewed) | https://arxiv.org/abs/2311.17919 | Images with exactly two readings (Q4) |
| Evaluating VLMs on Bistable Images | Panagopoulou et al. 2024 | CMCL workshop, ACL (peer-reviewed workshop) | https://aclanthology.org/2024.cmcl-1.2 | Models pick one reading; language prior dominates (Q4, direction a) |
| Ambiguous Images With Human Judgments (SQUID-E) | Sanders et al. 2022 | NeurIPS Datasets and Benchmarks (peer-reviewed) | https://arxiv.org/abs/2210.03102 | Ambiguous images with human uncertainty (Q4) |
| Human Uncertainty Makes Classification More Robust (CIFAR-10H) | Peterson et al. 2019 | ICCV (peer-reviewed) | https://mlanthology.org/iccv/2019/peterson2019iccv-human/ | Label distributions from many annotators (Q4) |
| ArtEmis / ArtELingo | Achlioptas et al. 2021; Mohamed et al. 2022 | CVPR; EMNLP (peer-reviewed) | https://arxiv.org/abs/2101.07396 ; https://arxiv.org/abs/2211.10780 | Emotion votes per painting (Q4) |
| A Metric Learning Reality Check | Musgrave et al. 2020 | ECCV (peer-reviewed) | https://arxiv.org/abs/2003.08505 | MAP@R and R-Precision (Q4) |
| MixGen (follow-up search) | Hao et al. 2023 | WACV workshop | https://arxiv.org/abs/2206.08358 | No direct CLIP-retrieval follow-up found (Q5) |

### Open points
1. **A masked-token loss with a minimum-over-captions or winner-takes-all target in vision-language.** One search found nothing. It would be settled by a Semantic Scholar search on the papers that cite stochastic multiple choice learning and Shen et al. 2019, filtered to vision-language.
2. **Grounding by Separation's full method**, especially its failure modes and whether text conditioning was ever ignored. Only the abstract was read; reading the ICCV PDF's method and ablation sections would settle it.
3. **Two benchmarks known from snippets only**: "What Do VLMs See?" and "Looking Beyond the One". Their OpenReview and ACL Anthology pages would settle what they contain.
4. **Maximal Matching on ECCV Caption mAP@R.** No numbers were found; its appendix or a citing paper may report them.
5. **2025 to 2026 image-text mixing for CLIP retrieval.** RobustMixGen (Neurocomputing 2025) and ARMADA were seen only in reference lists. A direct search for each would settle whether they report one-to-many metrics.
6. **The "Takatsuki et al." attribution** in the 2026-10-06 note (resolved by the controller, see the header).
