# Three candidate ideas for loop r1, from the two literature checks and our own results: decoder-assigned image slots, text-selected unmixing of blends, and a context-conditioned image embedding trained through the masked decoder

Date: 2026-10-26. Research-loop step 3's note (loop r1): the candidates for the user's choice at step 4. Sources: the two literature checks of today ([masking under augmentation](../../literature/2026-10-26_masking_under_augmentation.md), [masked sub-spaces](../../literature/2026-10-26_masked_subspace_pvse.md)), the direction log's parked ideas, and our results ([H-b report](2026-10-08_hb_emotion_readout.md), [Stages 1 and 2](2026-10-06_masking_stages_1_2.md)). The user's principle for this loop: the masked objective must be tied to the polysemy claim by construction, since a masked loss as a plain auxiliary loss is ill-defined as a test.

## What our results and the field say, in four lines

- One pooled vector per image carries the emotion spread as well as any decoder did (the linear probe, JSD 0.265), and theory says a single vector cannot hold several readings for retrieval (contrastive learning keeps only shared content; the softmax bottleneck; the dimension cap of single-vector retrieval). A polysemy gain in retrieval needs more than one vector per image, or a vector conditioned on the query's context.
- The masked decoder is steerable by context: visible words move its reading (D7, D2's selection index). That is a masking-specific property, and the selector the designs below need. Its known failure is that the words alone may do the work (D7's null image; the bistable-image study).
- DML (NeurIPS 2022) shows that a fully masked caption decoder with Hungarian assignment of an image's captions to K codes keeps 29 of 64 codes in use, against 5 or 6 when masking is partial or absent. Masking is the knob that makes a code carry a reading. No one has used this for image-side retrieval embeddings.
- Partial views carry nothing unless a discriminative or cross-modal loss trains them (our collapse; U-MAE, MSN). K readings cannot come from K pixel-reconstructed views alone.

## The candidates

### 1. Decoder-assigned image slots (idea 2, reading b; the parked candidate B of 2026-10-05)

- **Design.** K slot embeddings per image (learned queries over the image tokens, as in DivE and Llip). Each of an image's captions goes to the slot whose slot-conditioned masked caption decoder rebuilds it best (Hungarian assignment over the image's captions × K, or annealed winner-takes-all). InfoNCE trains the assigned (caption, slot) pair; retrieval scores the max or smooth-Chamfer over slots. The masking ratio is the design knob: at 80% the slot must carry the reading; at 15% DML predicts the slots collapse.
- **Why masking is tied to the claim.** The masked decoder is the assignment signal; the prediction that slot use falls with the masking ratio is the test of the mechanism, and its control (the same slots assigned by embedding similarity, as in MaxMatch or DivE) isolates what masking adds.
- **What we measure.** ECCV Caption mAP@R and R-Precision (coverage of the valid matches), slot usage and diversity, and on blends (we have the D2 code) whether slot j retrieves source A's captions and slot k source B's. On ArtELingo, whether different slots retrieve different annotators' captions.
- **Nearest prior work; comparator.** DML (caption-side codes for generation), CITE (learned phrase-to-embedding assignment), DivE and MaxMatch (sets assigned by similarity), Lenses (CVPR 2026, supervised slots with an emotional lens: the framing risk). The external bar at CLIP B/32 is VSE∞ or PCME++ at 40.0 to 40.2 mAP@R (our InfoNCE baseline is 37.11 on our recipe, so the fair test is within our recipe).
- **Risks.** Slot collapse (the known fixes: Hungarian or annealed assignment, shared parameters, no dropout in the assignment step); slots that encode "where" instead of "which reading"; the NLP lesson that fixed K prototypes lose to a context-conditioned vector (candidate 3 is that alternative).
- **Cost.** Moderate code (slot module, assignment, slot-conditioned decoder; about 300 to 500 lines plus tests), about 4 arms × 3 seeds × 3 to 7 GPU-hours, about 60 GPU-hours. Novelty risk: medium.

### 2. Text-selected unmixing of blends through the masked caption (idea 1; the parked D10)

- **Design.** Blend two images (a share of each batch); keep each source caption as a separate target; the masked caption decoder reads the blend and the visible words and must rebuild the hidden words of the right source; the contrastive loss treats both captions as positives of the blend (or weights them by the blend ratio, as PCME++ does).
- **Why masking is tied to the claim.** The visible words select the reading, and the hidden words are the target the image must help with. Without masking there is no selector.
- **What we measure.** On blends: both-covered and balance (the D2 probes), and the must-have control from the literature: the visible words of A with an image that holds no A, which tells whether the decoder reads the image or the words. On natural data: ECCV mAP@R, PMRP, and ArtELingo coverage.
- **Nearest prior work; comparator.** Grounding by Separation (ICCV 2021: a caption selects a source in an alpha blend, target an alpha map), VLMixer (ICML 2022: a masked loss on a cross-modal mix), CLIPSep (audio). Comparators: MixGen on our fine-tune, a prefix captioner on blends without masking (the same selector, no masking), PCME++ with its mixing.
- **Risks.** The language prior resolves the blend without the image (our D7 null-image problem, in a new form); blends are out of distribution for natural retrieval; the prior that mixing moves a natural one-to-many metric is weak (PCME++'s mixing row 38.9 to 39.0; MixGen's single-caption variant at baseline).
- **Cost.** Small code (the blend code exists from D2; the either-one target and the multi-positive loss are small), about 5 arms × 3 seeds, about 60 to 75 GPU-hours. Novelty risk: low for the design; the open question is whether it moves a natural metric.

### 3. A context-conditioned image embedding trained through the masked decoder (the NLP lesson applied)

- **Design.** No fixed K. The visible words of a masked caption pool the image tokens (cross-attention, as in Llip) into one embedding that must both rebuild the hidden words and match the caption contrastively. At retrieval the full query caption conditions the pooling. Masking by construction: the conditioning context is the partial caption, so the embedding learns to resolve the reading from context.
- **Why masking is tied to the claim.** The same embedding serves the masked reconstruction and the match; the masking ratio sets how much of the reading the context gives and how much the image must supply.
- **What we measure.** ECCV mAP@R and R-Precision; on ArtELingo, whether the conditioned embedding retrieves the annotator's own caption better than the unconditioned one; a no-image control for the language prior.
- **Nearest prior work; comparator.** Llip (ICML 2024: text-conditioned pooling, no masking; its gain comes from the conditioning), FLAIR, DreamLIP (sub-captions aligned to patch groups). Comparators: Llip-style pooling without the masked decoder, and ML-80.
- **Risks.** Thin novelty next to Llip and FLAIR; text-conditioned pooling costs a pass per query-image pair at retrieval (Llip pays this too); the masked decoder may add nothing over the conditioning, as it added nothing over the probe in D4.
- **Cost.** Moderate code (conditioned pooling in the retrieval path, which today bypasses the fusion module), about 3 arms × 3 seeds, about 45 GPU-hours. Novelty risk: medium to high.

## Parked ideas considered and not carried

The 2×2 on ArtELingo, the autoregressive captioner for D7 and the emotion-loss checkpoint rerun test the old frame, which the user set aside; they stay parked. The collapse correlate (a study of the existing checkpoints) is an exploration that fits beside any candidate.

<details>
<summary>Our recommendation (open after choosing)</summary>

**Our view: candidate 1 first.** It follows from the two things we know best: one vector cannot hold several readings, and the masked decoder is steerable by context. DML gives a tested recipe in which masking is the knob that makes a slot carry a reading, and the design has a clean control (the same slots assigned by embedding similarity) that isolates masking's part. Candidate 2's blends serve it as the test bed where the two readings are known exactly, so the first stage can combine them: slots trained on natural captions, probed on blends with the D2 code. Candidate 2 alone risks the language-prior failure and a null on natural metrics, as the mixing rows in the literature suggest. Candidate 3 is the strongest alternative if the K-slot structure fails, and its comparator (Llip) is close enough that novelty is the main risk.

</details>
