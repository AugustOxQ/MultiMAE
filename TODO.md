# TODO

Project development tracking and planning.

## 🎯 Current Sprint

### High Priority
- [x] Complete CLIP backbone support
- [x] Add Cluster support (main_fusion_mmae.py)

### Medium Priority

### Low Priority
- [x] Add Multi-gpu/node support
- [ ] Unifying all model supports format


## 📅 This Week's Progress (Sep 01-05)

### ✅ Completed
- [x] Add separate multi-tasl learner head.

### ⏸️ In Progress
- [ ] Test CLIP-ViT-B/16 ability
- [ ] Optimize for multimodal reconstruction 
- [ ] Add a version to compare contrasting masked CLS / pooled features  # currently we doing full feature contrast
- [ ] Add back img shuffle in the reconstruction
- [ ] Multigpu error with eval script, fix it.

### ❌ Blocked/Postponed


### 📓 Quick Log

2025-08-26
- Need to double check accelerate's working condition on eval_fusionmmae
- ‼️ Super slow contrastive loss decrease speed when only CLS token is used in contrastive training

2025-08-29
- Add two stages training strategy

2025-09-02
- Add transformer learner head to perform multi-task learning.

image -> encoder -> masked image feature ->                         -> Image learner -> 
                                                                                        MAE learner -> image decoder -> reconstruction loss
                                            fusion -> Fused feature -> Joint learner ->
                                                                                        MLM leaner -> text decoder -> reconstruction loss
text ->  encoder -> masked text feature  ->                         -> Text learner  ->

## 📋 Backlog

- [x] Add CLIP support (Huggingface format)
- [x] Training strategy add: two stages training (first reconstruct, then contrastive)
- [x] Optimize FusionMMAE decoder
- [x] Optimize FusionMMAE decoder

## Milestones

### Basic Code Created (? ~ 2025-08-25)
- Complete single modality MAE and MLM code
- Complete basic ViT + BERT MMAE training code

### Add CLIP support (2025-08-26)
- Create repo and make it public
- Add CLIP support