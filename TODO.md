# TODO

Project development tracking and planning.

## 🎯 Current Sprint

### High Priority
- [x] Complete CLIP backbone support
- [x] Add Cluster support (main_fusion_mmae.py)

### Medium Priority
- [ ] Now main_fusion_mmae_clip is the focused main file, after all basic changes are made, also apply changes to other main files.

### Low Priority
- [ ] Add Multi-gpu/node support
- [ ] Unifying all model supports format


## 📅 This Week's Progress (Aug 26-30)

### ✅ Completed
- [x] Add CLIP support (Huggingface format)

### ⏸️ In Progress
- [ ] Test CLIP-ViT-B/16 ability
- [ ] Optimize for multimodal reconstruction 
- [ ] Add accelerate support，first in main_fusion_mmae.py
- [ ] Add a version to compare contrasting masked CLS / pooled features  # currently we doing full feature contrast
- [ ] Optimize FusionMMAE decoder
- [ ] Add back img shuffle in the reconstruction
- [ ] Training strategy add: two stages training (first reconstruct, then contrastive)
- [ ] 

### 📓 Quick Log

2025-08-26
- Need to double check accelerate's working condition on eval_fusionmmae
- ‼️ Super slow contrastive loss decrease speed when only CLS token is used in contrastive training

2025-08-27

### ❌ Blocked/Postponed


<!-- ## 🚀 Next Week Goals (Sep 2-8)

### Must Have

### Should Have

### Could Have

## 📋 Backlog -->

## Milestones

### Basic Code Created (? ~ 2025-08-25)
- Complete single modality MAE and MLM code
- Complete basic ViT + BERT MMAE training code

### Add CLIP support (2025-08-26)
- Create repo and make it public
- Add CLIP support