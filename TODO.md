# TODO

Project development tracking and planning.

## 🎯 Current Sprint (Week 35 - Aug 26-30, 2025)

### High Priority
- [x] Complete CLIP backbone support

### Medium Priority
- [ ] Unifying all model supports format
- [ ] Now main_fusion_mmae_clip is the focused main file, after all basic changes are made, also apply changes to other main files.
- [ ] Add Cluster support (main_fusion_mmae.py)

### Low Priority
- [ ] Add Multi-gpu support


## 📅 This Week's Progress (Aug 19-25)

### ✅ Completed
- [x] Add CLIP support (Huggingface format)

### ⏸️ In Progress
- [ ] Test CLIP-ViT-B/16 ability
- [ ] Optimize for multimodal reconstruction 
- [ ] Add accelerate support，先在main_fusion_mmae.py实现

### 📓 Quick Log

2025-08-27

- Need to double check accelerate's working condition on eval_fusionmmae
- ‼️ Super slow contrastive loss decrease speed when only CLS token is used in contrastive training

### ❌ Blocked/Postponed


<!-- ## 🚀 Next Week Goals (Sep 2-8)

### Must Have

### Should Have

### Could Have

## 📋 Backlog -->

## Milestones

### Repo Created (2025-08-26)
- Complete basic ViT + BERT MMAE training code
- Create repo and make it public

## 📝 Notes