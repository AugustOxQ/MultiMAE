import sys, os; sys.path.insert(0, os.getcwd())
import torch
from src.model import MultiModalMAE
from src.metrics.losses import calculate_mae_loss, calculate_mlm_loss
for bv, sz in [(None,32),(None,48),(None,64),(None,224)]:
    m = MultiModalMAE(image_size=sz, patch_size=16, emb_dim=192, encoder_layer=12, encoder_head=3, mask_ratio=0.75,
        backbone_vision=bv, text_backbone="bert-base-uncased", proj_dim=128)
    img = torch.randn(4,3,sz,sz); ids = torch.randint(1000,2000,(4,32))
    out, mask = m.vision(img)
    print(bv, sz, "mae out finite", torch.isfinite(out).all().item(), "loss", calculate_mae_loss(out,img,mask,mask_ratio=0.75).item())
    o,mm = m.text(ids); print(" mlm", calculate_mlm_loss(o,ids,mm).item())
    for n,t in zip(["img","txt"], [m.encode_image_split(img), m.encode_text_split(ids)]):
        print(" ",n,[torch.isfinite(x).all().item() for x in t])
