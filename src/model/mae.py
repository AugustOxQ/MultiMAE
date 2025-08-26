import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block, resize_pos_embed


def random_indexes(size: int):
    # Generate random permutation of indexes
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    # Generate inverse permutation
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    # sequences shape: (T, B, C)
    # indexes shape: (T, B)
    # Output shape: (T, B, C)
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        # patches shape: (T, B, C)
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)

        # Shuffle and mask patches
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]  # Keep only unmasked patches

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        num_layer=12,
        num_head=3,
        mask_ratio=0.75,
    ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2, 1, emb_dim)
        )
        self.shuffle = PatchShuffle(mask_ratio)

        # Convert image patches to embedding dimension
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img):
        # img shape: (B, 3, H, W)
        patches = self.patchify(img)  # Shape: (B, emb_dim, H/patch_size, W/patch_size)
        patches = rearrange(
            patches, "b c h w -> (h w) b c"
        )  # Shape: (T, B, C) where T = H*W/patch_size^2
        patches = patches + self.pos_embedding  # Add positional embedding

        # Shuffle and mask patches
        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        # Add cls token
        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )
        patches = rearrange(patches, "t b c -> b t c")  # Shape: (B, T+1, C)
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")  # Shape: (T+1, B, C)

        return features, backward_indexes


class TimmViTEncoder(torch.nn.Module):
    def __init__(
        self, backbone: str = "vit_tiny_patch16_224", mask_ratio: float = 0.75
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.shuffle = PatchShuffle(mask_ratio)
        # default grid from the pretrained model's image size
        self.old_grid = self.vit.patch_embed.grid_size
        self.patch_size = self.vit.patch_embed.patch_size

    def forward(self, img: torch.Tensor, return_unmasked_features: bool = False):
        # img: (B, 3, H, W)
        x = self.vit.patch_embed(img)  # (B, N, C)
        B, N, C = x.shape

        # compute target grid size based on current input spatial dims
        if isinstance(self.patch_size, tuple):
            ph, pw = self.patch_size
        else:
            ph, pw = self.patch_size, self.patch_size
        new_grid = (img.shape[-2] // ph, img.shape[-1] // pw)

        pos = self.vit.pos_embed  # (1, 1+N, C)
        if pos.shape[1] != N + 1 or new_grid != self.old_grid:
            pos = resize_pos_embed(pos, self.old_grid, new_grid)

        # add positional encoding to patch tokens (exclude cls)
        x = x + pos[:, 1:, :]

        # shuffle & mask patch tokens
        x = rearrange(x, "b t c -> t b c")
        x, fwd_idx, bwd_idx = self.shuffle(x)

        # prepend cls token with its positional embedding
        cls = (self.vit.cls_token + pos[:, :1, :]).expand(B, -1, -1)

        cls = rearrange(cls, "b t c -> t b c")  # t=1
        x = torch.cat([cls, x], dim=0)  # (T+1, B, C)

        # transformer blocks + norm
        x = rearrange(x, "t b c -> b t c")
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        x = rearrange(x, "b t c -> t b c")

        return x, bwd_idx

    def encode_image_clean(self, images: torch.Tensor) -> torch.Tensor:
        """获取完整的、有序的图像特征，用于对比学习"""
        with torch.no_grad():  # 如果不需要训练这个路径
            x = self.vit.patch_embed(images)  # (B, N, C)
            B, N, C = x.shape

            # 添加位置编码
            pos = self.vit.pos_embed
            if pos.shape[1] != N + 1:
                pos = resize_pos_embed(pos, self.old_grid, (N, N))

            x = x + pos[:, 1:, :]

            # 添加CLS token
            cls = (self.vit.cls_token + pos[:, :1, :]).expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, N+1, C)

            # 通过transformer blocks
            for blk in self.vit.blocks:
                x = blk(x)
            x = self.vit.norm(x)  # (B, N+1, C)

            return x  # 返回完整的图像特征


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        num_layer=4,
        num_head=3,
    ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim)
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size**2)
        self.patch2img = Rearrange(
            "(h w) b (c p1 p2) -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            h=image_size // patch_size,
        )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features, backward_indexes):
        # features shape: (T, B, C)
        # backward_indexes shape: (T', B) where T' is the total number of patches
        T = features.shape[0]
        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )
        features = take_indexes(features, backward_indexes)  # Unshuffle patches
        features = features + self.pos_embedding  # Add positional embedding

        features = rearrange(features, "t b c -> b t c")  # Shape: (B, T', C)
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")  # Shape: (T', B, C)
        features = features[1:]  # Remove cls token

        patches = self.head(features)  # Shape: (T'-1, B, 3*patch_size^2)
        mask = torch.zeros_like(patches)
        mask[T - 1 :] = 1  # Create binary mask for original masked patches
        mask = take_indexes(mask, backward_indexes[1:] - 1)  # Unshuffle mask
        img = self.patch2img(patches)  # Shape: (B, 3, H, W)
        mask = self.patch2img(mask)  # Shape: (B, 3, H, W)

        return img, mask


class MAE_ViT(torch.nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        encoder_layer: int = 12,
        encoder_head: int = 3,
        decoder_layer: int = 4,
        decoder_head: int = 3,
        mask_ratio: float = 0.75,
        backbone: str | None = None,
    ) -> None:
        super().__init__()

        self.mask_ratio = mask_ratio

        if backbone and len(str(backbone)) > 0:
            # use pretrained timm ViT as encoder
            self.encoder = TimmViTEncoder(backbone=backbone, mask_ratio=mask_ratio)
            # derive emb_dim & patch_size for decoder from backbone
            backbone_emb_dim = getattr(self.encoder.vit, "embed_dim", emb_dim)
            ps = (
                self.encoder.patch_size[0]
                if isinstance(self.encoder.patch_size, tuple)
                else self.encoder.patch_size
            )

            self.decoder = MAE_Decoder(
                image_size=image_size,
                patch_size=ps,
                emb_dim=backbone_emb_dim,
                num_layer=decoder_layer,
                num_head=decoder_head,
            )
        else:
            # fallback to lightweight custom encoder
            self.encoder = MAE_Encoder(
                image_size=image_size,
                patch_size=patch_size,
                emb_dim=emb_dim,
                num_layer=encoder_layer,
                num_head=encoder_head,
                mask_ratio=mask_ratio,
            )

            self.decoder = MAE_Decoder(
                image_size=image_size,
                patch_size=patch_size,
                emb_dim=emb_dim,
                num_layer=decoder_layer,
                num_head=decoder_head,
            )

    def forward(self, img):
        # img shape: (B, 3, H, W)
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask


class ViT_Classifier(torch.nn.Module):
    def __init__(self, model: MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.cls_token = self.encoder.cls_token
        self.pos_embedding = self.encoder.pos_embedding
        self.patchify = self.encoder.patchify
        self.transformer = self.encoder.transformer
        self.layer_norm = self.encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        # img shape: (B, 3, H, W)
        patches = self.patchify(img)  # Shape: (B, emb_dim, H/patch_size, W/patch_size)
        patches = rearrange(
            patches, "b c h w -> (h w) b c"
        )  # Shape: (T, B, C) where T = H*W/patch_size^2
        patches = patches + self.pos_embedding  # Add positional embedding
        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )  # Add cls token
        patches = rearrange(patches, "t b c -> b t c")  # Shape: (B, T+1, C)
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")  # Shape: (T+1, B, C)
        logits = self.head(features[0])  # Use cls token for classification
        return logits
