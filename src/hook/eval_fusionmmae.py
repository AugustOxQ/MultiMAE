from __future__ import print_function
import os

import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

import json
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
from typing import Optional
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderStateMixin

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


CLIP_Retrieval_Metrics = {
    "i2t_R1": 58.4,
    "i2t_R5": 81.5,
    "i2t_R10": 88.1,
    "t2i_R1": 37.8,
    "t2i_R5": 62.4,
    "t2i_R10": 72.2,
}


class IMPDataset_test(Dataset):
    def __init__(self, annotation_path, image_path, vis_processors, txt_processors):
        self.annotations = json.load(open(annotation_path))
        self.image_path = image_path
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        raw_image = Image.open(
            os.path.join(self.image_path, self.annotations[idx]["image"])
        ).convert("RGB")
        image_input = self.vis_processors(raw_image)
        caption = self.annotations[idx]["caption"][:5]
        text_input = self.txt_processors(caption)

        return image_input, text_input


def calculate_average_precision(correct_positions, total_relevant):
    """
    Calculate Average Precision (AP) for the given ranks of relevant documents.
    correct_positions: Tensor of ranks where relevant documents were retrieved.
    total_relevant: Total number of relevant documents for the query.
    """
    if total_relevant == 0 or correct_positions.numel() == 0:
        return 0.0  # Return 0 if no relevant documents

    ap_sum = 0.0
    for i, rank in enumerate(correct_positions.sort()[0], 1):
        precision_at_rank = i / float(rank + 1)  # Correct for 1-based indexing
        ap_sum += precision_at_rank

    return ap_sum / total_relevant


def calculate_metrics(inds, mappings, captions_per_image, device):
    """
    Calculate R-Precision and mAP for a set of rankings (inds) given the correct mappings.
    inds: Sorted indices for predictions.
    mappings: Correct mappings from queries (texts or images) to targets (images or texts).
    captions_per_image: Number of captions per image, used for calculating R-Precision for i2t.
    """
    num_queries = inds.size(0)
    R_precisions = []
    AP_scores = []
    all_ranks = []

    for query_idx in range(num_queries):
        correct_indices = mappings[query_idx].tolist()

        query_inds = inds[query_idx]

        # Find ranks of correct indices
        if type(correct_indices) == int:
            # For single correct index
            correct_mask = query_inds == torch.tensor(correct_indices, device=device)
            correct_positions = correct_mask.nonzero(as_tuple=True)[-1].item()
            ranks = correct_positions + 1  # Convert to 1-based indexing
        else:
            ranks = []
            correct_mask = []
            for correct_index in correct_indices:
                # Find the position of the correct caption index in the sorted indices
                position = (query_inds == correct_index).nonzero(as_tuple=True)[-1]
                correct_mask.append(position)
                rank = position.item() + 1
                ranks.append(rank)
            assert len(ranks) == captions_per_image

        if type(ranks) != list:
            ranks = [ranks]
        all_ranks.extend(ranks)

        # Calculate AP for this query
        AP = 0
        for j, rank in enumerate(sorted(ranks), start=1):
            precision_at_j = j / rank
            AP += precision_at_j
        AP /= captions_per_image
        AP_scores.append(AP)

    mean_ap = np.mean(AP_scores)
    meanR = np.mean(all_ranks)
    medR = np.median(all_ranks)

    return (meanR, medR, mean_ap)


def encode_data(model, data_loader, accelerator: Optional[Accelerator] = None):
    """Encode all images and captions loadable by `data_loader`.

    With an accelerator, `data_loader` should be prepared (sharded); embeddings are gathered
    from every process batch by batch, so the result covers the whole dataset exactly once.
    """
    if (
        accelerator is not None
        and accelerator.num_processes > 1
        and not isinstance(data_loader, DataLoaderStateMixin)
    ):
        # DataLoaderStateMixin is the base of both DataLoaderShard and DataLoaderDispatcher,
        # the types accelerator.prepare returns for a DataLoader.
        raise ValueError(
            "encode_data got a data_loader that was not prepared by accelerate while running "
            f"on {accelerator.num_processes} processes: every process would encode the full "
            "set and the gather would count each sample once per process. Pass "
            "`accelerator.prepare(data_loader)`."
        )
    if accelerator is not None:
        # DDP does not expose the model's custom encode_* methods
        model = accelerator.unwrap_model(model)
    # switch to evaluate mode
    model.eval()
    if accelerator:
        accelerator.print("Evaluating...")
    else:
        print("Evaluating...")

    # Lists to keep all the embeddings
    img_embs = []
    cap_embs = []
    captions_per_image = None

    device = next(model.parameters()).device
    with torch.no_grad():

        for i, (images, captions) in enumerate(
            tqdm(
                data_loader,
                disable=(
                    accelerator is not None and not accelerator.is_local_main_process
                ),
            )
        ):
            images = images.to(device)
            captions = captions.to(device)
            batch_size, captions_per_image, _ = captions.shape

            img_emb = model.encode_image_tokens_cls(images)  # (B, D)
            cap_emb = model.encode_text_tokens_cls(
                torch.flatten(captions, start_dim=0, end_dim=1)
            )  # (B * captions_per_image, D)
            cap_emb = cap_emb.view(batch_size, captions_per_image, -1)  # (B, cpi, D)

            if accelerator is not None:
                # Gather per batch, with dim 0 counting samples (images), so that on the last
                # batch gather_for_metrics drops the samples the sampler duplicated to give
                # every process a full batch.
                img_emb = accelerator.gather_for_metrics(img_emb)
                cap_emb = accelerator.gather_for_metrics(cap_emb)

            img_embs.append(img_emb)
            cap_embs.append(torch.flatten(cap_emb, start_dim=0, end_dim=1))

    image_embeddings = torch.cat(img_embs, axis=0)  # type: ignore
    text_embeddings = torch.cat(cap_embs, axis=0)  # type: ignore

    # Images and their captions stay aligned through the gather, so image i owns texts
    # i*cpi ... i*cpi + cpi - 1.
    num_images = image_embeddings.shape[0]
    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = torch.arange(num_images, device=device).repeat_interleave(
        captions_per_image
    )
    # image_to_text_map[i] gives the indices of the captions of the ith image
    image_to_text_map = torch.arange(
        num_images * captions_per_image, device=device
    ).view(num_images, captions_per_image)

    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    return image_embeddings, text_embeddings, text_to_image_map, image_to_text_map


def evalrank(model, data_loader, npts=None, accelerator: Optional[Accelerator] = None):
    # Extract Embeddings
    image_embeddings, text_embeddings, text_to_image_map, image_to_text_map = (
        encode_data(model, data_loader, accelerator=accelerator)
    )
    if accelerator:
        accelerator.print(image_embeddings.shape, text_embeddings.shape)
        accelerator.print(text_to_image_map.shape, image_to_text_map.shape)
    else:
        print(image_embeddings.shape, text_embeddings.shape)
        print(text_to_image_map.shape, image_to_text_map.shape)

    num_text = text_embeddings.shape[0]
    num_im = image_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    k_vals = [1, 5, 10, 50, 100]
    msg = f"Number of images: {num_im}, Number of texts: {num_text}, Captions per image: {captions_per_image}"
    accelerator.print(msg) if accelerator else print(msg)

    # text-to-image recall
    (
        accelerator.print("Text-to-image recall...")
        if accelerator
        else print("Text-to-image recall...")
    )

    dist_matrix = (
        text_embeddings @ image_embeddings.T
    )  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(image_embeddings.device)
    accelerator.print(inds.shape) if accelerator else print(inds.shape)

    text_to_image_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text * 100)

    meanR_t2i, medR_t2i, mAP_t2i = calculate_metrics(
        inds, text_to_image_map, 1, device=inds.device
    )

    # image-to-text recall
    (
        accelerator.print("Image-to-text recall...")
        if accelerator
        else print("Image-to-text recall...")
    )
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(text_embeddings.device)
    accelerator.print(inds.shape) if accelerator else print(inds.shape)

    image_to_text_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool, device=inds.device)

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(
                dim=1
            )
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im * 100)  #

    meanR_i2t, medR_i2t, mAP_i2t = calculate_metrics(
        inds, image_to_text_map, 5, device=inds.device
    )

    accelerator.print("Done.") if accelerator else print("Done.")
    metrics = {
        "i2t_R1": round(image_to_text_recall[0], 2),
        "i2t_R5": round(image_to_text_recall[1], 2),
        "i2t_R10": round(image_to_text_recall[2], 2),
        "i2t_meanR": int(round(meanR_i2t, 0)),
        "i2t_medR": int(round(medR_i2t, 0)),
        "i2t_mAP": round(mAP_i2t * 100, 2),
        "t2i_R1": round(text_to_image_recall[0], 2),
        "t2i_R5": round(text_to_image_recall[1], 2),
        "t2i_R10": round(text_to_image_recall[2], 2),
        "t2i_meanR": int(round(meanR_t2i, 0)),
        "t2i_medR": int(round(medR_t2i, 0)),
        "t2i_mAP": round(mAP_t2i * 100, 2),
        "t2i_rsum": round(sum(text_to_image_recall[:2]), 2),
        "i2t_rsum": round(sum(image_to_text_recall[:2]), 2),
        "r_sum": round(
            sum(text_to_image_recall[:2]) + sum(image_to_text_recall[:2]), 2
        ),  # 100
    }

    if accelerator:
        for key, value in metrics.items():
            accelerator.print(f"{key}: {value}")
    else:
        for key, value in metrics.items():
            print(f"{key}: {value}")

    return metrics
