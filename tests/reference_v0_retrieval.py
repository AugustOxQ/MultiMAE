"""v0's retrieval metrics (legacy-v0:src/hook/eval_fusionmmae.py), kept only as a test reference."""
import numpy as np
import torch


def calculate_metrics(inds, mappings, captions_per_image):
    num_queries = inds.size(0)
    AP_scores, all_ranks = [], []
    for query_idx in range(num_queries):
        correct_indices = mappings[query_idx].tolist()
        query_inds = inds[query_idx]
        if type(correct_indices) == int:
            correct_mask = query_inds == torch.tensor(correct_indices)
            ranks = correct_mask.nonzero(as_tuple=True)[-1].item() + 1
        else:
            ranks = []
            for correct_index in correct_indices:
                position = (query_inds == correct_index).nonzero(as_tuple=True)[-1]
                ranks.append(position.item() + 1)
            assert len(ranks) == captions_per_image
        if type(ranks) != list:
            ranks = [ranks]
        all_ranks.extend(ranks)
        AP = 0
        for j, rank in enumerate(sorted(ranks), start=1):
            AP += j / rank
        AP /= captions_per_image
        AP_scores.append(AP)
    return np.mean(all_ranks), np.median(all_ranks), np.mean(AP_scores)


def v0_metrics(image_embeddings, text_embeddings, text_to_image_map, image_to_text_map):
    num_text, num_im = text_embeddings.shape[0], image_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    k_vals = [1, 5, 10, 50, 100]
    dist_matrix = text_embeddings @ image_embeddings.T
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    t2i = []
    for k in k_vals:
        correct = torch.eq(inds[:, :k], text_to_image_map.unsqueeze(-1)).any(dim=1)
        t2i.append(correct.sum().item() / num_text * 100)
    meanR_t2i, medR_t2i, mAP_t2i = calculate_metrics(inds, text_to_image_map, 1)
    inds = torch.argsort(dist_matrix.T, dim=1, descending=True)
    i2t = []
    for k in k_vals:
        correct = torch.zeros((num_im,), dtype=torch.bool)
        for i in range(captions_per_image):
            correct = correct | torch.eq(inds[:, :k], image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
        i2t.append(correct.sum().item() / num_im * 100)
    meanR_i2t, medR_i2t, mAP_i2t = calculate_metrics(inds, image_to_text_map, captions_per_image)
    return {
        "i2t_R1": round(i2t[0], 2), "i2t_R5": round(i2t[1], 2), "i2t_R10": round(i2t[2], 2),
        "i2t_meanR": int(round(meanR_i2t, 0)), "i2t_medR": int(round(medR_i2t, 0)),
        "i2t_mAP": round(mAP_i2t * 100, 2),
        "t2i_R1": round(t2i[0], 2), "t2i_R5": round(t2i[1], 2), "t2i_R10": round(t2i[2], 2),
        "t2i_meanR": int(round(meanR_t2i, 0)), "t2i_medR": int(round(medR_t2i, 0)),
        "t2i_mAP": round(mAP_t2i * 100, 2),
    }
