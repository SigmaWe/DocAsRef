import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import numpy as np
import torch
from tqdm.auto import trange
import functools
import dar_type
from text_preprocess import list_segmentation
import warnings


def cos_sim_mat_f(cand_segments: dar_type.TextSegments, ref_segments: dar_type.TextSegments, embedder: dar_type.Embedder) -> np.ndarray:
    def bert_encode(piece_segments: dar_type.TextSegments):
        sent_emb = list()
        for sent in piece_segments:
            with torch.no_grad():
                sent_emb.append(embedder.encode(sent, convert_to_numpy=True))
        return sent_emb

    ref_sent_emb_list = bert_encode(ref_segments)
    cand_sent_emb_list = bert_encode(cand_segments)
    ref_sent_emb = np.stack(ref_sent_emb_list, axis=0)
    cand_sent_emb = np.stack(cand_sent_emb_list, axis=0)
    numerators = np.inner(ref_sent_emb, cand_sent_emb)
    ref_sent_emb_norms = np.linalg.norm(ref_sent_emb, axis=1)
    cand_sent_emb_norms = np.linalg.norm(cand_sent_emb, axis=1)
    denominators = np.outer(ref_sent_emb_norms, cand_sent_emb_norms)
    sim_mat = np.divide(numerators, denominators)
    return sim_mat


def score_np(predictions: dar_type.TextList, references: dar_type.TextList, sim_mat_f: dar_type.SimilarityMatrixFunc) -> np.ndarray:
    cands, refs = list_segmentation(predictions), list_segmentation(references)
    all_scores = np.empty((len(cands), 3))

    for index in trange(len(cands), desc="bertscore-sentence {}".format(sim_mat_f.__name__), leave=False):  # all pieces, len(cands) == len(refs)
        sim_mat = sim_mat_f(cand_segments=cands[index], ref_segments=refs[index])

        def sum_max(is_r: bool) -> float:
            if is_r:
                return np.sum(np.max(sim_mat, axis=1))
            else:
                return np.sum(np.max(sim_mat, axis=0))  # equals to np.sum(np.max(sim_mat.T, axis=1))

        has_empty = False
        if len(refs[index]) == 0:
            warnings.warn("empty ref str", dar_type.DocWarning)
            has_empty = True
        if len(cands[index]) == 0:
            warnings.warn("empty cand str", dar_type.DocWarning)
            has_empty = True
        if has_empty:
            warnings.warn("detail: [ref] {}; [cand] {}".format(refs[index], cands[index]), dar_type.DocWarning)

        if not has_empty:
            R = (1 / len(refs[index])) * sum_max(True)
            P = (1 / len(cands[index])) * sum_max(False)
            F = 2 * ((P * R) / (P + R))
            all_scores[index, :] = np.array([P, R, F])
        else:
            all_scores[index, :] = np.zeros((3,))

    return all_scores


def compute(predictions: dar_type.TextList, references: dar_type.TextList, sim_mat_f: dar_type.SimilarityMatrixFunc) -> dar_type.MetricScoreDict:
    cands, refs = predictions, references # simple renaming
    score_arr = score_np(predictions=cands, references=refs, sim_mat_f=sim_mat_f)
    return {
        "P": score_arr[:, 0].tolist(),
        "R": score_arr[:, 1].tolist(),
        "F": score_arr[:, 2].tolist()
    }


def compute_cos(predictions: dar_type.TextList, references: dar_type.TextList, embedder: dar_type.Embedder) -> dar_type.MetricScoreDict:
    cos_sim_mat_f_with_embedder: dar_type.SimilarityMatrixFunc = functools.partial(cos_sim_mat_f, embedder=embedder)
    cos_sim_mat_f_with_embedder.__name__ = " ".join(["cos", embedder.__name__])
    return compute(predictions=predictions, references=references, sim_mat_f=cos_sim_mat_f_with_embedder)
