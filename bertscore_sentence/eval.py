# Python standard modules
import functools
import warnings
import typing

# Common ML/math modules 
import numpy as np
import torch
from tqdm.auto import trange

from text_preprocess import list_segmentation

# Our own 
import dar_type
import mnli.sim 

# Approach 1.1: Cosine similarity for sentence similarity 
# originally called sim_mat_cos_f
def get_similarity_matrix_cos(
        cand_segments: dar_type.TextSegments, 
        ref_segments: dar_type.TextSegments, 
        embedder: dar_type.Embedder) -> typing.Optional[np.ndarray]:
    """Compute the similarity matrix between two sets of sentences using Cosine similarity.
    """

    if len(cand_segments) == 0 or len(ref_segments) == 0:
        warnings.warn("Empty cand_segments or ref_segments; len(cand_segments)={}, len(ref_segments)={}".format(len(cand_segments), len(ref_segments)))
        return None

    def bert_encode(piece_segments: dar_type.TextSegments):
        with torch.no_grad():
            return embedder(piece_segments)

    ref_sent_emb = bert_encode(ref_segments)
    cand_sent_emb = bert_encode(cand_segments)
    numerators = np.inner(ref_sent_emb, cand_sent_emb)
    ref_sent_emb_norms = np.linalg.norm(ref_sent_emb, axis=1)
    cand_sent_emb_norms = np.linalg.norm(cand_sent_emb, axis=1)
    denominators = np.outer(ref_sent_emb_norms, cand_sent_emb_norms)
    sim_mat = np.divide(numerators, denominators)
    return sim_mat  # shape: (len(ref_segments), len(cand_segments))

def score_np(
        predictions: dar_type.TextList, 
        references: dar_type.TextList, 
        sim_mat_f: dar_type.SimilarityMatrixFunc, # the function that computes the similarity matrix
        idf_f: typing.Optional[dar_type.IdfScoreFunction] = None) -> np.ndarray:
    cands, refs = list_segmentation(predictions), list_segmentation(references)
    all_scores = np.empty((len(cands), 3))

    for index in trange(len(cands), desc="bertscore-sentence {}".format(sim_mat_f.__name__), leave=False):  # all pieces, len(cands) == len(refs)
        sim_mat = sim_mat_f(cand_segments=cands[index], ref_segments=refs[index])
        if sim_mat is None:
            all_scores[index, :] = np.zeros((3,))
            continue
        if idf_f is None:
            idf_list_r = np.ones(len(refs[index]))
            idf_list_p = np.ones(len(cands[index]))
        else:
            idf_list_r = idf_f(cands[index], sim_mat.T, sim_mat_f)
            idf_list_p = idf_f(refs[index], sim_mat, sim_mat_f)
            if sum(idf_list_r) == 0:
                idf_list_r = np.ones(len(refs[index]))
            if sum(idf_list_p) == 0:
                idf_list_p = np.ones(len(cands[index]))
        R = (1 / np.sum(idf_list_r)) * np.sum(idf_list_r * np.max(sim_mat, axis=1))
        P = (1 / np.sum(idf_list_p)) * np.sum(idf_list_p * np.max(sim_mat, axis=0))
        F = 2 * ((P * R) / (P + R))
        all_scores[index, :] = np.array([P, R, F])
        if np.isnan(all_scores[index, :]).any():
            warnings.warn("nan score replaced. [ref] {}; [cand] {}".format(refs[index], cands[index]), dar_type.DocWarning)
            all_scores[index, :] = np.zeros((3,))
    return all_scores


def compute(
        predictions: dar_type.TextList, 
        references: dar_type.TextList, 
        sim_mat_f: dar_type.SimilarityMatrixFunc, 
        idf_f: typing.Optional[dar_type.IdfScoreFunction] = None
        ) -> dar_type.MetricScoreDict:
    cands, refs = predictions, references # simple renaming
    score_arr = score_np(predictions=cands, references=refs, sim_mat_f=sim_mat_f, idf_f=idf_f)
    return {
        "P": score_arr[:, 0].tolist(),
        "R": score_arr[:, 1].tolist(),
        "F": score_arr[:, 2].tolist()
    }

# Approach 1.1: use cosine similarity to estimate sentence similarity
def compute_cos(
        predictions: dar_type.TextList, 
        references: dar_type.TextList, 
        embedder: dar_type.Embedder, 
        idf_f: typing.Optional[dar_type.IdfScoreFunction] = None
        ) -> dar_type.MetricScoreDict:

    cos_sim_mat_f_with_embedder: dar_type.SimilarityMatrixFunc = \
        functools.partial(get_similarity_matrix_cos, embedder=embedder)
    cos_sim_mat_f_with_embedder.__name__ = " ".join(["cos", embedder.__name__])

    return compute(predictions=predictions, references=references, sim_mat_f=cos_sim_mat_f_with_embedder, idf_f=idf_f)
