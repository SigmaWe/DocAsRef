import numpy as np
import torch
from tqdm.auto import trange
import functools
import dar_type
from text_preprocess import list_segmentation
import warnings
import typing


def cos_sim_mat_f(cand_segments: dar_type.TextSegments, ref_segments: dar_type.TextSegments, embedder: dar_type.Embedder) -> np.ndarray:
    def bert_encode(piece_segments: dar_type.TextSegments):
        sent_emb_list = list()
        for sent in piece_segments:
            with torch.no_grad():
                sent_emb_list.append(embedder.encode(sent, convert_to_numpy=True))
        return np.stack(sent_emb_list, axis=0)

    # def bert_encode_multiprocess(piece_segments: dar_type.TextSegments):
    #     pool = embedder.start_multi_process_pool()
    #     sent_emb = embedder.encode_multi_process(sentences=piece_segments, pool=pool, batch_size=8)
    #     embedder.stop_multi_process_pool(pool)
    #     return sent_emb

    ref_sent_emb = bert_encode(ref_segments)
    cand_sent_emb = bert_encode(cand_segments)
    numerators = np.inner(ref_sent_emb, cand_sent_emb)
    ref_sent_emb_norms = np.linalg.norm(ref_sent_emb, axis=1)
    cand_sent_emb_norms = np.linalg.norm(cand_sent_emb, axis=1)
    denominators = np.outer(ref_sent_emb_norms, cand_sent_emb_norms)
    sim_mat = np.divide(numerators, denominators)
    return sim_mat


def score_np(predictions: dar_type.TextList, references: dar_type.TextList, sim_mat_f: dar_type.SimilarityMatrixFunc, idf_f: typing.Optional[dar_type.IdfScoreFunction] = None) -> np.ndarray:
    cands, refs = list_segmentation(predictions), list_segmentation(references)
    all_scores = np.empty((len(cands), 3))

    for index in trange(len(cands), desc="bertscore-sentence {}".format(sim_mat_f.__name__), leave=False):  # all pieces, len(cands) == len(refs)
        sim_mat = sim_mat_f(cand_segments=cands[index], ref_segments=refs[index])
        if idf_f is None:
            idf_list_r = np.ones(len(refs[index]))
            idf_list_p = np.ones(len(cands[index]))
        else:
            idf_list_r = idf_f(cands[index], sim_mat.T)
            idf_list_p = idf_f(refs[index], sim_mat)
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


def compute(predictions: dar_type.TextList, references: dar_type.TextList, sim_mat_f: dar_type.SimilarityMatrixFunc, idf_f: typing.Optional[dar_type.IdfScoreFunction] = None) -> dar_type.MetricScoreDict:
    cands, refs = predictions, references # simple renaming
    score_arr = score_np(predictions=cands, references=refs, sim_mat_f=sim_mat_f, idf_f=idf_f)
    return {
        "P": score_arr[:, 0].tolist(),
        "R": score_arr[:, 1].tolist(),
        "F": score_arr[:, 2].tolist()
    }


def compute_cos(predictions: dar_type.TextList, references: dar_type.TextList, embedder: dar_type.Embedder, idf_f: typing.Optional[dar_type.IdfScoreFunction] = None) -> dar_type.MetricScoreDict:
    cos_sim_mat_f_with_embedder: dar_type.SimilarityMatrixFunc = functools.partial(cos_sim_mat_f, embedder=embedder)
    cos_sim_mat_f_with_embedder.__name__ = " ".join(["cos", embedder.__name__])
    return compute(predictions=predictions, references=references, sim_mat_f=cos_sim_mat_f_with_embedder, idf_f=idf_f)
