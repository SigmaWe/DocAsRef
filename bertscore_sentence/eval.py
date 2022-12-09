import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import typing
import numpy as np
import torch
from tqdm.auto import trange
from dar_env import nlp_spacy
import functools
import sentence_transformers


def cos_sim_mat_f(cand, ref, embedder) -> np.ndarray:
    def bert_encode(piece: str):
        sentence_emb = list()
        doc = nlp_spacy(piece)
        doc_sents = [sent.text for sent in doc.sents]
        for sentence in doc_sents:
            with torch.no_grad():
                sentence_emb.append(embedder.encode(sentence, convert_to_numpy=True))
        return sentence_emb, doc_sents

    ref_sent_emb_list, ref_sents = bert_encode(ref)
    cand_sent_emb_list, cand_sents = bert_encode(cand)
    ref_sent_emb = np.stack(ref_sent_emb_list, axis=0)
    cand_sent_emb = np.stack(cand_sent_emb_list, axis=0)
    numerators = np.inner(ref_sent_emb, cand_sent_emb)
    ref_sent_emb_norms = np.linalg.norm(ref_sent_emb, axis=1)
    cand_sent_emb_norms = np.linalg.norm(cand_sent_emb, axis=1)
    denominators = np.outer(ref_sent_emb_norms, cand_sent_emb_norms)
    sim_mat = np.divide(numerators, denominators)
    return sim_mat, cand_sents, ref_sents


def score_np(predictions: typing.List[str], references: typing.List[str], sim_mat_f: typing.Callable) -> np.ndarray:
    cands, refs = predictions, references # simple renaming. 
    all_scores = np.empty((len(cands), 3))

    for index in trange(len(cands), desc="bertscore-sentence {}".format(sim_mat_f.__name__), leave=False):  # all pieces, len(cands) == len(refs)
        sim_mat, cand_sents, ref_sents = sim_mat_f(cand=cands[index], ref=refs[index])

        def sum_max(is_r: bool) -> float:
            if is_r:
                return np.sum(np.max(sim_mat, axis=1))
            else:
                return np.sum(np.max(sim_mat, axis=0))  # equals to np.sum(np.max(sim_mat.T, axis=1))

        R = (1 / len(ref_sents)) * sum_max(True)
        P = (1 / len(cand_sents)) * sum_max(False)
        F = 2 * ((P * R) / (P + R))
        all_scores[index, :] = np.array([P, R, F])
        del sim_mat

    return all_scores


def compute(predictions: typing.List[str], references: typing.List[str], sim_mat_f: typing.Optional[typing.Callable] = None, embedder: typing.Optional[sentence_transformers.SentenceTransformer] = None) -> typing.Dict:
    cands, refs = predictions, references # simple renaming
    if sim_mat_f is None:  # cosine similarity by default
        sim_mat_f = functools.partial(cos_sim_mat_f, embedder=embedder)
        sim_mat_f.__name__ = " ".join(["cos", embedder.__name__])
    score_arr = score_np(predictions=cands, references=refs, sim_mat_f=sim_mat_f)
    return {
        "P": score_arr[:, 0].tolist(),
        "R": score_arr[:, 1].tolist(),
        "F": score_arr[:, 2].tolist()
    }
