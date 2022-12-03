import sys
from os import path

file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import typing
import numpy as np
import torch
from tqdm.auto import trange
from dar_env import nlp
from dar_env import sent_embedder as embedder


def cos_sim_mat_f(cand, ref) -> np.ndarray:
    def bert_encode(piece: str):
        sentence_emb = list()
        doc = nlp(piece)
        doc_sents = [sent.text for sent in doc.sents]
        for sentence in doc_sents:
            with torch.no_grad():
                sentence_emb.append(embedder.encode(sentence, convert_to_numpy=True))
        return sentence_emb, doc_sents

    cand_sentence_emb, cand_sentences = bert_encode(cand)
    ref_sentence_emb, ref_sentences = bert_encode(ref)
    sim_mat = np.zeros((len(ref_sentence_emb), len(cand_sentence_emb)))
    for i in range(len(ref_sentence_emb)):
        for j in range(len(cand_sentence_emb)):
            numerator = np.dot(ref_sentence_emb[i], cand_sentence_emb[j])  # float32
            denominator = np.dot(np.linalg.norm(ref_sentence_emb[i]),
                                    np.linalg.norm(cand_sentence_emb[j]))  # float32
            cos_sim = np.divide(numerator, denominator)  # float32
            sim_mat[i][j] = cos_sim
            del numerator, denominator, cos_sim
    return sim_mat, cand_sentences, ref_sentences


def score_np(predictions: typing.List[str], references: typing.List[str], sim_mat_f: typing.Callable) -> np.ndarray:
    cands, refs = predictions, references # simple renaming. 

    all_scores = np.zeros((len(cands), 3))

    for index in trange(len(cands), desc="bertscore-sentence cands {}".format(sim_mat_f.__name__), leave=False):  # all pieces, len(cands) == len(refs)
        sim_mat, cand_sentences, ref_sentences = sim_mat_f(cand=cands[index], ref=refs[index])

        def sum_max(is_r: bool) -> float:
            sum_result = 0.0
            if is_r:
                for i in range(len(ref_sentences)):
                    sum_result += sim_mat[i].max()
            else:
                sim_mat_t = sim_mat.transpose()
                for j in range(len(cand_sentences)):
                    sum_result += sim_mat_t[j].max()
                del sim_mat_t
            return sum_result

        R = (1 / len(ref_sentences)) * sum_max(True)
        P = (1 / len(cand_sentences)) * sum_max(False)
        F = 2 * ((P * R) / (P + R))
        all_scores[index, :] = np.array([P, R, F])
        del sim_mat

    return all_scores


def compute(predictions: typing.List[str], references: typing.List[str], sim_mat_f: typing.Callable = cos_sim_mat_f) -> typing.Dict:
    cands, refs = predictions, references # simple renaming. 
    score_arr = score_np(predictions=cands, references=refs, sim_mat_f=sim_mat_f)
    return {
        "P": score_arr[:, 0].tolist(),
        "R": score_arr[:, 1].tolist(),
        "F": score_arr[:, 2].tolist()
    }
