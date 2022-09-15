import typing
from typing import List

import numpy as np
import sentence_transformers
import torch
from tqdm import tqdm

embedder = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")


def score_np(predictions: List[str], references: List[str]) -> np.ndarray:
    cands, refs = predictions, references # simple renaming. 

    all_scores = np.zeros((len(cands), 3))

    def bert_encode(piece: str):
        sentence_emb = list()
        sentences_raw = piece.split(".")
        sentences = list()
        for sentence in sentences_raw:
            stripped = sentence.strip()
            sentences.append(stripped) if stripped != "" else None
        for sentence in sentences:
            with torch.no_grad():
                sentence_emb.append(embedder.encode(sentence, convert_to_numpy=True))
        return sentence_emb, sentences

    for index in tqdm(range(len(cands))):  # all pieces, len(cands) == len(refs)
        cand_sentence_emb, cand_sentences = bert_encode(cands[index])
        ref_sentence_emb, ref_sentences = bert_encode(refs[index])
        product_mat = np.zeros((len(ref_sentence_emb), len(cand_sentence_emb)))
        cos_sim_mat = np.zeros((len(ref_sentence_emb), len(cand_sentence_emb)))
        for i in range(len(ref_sentence_emb)):
            for j in range(len(cand_sentence_emb)):
                numerator = np.dot(ref_sentence_emb[i], cand_sentence_emb[j])  # float32
                denominator = np.dot(np.linalg.norm(ref_sentence_emb[i]),
                                     np.linalg.norm(cand_sentence_emb[j]))  # float32
                cos_sim = np.divide(numerator, denominator)  # float32
                product_mat[i][j] = numerator
                cos_sim_mat[i][j] = cos_sim
                del numerator, denominator, cos_sim

        def sum_max(is_r: bool) -> float:
            sum_result = 0.0
            if is_r:
                for i in range(len(ref_sentence_emb)):
                    sum_result += cos_sim_mat[i].max()
            else:
                cos_sim_mat_t = cos_sim_mat.transpose()
                for j in range(len(cand_sentence_emb)):
                    sum_result += cos_sim_mat_t[j].max()
                del cos_sim_mat_t
            return sum_result

        R = (1 / len(ref_sentence_emb)) * sum_max(True)
        P = (1 / len(cand_sentence_emb)) * sum_max(False)
        F = 2 * ((P * R) / (P + R))
        all_scores[index, :] = np.array([P, R, F])
        del cand_sentence_emb, cand_sentences, ref_sentence_emb, ref_sentences, product_mat, cos_sim_mat

    return all_scores


def compute(predictions: List[str], references: List[str]) -> typing.Dict:
    cands, refs = predictions, references # simple renaming. 
    score_arr = score_np(cands, refs)
    return {
        "P": score_arr[:, 0].tolist(),
        "R": score_arr[:, 1].tolist(),
        "F": score_arr[:, 2].tolist()
    }
