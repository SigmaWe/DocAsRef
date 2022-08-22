from typing import List

import numpy as np
import torch
from tqdm import tqdm

from bertscore.bert_score import get_tokenizer, get_model, model2layers

model_type = "roberta-large"
num_layers = model2layers[model_type]
tokenizer = get_tokenizer(model_type, use_fast=False)
model = get_model(model_type, num_layers, all_layers=False)


def score(cands: List[str], refs: List[str]) -> np.ndarray:
    all_scores = np.zeros((len(cands), 3))

    def bert_encode(piece: str):
        cls_emb = list()
        sentences_raw = piece.split(".")
        sentences = list()
        for sentence in sentences_raw:
            stripped = sentence.strip()
            sentences.append(stripped) if stripped != "" else None
        for sentence in tqdm(sentences):
            with torch.no_grad():
                sentence_enc = tokenizer.encode(sentence, return_tensors="pt")
                sentence_emb = model(sentence_enc)
            cls_emb.append(sentence_emb['last_hidden_state'][0, 0, :].numpy())  # only need embedding for CLS token
            del sentence_enc, sentence_emb
        return cls_emb, sentences

    for index in tqdm(range(len(cands))):  # all pieces, len(cands) == len(refs)
        cand_cls_emb, cand_sentences = bert_encode(cands[index])
        ref_cls_emb, ref_sentences = bert_encode(refs[index])
        product_mat = np.zeros((len(ref_cls_emb), len(cand_cls_emb)))
        cos_sim_mat = np.zeros((len(ref_cls_emb), len(cand_cls_emb)))
        for i in range(len(ref_cls_emb)):
            for j in range(len(cand_cls_emb)):
                numerator = np.inner(np.reshape(ref_cls_emb[i], (1, -1)), cand_cls_emb[j])[
                    0]  # ndarray: (1,) -> float32
                denominator = np.inner(np.linalg.norm(ref_cls_emb[i]), np.linalg.norm(cand_cls_emb[j]))  # float32
                cos_sim = np.divide(numerator, denominator)  # float32
                product_mat[i][j] = numerator
                cos_sim_mat[i][j] = cos_sim
                del numerator, denominator, cos_sim

        def sum_max(is_r: bool) -> float:
            sum_result = 0.0
            if is_r:
                for i in range(len(ref_cls_emb)):
                    sum_result += product_mat[i].max()
            else:
                product_mat_t = product_mat.transpose()
                for j in range(len(cand_cls_emb)):
                    sum_result += product_mat_t[j].max()
                del product_mat_t
            return sum_result

        R = (1 / len(ref_cls_emb)) * sum_max(True)
        P = (1 / len(cand_cls_emb)) * sum_max(False)
        F = 2 * ((P * R) / (P + R))
        all_scores[index, :] = np.array([P, R, F])
        del cand_cls_emb, cand_sentences, ref_cls_emb, ref_sentences, product_mat, cos_sim_mat

    return all_scores


if __name__ == '__main__':
    pass
