import sys
from os import path

file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import typing
from bertscore_sentence import eval
import numpy as np
from mnli.sim import similarity
from dar_env import nlp


def mnli_sim_mat(cand, ref) -> np.ndarray:
    def segmentation(piece: str):
        doc = nlp(piece)
        doc_sents = [sent.text for sent in doc.sents]
        return doc_sents

    cand_sentences = segmentation(cand)
    ref_sentences = segmentation(ref)
    sim_mat = np.zeros((len(ref_sentences), len(cand_sentences)))
    for i in range(len(ref_sentences)):
        for j in range(len(cand_sentences)):
            sim_mat[i][j] = similarity(ref_sentences[i], cand_sentences[j])
    return sim_mat, cand_sentences, ref_sentences


def bertscore_sentence_compute(predictions: typing.List[str], references: typing.List[str]) -> typing.Dict:
    return eval.compute(predictions=predictions, references=references, sim_mat_f=mnli_sim_mat)
