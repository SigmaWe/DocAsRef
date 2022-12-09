import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import typing
from bertscore_sentence import eval
import numpy as np
from mnli.sim import similarity
from dar_env import nlp_spacy
import functools
import transformers


def mnli_sim_mat(cand: str, ref: str, classifier: transformers.Pipeline) -> np.ndarray:
    def segmentation(piece: str):
        doc = nlp_spacy(piece)
        doc_sents = [sent.text for sent in doc.sents]
        return doc_sents

    cand_sents = segmentation(cand)
    ref_sents = segmentation(ref)
    sent_pairs = [" ".join([x, y]) for x in ref_sents for y in cand_sents]
    sim_mat = np.empty((len(ref_sents), len(cand_sents)))
    sim_mat.flat = similarity(sent_pairs, classifier)
    return sim_mat, cand_sents, ref_sents


def bertscore_sentence_compute(predictions: typing.List[str], references: typing.List[str], classifier: transformers.Pipeline) -> typing.Dict:
    sim_mat_f = functools.partial(mnli_sim_mat, classifier=classifier)
    sim_mat_f.__name__ = " ".join(["mnli", classifier.__name__])
    return eval.compute(predictions=predictions, references=references, sim_mat_f=sim_mat_f)
