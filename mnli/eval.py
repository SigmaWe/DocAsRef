import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import typing
from bertscore_sentence import eval
import numpy as np
from mnli.sim import similarity
import functools
import transformers
from type_piece import EvalPieces


def mnli_sim_mat(cand_segments: typing.List[str], ref_segments: typing.List[str], classifier: transformers.Pipeline) -> np.ndarray:
    sent_pairs = [" ".join([x, y]) for x in ref_segments for y in cand_segments]
    sim_mat = np.empty((len(ref_segments), len(cand_segments)))
    sim_mat.flat = similarity(sent_pairs, classifier)
    return sim_mat


def bertscore_sentence_compute(predictions: EvalPieces, references: EvalPieces, classifier: transformers.Pipeline) -> typing.Dict:
    sim_mat_f = functools.partial(mnli_sim_mat, classifier=classifier)
    sim_mat_f.__name__ = " ".join(["mnli", classifier.__name__])
    return eval.compute(predictions=predictions, references=references, sim_mat_f=sim_mat_f)
