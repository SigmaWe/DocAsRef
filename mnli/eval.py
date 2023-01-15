import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from bertscore_sentence import eval
import numpy as np
from mnli.sim import similarity
import functools
import dar_type
import typing


def mnli_sim_mat(cand_segments: dar_type.TextSegments, ref_segments: dar_type.TextSegments, classifiers: dar_type.PipelinesList, expr: dar_type.MNLISimilarityExpression) -> np.ndarray:
    sent_pairs = ["[SEP]".join([x, y]) for x in ref_segments for y in cand_segments]
    sim_mat = np.array(similarity(sent_pairs, classifiers, expr)).reshape((len(ref_segments), len(cand_segments)))
    return sim_mat


def bertscore_sentence_compute(predictions: dar_type.TextList, references: dar_type.TextList, classifiers: dar_type.PipelinesList, expr: dar_type.MNLISimilarityExpression, idf_f: typing.Optional[dar_type.IdfScoreFunction] = None) -> dar_type.MetricScoreDict:
    sim_mat_f: dar_type.SimilarityMatrixFunc = functools.partial(mnli_sim_mat, classifiers=classifiers, expr=expr)
    sim_mat_f.__name__ = " ".join(["mnli", classifiers.__name__])
    return eval.compute(predictions=predictions, references=references, sim_mat_f=sim_mat_f, idf_f=idf_f)
