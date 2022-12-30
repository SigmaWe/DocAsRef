import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from bertscore_sentence import eval
import numpy as np
from mnli.sim import similarity
import functools
import dar_type


def mnli_sim_mat(cand_segments: dar_type.TextSegments, ref_segments: dar_type.TextSegments, classifiers: dar_type.PipelinesList) -> np.ndarray:
    sent_pairs = [" ".join([x, "[SEP]", y]) for x in ref_segments for y in cand_segments]
    sim_mat = np.empty((len(ref_segments), len(cand_segments)))
    sim_mat.flat = np.array(similarity(sent_pairs, classifiers))
    return sim_mat


def bertscore_sentence_compute(predictions: dar_type.TextList, references: dar_type.TextList, classifiers: dar_type.PipelinesList) -> dar_type.MetricScoreDict:
    sim_mat_f: dar_type.SimilarityMatrixFunc = functools.partial(mnli_sim_mat, classifiers=classifiers)
    sim_mat_f.__name__ = " ".join(["mnli", classifiers.__name__])
    return eval.compute(predictions=predictions, references=references, sim_mat_f=sim_mat_f)
