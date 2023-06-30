import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import dar_type
import numpy as np


def text_weight(
        segments: dar_type.TextSegments, 
        sim_mat_f: dar_type.SimpleSimilarityMatrixFunc, 
        weight_f: dar_type.SentenceWeightFunction) -> np.ndarray:
    sim_mat = sim_mat_f(segments, segments)
    # return np.array(weight_f(sim_mat[i].tolist()) for i in range(len(segments)))
    return np.apply_along_axis(func1d=weight_f, axis=0, arr=sim_mat)


def get_idf(
        ref_segments: dar_type.TextSegments, 
        sim_mat: np.ndarray, 
        sim_mat_f: dar_type.SimpleSimilarityMatrixFunc, 
        weight_f: dar_type.SentenceWeightFunction) -> np.ndarray:
    w_refs = np.array(text_weight(ref_segments, sim_mat_f, weight_f))
    wd_mat = (sim_mat.T * w_refs).T
    v = np.sum(wd_mat, axis=0)
    np.nan_to_num(v, copy=False, nan=0, posinf=1, neginf=-1)
    return v
