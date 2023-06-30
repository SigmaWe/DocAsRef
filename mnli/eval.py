import sys
import os 
import functools
import typing
import warnings
# file_path = os.path.abspath(__file__)
# sys.path.append(path.dirname(path.dirname(file_path)))

import numpy as np

import bertscore_sentence.eval
import mnli.sim
import dar_type


# Approach 1.2: MNLI probabilties for sentence similarity
def get_similarity_matrix_mnli(
        cand_segments: dar_type.TextSegments, 
        ref_segments: dar_type.TextSegments, 
        classifiers: dar_type.PipelinesList, 
        expr: dar_type.MNLISimilarityExpression) -> typing.Optional[np.ndarray]:

    """Compute the similarity matrix between two sets of sentences using MNLI.
    """ 

    if len(cand_segments) == 0 or len(ref_segments) == 0:
        warnings.warn("Empty cand_segments or ref_segments; len(cand_segments)={}, len(ref_segments)={}".format(len(cand_segments), len(ref_segments)))
        return None
    sent_pairs = ["[SEP]".join([x, y]) for x in ref_segments for y in cand_segments]
    sim_mat = np.array(mnli.sim.similarity(sent_pairs, classifiers, expr)).reshape((len(ref_segments), len(cand_segments)))
    return sim_mat

# Approach 1.2: use NLI probabilities to estimate sentence similarity
def compute_mnli(
        predictions: dar_type.TextList, 
        references: dar_type.TextList, 
        classifiers: dar_type.PipelinesList, 
        expr: dar_type.MNLISimilarityExpression, 
        idf_f: typing.Optional[dar_type.IdfScoreFunction] = None
        ) -> dar_type.MetricScoreDict:
    sim_mat_f: dar_type.SimilarityMatrixFunc = \
        functools.partial(get_similarity_matrix_mnli, classifiers=classifiers, expr=expr)
    sim_mat_f.__name__ = " ".join(["mnli", classifiers.__name__])
    return bertscore_sentence.eval.compute(predictions=predictions, references=references, sim_mat_f=sim_mat_f, idf_f=idf_f)


