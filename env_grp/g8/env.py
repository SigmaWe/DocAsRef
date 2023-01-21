# Group 8: PageRank + Group 2

from env_root import *
from env_grp.g2.env import *

### MODELS ###

import bertscore_sentence.eval as bertscore_sentence
import dar_type
import functools
import mnli.eval
import mnli.sim_expr
import numpy as np
import pagerank.eval as pagerank
import scipy.stats
import typing

g2_metrics = metrics
metrics = dict()

weight_fs: typing.Dict[str, dar_type.SentenceWeightFunction] = {
    "entropy": scipy.stats.entropy,
    "sum": np.sum,
}

sim_mat_fs: typing.Dict[str, dar_type.SimpleSimilarityMatrixFunc] = {
    "cos-mpnet": functools.partial(bertscore_sentence.cos_sim_mat_f, embedder=sent_embedders["mpnet"]),
    "cos-deberta-large": functools.partial(bertscore_sentence.cos_sim_mat_f, embedder=sent_embedders["deberta-large"]),
    "cos-deberta-xlarge": functools.partial(bertscore_sentence.cos_sim_mat_f, embedder=sent_embedders["deberta-xlarge"]),
}

for mnli_name in ["deberta-large"]:
    for mnli_expr in [mnli.sim_expr.entail_contradict]:
        sim_mat_fs["mnli-{}-{}".format(mnli_name, mnli_expr.__name__)] = functools.partial(mnli.eval.mnli_sim_mat, classifiers=mnli_classifiers[mnli_name], expr=mnli_expr)

### METRICS ###

metrics: typing.Dict[str, dar_type.MetricComputeFunc] = dict()

for sim_mat_f_name, sim_mat_f in sim_mat_fs.items():
    for weight_f_name, weight_f in weight_fs.items():
        metrics["bertscore-sentence-{}".format(sim_mat_f_name)] = g2_metrics["bertscore-sentence-{}".format(sim_mat_f_name)]
        pagerank_get_idf_list = functools.partial(pagerank.get_idf, sim_mat_f=sim_mat_f, weight_f=weight_f)
        metrics["bertscore-sentence-pagerank-{}-{}".format(sim_mat_f_name, weight_f_name)] = functools.partial(g2_metrics["bertscore-sentence-{}".format(sim_mat_f_name)], idf_f = pagerank_get_idf_list)

del g2_metrics
