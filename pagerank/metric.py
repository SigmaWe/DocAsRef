# PageRank

from mnli.classifiers import mnli_classifiers
from bertscore_sentence.embedders import sent_embedders
import bertscore_sentence.eval as bertscore_sentence
import dar_type
import functools
import mnli.eval
import mnli.sim_expr
import numpy as np
import pagerank.eval as pagerank
import scipy.stats
import typing

def additional_metrics(bs_sent_metrics: dar_type.MetricDict, model_names: typing.List[str], category: typing.Literal["cos", "mnli"], weight_f_names: typing.List[str], mnli_exprs: typing.List[dar_type.MNLISimilarityExpression] = []) -> dar_type.MetricDict:
    metrics = dict()

    weight_fs: typing.Dict[str, dar_type.SentenceWeightFunction] = {
        "entropy": scipy.stats.entropy,
        "sum": np.sum,
    }

    for weight_f_name in list(weight_fs.keys()):
        if weight_f_name not in weight_f_names:
            weight_fs.pop(weight_f_name)

    sim_mat_fs: typing.Dict[str, dar_type.SimpleSimilarityMatrixFunc] = dict()

    if category == "cos":
        for model_name in model_names:
            sim_mat_fs["cos-{}".format(model_name)] = functools.partial(bertscore_sentence.cos_sim_mat_f, embedder=sent_embedders[model_name])
    elif category == "mnli":
        for mnli_name in model_names:
            for mnli_expr in mnli_exprs:
                sim_mat_fs["mnli-{}-{}".format(mnli_name, mnli_expr.__name__)] = functools.partial(mnli.eval.mnli_sim_mat, classifiers=mnli_classifiers[mnli_name], expr=mnli_expr)

    metrics = dict()

    for sim_mat_f_name, sim_mat_f in sim_mat_fs.items():
        for weight_f_name, weight_f in weight_fs.items():
            pagerank_get_idf_list = functools.partial(pagerank.get_idf, sim_mat_f=sim_mat_f, weight_f=weight_f)
            metrics["bertscore-sentence-pagerank-{}-{}".format(sim_mat_f_name, weight_f_name)] = functools.partial(bs_sent_metrics["bertscore-sentence-{}".format(sim_mat_f_name)], idf_f = pagerank_get_idf_list)

    return metrics
