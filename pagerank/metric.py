# PageRank

import functools
import typing

import numpy as np
import scipy.stats


from mnli.classifiers import mnli_classifiers
import bertscore_sentence.embedders
import bertscore_sentence.eval 
import bertscore_sentence.metric  
import dar_type
import mnli.eval
import mnli.sim_expr
import pagerank.eval 

# Deprecated function. Remove in three months. 
def additional_metrics(
        bs_sent_metrics: dar_type.MetricDict, 
        model_names: typing.List[str], 
        category: typing.Literal["cos", "mnli"], 
        weight_f_names: typing.List[str], 
        mnli_exprs: typing.List[dar_type.MNLISimilarityExpression] = []
        ) -> dar_type.MetricDict:
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
            sim_mat_fs["cos-{}".format(model_name)] = functools.partial(bertscore_sentence.eval.get_similarity_matrix_cos, embedder=bertscore_sentence.embedders.sent_embedders[model_name])
    elif category == "mnli":
        for mnli_name in model_names:
            for mnli_expr in mnli_exprs:
                sim_mat_fs["mnli-{}-{}".format(mnli_name, mnli_expr.__name__)] = functools.partial(bertscore_sentence.eval.get_similarity_matrix_mnli, classifiers=mnli_classifiers[mnli_name], expr=mnli_expr)

    metrics = dict()

    for sim_mat_f_name, sim_mat_f in sim_mat_fs.items():
        for weight_f_name, weight_f in weight_fs.items():
            pagerank_get_idf_list = functools.partial(pagerank.eval.get_idf, sim_mat_f=sim_mat_f, weight_f=weight_f)
            metrics["bertscore-sentence-pagerank-{}-{}".format(sim_mat_f_name, weight_f_name)] = functools.partial(bs_sent_metrics["bertscore-sentence-{}".format(sim_mat_f_name)], idf_f = pagerank_get_idf_list)

    return metrics

weight_fs: typing.Dict[str, dar_type.SentenceWeightFunction] = {
    "entropy": scipy.stats.entropy,
    "sum": np.sum,
}

metrics = dict()

# FIXME: Simply add IDF_F onto preexisting metrics 

# dot-product based sentence metrics 
for model_name in bertscore_sentence.embedders.sent_embedders.keys(): # mpnet, roberta-large, deberta-large
    for weight_f_name, weight_f in weight_fs.items():
        metrics[f"bertscore-sentence-pagerank-cos-{model_name}-{weight_f_name}"] = \
            functools.partial(
                bertscore_sentence.eval.compute_cos, 
                embedder=bertscore_sentence.embedders.sent_embedders[model_name], 
                idf_f = functools.partial(
                    pagerank.eval.get_idf,
                    weight_f=weight_f
                )   
            )

        # 'bertscore-sentence-cos-mpnet',
        # 'bertscore-sentence-cos-roberta-large',
        # 'bertscore-sentence-cos-deberta-large', 

# MNLI probability based sentence metrics 
for model_name in mnli_classifiers.keys(): # roberta-large-mnli, bart-large-mnli, deberta-large-mnli 
    for mnli_expr in [mnli.sim_expr.not_neutral, mnli.sim_expr.entail_only, mnli.sim_expr.entail_contradict]:
        for weight_f_name, weight_f in weight_fs.items():
            metrics[f"bertscore-sentence-pagerank-mnli-{model_name}-{mnli_expr.__name__}-{weight_f_name}"] = \
                functools.partial(
                    mnli.eval.compute_mnli, 
                    classifiers=mnli_classifiers[model_name], 
                    expr=mnli_expr, 
                    idf_f = functools.partial(
                        pagerank.eval.get_idf,
                        weight_f=weight_f
                    )
                )