# bertscore-sentence (cos, mnli)

import dar_type
import typing

### MODELS ###

from mnli.classifiers import mnli_classifiers
from bertscore_sentence.embedders import sent_embedders

### METRICS ###

import functools
import bertscore_sentence.eval as bertscore_sentence
import mnli.eval
import mnli.sim_expr

def additional_metrics(model_names: typing.List[str], category: typing.Literal["cos", "mnli"], mnli_exprs: typing.List[dar_type.MNLISimilarityExpression] = []) -> dar_type.MetricDict:
    metrics = dict()

    if category == "cos":
        for model_name in model_names:
            metrics["bertscore-sentence-cos-{}".format(model_name)] = functools.partial(bertscore_sentence.compute_cos, embedder=sent_embedders[model_name])
    elif category == "mnli":
        for mnli_name in model_names:
            for mnli_expr in mnli_exprs:
                metrics["bertscore-sentence-mnli-{}-{}".format(mnli_name, mnli_expr.__name__)] = functools.partial(mnli.eval.bertscore_sentence_compute, classifiers=mnli_classifiers[mnli_name], expr=mnli_expr)

    return metrics
