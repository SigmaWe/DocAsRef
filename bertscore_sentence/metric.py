# bertscore-sentence (cos, mnli)

import dar_type
import typing

### MODELS ###

from mnli.classifiers import mnli_classifiers
from bertscore_sentence.embedders import sent_embedders
# from embedders import sent_embedders

### METRICS ###

import functools
import bertscore_sentence.eval
import mnli.eval
import mnli.sim_expr

# FIXME: Deprecated function. Keep for reference. To be removed in 3 months. 
def additional_metrics(
    model_names: typing.List[str], 
    category: typing.Literal["cos", "mnli"], 
    mnli_exprs: typing.List[dar_type.MNLISimilarityExpression] = []
    ) -> dar_type.MetricDict:
    metrics = dict()

    if category == "cos":
        for model_name in model_names:
            metrics["bertscore-sentence-cos-{}".format(model_name)] = functools.partial(bertscore_sentence.compute_cos, embedder=sent_embedders[model_name])
    elif category == "mnli":
        for mnli_name in model_names:
            for mnli_expr in mnli_exprs:
                metrics["bertscore-sentence-mnli-{}-{}".format(mnli_name, mnli_expr.__name__)] = functools.partial(mnli.eval.bertscore_sentence_compute, classifiers=mnli_classifiers[mnli_name], expr=mnli_expr)

    return metrics

metrics = dict()

# dot-product based sentence metrics 
for model_name in sent_embedders.keys(): # mpnet, roberta-large, deberta-large
    metrics["bertscore-sentence-cos-{}".format(model_name)] = \
        functools.partial(
            bertscore_sentence.eval.compute_cos, 
            embedder=sent_embedders[model_name]   
        )

        # 'bertscore-sentence-cos-mpnet',
        # 'bertscore-sentence-cos-roberta-large',
        # 'bertscore-sentence-cos-deberta-large', 

# MNLI probability based sentence metrics 
for model_name in mnli_classifiers.keys(): # roberta-large-mnli, bart-large-mnli, deberta-large-mnli 
    for mnli_expr in [mnli.sim_expr.not_neutral, mnli.sim_expr.entail_only, mnli.sim_expr.entail_contradict]:
        metrics["bertscore-sentence-mnli-{}-{}".format(model_name, mnli_expr.__name__)] = \
            functools.partial(
                mnli.eval.compute_mnli, 
                classifiers=mnli_classifiers[model_name], 
                expr=mnli_expr
            )
        
        # 'bertscore-sentence-mnli-roberta-large-mnli-not_neutral',
        # 'bertscore-sentence-mnli-roberta-large-mnli-entail_only',
        # 'bertscore-sentence-mnli-roberta-large-mnli-entail_contradict',
        # 'bertscore-sentence-mnli-bart-large-mnli-not_neutral',
        # 'bertscore-sentence-mnli-bart-large-mnli-entail_only',
        # 'bertscore-sentence-mnli-bart-large-mnli-entail_contradict',
        # 'bertscore-sentence-mnli-deberta-large-mnli-not_neutral',
        # 'bertscore-sentence-mnli-deberta-large-mnli-entail_only',
        # 'bertscore-sentence-mnli-deberta-large-mnli-entail_contradict'
