import env
import os
import typing

import evalbase # the evaluation framework dependency

import dar_type # data type definitions for DocAsRef 
import dataset_config # configurations for datasets used in DocAsRef benchmarking 

# from anyref.metric import additional_metrics as anyref_update_metrics
# from top.metric import additional_metrics as top_update_metrics
# from baseline.metric import additional_metrics as baseline_update_metrics

# Metrics to evaluate # 

def enable_metrics(
    metric_dict: dar_type.MetricDict,
    metric_names: typing.List[str]) -> dar_type.MetricDict:

    metrics_enabled = {
        metric_name:metric_fn 
        for metric_name, metric_fn 
        in metric_dict.items() 
        if metric_name in metric_names} 
    return metrics_enabled


# nlg_metrics.update(anyref_update_metrics(classic_metrics, [
#     "bart",
#     "pegasus-xsum",
#     "pegasus-newsroom",
#     "pegasus-cnndm",
#     "pegasus-large"
# ]))
# nlg_metrics.update(anyref_update_metrics(bs_sent_metrics, [
#     "pegasus-newsroom"
# ]))


import classic.metric
# To disable/enable benchmarking the metrics, 
# comment/uncomment the corresponding line below
names_of_enabled_classic_metrics = [
    "rouge", 
    "bleurt", # requires datasets-2.10.0 per https://github.com/huggingface/evaluate/issues/449
    "moverscore-1gram", 
    "moverscore-2gram", 
    "bertscore-roberta-base", 
    "bertscore-deberta-base", 
    "bertscore-bart-base", 
    "bertscore-deberta-large",
    "bertscore-roberta-large",
    "bertscore-bart-large",
    "bertscore-deberta-large-mnli",
    "bertscore-roberta-large-mnli",
    "bertscore-bart-large-mnli",
    ""
    ]
classic_metrics_enabled = enable_metrics(
    classic.metric.metrics, 
    names_of_enabled_classic_metrics
    )

import bertscore_sentence.metric 
# To disable/enable benchmarking the metrics, 
# comment/uncomment the corresponding line below
names_of_enabled_bertscore_sentence_metrics =     [
        'bertscore-sentence-cos-mpnet',
        'bertscore-sentence-cos-roberta-large',
        'bertscore-sentence-cos-deberta-large', 
        'bertscore-sentence-mnli-roberta-large-mnli-not_neutral',
        'bertscore-sentence-mnli-roberta-large-mnli-entail_only',
        'bertscore-sentence-mnli-roberta-large-mnli-entail_contradict',
        'bertscore-sentence-mnli-bart-large-mnli-not_neutral',
        'bertscore-sentence-mnli-bart-large-mnli-entail_only',
        'bertscore-sentence-mnli-bart-large-mnli-entail_contradict',
        'bertscore-sentence-mnli-deberta-large-mnli-not_neutral',
        'bertscore-sentence-mnli-deberta-large-mnli-entail_only',
        'bertscore-sentence-mnli-deberta-large-mnli-entail_contradict'
    ]
# names_of_enabled_bertscore_sentence_metrics = \
#         [ "bertscore-sentence-cos-{}".format(model_name)
#              for model_name in ["mpnet", "roberta-large", "deberta-large"]
#         ] + \
#          [ "bertscore-sentence-mnli-{}-{}".format(model_name, mnli_expr)
#                for model_name in ["roberta-large-mnli", "bart-large-mnli", "deberta-large-mnli"]
#                for mnli_expr in ["not_neutral", "entail_only", "entail_contradict"]
#          ]
bertscore_sentence_metrics_enabled = enable_metrics(
    bertscore_sentence.metric.metrics, 
    names_of_enabled_bertscore_sentence_metrics
    )


import pagerank.metric
# To disable/enable benchmarking the metrics,
# comment/uncomment/edit the corresponding line below
names_of_enabled_pagerank_metrics = \
    [ f"bertscore-sentence-pagerank-mnli-{model_name}-{mnli_expr}-{weight_f_name}"
        for model_name in ["roberta-large-mnli", "bart-large-mnli", "deberta-large-mnli"]
        for mnli_expr in ["not_neutral", "entail_only", "entail_contradict"]
        for weight_f_name in ["entropy", "sum"]
    ] + \
    [ f"bertscore-sentence-pagerank-cos-{model_name}-{weight_f_name}"
            for model_name in ["mpnet", "roberta-large", "deberta-large"]
            for weight_f_name in ["entropy", "sum"]
            # for model_name in ["mpnet"]
            # for weight_f_name in [ "sum"]
    ]
pagerank_metrics_enabled = enable_metrics(
    pagerank.metric.metrics,
    names_of_enabled_pagerank_metrics
    )

# baseline_metrics = dict()
# baseline_metrics.update(baseline_update_metrics([
#     "sacrebleu",
#     "meteor",
#     "bart",
#     # "reuse",  # TOFIX
#     # "sdc*",  # Slow
#     "smd"  # Linux only
# ]))
# nlg_metrics.update(baseline_metrics)

nlg_metrics = dict()
nlg_metrics.update(classic_metrics_enabled)
nlg_metrics.update(bertscore_sentence_metrics_enabled)
nlg_metrics.update(pagerank_metrics_enabled)

# Running experiments on different datasets #
## Experiment configurations for all datasets 
common_exp_config = {
    "nlg_metrics" : nlg_metrics,
    "corr_metrics" : ["spearmanr", "pearsonr", "kendalltau"],
    # "approaches": ["trad", "new"],
    "approaches": ["trad"],
    # "eval_levels": ["summary", "system"],
    "eval_levels": ["summary"],
    "result_path_root": "./results/",
    "debug": False, 
}

import dataset_config

# To disable/enable the experiment on one dataset, 
# comment/uncomment the corresponding line below
experiment_fn_and_configs = [
    (evalbase.summeval.main, dataset_config.summeval_config),
    (evalbase.newsroom.main, dataset_config.newsroom_config),
    # (evalbase.realsumm.main, dataset_config.realsumm_abs_config),
    # (evalbase.realsumm.main, dataset_config.realsumm_ext_config),
    # (evalbase.tac2010.main, dataset_config.tac2010_config),
    # (evalbase.qags.main, dataset_config.qags_config),
    # (evalbase.frank.main, dataset_config.frank_config),
    # (evalbase.fastcc.main, dataset_config.fastcc_config),
]

for (exp_fn, exp_config) in experiment_fn_and_configs:
    exp_config.update(common_exp_config) 
    exp_fn(exp_config)
