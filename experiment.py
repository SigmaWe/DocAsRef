import env
import os
import typing

import evalbase # the evaluation framework dependency

import dar_type # data type definitions for DocAsRef 
import dataset_config # configurations for datasets used in DocAsRef benchmarking 

# import mnli.sim_expr as mnli_sim
# from bertscore_sentence.metric import additional_metrics as bs_update_metrics
# from anyref.metric import additional_metrics as anyref_update_metrics
# from top.metric import additional_metrics as top_update_metrics
# from pagerank.metric import additional_metrics as pagerank_update_metrics
# from baseline.metric import additional_metrics as baseline_update_metrics


def enable_metrics(
    metric_dict: dar_type.MetricDict,
    metric_names: typing.List[str]) -> dar_type.MetricDict:

    metrics_enabled = {
        metric_name:metric_fn 
        for metric_name, metric_fn 
        in metric_dict.items() 
        if metric_name in metric_names} 
    return metrics_enabled

# Common configurations for all datasets

nlg_metrics = dict()

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
# nlg_metrics.update(pagerank_update_metrics(
#     bs_sent_metrics,
#     model_names=[
#         "mpnet",
#         "roberta",
#         "deberta-large",
#         # "deberta-xlarge"
#     ],
#     category="cos",
#     weight_f_names=[
#         "entropy",
#         "sum"
#     ]
# ))
# nlg_metrics.update(pagerank_update_metrics(
#     bs_sent_metrics,
#     model_names=[
#         "roberta",
#         "bart",
#         "deberta-large",
#         # "deberta-xlarge"
#     ],
#     category="mnli",
#     weight_f_names=[
#         "entropy",
#         "sum"
#     ],
#     mnli_exprs=[
#         mnli_sim.not_neutral,
#         mnli_sim.entail_only,
#         mnli_sim.entail_contradict
#     ]
# ))
# classic_metrics.update(classic_update_metrics([
#     "bertscore-deberta-large-mnli",
#     "bertscore-roberta-large-mnli",
#     "bertscore-bart-large-mnli",
#     # "bertscore-deberta-xlarge-mnli",
#     "bertscore-deberta-large",
#     "bertscore-bart-large",
#     "bertscore-roberta-base",
#     "bertscore-deberta-base",
#     "bertscore-bart-base",
# ]))
# nlg_metrics.update(classic_metrics)
# nlg_metrics.update(top_update_metrics(classic_metrics))

import classic.metric
classic_metrics = classic.metric.metrics 
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
classic_metrics_enabled = enable_metrics(classic_metrics, names_of_enabled_classic_metrics)

import bertscore_sentence.metric 
bertscore_sentence_metrics = bertscore_sentence.metric.metrics
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
#          [ "bertscore-sentence-mnli-{}-{}".format(model_name, mnli_expr)
#                for model_name in ["roberta-large-mnli", "bart-large-mnli", "deberta-large-mnli"]
#                for mnli_expr in ["not_neutral", "entail_only", "entail_contradict"]
#          ]
bertscore_sentence_metrics_enabled = enable_metrics(
    bertscore_sentence_metrics, 
    names_of_enabled_bertscore_sentence_metrics
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

# nlg_metrics.update(classic_metrics_enabled)
nlg_metrics.update(bertscore_sentence_metrics_enabled)

# Running experiments on different datasets

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
    # dataset_config.newsroom_config,
    # dataset_config.realsumm_abs_config,
    # dataset_config.realsumm_ext_config,
    # dataset_config.tac2010_config,
    # dataset_config.qags_config, 
    # dataset_config.frank_config, 
    # dataset_config.fastcc_config
]

for (exp_fn, exp_config) in experiment_fn_and_configs:
    exp_config.update(common_exp_config) 
    exp_fn(exp_config)
