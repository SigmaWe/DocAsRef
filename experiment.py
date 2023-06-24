import env
import os

from config import *
import mnli.sim_expr as mnli_sim
# from classic.metric import additional_metrics as classic_update_metrics
# from bertscore_sentence.metric import additional_metrics as bs_update_metrics
# from anyref.metric import additional_metrics as anyref_update_metrics
# from top.metric import additional_metrics as top_update_metrics
# from pagerank.metric import additional_metrics as pagerank_update_metrics
from baseline.metric import additional_metrics as baseline_update_metrics

# Common configurations for all datasets

nlg_metrics = dict()
# classic_metrics = dict()
# classic_metrics.update(classic_update_metrics([
#     "bertscore",
#     "rouge",
#     "bleurt",  # requires datasets-2.10.0 per https://github.com/huggingface/evaluate/issues/449
#     "moverscore-1gram",
#     "moverscore-2gram"
# ]))
# nlg_metrics.update(classic_metrics)
# bs_sent_metrics = dict()
# bs_sent_metrics.update(bs_update_metrics([
#     "mpnet",
#     "roberta",
#     "deberta-large",
#     # "deberta-xlarge"
# ], "cos"))
# bs_sent_metrics.update(bs_update_metrics(
#     model_names=[
#         "roberta",
#         "bart",
#         "deberta-large",
#         # "deberta-xlarge"
#     ],
#     category="mnli",
#     mnli_exprs=[
#         mnli_sim.not_neutral,
#         mnli_sim.entail_only,
#         mnli_sim.entail_contradict
#     ]
# ))
# nlg_metrics.update(bs_sent_metrics)
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
baseline_metrics = dict()
baseline_metrics.update(baseline_update_metrics([
    "sacrebleu",
    "meteor",
    "bart",
    # "reuse",  # TOFIX
    # "sdc*",  # Slow
    "smd"  # Linux only
]))
nlg_metrics.update(baseline_metrics)


common_exp_config = {
    "nlg_metrics" : nlg_metrics,
    "corr_metrics" : ["spearmanr", "pearsonr", "kendalltau"],
    "approaches": ["trad", "new"],
    "eval_levels": ["summary", "system"],
    "result_path_root": "./results/",
    "debug": False
}

### Example configurations for SummEval ###
summeval_config.update(common_exp_config)
evalbase.summeval.main(summeval_config)

## End of SummEval example ##

### Example configurations for the Newsroom dataset ###
newsroom_config.update(common_exp_config)
evalbase.newsroom.main(newsroom_config)
### End of configuration for the Newsroom dataset ###

### Example configurations for the ABStractive track in Realsumm ###
realsumm_abs_config.update(common_exp_config)
evalbase.realsumm.main(realsumm_abs_config)

### End of example for the ABStractive track in Realsumm ###

### Example configurations for the EXtractive track in Realsumm ###
realsumm_ext_config.update(common_exp_config)
evalbase.realsumm.main(realsumm_ext_config)
### End of example for the EXtractive track in Realsumm ###

### Example configurations for the TAC 2010 dataset ###
tac2010_config.update(common_exp_config)
evalbase.tac2010.main(tac2010_config)
### End of example for the TAC 2010 dataset ###

### Example configurations for the QAGS dataset ###
print("factcc.qags_main(), size: 235")
qags_config.update(common_exp_config)
evalbase.qaqs.main(qags_config)
### End of example for the QAGS dataset ###

### Example configurations for the Frank dataset ###
print("factcc.frank_main(): size: 1250")
frank_config.update(common_exp_config)
evalbase.frank.main(frank_config)
### End of example for the Frank dataset ###

### Example configurations for the FastCC dataset ###
print("factcc.factCC_main(): size: large")
fastcc_config.update(common_exp_config)
evalbase.factcc.main(fastcc_config)
### End of example for the FastCC dataset ###
