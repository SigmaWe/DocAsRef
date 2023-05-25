import env
import os
import sys

evalbase_path = "/home/turx/EvalBase"
sys.path.append(evalbase_path)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import evalbase
import mnli.sim_expr as mnli_sim
from classic.metric import additional_metrics as classic_update_metrics
from bertscore_sentence.metric import additional_metrics as bs_update_metrics
from anyref.metric import additional_metrics as anyref_update_metrics
from top.metric import additional_metrics as top_update_metrics
from pagerank.metric import additional_metrics as pagerank_update_metrics

# Common configurations for all datasets

nlg_metrics = dict()
classic_metrics = dict()
classic_metrics.update(classic_update_metrics([
    "bertscore",
    "rouge",
    # "bleurt"
]))
nlg_metrics.update(classic_metrics)
bs_sent_metrics = dict()
bs_sent_metrics.update(bs_update_metrics([
    "mpnet",
    "roberta",
    "deberta-large",
    # "deberta-xlarge"
], "cos"))
bs_sent_metrics.update(bs_update_metrics(
    model_names=[
        "roberta",
        "bart",
        "deberta-large",
        # "deberta-xlarge"
    ],
    category="mnli",
    mnli_exprs=[
        mnli_sim.not_neutral,
        mnli_sim.entail_only,
        mnli_sim.entail_contradict
    ]
))
nlg_metrics.update(bs_sent_metrics)
nlg_metrics.update(anyref_update_metrics(classic_metrics, [
    "bart",
    "pegasus-xsum",
    "pegasus-newsroom",
    "pegasus-cnndm",
    "pegasus-large"
]))
nlg_metrics.update(anyref_update_metrics(bs_sent_metrics, [
    "pegasus-newsroom"
]))
nlg_metrics.update(pagerank_update_metrics(
    bs_sent_metrics,
    model_names=[
        "mpnet",
        "roberta",
        "deberta-large",
        # "deberta-xlarge"
    ],
    category="cos",
    weight_f_names=[
        "entropy",
        "sum"
    ]
))
nlg_metrics.update(pagerank_update_metrics(
    bs_sent_metrics,
    model_names=[
        "roberta",
        "bart",
        "deberta-large",
        # "deberta-xlarge"
    ],
    category="mnli",
    weight_f_names=[
        "entropy",
        "sum"
    ],
    mnli_exprs=[
        mnli_sim.not_neutral,
        mnli_sim.entail_only,
        mnli_sim.entail_contradict
    ]
))
classic_metrics.update(classic_update_metrics([
    "bertscore-deberta-large-mnli",
    "bertscore-roberta-large-mnli",
    "bertscore-bart-large-mnli",
    # "bertscore-deberta-xlarge-mnli",
    "bertscore-deberta-large",
    "bertscore-bart-large"
]))
nlg_metrics.update(classic_metrics)
nlg_metrics.update(top_update_metrics(classic_metrics))

# TODO: fix moverscore
# nlg_metrics.update(classic_update_metrics([
#     "moverscore-1gram",
#     "moverscore-2gram"
# ]))


common_exp_config = {
    "nlg_metrics" : nlg_metrics,
    "corr_metrics" : ["spearmanr", "pearsonr", "kendalltau"],
    "approaches": ["trad", "new"],
    "eval_levels": ["summary", "system"],
    "result_path_root": "./results/",
    "debug": False
}

### Example configurations for SummEval ###
summeval_config = {
    "dataset_name": "summeval",
    "human_metrics": ["consistency", "relevance", "coherence", "fluency"],
    "docID_column": "id",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary_0",  # the id ranges from 0 to 10
    "is_multi": False, # must be False for SummEval
    "data_path": os.path.join(evalbase.path, "dataloader", "summeval_annotations.aligned.paired.scored.jsonl"),    
    "precalc_metrics": [  # keys from original SummEval json file
        'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score',
        'rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score',
        'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f_score',
        'rouge_we_1_p', 'rouge_we_1_r', 'rouge_we_1_f',
        'rouge_we_2_p', 'rouge_we_2_r', 'rouge_we_2_f',
        'meteor', 'cider', 's3_pyr', 's3_resp',
        'mover_score', 'sentence_movers_glove_sms', 'bleu',
        'bert_score_precision', 'bert_score_recall', 'bert_score_f1',
        'blanc', 'summaqa_avg_prob', 'summaqa_avg_fscore', 'supert'], 
    "debug": False
}
summeval_config.update(common_exp_config)
evalbase.summeval.main(summeval_config)

## End of SummEval example ##

### Example configurations for the ABStractive track in Realsumm ###
realsumm_abs_config = {
    "dataset_name": "realsumm_abs",
    "human_metrics": ["litepyramid_recall"],
    "docID_column": "doc_id",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary",
    "data_path": os.path.join(evalbase.path, "dataloader", "abs.pkl"),  # you need to get this file. See ReadMe.
    "result_path_root": "./results/",
    "precalc_metrics": ['rouge_1_f_score', 'rouge_2_recall', 'rouge_l_recall', 'rouge_2_precision',
                                'rouge_2_f_score', 'rouge_1_precision', 'rouge_1_recall', 'rouge_l_precision',
                                'rouge_l_f_score', 'js-2', 'mover_score', 'bert_recall_score', 'bert_precision_score',
                                'bert_f_score'],
    "debug": False                    
}
realsumm_abs_config.update(common_exp_config)
evalbase.realsumm.main(realsumm_abs_config)

### End of example for the ABStractive track in Realsumm ###

### Example configurations for the EXtractive track in Realsumm ###
realsumm_ext_config = realsumm_abs_config
realsumm_ext_config["dataset_name"] = "realsumm_ext"
realsumm_ext_config["data_path"] = os.path.join(evalbase.path, "dataloader", "ext.pkl")  # you need to get this file. See ReadMe.
evalbase.realsumm.main(realsumm_ext_config)
### End of example for the EXtractive track in Realsumm ###

### Example configurations for the Newsroom dataset ### 
newsroom_config = {
    "dataset_name": "newsroom",
    "human_metrics": ["InformativenessRating", "RelevanceRating", "CoherenceRating", "FluencyRating"],
    "docID_column": "ArticleID",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary",
    "human_eval_only_path": os.path.join(evalbase.path, "dataloader", "newsroom-human-eval.csv"),  # you need to get this file. See ReadMe.
    "refs_path": os.path.join(evalbase.path, "dataloader", "test.jsonl"),  # you need to get this file. See ReadMe.
    "human_eval_w_refs_path": os.path.join(evalbase.path, "dataloader", "newsroom_human_eval_with_refs.csv"), 
    "precalc_metrics": [],
}
# print (newsroom_config)
newsroom_config.update(common_exp_config)
evalbase.newsroom.main(newsroom_config)
### End of configuration for the Newsroom dataset ###

### Example configurations for the TAC 2010 dataset ###
tac2010_config = {
    "dataset_name": "tac2010",
    "human_metrics": ["Pyramid", "Linguistic", "Overall"],
    "approaches": ["new"],
    "docID_column": "docsetID",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary",
    "data_path": os.path.join(evalbase.path, "dataloader", "TAC2010"),  # This is a folder. See ReadMe.
    "precalc_metrics": [],
    "is_multi": True, # very important for TAC2010, multi-document summarization
    "debug": False
}    
tac2010_config.update(common_exp_config)
evalbase.tac2010.main(tac2010_config)
### End of example for the TAC 2010 dataset ###

### Example configurations for the QAGS dataset ###
print("factcc.qags_main(), size: 235")
qags_config = {
    "human_metrics": ["human"],
    "docID_column": "id",
    "document_column": "doc",
    "system_summary_column": "sum",
    # FIXME only one summary is available
    "reference_summary_column": "sum",
    "approaches": ["new"],
    "data_path": os.path.join(evalbase.path, "dataloader/qags/data"),
    "precalc_metrics": []
}
qags_config.update(common_exp_config)
evalbase.qaqs.main(qags_config)
### End of example for the QAGS dataset ###

### Example configurations for the Frank dataset ###
print("factcc.frank_main(): size: 1250")
frank_config = {
    "human_metrics": ["human"],
    "docID_column": "id",
    "document_column": "doc",
    "system_summary_column": "sum",
    "reference_summary_column": "ref",
    "approaches": ["new"],
    "data_path": os.path.join(evalbase.path, "dataloader/frank/data"),
    "precalc_metrics": []
}
frank_config.update(common_exp_config)
evalbase.frank.main(frank_config)
### End of example for the Frank dataset ###

### Example configurations for the FastCC dataset ###
print("factcc.factCC_main(): size: large")
fastcc_config = {
    "human_metrics": ["human"],
    "docID_column": "id",
    "document_column": "doc",
    "system_summary_column": "sum",
    # FIXME only one summary is available
    "reference_summary_column": "sum",
    "approaches": ["new"],
    "split": {
        "train": "data-train.jsonl",
        "dev": "data-dev.jsonl",
        "test": "data-test.jsonl"
    },
    "data_path": os.path.join(evalbase.path, "dataloader/factCC/data_pairing/data/generated_data/data-clipped"),
    "precalc_metrics": []
}
fastcc_config.update(common_exp_config)
evalbase.factcc.main(fastcc_config)
### End of example for the FastCC dataset ###
