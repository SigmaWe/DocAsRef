import os

import evalbase # be sure that evalbase is in your PYTHONPATH
data_path_root = os.path.join(evalbase.path, "dataloader")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# FIXME: What is the line above for? 

summeval_config = {
    "dataset_name": "summeval",
    "human_metrics": ["consistency", "relevance", "coherence", "fluency"],
    "docID_column": "id",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary_0",  # the id ranges from 0 to 10
    "is_multi": False, # must be False for SummEval
    "data_path": os.path.join(data_path_root, "summeval_annotations.aligned.paired.scored.jsonl"),    
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

realsumm_abs_config = {
    "dataset_name": "realsumm_abs",
    "human_metrics": ["litepyramid_recall"],
    "docID_column": "doc_id",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary",
    "data_path": os.path.join(data_path_root,  "abs.pkl"),  # you need to get this file. See ReadMe.
    "result_path_root": "./results/",
    "precalc_metrics": ['rouge_1_f_score', 'rouge_2_recall', 'rouge_l_recall', 'rouge_2_precision',
                                'rouge_2_f_score', 'rouge_1_precision', 'rouge_1_recall', 'rouge_l_precision',
                                'rouge_l_f_score', 'js-2', 'mover_score', 'bert_recall_score', 'bert_precision_score',
                                'bert_f_score'],
    "debug": False                    
}

realsumm_ext_config = realsumm_abs_config.copy()
realsumm_ext_config["dataset_name"] = "realsumm_ext"
realsumm_ext_config["data_path"] = os.path.join(data_path_root,  "ext.pkl")  # you need to get this file. See ReadMe.

newsroom_config = {
    "dataset_name": "newsroom",
    "human_metrics": ["InformativenessRating", "RelevanceRating", "CoherenceRating", "FluencyRating"],
    "docID_column": "ArticleID",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary",
    "human_eval_only_path": os.path.join(data_path_root,  "newsroom-human-eval.csv"),  # you need to get this file. See ReadMe.
    "refs_path": os.path.join(data_path_root, "test.jsonl"),  # you need to get this file. See ReadMe.
    "human_eval_w_refs_path": os.path.join(data_path_root,  "newsroom_human_eval_with_refs.csv"), 
    "precalc_metrics": [],
}

tac2010_config = {
    "dataset_name": "tac2010",
    "human_metrics": ["Pyramid", "Linguistic", "Overall"],
    "approaches": ["new"],
    "docID_column": "docsetID",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary",
    "data_path": os.path.join(data_path_root,  "TAC2010"),  # This is a folder. See ReadMe.
    "precalc_metrics": [],
    "is_multi": True, # very important for TAC2010, multi-document summarization
    "debug": False
}

qags_config = {
    "human_metrics": ["human"],
    "docID_column": "id",
    "document_column": "doc",
    "system_summary_column": "sum",
    # FIXME only one summary is available
    "reference_summary_column": "sum",
    "approaches": ["new"],
    "data_path": os.path.join(data_path_root, "qags/data"),
    "precalc_metrics": []
}

frank_config = {
    "human_metrics": ["human"],
    "docID_column": "id",
    "document_column": "doc",
    "system_summary_column": "sum",
    "reference_summary_column": "ref",
    "approaches": ["new"],
    "data_path": os.path.join(data_path_root, "frank/data"),
    "precalc_metrics": []
}

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
    "data_path": os.path.join(data_path_root, "factCC/data_pairing/data/generated_data/data-clipped"),
    "precalc_metrics": []
}
