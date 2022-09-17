from xml.dom.minidom import Document


approaches = ['trad', 'new']
models = ['rouge', 'bertscore', 'bleu', 'bleurt', 'bertscore-sentence']
models = ['rouge', 'bertscore', 'bleurt', 'bertscore-sentence']
# datasets = ['newsroom', 'realsumm_abs', 'realsumm_ext']
# eval_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore', 'bleurt', 'bertscore-sentence']
# corr_metrics = ['pearsonr', 'kendalltau', 'spearmanr']
corr_metrics = ['pearsonr', 'spearmanr']


document_column="ArticleText"
system_summary_column="SystemSummary"
reference_summary_column="ReferenceSummary"

# Newsroom 
human_metrics= ["CoherenceRating", "FluencyRating", "InformativenessRating", "RelevanceRating"]

# RealSumm
human_metrics = ["litepyramid_recall"] 

