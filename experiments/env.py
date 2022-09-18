

approaches = ['trad', 'new']
# models = ['rouge', 'bertscore', 'bleu', 'bleurt', 'bertscore-sentence']
models = ['rouge',  'bleurt', 'bertscore', 'bertscore-sentence']
# datasets = ['newsroom', 'realsumm_abs', 'realsumm_ext']
# eval_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore', 'bleurt', 'bertscore-sentence']
# corr_metrics = ['pearsonr', 'kendalltau', 'spearmanr']
corr_metrics = ['pearsonr', 'spearmanr']


document_column="ArticleText"
system_summary_column="SystemSummary"
reference_summary_column="ReferenceSummary"

# Newsroom 
human_metrics= ["InformativenessRating", "RelevanceRating", "CoherenceRating", "FluencyRating"]

# RealSumm
human_metrics = ["litepyramid_recall"] 

# SummEval 
human_metrics = ['consistency', 'relevance', 'coherence', 'fluency'] 
reference_summary_column="ReferenceSummary_0"  # the id ranges from 0 to 10