import sys

# from os import path
# file_path = path.abspath(__file__)
# sys.path.append(path.dirname(path.dirname(file_path)))

from moverscore_v2 import word_mover_score
import evaluate
import functools
import dar_type
from collections import defaultdict

# Load basic HF evaluate functions 
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")
bleurt = evaluate.load("bleurt", module_type="metric")

# Define variants of BERTScores 
bertscore_compute: dar_type.MetricComputeFunc = functools.partial(bertscore.compute, lang='en', use_fast_tokenizer=True)
rouge_compute: dar_type.MetricComputeFunc = functools.partial(rouge.compute, use_aggregator=False)
bleurt_compute: dar_type.MetricComputeFunc = bleurt.compute

def moverscore_partial(predictions: dar_type.TextList, references: dar_type.TextList, n_gram: int) -> dar_type.MetricScoreDict:
    # https://github.com/AIPHES/emnlp19-moverscore
    idf_dict_hyp = defaultdict(lambda: 1.)
    # idf_dict_hyp = get_idf_dict(predictions)
    idf_dict_ref = defaultdict(lambda: 1.)
    # idf_dict_ref = get_idf_dict(references)
    scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=n_gram, remove_subwords=True)
    return {"scores": scores}

moverscore_compute_1gram: dar_type.MetricComputeFunc = functools.partial(moverscore_partial, n_gram=1)
moverscore_compute_2gram: dar_type.MetricComputeFunc = functools.partial(moverscore_partial, n_gram=2)

metrics = {
    "rouge": rouge_compute,
    "bleurt": bleurt_compute,
    "moverscore-1gram": moverscore_compute_1gram,
    "moverscore-2gram": moverscore_compute_2gram,
    "bertscore-deberta-large-mnli": functools.partial(bertscore_compute, model_type="microsoft/deberta-large-mnli"),
    "bertscore-roberta-large-mnli": functools.partial(bertscore_compute, model_type="roberta-large-mnli"),
    "bertscore-bart-large-mnli": functools.partial(bertscore_compute, model_type="facebook/bart-large-mnli"),
    "bertscore-deberta-xlarge-mnli": functools.partial(bertscore_compute, model_type="microsoft/deberta-xlarge-mnli"),
    "bertscore-deberta-large": functools.partial(bertscore_compute, model_type="microsoft/deberta-large"),
    "bertscore-bart-large": functools.partial(bertscore_compute, model_type="facebook/bart-large"),
    "bertscore-roberta-base": functools.partial(bertscore_compute, model_type="roberta-base"),
    "bertscore-deberta-base": functools.partial(bertscore_compute, model_type="microsoft/deberta-base"),
    "bertscore-bart-base": functools.partial(bertscore_compute, model_type="facebook/bart-base"),
}
