import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from moverscore_v2 import get_idf_dict, word_mover_score
import evaluate
import functools
import dar_type


bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt", config_name="BLEURT-20", module_type="metric")


bertscore_compute: dar_type.MetricComputeFunc = functools.partial(bertscore.compute, lang='en', use_fast_tokenizer=True)
rouge_compute: dar_type.MetricComputeFunc = functools.partial(rouge.compute, use_aggregator=False)
bleurt_compute: dar_type.MetricComputeFunc = bleurt.compute


def moverscore_partial(predictions: dar_type.TextList, references: dar_type.TextList, n_gram: int) -> dar_type.MetricScoreDict:
    # https://github.com/AIPHES/emnlp19-moverscore
    # idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_hyp = get_idf_dict(predictions)
    # idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_ref = get_idf_dict(references)
    scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp,
                              stop_words=[], n_gram=n_gram, remove_subwords=True)
    return {"scores": scores}


moverscore_compute_1gram: dar_type.MetricComputeFunc = functools.partial(moverscore_partial, n_gram=1)
moverscore_compute_2gram: dar_type.MetricComputeFunc = functools.partial(moverscore_partial, n_gram=2)
