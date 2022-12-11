import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from type_piece import EvalPieces
import typing
from dar_env import bertscore, rouge, bleurt, get_idf_dict, word_mover_score
import functools


bertscore_partial = functools.partial(bertscore.compute, lang='en', use_fast_tokenizer=True)
rouge_partial = functools.partial(rouge.compute, use_aggregator=False)
bleurt_partial = functools.partial(bleurt.compute)


def moverscore_partial(predictions: typing.List[str], references: typing.List[str], n_gram: int) -> typing.Dict:
    # https://github.com/AIPHES/emnlp19-moverscore
    idf_dict_hyp = get_idf_dict(predictions) # idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = get_idf_dict(references) # idf_dict_ref = defaultdict(lambda: 1.)
    scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp, \
                          stop_words=[], n_gram=n_gram, remove_subwords=True)
    return { "scores": scores }


def data_raw(predictions: EvalPieces, references: EvalPieces) -> typing.Tuple[typing.List[str], typing.List[str]]:
    preds = predictions.raw_list
    refs = references.raw_list
    return preds, refs


def bertscore_compute(predictions: EvalPieces, references: EvalPieces) -> typing.Dict:
    preds, refs = data_raw(predictions, references)
    return bertscore_partial(predictions=preds, references=refs)


def rouge_compute(predictions: EvalPieces, references: EvalPieces) -> typing.Dict:
    preds, refs = data_raw(predictions, references)
    return rouge_partial(predictions=preds, references=refs)


def bleurt_compute(predictions: EvalPieces, references: EvalPieces) -> typing.Dict:
    preds, refs = data_raw(predictions, references)
    return bleurt_partial(predictions=preds, references=refs)


def moverscore_compute(predictions: EvalPieces, references: EvalPieces, n_gram: int) -> typing.Dict:
    preds, refs = data_raw(predictions, references)
    return moverscore_partial(predictions=preds, references=refs, n_gram=n_gram)
