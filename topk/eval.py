import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import typing
import classic.eval as classic
from type_piece import EvalPieces


def extract_topk_doc(ref_segments: typing.List[str], topk: int) -> str:
    topk_sents = ref_segments[0:topk]
    return " ".join(topk_sents)


def extract_topk(ref_segments_list: typing.List[typing.List[str]], topk: int) -> typing.List[str]:
    return [extract_topk_doc(ref_segments, topk) for ref_segments in ref_segments_list]


def data_topk(predictions: EvalPieces, references: EvalPieces, topk: int) -> typing.Tuple[str, str]:
    preds = predictions.raw_list
    refs = extract_topk(references.segments_list, topk)
    return preds, refs


def bertscore_compute(predictions: EvalPieces, references: EvalPieces, topk: int) -> typing.Dict:
    preds, refs = data_topk(predictions, references, topk)
    return classic.bertscore_partial(predictions=preds, references=refs)


def rouge_compute(predictions: EvalPieces, references: EvalPieces, topk: int) -> typing.Dict:
    preds, refs = data_topk(predictions, references, topk)
    return classic.rouge_partial(predictions=preds, references=refs)


def bleurt_compute(predictions: EvalPieces, references: EvalPieces, topk: int) -> typing.Dict:
    preds, refs = data_topk(predictions, references, topk)
    return classic.bleurt_partial(predictions=preds, references=refs)


def moverscore_compute(predictions: EvalPieces, references: EvalPieces, n_gram: int, topk: int) -> typing.Dict:
    preds, refs = data_topk(predictions, references, topk)
    return classic.moverscore_partial(predictions=preds, references=refs, n_gram=n_gram)
