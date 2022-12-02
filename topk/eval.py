import sys
from os import path

file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import typing
from dar_env import nlp, bertscore, rouge, bleurt


def extract_topk_doc(ref: str, topk: int) -> str:
    doc = nlp(ref)
    doc_sents = [sent.text for sent in doc.sents]
    topk_sents = doc_sents[0:topk]
    return " ".join(topk_sents)


def extract_topk(references: typing.List[str], topk: int) -> typing.List[str]:
    return [extract_topk_doc(ref, topk) for ref in references]


def bertscore_compute(predictions: typing.List[str], references: typing.List[str], topk: int) -> typing.Dict:
    refs = extract_topk(references, topk)
    return bertscore.compute(
        predictions=predictions,
        references=refs,
        lang='en',
        use_fast_tokenizer=True
    )


def rouge_compute(predictions: typing.List[str], references: typing.List[str], topk: int) -> typing.Dict:
    refs = extract_topk(references, topk)
    return rouge.compute(
        predictions=predictions,
        references=refs,
        use_aggregator=False
    )


def bleurt_compute(predictions: typing.List[str], references: typing.List[str], topk: int) -> typing.Dict:
    refs = extract_topk(references, topk)
    return bleurt.compute(
        predictions=predictions,
        references=refs
    )
