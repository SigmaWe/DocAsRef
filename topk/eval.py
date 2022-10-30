import typing
import evaluate


def extract_topk_sentence(ref: str, topk: int) -> str:
    topk_tokens = ref.split(" ")[0:topk]
    return " ".join(topk_tokens)


def extract_topk(references: typing.List[str], topk: int) -> typing.List[str]:
    return [extract_topk_sentence(ref, topk) for ref in references]


def bertscore_compute(predictions: typing.List[str], references: typing.List[str], topk: int) -> typing.Dict:
    refs = extract_topk(references, topk)
    return evaluate.load("bertscore").compute(
        predictions=predictions,
        references=refs,
        lang='en',
        use_fast_tokenizer=True
    )


def rouge_compute(predictions: typing.List[str], references: typing.List[str], topk: int) -> typing.Dict:
    refs = extract_topk(references, topk)
    return evaluate.load("rouge").compute(
        predictions=predictions,
        references=refs,
        use_aggregator=False
    )


def bleurt_compute(predictions: typing.List[str], references: typing.List[str], topk: int) -> typing.Dict:
    refs = extract_topk(references, topk)
    return evaluate.load("bleurt", config_name="BLEURT-20", module_type="metric").compute(
        predictions=predictions,
        references=refs
    )
