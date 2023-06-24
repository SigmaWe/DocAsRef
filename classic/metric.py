# classic (bertscore, rouge, bleurt)

import os
import typing
import dar_type
import classic.eval as classic
import functools

### LIBRARY VARS ###

os.environ["MOVERSCORE_MODEL"] = "roberta-large"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

### METRICS ###

def additional_metrics(metric_names: typing.List[str]) -> dar_type.MetricDict:
    metrics = {
        "bertscore": classic.bertscore_compute,
        "rouge": classic.rouge_compute,
        "bleurt": classic.bleurt_compute,
        "moverscore-1gram": classic.moverscore_compute_1gram,
        "moverscore-2gram": classic.moverscore_compute_2gram,
        "bertscore-deberta-large-mnli": functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-large-mnli"),
        "bertscore-roberta-large-mnli": functools.partial(classic.bertscore_compute, model_type="roberta-large-mnli"),
        "bertscore-bart-large-mnli": functools.partial(classic.bertscore_compute, model_type="facebook/bart-large-mnli"),
        "bertscore-deberta-xlarge-mnli": functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-xlarge-mnli"),
        "bertscore-deberta-large": functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-large"),
        "bertscore-bart-large": functools.partial(classic.bertscore_compute, model_type="facebook/bart-large"),
        "bertscore-roberta-base": functools.partial(classic.bertscore_compute, model_type="roberta-base"),
        "bertscore-deberta-base": functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-base"),
        "bertscore-bart-base": functools.partial(classic.bertscore_compute, model_type="facebook/bart-base"),
    }

    for metric_name in list(metrics.keys()):
        if metric_name not in metric_names:
            metrics.pop(metric_name)

    return metrics
