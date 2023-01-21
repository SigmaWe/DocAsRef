# Group 11: classic bertscore + other models + Top

from env_root import *

### METRICS ###

import classic.eval as classic
import functools
import top.eval as top

metrics = {
    "bertscore-deberta-large-mnli": functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-large-mnli"),
    "bertscore-roberta-large-mnli": functools.partial(classic.bertscore_compute, model_type="roberta-large-mnli"),
    "bertscore-bart-large-mnli": functools.partial(classic.bertscore_compute, model_type="facebook/bart-large-mnli"),
    "bertscore-deberta-xlarge-mnli": functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-xlarge-mnli"),
    "bertscore-deberta-large": functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-large"),
    "bertscore-bart-large": functools.partial(classic.bertscore_compute, model_type="facebook/bart-large"),
}

raw_metrics = metrics.copy()
metrics = dict()

for metric_name, metric_f in raw_metrics.items():
    metrics[metric_name] = metric_f
    for topk in [5, 10, 20]:
        metrics["-".join([metric_name, "topk", str(topk)])] = functools.partial(top.topk_compute, metric_compute_f=metric_f, topk=topk)
    for topp in [0.2, 0.5, 0.7]:
        metrics["-".join([metric_name, "topp", str(topp)])] = functools.partial(top.topp_compute, metric_compute_f=metric_f, topp=topp)
