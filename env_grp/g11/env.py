# Group 11: classic bertscore + deberta-large-mnli + Top

from env_root import *

### METRICS ###

import classic.eval as classic
import functools
import top.eval as top

bertscore_compute_deberta_large_mnli =  functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-large-mnli")
bertscore_compute_deberta_xlarge_mnli =  functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-large-mnli")
# bertscore_compute_deberta_v3_large =  functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-v3-large", use_fast_tokenizer=False)

metrics = {
    "bertscore-deberta-large-mnli": bertscore_compute_deberta_large_mnli,
    "bertscore-deberta-xlarge-mnli": bertscore_compute_deberta_xlarge_mnli,
    # "bertscore-deberta-v3-large": bertscore_compute_deberta_v3_large,
}

raw_metrics = metrics.copy()
metrics = dict()

for metric_name, metric_f in raw_metrics.items():
    metrics[metric_name] = metric_f
    for topk in [5, 10, 20]:
        metrics["-".join([metric_name, "topk", str(topk)])] = functools.partial(top.topk_compute, metric_compute_f=metric_f, topk=topk)
    for topp in [0.2, 0.5, 0.7]:
        metrics["-".join([metric_name, "topp", str(topp)])] = functools.partial(top.topp_compute, metric_compute_f=metric_f, topp=topp)
