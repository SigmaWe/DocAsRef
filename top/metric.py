# Top-k or Top-p

import functools, typing

import top.eval as top
import bertscore_sentence.metric  

import dar_type

def create_metrics(
        base_metrics: dar_type.MetricDict, 
        k_range: typing.List[int] = [5, 10, 20],
        p_range: typing.List[float] = [0.2, 0.5, 0.7],
        ) -> dar_type.MetricDict:
    metrics = dict()

    for metric_name, metric_f in base_metrics.items():
        for topk in k_range:
            metrics["-".join(["topK", metric_name, str(topk)])] = functools.partial(top.topk_compute, metric_compute_f=metric_f, topk=topk)
        for topp in p_range:
            metrics["-".join(["topP", metric_name, str(topp)])] = functools.partial(top.topp_compute, metric_compute_f=metric_f, topp=topp)

    return metrics
