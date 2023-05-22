# Top

import functools
import top.eval as top
import dar_type

def additional_metrics(raw_metrics: dar_type.MetricDict) -> dar_type.MetricDict:
    metrics = dict()

    for metric_name, metric_f in raw_metrics.items():
        for topk in [5, 10, 20]:
            metrics["-".join([metric_name, "topk", str(topk)])] = functools.partial(top.topk_compute, metric_compute_f=metric_f, topk=topk)
        for topp in [0.2, 0.5, 0.7]:
            metrics["-".join([metric_name, "topp", str(topp)])] = functools.partial(top.topp_compute, metric_compute_f=metric_f, topp=topp)

    return metrics
