# Group 4: Top + Group 1

from env_grp.g1.env import *

### METRICS ###

import functools
import top.eval as top

raw_metrics = metrics.copy()
metrics = dict()

for metric_name, metric_f in raw_metrics.items():
    metrics[metric_name] = metric_f
    for topk in [5, 10, 20]:
        metrics["-".join([metric_name, "topk", str(topk)])] = functools.partial(top.topk_compute, metric_compute_f=metric_f, topk=topk)
    for topp in [0.2, 0.5, 0.7]:
        metrics["-".join([metric_name, "topp", str(topp)])] = functools.partial(top.topp_compute, metric_compute_f=metric_f, topp=topp)
