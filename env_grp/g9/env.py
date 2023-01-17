# Group 9: AnyRef (pegasus-newsroom, pegasus-cnndm) + Group 2

from env_root import *
from env_grp.g2.env import *

### MODELS ###

anyref_whitelist = ["pegasus-newsroom", "pegasus-cnndm"]
if "anyref.summarizers" in sys.modules:
    del sys.modules["anyref.summarizers"]
from anyref.summarizers import anyref_summarizers

### METRICS ###

raw_metrics = metrics.copy()
metrics = dict()

import functools
import anyref.eval as anyref
import anyref.summary_length as summary_len_expr

ratios = [(0.2, 0.5), (0.5, 0.7)]
constants = [(16, 32), (32, 64)]

# min-0.2-0.5, min-0.5-0.7
# mean-0.2-0.5, mean-0.5-0.7
# constant-32-64, constant-64-128

for metric_name, metric_f in raw_metrics.items():
    metrics[metric_name] = metric_f
    for model in anyref_whitelist:
        metrics["-".join([metric_name, "anyref", model, "default"])] = functools.partial(
            anyref.anyref_compute,
            metric_compute_f=metric_f,
            summarizers=anyref_summarizers[model],
            min_len_expr=summary_len_expr.default,
            max_len_expr=summary_len_expr.default
        )
        for expr in [summary_len_expr.min, summary_len_expr.mean]:
            for min_ratio, max_ratio in ratios:
                min_len_expr = functools.partial(expr, ratio=min_ratio)
                max_len_expr = functools.partial(expr, ratio=max_ratio)
                metrics["-".join([metric_name, "anyref", model, expr.__name__, str(min_ratio), str(max_ratio)])] = functools.partial(
                    anyref.anyref_compute,
                    metric_compute_f=metric_f,
                    summarizers=anyref_summarizers[model],
                    min_len_expr=min_len_expr,
                    max_len_expr=max_len_expr
                )
        for min_const, max_const in constants:
            min_len_expr = functools.partial(summary_len_expr.constant, len=min_const)
            max_len_expr = functools.partial(summary_len_expr.constant, len=max_const)
            metrics["-".join([metric_name, "anyref", model, "constant", str(min_const), str(max_const)])] = functools.partial(
                anyref.anyref_compute,
                metric_compute_f=metric_f,
                summarizers=anyref_summarizers[model],
                min_len_expr=min_len_expr,
                max_len_expr=max_len_expr
            )
