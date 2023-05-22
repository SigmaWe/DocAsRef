# AnyRef

import dar_type
import typing
import functools
from anyref.eval import anyref_compute
import anyref.summary_length as summary_len_expr
import anyref.summarizers


def additional_metrics(raw_metrics: dar_type.MetricDict, model_names: typing.List[str]) -> dar_type.MetricDict:
    metrics = dict()

    ratios = [(0.2, 0.5), (0.5, 0.7)]
    constants = [(16, 32), (32, 64)]

    anyref_summarizers = anyref.summarizers.get_summarizers(model_names)

    # min-0.2-0.5, min-0.5-0.7
    # mean-0.2-0.5, mean-0.5-0.7
    # constant-32-64, constant-64-128

    for metric_name, metric_f in raw_metrics.items():
        for model in model_names:
            metrics["-".join([metric_name, "anyref", model, "default"])] = functools.partial(
                anyref_compute,
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
                        anyref_compute,
                        metric_compute_f=metric_f,
                        summarizers=anyref_summarizers[model],
                        min_len_expr=min_len_expr,
                        max_len_expr=max_len_expr
                    )
            for min_const, max_const in constants:
                min_len_expr = functools.partial(summary_len_expr.constant, len=min_const)
                max_len_expr = functools.partial(summary_len_expr.constant, len=max_const)
                metrics["-".join([metric_name, "anyref", model, "constant", str(min_const), str(max_const)])] = functools.partial(
                    anyref_compute,
                    metric_compute_f=metric_f,
                    summarizers=anyref_summarizers[model],
                    min_len_expr=min_len_expr,
                    max_len_expr=max_len_expr
                )

    return metrics
