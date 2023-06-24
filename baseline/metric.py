# Baselines

from baseline.baseline_additional_metics import *
import typing
import dar_type
from baseline.wmd_master import SMD_scorer

### METRICS ###

def additional_metrics(metric_names: typing.List[str]) -> dar_type.MetricDict:
    # cant run bart, resue and sdc at the same time, requires more than 8GB+ gpu memory
    metrics = {
        "sacrebleu": sacrebleu_score_formatted().compute,
        "meteor": meteor_score_formatted().compute,
        "bart": BART_Score_Eval().compute,
        # "reuse": REUSE_score().compute,      # TOFIX: model
        "sdc*": SDC_Star().compute,            # Slow
        'smd': SMD_scorer.calculate_score      # Linux only
    }

    for metric_name in list(metrics.keys()):
        if metric_name not in metric_names:
            metrics.pop(metric_name)

    return metrics
