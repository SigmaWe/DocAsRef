# Baselines

import typing

from baseline.baseline_additional_metics import *

import dar_type
from baseline.wmd_master import SMD_scorer

### METRICS ###

# cant run bart, resue and sdc at the same time, requires more than 8GB+ gpu memory
metrics = {
    "sacrebleu": sacrebleu_score_formatted().compute,
    "meteor": meteor_score_formatted().compute,
    "bart": BART_Score_Eval().compute,
    # "reuse": REUSE_score().compute,      # TOFIX: model
    "sdc*": SDC_Star().compute,            # Slow
    'smd': SMD_scorer.calculate_score      # Linux only
}
