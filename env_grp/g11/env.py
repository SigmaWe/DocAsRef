# Group 11: classic bertscore + deberta-large-mnli

from env_root import *

### METRICS ###

import classic.eval as classic
import functools

bertscore_compute_deberta_large_mnli =  functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-large-mnli")
# bertscore_compute_deberta_v3_large =  functools.partial(classic.bertscore_compute, model_type="microsoft/deberta-v3-large", use_fast_tokenizer=False)

metrics = {
    "bertscore-deberta-large-mnli": bertscore_compute_deberta_large_mnli,
    # "bertscore-deberta-v3-large": bertscore_compute_deberta_v3_large,
}
