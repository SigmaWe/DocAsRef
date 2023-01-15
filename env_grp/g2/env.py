# Group 2: bertscore-sentence (cos, mnli)

from env_root import *

### MODELS ###

from mnli.classifiers import mnli_classifiers
from bertscore_sentence.embedders import sent_embedders


### METRICS ###

import functools
import bertscore_sentence.eval as bertscore_sentence
import mnli.eval
import mnli.sim_expr

metrics = {
    "bertscore-sentence-cos-mpnet": functools.partial(bertscore_sentence.compute_cos, embedder=sent_embedders["mpnet"]),
    "bertscore-sentence-cos-roberta": functools.partial(bertscore_sentence.compute_cos, embedder=sent_embedders["roberta"]),
}

for mnli_name in ["roberta", "bart", "deberta-large", "deberta-xlarge"]:
    for mnli_expr in [mnli.sim_expr.not_neutral, mnli.sim_expr.entail_only, mnli.sim_expr.entail_contradict]:
        metrics["bertscore-sentence-mnli-{}-{}".format(mnli_name, mnli_expr.__name__)] = functools.partial(mnli.eval.bertscore_sentence_compute, classifiers=mnli_classifiers[mnli_name], expr=mnli_expr)
