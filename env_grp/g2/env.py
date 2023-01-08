# Group 2: bertscore-sentence (cos, mnli)

from env_root import *

### MODELS ###

import dar_type
import sentence_transformers

from mnli.classifiers import mnli_classifiers
sent_embedder_mpnet: dar_type.Embedder = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
sent_embedder_mpnet.__name__ = "all-mpnet-base-v2"
sent_embedder_roberta: dar_type.Embedder = sentence_transformers.SentenceTransformer("all-roberta-large-v1")
sent_embedder_roberta.__name__ = "all-roberta-large-v1"

### METRICS ###

import functools
import bertscore_sentence.eval as bertscore_sentence
import mnli.eval
import mnli.sim_expr

metrics = {
    "bertscore-sentence-cos-mpnet": functools.partial(bertscore_sentence.compute_cos, embedder=sent_embedder_mpnet),
    "bertscore-sentence-cos-roberta": functools.partial(bertscore_sentence.compute_cos, embedder=sent_embedder_roberta),
}

for mnli_name in ["roberta", "bart", "deberta"]:
    for mnli_expr in [mnli.sim_expr.not_neutral, mnli.sim_expr.entail_only, mnli.sim_expr.entail_contradict]:
        metrics["bertscore-sentence-mnli-{}-{}".format(mnli_name, mnli_expr.__name__)] = functools.partial(mnli.eval.bertscore_sentence_compute, classifiers=mnli_classifiers[mnli_name], expr=mnli_expr)
