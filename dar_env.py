import datasets
import re
import torch
import spacy
import dar_type
import sentence_transformers
from transformers import pipeline
import os

os.environ["MOVERSCORE_MODEL"] = "roberta-large"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


datasets.disable_progress_bar()
n_gpu = torch.cuda.device_count()
nlp_spacy = spacy.load("en_core_web_lg")
mnli_models = ["roberta-large-mnli", "facebook/bart-large-mnli", "microsoft/deberta-xlarge-mnli"]
mnli_classifiers: dar_type.PipelinesDict = dict()
mnli_classifier_name_pattern = re.compile("(?:.*\/)?(?P<name>.*)-.*-mnli")
for model in mnli_models:
    mnli_model_classifiers: dar_type.PipelinesList = list()
    model_name = mnli_classifier_name_pattern.match(model).group("name")
    classifiers = dar_type.PipelinesList()
    classifiers.pipelines = [pipeline("text-classification", model=model, top_k=None, device=device) for device in range(n_gpu)]
    classifiers.__name__ = model_name
    mnli_classifiers[model_name] = classifiers
sent_embedder_mpnet: dar_type.Embedder = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
sent_embedder_mpnet.__name__ = "all-mpnet-base-v2"
sent_embedder_roberta: dar_type.Embedder = sentence_transformers.SentenceTransformer("all-roberta-large-v1")
sent_embedder_roberta.__name__ = "all-roberta-large-v1"
