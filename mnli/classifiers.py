import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from transformers import pipeline
import env
import dar_type


mnli_models = {
    "roberta": "roberta-large-mnli",
    "bart": "facebook/bart-large-mnli",
    "deberta-large": "microsoft/deberta-large-mnli",
    "deberta-xlarge": "microsoft/deberta-xlarge-mnli"
}
mnli_classifiers: dar_type.PipelinesDict = dict()
for model_name, model in mnli_models.items():
    mnli_model_classifiers: dar_type.PipelinesList = list()
    classifiers = dar_type.PipelinesList()
    classifiers.pipelines = [pipeline("text-classification", model=model, top_k=None, device=device, truncation=True) for device in range(env.n_gpu)]
    classifiers.__name__ = model_name
    mnli_classifiers[model_name] = classifiers
