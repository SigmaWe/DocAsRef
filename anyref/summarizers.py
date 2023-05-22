import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from transformers import pipeline
import env
import dar_type
import typing


def get_summarizers(model_names: typing.List[str]) -> dar_type.PipelinesDict:
    anyref_models = {
        "bart": "facebook/bart-large-cnn",
        "pegasus-xsum": "google/pegasus-xsum",
        "pegasus-cnndm": "google/pegasus-cnn_dailymail",
        "pegasus-newsroom": "google/pegasus-newsroom",
        "pegasus-large": "google/pegasus-large",
        "pegasus-x-large": "google/pegasus-x-large"
    }
    for key in list(anyref_models.keys()):
        if key not in model_names:
            anyref_models.pop(key)
    anyref_summarizers: dar_type.PipelinesDict = dict()
    for model_name, model in anyref_models.items():
        summarizers = dar_type.PipelinesList()
        summarizers.pipelines = [pipeline("summarization", model=model, device=device, truncation=True) for device in range(env.n_gpu)]
        summarizers.__name__ = model_name
        anyref_summarizers[model_name] = summarizers
    return anyref_summarizers
