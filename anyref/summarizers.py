import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from transformers import pipeline
import env
import dar_type


anyref_models = {
    "bart": "facebook/bart-large-cnn",
    "pegasus-xsum": "google/pegasus-xsum",
    "pegasus-cnndm": "google/pegasus-cnn_dailymail",
    "pegasus-newsroom": "google/pegasus-newsroom",
    "pegasus-large": "google/pegasus-large",
    "pegasus-x-large": "google/pegasus-x-large"
}
whitelist_exist = False
try:
    env.anyref_whitelist
    whitelist_exist = True
except NameError:
    pass
if whitelist_exist:
    for key in list(anyref_models.keys()):
        if key not in env.anyref_whitelist:
            anyref_models.pop(key)
anyref_summarizers: dar_type.PipelinesDict = dict()
for model_name, model in anyref_models.items():
    anyref_model_summarizers: dar_type.PipelinesList = list()
    summarizers = dar_type.PipelinesList()
    summarizers.pipelines = [pipeline("summarization", model=model, device=device, truncation=True) for device in range(env.n_gpu)]
    summarizers.__name__ = model_name
    anyref_summarizers[model_name] = summarizers
