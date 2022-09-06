import json
from os import path

from experiments.env import datasets, approaches, models

results, formatted_scores = dict(), dict()


def read_result(path_result: str) -> dict:
    with open(path_result, 'r') as infile:
        return json.load(infile)


def read_results() -> dict:
    path_results = 'experiments/results/model'
    newsroom_results = read_result(path.join(path_results, 'newsroom.json'))
    realsumm_abs_results = read_result(path.join(path_results, 'realsumm_abs.json'))
    realsumm_ext_results = read_result(path.join(path_results, 'realsumm_ext.json'))
    return {
        'newsroom': newsroom_results,
        'realsumm_abs': realsumm_abs_results,
        'realsumm_ext': realsumm_ext_results
    }


def format(model_name: str) -> None:
    if model_name == 'rouge':
        metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        for dataset in datasets:
            rouge_scores = dict()
            for metric in metrics:
                rouge_scores[metric] = dict()
                for approach in approaches:
                    approach_scores = results[dataset]['rouge'][approach][metric]
                    formatted_approach_scores = dict()
                    formatted_approach_scores['scores'] = approach_scores
                    rouge_scores[metric][approach] = formatted_approach_scores
            formatted_scores[dataset].update(rouge_scores)
    elif model_name == 'bertscore':
        for dataset in datasets:
            formatted_scores[dataset]['bertscore'] = dict()
            for approach in approaches:
                approach_dict = results[dataset]['bertscore'][approach]
                del approach_dict['hashcode']
                formatted_scores[dataset]['bertscore'][approach] = approach_dict
    else:
        for dataset in datasets:
            formatted_scores[dataset][model_name] = results[dataset][model_name]


if __name__ == '__main__':
    results = read_results()
    for dataset in datasets:
        formatted_scores[dataset] = dict()
    for model_name in models:
        format(model_name)
    with open('experiments/results/model/scores.json', 'w') as outfile:
        json.dump(formatted_scores, outfile, indent=4)
