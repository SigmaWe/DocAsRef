import json
from os import path

results, formatted_scores = dict(), dict()
approaches = ['trad', 'new']
datasets = ['newsroom', 'realsumm_abs', 'realsumm_ext']


def read_result(path_result: str) -> dict:
    with open(path_result, 'r') as infile:
        return json.load(infile)


def read_results() -> dict:
    path_results = 'results/model'
    newsroom_results = read_result(path.join(path_results, 'newsroom.json'))
    realsumm_abs_results = read_result(path.join(path_results, 'realsumm_abs.json'))
    realsumm_ext_results = read_result(path.join(path_results, 'realsumm_ext.json'))
    return {
        'newsroom': newsroom_results,
        'realsumm_abs': realsumm_abs_results,
        'realsumm_ext': realsumm_ext_results
    }


def rouge_format() -> None:
    metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    for dataset in datasets:
        dataset_scores = dict()
        for metric in metrics:
            dataset_scores[metric] = dict()
            for approach in approaches:
                approach_scores = results[dataset]['rouge'][approach][metric]
                formatted_approach_scores = dict()
                formatted_approach_scores['scores'] = approach_scores
                dataset_scores[metric][approach] = formatted_approach_scores
        formatted_scores[dataset] = dataset_scores


def bertscore_format() -> None:
    for dataset in datasets:
        formatted_scores[dataset]['bertscore'] = dict()
        for approach in approaches:
            approach_dict = results[dataset]['bertscore'][approach]
            del approach_dict['hashcode']
            formatted_scores[dataset]['bertscore'][approach] = approach_dict


def bleurt_format() -> None:
    for dataset in datasets:
        formatted_scores[dataset]['bleurt'] = results[dataset]['bleurt']


def bleu_format() -> None:
    for dataset in datasets:
        formatted_scores[dataset]['bleu'] = results[dataset]['bleu']


if __name__ == '__main__':
    results = read_results()
    rouge_format()
    bertscore_format()
    bleurt_format()
    bleu_format()
    with open('results/model/scores.json', 'w') as outfile:
        json.dump(formatted_scores, outfile, indent=4)
