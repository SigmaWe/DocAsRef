import json
import pickle
from os import path

import numpy as np

datasets = ['newsroom', 'realsumm_abs', 'realsumm_ext']
approaches = ['trad', 'new']


def read_json(path_result: str) -> dict:
    with open(path_result, 'r') as infile:
        return json.load(infile)


def read_pkl(path_result: str) -> dict:
    with open(path_result, 'rb') as infile:
        return pickle.load(infile)


def read_results() -> dict:
    path_results = 'results'
    results = dict()
    for dataset in datasets:
        results[dataset] = read_json(path.join(path_results, 'model/' + dataset + '.json'))
    results['corr'] = read_pkl(path.join(path_results, 'model/corr.pkl'))
    return results


def rouge_analysis(results: dict) -> None:
    rouge_scores = dict()
    metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    for metric in metrics:
        metric_score = dict()
        for dataset in datasets:
            metric_score[dataset] = dict()
            for approach in approaches:
                metric_score[dataset][approach] = np.median(results[dataset]['rouge'][approach][metric])
        rouge_scores[metric] = metric_score
    with open('results/analysis/rouge.json', 'w') as outfile:
        json.dump(rouge_scores, outfile, indent=4)


def bertscore_analysis(results: dict) -> None:
    bertscore_score = dict()
    for dataset in datasets:
        bertscore_score[dataset] = dict()
        for approach in approaches:
            bertscore_score[dataset][approach] = np.median(results[dataset]['bertscore'][approach]['f1'])
    with open('results/analysis/bertscore.json', 'w') as outfile:
        json.dump(bertscore_score, outfile, indent=4)


def bleu_analysis(results: dict) -> None:
    bleu_score = dict()
    for dataset in datasets:
        bleu_score[dataset] = dict()
        for approach in approaches:
            bleu_score[dataset][approach] = np.median(results[dataset]['bleu'][approach]['bleu'])
    with open('results/analysis/bleu.json', 'w') as outfile:
        json.dump(bleu_score, outfile, indent=4)


def bleurt_analysis(results: dict) -> None:
    bleurt_score = dict()
    for dataset in datasets:
        bleurt_score[dataset] = dict()
        for approach in approaches:
            bleurt_score[dataset][approach] = np.median(results[dataset]['bleurt'][approach]['scores'])
    with open('results/analysis/bleurt.json', 'w') as outfile:
        json.dump(bleurt_score, outfile, indent=4)


def corr_analysis(corr_results: dict) -> None:
    results = dict()
    metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore', 'bleurt']
    for dataset in datasets:
        results[dataset] = dict()
        for metric in metrics:
            results[dataset][metric] = dict()
            for approach in approaches:
                results[dataset][metric][approach] = dict()
                for corr_type in corr_results[dataset][metric][approach].keys():
                    values = list(corr_results[dataset][metric][approach][corr_type].values())
                    results[dataset][metric][approach][corr_type] = np.mean(values)
    with open('results/analysis/corr.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == '__main__':
    results = read_results()
    rouge_analysis(results)
    bertscore_analysis(results)
    bleu_analysis(results)
    bleurt_analysis(results)
    corr_analysis(results['corr'])
