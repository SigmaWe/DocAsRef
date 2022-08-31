# See /vanilla/corr.py, without approach (trad, new)

import json

import pandas as pd
import numpy as np

import vanilla.corr as vcorr
from dataloader import newsroom, realsumm

datasets = ["newsroom", "realsumm_abs", "realsumm_ext"]
model_scores = dict()
corr = dict()
corr_json = dict()


def read_system_scores() -> dict:
    results = dict()
    for dataset in datasets:
        results[dataset] = dict()
        results[dataset]["bertscore-sentence"] = pd.read_csv("bertscore-sentence/results/" + dataset + ".csv").to_dict("list")
    return results


def newsroom_read(metrics: list) -> dict:
    """
    Return data structure:
    {
        docID: {
            system1: {
                "Coherence":       float,
                "Fluency":         float,
                "Informativeness": float,
                "Relevance":       float,
                "precision":       float,
                "recall":          float,
                "f1":              float
            }
            system2: { ... }
            ...
            system7: {... }
        }
    }
    """
    system_scores = dict()
    _, _, _, human_scores = newsroom.read('dataloader')
    for i in range(len(human_scores)):
        system_scores[i] = dict()
        human_keys = human_scores[i].keys()
        for metric in metrics:
            if metric != 'bleu':
                system_scores[i][metric] = dict()
                for key in human_keys:
                    system_scores[i][metric][key] = human_scores[i][key]
                system_keys = model_scores['newsroom'][metric].keys()
                for key in system_keys:
                    system_scores[i][metric][key] = model_scores['newsroom'][metric][key][i]
    return system_scores


def realsumm_read(metrics: list, abs: bool) -> dict:
    _, _, _, dataset_scores = realsumm.read('suenes/human/realsumm/scores_dicts/',
                                            'suenes/human/realsumm/analysis/test.tsv', abs)
    if abs:
        name = 'realsumm_abs'
    else:
        name = 'realsumm_ext'
    system_scores = dict()
    for i in range(len(dataset_scores)):
        system_scores[i] = dict()
        for metric in metrics:
            system_scores[i][metric] = dict()
            system_scores[i][metric]['litepyramid_recall'] = dataset_scores[i][
                'litepyramid_recall']  # human score
            system_keys = model_scores[name][metric].keys()
            for key in system_keys:
                system_scores[i][metric][key] = model_scores[name][metric][key][i]
    return system_scores


def calculate(dataset: str):
    corr[dataset] = dict()
    metric_systems = ["bertscore-sentence"]
    metrics_system = ['P', 'R', 'F']
    if dataset == 'newsroom':
        system_scores = newsroom_read(metric_systems)
        metrics_human = ['Coherence', 'Informativeness', 'Fluency', 'Relevance']
    elif dataset == 'realsumm_abs':
        system_scores = realsumm_read(metric_systems, abs=True)
        metrics_human = ['litepyramid_recall']
    elif dataset == 'realsumm_ext':
        system_scores = realsumm_read(metric_systems, abs=False)
        metrics_human = ['litepyramid_recall']
    else:
        raise NotImplementedError()
    correlation_types = ['pearsonr', 'kendalltau', 'spearmanr']
    my_corr = vcorr.system_judge(system_scores, metrics_human, metrics_system, correlation_types)
    corr[dataset]["bertscore-sentence"] = my_corr


def convert():
    for i in corr.keys():
        corr_json[i] = dict()
        for j in corr[i].keys():
            corr_json[i][j] = dict()
            for k in corr[i][j].keys():
                corr_json[i][j][k] = dict()
                for (m, n) in corr[i][j][k].keys():
                    corr_json[i][j][k]['(' + m + ', ' + n + ')'] = corr[i][j][k][(m, n)]


def summarize(corr_results: dict) -> None:
    results = dict()
    for dataset in datasets:
        results[dataset] = dict()
        results[dataset]["bertscore-sentence"] = dict()
        results[dataset]["bertscore-sentence"] = dict()
        for corr_type in corr_results[dataset]["bertscore-sentence"].keys():
            values = list(corr_results[dataset]["bertscore-sentence"][corr_type].values())
            results[dataset]["bertscore-sentence"][corr_type] = np.mean(values)
    with open('bertscore-sentence/results/corr.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == '__main__':
    model_scores = read_system_scores()
    for dataset in datasets:
        calculate(dataset)
    convert()
    with open('bertscore-sentence/results/corr_full.json', 'w') as outfile:
        json.dump(corr_json, outfile, indent=4)
    summarize(corr_json)
