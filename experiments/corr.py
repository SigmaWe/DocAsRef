import json

import numpy as np
import scipy

import dataloader.newsroom as newsroom
import dataloader.realsumm as realsumm
from experiments.env import datasets, approaches, eval_metrics

model_scores = dict()
corr = dict()
corr_json = dict()


def read_system_scores() -> dict:
    with open('experiments/results/model/scores.json', 'r') as infile:
        return json.load(infile)


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
    for approach in approaches:
        system_scores[approach] = dict()
    _, _, _, human_scores = newsroom.read('dataloader')
    for i in range(len(human_scores)):
        for approach in approaches:
            system_scores[approach][i] = dict()
            human_keys = human_scores[i].keys()
            for metric in metrics:
                if metric != 'bleu':
                    system_scores[approach][i][metric] = dict()
                    for key in human_keys:
                        system_scores[approach][i][metric][key] = human_scores[i][key]
                    system_keys = model_scores['newsroom'][metric][approach].keys()
                    for key in system_keys:
                        system_scores[approach][i][metric][key] = model_scores['newsroom'][metric][approach][key][i]
    return system_scores


def system_judge(scores, metrics_human, metrics_system, correlation_types) -> dict:
    # ref: suenes.human.newsroom.test_eval
    all_system_names = list(scores[list(scores.keys())[0]].keys())

    def get_correlation_two_metrics(scores, metric_human, metric_system, correlation_type):
        mean_score_vector_newsroom = []
        mean_score_vector_other = []
        for system in all_system_names:
            vector_human = []  # scores from a human metric
            vector_system = []  # scores from a non-human metric
            for docID in scores.keys():
                score_local = scores[docID][system]
                score_newsroom = score_local[metric_human]  # one float
                score_other = score_local[metric_system]  # one float
                vector_human.append(score_newsroom)
                vector_system.append(score_other)

            mean_score_vector_newsroom.append(np.mean(vector_human))
            mean_score_vector_other.append(np.mean(vector_system))
        return eval(f"scipy.stats.{correlation_type}(vector_human, vector_system)")[0]

    # now begins the system-level judge
    correlations = {}
    for correlation_type in correlation_types:
        correlations[correlation_type] = {}
        for metric_human in metrics_human:  # one metric from human
            for metric_system in metrics_system:  # one metric to evaluate against human
                correlations[correlation_type] \
                    [(metric_human, metric_system)] = \
                    get_correlation_two_metrics(scores, metric_human, metric_system, correlation_type)

    return correlations


def realsumm_read(metrics: list, abs: bool) -> dict:
    _, _, _, dataset_scores = realsumm.read('suenes/human/realsumm/scores_dicts/',
                                            'suenes/human/realsumm/analysis/test.tsv', abs)
    if abs:
        name = 'realsumm_abs'
    else:
        name = 'realsumm_ext'
    system_scores = dict()
    for approach in approaches:
        system_scores[approach] = dict()
        for i in range(len(dataset_scores)):
            system_scores[approach][i] = dict()
            for metric in metrics:
                system_scores[approach][i][metric] = dict()
                system_scores[approach][i][metric]['litepyramid_recall'] = dataset_scores[i][
                    'litepyramid_recall']  # human score
                system_keys = model_scores[name][metric]['trad'].keys()
                for key in system_keys:
                    system_scores[approach][i][metric][key] = model_scores[name][metric][approach][key][i]
    return system_scores


def calculate(dataset: str) -> None:
    corr[dataset] = dict()
    for metric_systems_name in eval_metrics:
        metric_systems = [metric_systems_name]
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
        if metric_systems_name == 'bertscore':
            metrics_system = ['precision', 'recall', 'f1']
        elif metric_systems_name == 'bertscore-sentence':
            metrics_system = ['P', 'R', 'F']
        else:
            metrics_system = ['scores']
        correlation_types = ['pearsonr', 'kendalltau', 'spearmanr']
        my_corr = dict()
        for approach in approaches:
            my_corr[approach] = system_judge(system_scores[approach], metrics_human, metrics_system,
                                             correlation_types)
        corr[dataset][metric_systems_name] = my_corr


def convert():
    # corr: key(dataset), key(eval_metric), key(approach), key(corr_metric), tuple(human_metric, system_metric),
    #       value(corr)
    # corr_json: key(dataset), key(eval_metric), key(approach), key(corr_metric),
    #            key(string(human_metric, system_metric)), value(corr)
    for i in corr.keys():
        corr_json[i] = dict()
        for j in corr[i].keys():
            corr_json[i][j] = dict()
            for k in corr[i][j].keys():
                corr_json[i][j][k] = dict()
                for l in corr[i][j][k].keys():
                    corr_json[i][j][k][l] = dict()
                    for (m, n) in corr[i][j][k][l].keys():
                        corr_json[i][j][k][l]['(' + m + ', ' + n + ')'] = corr[i][j][k][l][(m, n)]


def summarize(corr_results: dict) -> None:
    # corr_json_summarized: key(dataset), key(eval_metric), key(approach), key(corr_metric), value(mean_corr)
    results = dict()
    for dataset in datasets:
        results[dataset] = dict()
        for metric in eval_metrics:
            results[dataset][metric] = dict()
            for approach in approaches:
                results[dataset][metric][approach] = dict()
                for corr_type in corr_results[dataset][metric][approach].keys():
                    values = list(corr_results[dataset][metric][approach][corr_type].values())
                    results[dataset][metric][approach][corr_type] = np.mean(values)
    with open('experiments/results/analysis/corr.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == '__main__':
    model_scores = read_system_scores()
    for dataset in datasets:
        calculate(dataset)
    convert()
    with open('experiments/results/analysis/corr_full.json', 'w') as outfile:
        json.dump(corr_json, outfile, indent=4)
    summarize(corr_json)
