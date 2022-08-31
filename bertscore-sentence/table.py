# See /vanilla/table.py, without approach (trad, new)

import csv
import json
from os import path

datasets = ['newsroom', 'realsumm_abs', 'realsumm_ext']
eval_metrics = ['bertscore-sentence']
corr_metrics = ['pearsonr', 'kendalltau', 'spearmanr']


def read_json(path_result: str) -> dict:
    with open(path_result, 'r') as infile:
        return json.load(infile)


def generate_table(corr: dict):
    fields = ['dataset', 'eval_metric', 'corr_metric', 'value']
    rows = list()
    for dataset in datasets:
        for eval_metric in eval_metrics:
            for corr_metric in corr_metrics:
                rows.append([dataset, eval_metric, corr_metric, corr[dataset][eval_metric][corr_metric]])
    with open('bertscore-sentence/results/corr.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(fields)
        writer.writerows(rows)


def generate_full_table(corr: dict):
    fields = ['dataset', 'eval_metric', 'corr_metric', 'pair', 'value']
    rows = list()
    for dataset in datasets:
        for eval_metric in eval_metrics:
            for corr_metric in corr_metrics:
                for pair in corr[dataset][eval_metric][corr_metric].keys():
                    rows.append([dataset, eval_metric, corr_metric, pair, corr[dataset][eval_metric][corr_metric][pair]])
    with open('bertscore-sentence/results/corr_full.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(fields)
        writer.writerows(rows)


if __name__ == '__main__':
    generate_table(read_json(path.join('bertscore-sentence/results/corr.json')))
    generate_full_table(read_json(path.join('bertscore-sentence/results/corr_full.json')))
