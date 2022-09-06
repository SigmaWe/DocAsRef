import csv
import json
from os import path

from experiments.env import approaches, datasets, eval_metrics, corr_metrics


def read_json(path_result: str) -> dict:
    with open(path_result, 'r') as infile:
        return json.load(infile)


def generate_full_table(corr: dict):
    fields = ['approach', 'dataset', 'eval_metric', 'corr_metric', 'pair', 'value']
    rows = list()
    for approach in approaches:
        for dataset in datasets:
            for eval_metric in eval_metrics:
                for corr_metric in corr_metrics:
                    for pair in corr[dataset][eval_metric][approach][corr_metric].keys():
                        value = corr[dataset][eval_metric][approach][corr_metric][pair]
                        rows.append([approach, dataset, eval_metric, corr_metric, str(pair), value])
    with open('experiments/results/analysis/corr_full.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(fields)
        writer.writerows(rows)


def generate_table(corr: dict):
    fields = ['approach', 'dataset', 'eval_metric', 'corr_metric', 'value']
    rows = list()
    for approach in approaches:
        for dataset in datasets:
            for eval_metric in eval_metrics:
                for corr_metric in corr_metrics:
                    value = corr[dataset][eval_metric][approach][corr_metric]
                    rows.append([approach, dataset, eval_metric, corr_metric, value])
    with open('experiments/results/analysis/corr.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(fields)
        writer.writerows(rows)


if __name__ == '__main__':
    generate_table(read_json(path.join('experiments/results/analysis/corr.json')))
    generate_full_table(read_json(path.join('experiments/results/analysis/corr_full.json')))