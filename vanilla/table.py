import csv
import json
from os import path

approaches = ['trad', 'new']
datasets = ['newsroom', 'realsumm_abs', 'realsumm_ext']
eval_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore', 'bleurt']
corr_metrics = ['pearsonr', 'kendalltau', 'spearmanr']

def read_json(path_result: str) -> dict:
    with open(path_result, 'r') as infile:
        return json.load(infile)

def generate_table(corr: dict):
    fields = ['approach', 'dataset', 'eval_metric', 'corr_metric', 'value']
    rows = list()
    for approach in approaches:
        for dataset in datasets:
            for eval_metric in eval_metrics:
                for corr_metric in corr_metrics:
                    value = corr[dataset][eval_metric][approach][corr_metric]
                    rows.append([approach, dataset, eval_metric, corr_metric, value])
    with open('results/analysis/corr.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(fields)
        writer.writerows(rows)


if __name__ == '__main__':
    corr = read_json(path.join('results/analysis/corr.json'))
    generate_table(corr)
