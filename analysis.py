import json
from os import path

import numpy as np


def read_result(path_result: str) -> dict:
    with open(path_result, 'r') as infile:
        return json.load(infile)


def read_results() -> (dict, dict):
    path_results = 'results'
    newsroom_results = read_result(path.join(path_results, 'newsroom.json'))
    realsumm_results = read_result(path.join(path_results, 'realsumm.json'))
    return newsroom_results, realsumm_results


def extract_results(metric_name: str, newsroom_results: dict, realsumm_results: dict) -> (dict, dict, dict, dict):
    newsroom_trad = newsroom_results[metric_name]['trad']
    newsroom_new = newsroom_results[metric_name]['new']
    realsumm_trad = realsumm_results[metric_name]['trad']
    realsumm_new = realsumm_results[metric_name]['new']
    return newsroom_trad, newsroom_new, realsumm_trad, realsumm_new


def rouge_analysis(newsroom_results: dict, realsumm_results: dict) -> None:
    rouge_scores = dict()
    newsroom_trad, newsroom_new, realsumm_trad, realsumm_new = extract_results('rouge', newsroom_results,
                                                                               realsumm_results)
    metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    for metric in metrics:
        metric_score = dict()
        metric_score['newsroom_trad'] = newsroom_trad[metric][1][2]
        metric_score['newsroom_new'] = newsroom_new[metric][1][2]
        metric_score['realsumm_trad'] = realsumm_trad[metric][1][2]
        metric_score['realsumm_new'] = realsumm_new[metric][1][2]
        rouge_scores[metric] = metric_score
    with open('results/analysis/rouge.json', 'w') as outfile:
        json.dump(rouge_scores, outfile, indent=4)


def bertscore_analysis(newsroom_results: dict, realsumm_results: dict) -> None:
    bertscore_score = dict()
    newsroom_trad, newsroom_new, realsumm_trad, realsumm_new = extract_results('bertscore', newsroom_results,
                                                                               realsumm_results)
    bertscore_score['newsroom_trad'] = np.median(newsroom_trad['f1'])
    bertscore_score['newsroom_new'] = np.median(newsroom_new['f1'])
    bertscore_score['realsumm_trad'] = np.median(realsumm_trad['f1'])
    bertscore_score['realsumm_new'] = np.median(realsumm_new['f1'])
    with open('results/analysis/bertscore.json', 'w') as outfile:
        json.dump(bertscore_score, outfile, indent=4)


def bleu_analysis(newsroom_results: dict, realsumm_results: dict) -> None:
    bleu_score = dict()
    newsroom_trad, newsroom_new, realsumm_trad, realsumm_new = extract_results('bleu', newsroom_results,
                                                                               realsumm_results)
    bleu_score['newsroom_trad'] = newsroom_trad['bleu']
    bleu_score['newsroom_new'] = newsroom_new['bleu']
    bleu_score['realsumm_trad'] = realsumm_trad['bleu']
    bleu_score['realsumm_new'] = realsumm_new['bleu']
    with open('results/analysis/bleu.json', 'w') as outfile:
        json.dump(bleu_score, outfile, indent=4)


def bleurt_analysis(newsroom_results: dict, realsumm_results: dict) -> None:
    bleurt_score = dict()
    newsroom_trad, newsroom_new, realsumm_trad, realsumm_new = extract_results('bleurt', newsroom_results,
                                                                               realsumm_results)
    bleurt_score['newsroom_trad'] = np.median(newsroom_trad['scores'])
    bleurt_score['newsroom_new'] = np.median(newsroom_new['scores'])
    bleurt_score['realsumm_trad'] = np.median(realsumm_trad['scores'])
    bleurt_score['realsumm_new'] = np.median(realsumm_new['scores'])
    with open('results/analysis/bleurt.json', 'w') as outfile:
        json.dump(bleurt_score, outfile, indent=4)


if __name__ == '__main__':
    newsroom_results, realsumm_results = read_results()
    rouge_analysis(newsroom_results, realsumm_results)
    bertscore_analysis(newsroom_results, realsumm_results)
    bleu_analysis(newsroom_results, realsumm_results)
    bleurt_analysis(newsroom_results, realsumm_results)
