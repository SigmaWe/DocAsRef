import csv
import string
from os import path


def read(path_read: string) -> (list, list, list, list):
    data = list()
    with open(path.join(path_read, "newsroom-human-eval.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    sys_summaries, ref_summaries, docs, scores = list(), list(), list(), list()
    for datum in data:
        sys_summaries.append(datum['SystemSummary'])
        ref_summaries.append(datum['ArticleTitle'])
        docs.append(datum['ArticleText'])
        score = dict()
        to_copy = ['Coherence', 'Fluency', 'Informativeness', 'Relevance']
        for i in to_copy:
            score[i] = int(datum[i + 'Rating'])
        scores.append(score)
    return sys_summaries, ref_summaries, docs, scores


if __name__ == '__main__':
    data = read('.')
    print(data)
