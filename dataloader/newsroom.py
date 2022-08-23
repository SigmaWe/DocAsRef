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
    for index in range(int(len(data) / 3)):  # 3 experts for the same pair
        sys_summaries.append(data[index * 3]['SystemSummary'])
        ref_summaries.append(data[index * 3]['ArticleTitle'])
        docs.append(data[index * 3]['ArticleText'])
        score = dict()
        to_copy = ['Coherence', 'Fluency', 'Informativeness', 'Relevance']
        for i in to_copy:
            sum_score = 0
            for j in range(3):
                sum_score += int(data[index * 3 + j][i + 'Rating'])
            score[i] = sum_score / 3  # mean
        scores.append(score)
    return sys_summaries, ref_summaries, docs, scores


if __name__ == '__main__':
    data = read('.')
    print(data)
