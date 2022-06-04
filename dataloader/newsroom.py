import csv
import string
from os import path

def read(path_read: string) -> (list, list, list):
    data = list()
    with open(path.join(path_read, "newsroom-human-eval.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    sys_summaries = list()
    ref_summaries = list()
    docs = list()
    for datum in data:
        sys_summaries.append(datum['SystemSummary'])
        ref_summaries.append(datum['ArticleTitle'])
        docs.append(datum['ArticleText'])
    return sys_summaries, ref_summaries, docs
