import copy
import string
from os import path

import suenes.human.realsumm.analysis.utils as utils


def read_summary(path_summary: string, abs: bool):
    sd_abs_path = path.join(path_summary, "abs_ours.pkl")
    sd_ext_path = path.join(path_summary, "ext_ours.pkl")
    if abs:
        sd = utils.get_pickle(sd_abs_path)
    else:
        sd = utils.get_pickle(sd_ext_path)
    sys_summaries, ref_summaries, scores = list(), list(), list()
    for sd_item in sd.items():
        for sys_item in sd_item[1]['system_summaries'].items():
            ref_summaries.append(sd_item[1]['ref_summ'])
            sys_summaries.append(sys_item[1]['system_summary'])
            scores.append(sys_item[1]['scores'])
    return sys_summaries, ref_summaries, scores


def read_docs(path_docs: string, abs: bool) -> list:
    if abs:
        col = 14
    else:
        col = 12
    with open(path_docs, 'r') as infile:
        docs_orig = infile.readlines()
    docs = list()
    for row in docs_orig:
        for i in range(col):
            docs.append(row)
    return docs


def read(path_summary: string, path_docs: string, abs: bool):
    sys_summaries, ref_summaries, scores = read_summary(path_summary, abs)
    docs = read_docs(path_docs, abs)
    return sys_summaries, ref_summaries, docs, scores


if __name__ == '__main__':
    data_abs = read('../suenes/human/realsumm/scores_dicts/', '../suenes/human/realsumm/analysis/test.tsv', abs=True)
    print(data_abs)
    data_ext = read('../suenes/human/realsumm/scores_dicts/', '../suenes/human/realsumm/analysis/test.tsv', abs=False)
    print(data_ext)
