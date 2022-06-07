import copy
import string
from os import path

import suenes.human.realsumm.analysis.utils as utils


def read_summary(path_summary: string) -> (list, list):
    sd_abs_path = path.join(path_summary, "abs_ours.pkl")
    sd_ext_path = path.join(path_summary, "ext_ours.pkl")
    sd_abs = utils.get_pickle(sd_abs_path)
    sd_ext = utils.get_pickle(sd_ext_path)
    sd = copy.deepcopy(sd_abs)
    for doc_id in sd:
        isd_sota_ext = sd_ext[doc_id]
        sd[doc_id]['system_summaries'].update(isd_sota_ext['system_summaries'])
    sys_summaries, ref_summaries, scores = list(), list(), list()
    for sd_item in sd.items():
        for sys_item in sd_item[1]['system_summaries'].items():
            ref_summaries.append(sd_item[1]['ref_summ'])
            sys_summaries.append(sys_item[1]['system_summary'])
            scores.append(sys_item[1]['scores'])
    return sys_summaries, ref_summaries, scores


def read_docs(path_docs: string) -> list:
    with open(path_docs, 'r') as infile:
        docs_orig = infile.readlines()
    docs = list()
    for row in docs_orig:
        for i in range(25):
            docs.append(row)
    return docs


def read(path_summary: string, path_docs: string) -> (list, list, list):
    sys_summaries, ref_summaries, scores = read_summary(path_summary)
    docs = read_docs(path_docs)
    return sys_summaries, ref_summaries, docs, scores


if __name__ == '__main__':
    data = read('../suenes/human/realsumm/scores_dicts/', '../suenes/human/realsumm/analysis/test.tsv')
    print(data)
