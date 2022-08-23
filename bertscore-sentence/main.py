import numpy as np

from dataloader import newsroom, realsumm
from eval import score


def newsroom_eval():
    print('[Newsroom]')
    sys_summaries, ref_summaries, docs, _ = newsroom.read('dataloader')
    results = score(sys_summaries, docs)
    np.savetxt('results/model/newsroom.json', results, delimiter=',')


def realsumm_eval(abs: bool):
    print('[RealSumm] abs=' + str(abs))
    sys_summaries, ref_summaries, docs, _ = realsumm.read('suenes/human/realsumm/scores_dicts/',
                                                          'suenes/human/realsumm/analysis/test.tsv', abs)
    results = score(sys_summaries, docs)
    if abs:
        filename = 'realsumm_abs.csv'
    else:
        filename = 'realsumm_ext.csv'
    np.savetxt('results/' + filename, results, delimiter=',')


if __name__ == '__main__':
    newsroom_eval()
    realsumm_eval(abs=True)
    realsumm_eval(abs=False)
