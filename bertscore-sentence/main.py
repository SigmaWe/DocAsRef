import numpy as np

from dataloader import newsroom, realsumm
from eval import score


def newsroom_eval():
    print('[Newsroom]')
    sys_summaries, ref_summaries, docs, _ = newsroom.read('dataloader')
    results = score(sys_summaries, docs)
    np.savetxt('bertscore-sentence/results/newsroom.csv', results, delimiter=',', header='P,R,F', comments='')


def realsumm_eval(abs: bool):
    print('[RealSumm] abs=' + str(abs))
    sys_summaries, ref_summaries, docs, _ = realsumm.read('suenes/human/realsumm/scores_dicts/',
                                                          'suenes/human/realsumm/analysis/test.tsv', abs)
    results = score(sys_summaries, docs)
    if abs:
        filename = 'realsumm_abs.csv'
    else:
        filename = 'realsumm_ext.csv'
    np.savetxt('bertscore-sentence/results/' + filename, results, delimiter=',', header='P,R,F', comments='')


if __name__ == '__main__':
    newsroom_eval()
    realsumm_eval(abs=True)
    realsumm_eval(abs=False)
