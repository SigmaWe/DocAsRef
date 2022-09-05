import json

import evaluate

import bertscore_sentence.eval as bertscore_sentence
import dataloader.newsroom as newsroom
import dataloader.realsumm as realsumm
from experiments.env import models, approaches


def model_eval(sys_summaries: list, ref_summaries: list, docs: list) -> dict:
    results = dict()

    for model_name in models:
        print('Model: ' + model_name)
        if model_name == 'bleurt':
            model = evaluate.load('bleurt', config_name='BLEURT-20', module_type='metric')
        elif model_name == 'bertscore-sentence':
            model = bertscore_sentence
        else:
            model = evaluate.load(model_name)

        model_result = dict()

        # calculate traditional (reference, system summary) pairs and new (document, system summary) pairs
        for approach in approaches:
            print('Evaluating on ' + approach + ' approach')
            cands = sys_summaries
            refs = ref_summaries if approach == "trad" else docs
            if model_name == 'bertscore':
                model_result[approach] = model.compute(predictions=cands, references=refs, lang='en')
            elif model_name == 'rouge':
                model_result[approach] = model.compute(predictions=cands, references=refs, use_aggregator=False)
            elif model_name == 'bertscore-sentence':
                model_result[approach] = model.score(cands=cands, refs=refs)
            else:
                model_result[approach] = model.compute(predictions=cands, references=refs)

        results[model_name] = model_result

    return results


def realsumm_eval(abs: bool):
    print('[RealSumm] abs=' + str(abs))
    sys_summaries, ref_summaries, docs, _ = realsumm.read('suenes/human/realsumm/scores_dicts/',
                                                          'suenes/human/realsumm/analysis/test.tsv', abs)
    results = model_eval(sys_summaries, ref_summaries, docs)
    if abs:
        filename = 'realsumm_abs.json'
    else:
        filename = 'realsumm_ext.json'
    with open('experiments/results/model/' + filename, 'w') as outfile:
        json.dump(results, outfile, indent=4)


def newsroom_eval():
    print('[Newsroom]')
    sys_summaries, ref_summaries, docs, _ = newsroom.read('dataloader')
    results = model_eval(sys_summaries, ref_summaries, docs)
    with open('experiments/results/model/newsroom.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == '__main__':
    evaluate.enable_progress_bar()
    newsroom_eval()
    realsumm_eval(abs=True)
    realsumm_eval(abs=False)
