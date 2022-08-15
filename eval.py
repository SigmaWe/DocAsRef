import json

import evaluate

import dataloader.newsroom as newsroom
import dataloader.realsumm as realsumm


def model_eval(sys_summaries: list, ref_summaries: list, docs: list) -> dict:
    models_name = ['rouge', 'bertscore', 'bleu', 'bleurt']
    results = dict()

    for model_name in models_name:
        print(model_name + ' -')
        if model_name != 'bleurt':
            model = evaluate.load(model_name)
        else:
            model = evaluate.load('bleurt', config_name='BLEURT-20', module_type='metric')

        model_result = dict()

        # calculate traditional (reference, system summary) pairs
        print('Eval trad')
        if model_name == 'bertscore':
            model_result['trad'] = model.compute(predictions=sys_summaries, references=ref_summaries, lang='en')
        elif model_name == 'rouge':
            model_result['trad'] = model.compute(predictions=sys_summaries, references=ref_summaries,
                                                 use_aggregator=False)
        else:
            model_result['trad'] = model.compute(predictions=sys_summaries, references=ref_summaries)

        # calculate new (document, system summary) pairs
        print('Eval new')
        if model_name == 'bertscore':
            model_result['new'] = model.compute(predictions=sys_summaries, references=docs, lang='en')
        elif model_name == 'rouge':
            model_result['new'] = model.compute(predictions=sys_summaries, references=docs, use_aggregator=False)
        else:
            model_result['new'] = model.compute(predictions=sys_summaries, references=docs)

        results[model_name] = model_result

    return results


def realsumm_eval(abs: bool):
    print('[RealSumm]')
    sys_summaries, ref_summaries, docs, _ = realsumm.read('suenes/human/realsumm/scores_dicts/',
                                                          'suenes/human/realsumm/analysis/test.tsv', abs)
    results = model_eval(sys_summaries, ref_summaries, docs)
    if abs:
        filename = 'realsumm_abs.json'
    else:
        filename = 'realsumm_ext.json'
    with open('results/model/' + filename, 'w') as outfile:
        json.dump(results, outfile, indent=4)


def newsroom_eval():
    print('[Newsroom]')
    sys_summaries, ref_summaries, docs, _ = newsroom.read('dataloader')
    results = model_eval(sys_summaries, ref_summaries, docs)
    with open('results/model/newsroom.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == '__main__':
    newsroom_eval()
    realsumm_eval(abs=True)
    realsumm_eval(abs=False)
