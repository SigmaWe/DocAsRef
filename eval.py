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
            model = evaluate.load('bleurt', config_name='bleurt-large-512', module_type='metric')

        model_result = dict()

        # calculate traditional (reference, system summary) pairs
        print('Eval trad')
        if model_name != 'bertscore':
            model_result['trad'] = model.compute(predictions=sys_summaries, references=ref_summaries)
        else:
            model_result['trad'] = model.compute(predictions=sys_summaries, references=ref_summaries, lang='en')

        # calculate new (document, system summary) pairs
        print('Eval new')
        if model_name != 'bertscore':
            model_result['new'] = model.compute(predictions=sys_summaries, references=docs)
        else:
            model_result['new'] = model.compute(predictions=sys_summaries, references=docs, lang='en')

        results[model_name] = model_result

    return results


def realsumm_eval():
    print('[Realsumm]')
    sys_summaries, ref_summaries, docs = realsumm.read('suenes/human/realsumm/scores_dicts/',
                                                       'suenes/human/realsumm/analysis/test.tsv')
    results = model_eval(sys_summaries, ref_summaries, docs)
    with open('results_old/realsumm.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


def newsroom_eval():
    print('[Newsroom]')
    sys_summaries, ref_summaries, docs = newsroom.read('dataloader')
    results = model_eval(sys_summaries, ref_summaries, docs)
    with open('results_old/newsroom.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == '__main__':
    newsroom_eval()
    realsumm_eval()
