import json

import evaluate

# import bertscore_sentence.eval as bertscore_sentence
# import dataloader.newsroom as newsroom
# import dataloader.realsumm as realsumm
from env import models, approaches
import typing 


def model_eval(sys_summaries: list, ref_summaries: list, docs: list, models:typing.List[str], approaches:typing.List[str]) -> dict:
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
                model_result[approach] = model.compute(cands=cands, refs=refs)
            else:
                model_result[approach] = model.compute(predictions=cands, references=refs)

        results[model_name] = model_result

    return results
