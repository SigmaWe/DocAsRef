import json
import evaluate as evaluate
import dataloader.realsumm as realsumm
import dataloader.newsroom as newsroom

def rouge_eval(sys_summaries: list, ref_summaries: list, docs: list) -> (dict, dict):
    rouge = evaluate.load('rouge')

    # calculate traditional (reference, system summary) pairs
    print("Eval trad")
    results_trad = rouge.compute(predictions=sys_summaries, references=ref_summaries)

    # calculate new (document, system summary) pairs
    print("Eval new")
    results_new = rouge.compute(predictions=sys_summaries, references=docs)

    return results_trad, results_new

def realsumm_eval():
    print("[Realsumm]")
    sys_summaries, ref_summaries, docs = realsumm.read("suenes/human/realsumm/scores_dicts/", "suenes/human/realsumm/analysis/test.tsv")
    results_trad, results_new = rouge_eval(sys_summaries, ref_summaries, docs)
    with open('results/rouge_realsumm_trad.json', 'w') as outfile:
        json.dump(results_trad, outfile, indent=4)
    with open('results/rouge_realsumm_new.json', 'w') as outfile:
        json.dump(results_new, outfile, indent=4)

def newsroom_eval():
    print("[Newsroom]")
    sys_summaries, ref_summaries, docs = newsroom.read("dataloader")
    results_trad, results_new = rouge_eval(sys_summaries, ref_summaries, docs)
    with open('results/rouge_newsroom_trad.json', 'w') as outfile:
        json.dump(results_trad, outfile, indent=4)
    with open('results/rouge_newsroom_new.json', 'w') as outfile:
        json.dump(results_new, outfile, indent=4)

if __name__ == '__main__':
    newsroom_eval()
    realsumm_eval()
