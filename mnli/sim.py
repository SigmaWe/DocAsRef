import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import env
import dar_type
from datasets.arrow_dataset import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import typing
import math
import threading
from mnli import sim_expr
import text_preprocess


def similarity_thread(dataset: Dataset, classifier: dar_type.Pipeline, scores: dar_type.MetricScoreList, expr: dar_type.MNLISimilarityExpression) -> None:
    for out in classifier(KeyDataset(dataset, "text"), batch_size=8):
        scores.append(expr(out))


def similarity_ngpu(text_dataset: Dataset, classifiers: dar_type.PipelinesList, n_pairs: int, expr: dar_type.MNLISimilarityExpression):
    # split dataset to n_gpu, batch 8 on every gpu at same time
    # ds[0]: [0, n_pairs / n_gpu); ds[1]: [n_pairs / n_gpu, 2 * (n_pairs / n_gpu))
    ds_batch_sz = int(math.ceil(n_pairs / env.n_gpu))
    thread_pool: typing.List[threading.Thread] = []
    results: typing.List[dar_type.MetricScoreList] = []
    for i_gpu in range(env.n_gpu):
        lower_bound = ds_batch_sz * i_gpu
        upper_bound = ds_batch_sz * (i_gpu + 1)
        if upper_bound > n_pairs:
            upper_bound = n_pairs
        if lower_bound >= upper_bound:
            break
        scores_gpu: dar_type.MetricScoreList = []
        results.append(scores_gpu)
        thread_pool.append(threading.Thread(target=similarity_thread, kwargs={
            "dataset": text_dataset.select(range(lower_bound, upper_bound, 1)),
            "classifier": classifiers.pipelines[i_gpu],
            "scores": scores_gpu,
            "expr": expr
        }))
    for thread in thread_pool:
        thread.start()
    for thread in thread_pool:
        thread.join()
    scores = text_preprocess.flatten(results)
    if len(scores) != n_pairs:
        raise Exception("missing score")
    return scores


def similarity(sent_pairs: dar_type.TextList, classifiers: dar_type.PipelinesList, expr: dar_type.MNLISimilarityExpression) -> dar_type.MetricScoreList:
    text_dataset = Dataset.from_dict({"text": sent_pairs})
    n_pairs = len(sent_pairs)
    if n_pairs < env.n_gpu:
        scores = []
        similarity_thread(dataset=text_dataset, classifier=classifiers.pipelines[0], expr=expr, scores=scores)
        return scores
    return similarity_ngpu(text_dataset, classifiers, n_pairs, expr)


if __name__ == "__main__":
    sample_a = "Each computer program uses a region of memory called the stack to enable functions to work properly."
    sample_b = "From the outside, Les 4G, a Lyonnais bouchon (traditional restaurant), looked much like the nondescript cafe-cum-tobacco shops that can be found in most small French towns, but inside the decor was as warm and inviting as a country pub."
    sent_pairs = ["[SEP]".join([sample_a, sample_b])]
    for i in range(100):
        sent_pairs.append(sent_pairs[0])
    results = similarity(
        sent_pairs=sent_pairs,
        classifiers=env.mnli_classifiers["deberta"],
        expr=sim_expr.not_neutral
    )
    print(len(results))
    print(results)
