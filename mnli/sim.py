import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import dar_env
import dar_type
from datasets.arrow_dataset import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import typing
import math
import threading


def similarity_row(row: str, classifier: dar_type.Pipeline) -> float:
    classes = classifier(row)
    for c in classes:
        for category in c:
            if category["label"] == "NEUTRAL" or category["label"] == "neutral":
                return 1 - category["score"]
    raise Exception("no NEUTRAL label")


def similarity_gpu(dataset: Dataset, classifier: dar_type.Pipeline, scores: dar_type.MetricScoreList) -> None:
    for out in classifier(KeyDataset(dataset, "text"), batch_size=8):
        for category in out:
            if category["label"] == "NEUTRAL" or category["label"] == "neutral":
                scores.append(1 - category["score"])
                break


def flatten(l: typing.List[typing.List]):
    return [item for sublist in l for item in sublist]


def similarity_ngpu(text_dataset: Dataset, classifiers: dar_type.PipelinesList, n_pairs: int):
    # split dataset to n_gpu, batch 8 on every gpu at same time
    # ds[0]: [0, n_pairs / n_gpu); ds[1]: [n_pairs / n_gpu, 2 * (n_pairs / n_gpu))
    ds_batch_sz = int(math.ceil(n_pairs / dar_env.n_gpu))
    thread_pool: typing.List[threading.Thread] = []
    results: typing.List[dar_type.MetricScoreList] = []
    for i_gpu in range(dar_env.n_gpu):
        lower_bound = ds_batch_sz * i_gpu
        upper_bound = ds_batch_sz * (i_gpu + 1)
        if upper_bound > n_pairs:
            upper_bound = n_pairs
        if lower_bound >= upper_bound:
            break
        scores_gpu: dar_type.MetricScoreList = []
        results.append(scores_gpu)
        thread_pool.append(threading.Thread(target=similarity_gpu, kwargs={
            "dataset": text_dataset.select(range(lower_bound, upper_bound, 1)),
            "classifier": classifiers.pipelines[i_gpu],
            "scores": scores_gpu
        }))
    for thread in thread_pool:
        thread.start()
    for thread in thread_pool:
        thread.join()
    scores = flatten(results)
    if len(scores) != n_pairs:
        raise Exception("missing score")
    return scores


def similarity(sent_pairs: dar_type.TextList, classifiers: dar_type.PipelinesList) -> dar_type.MetricScoreList:
    text_dataset = Dataset.from_dict({"text": sent_pairs})
    n_pairs = len(sent_pairs)
    if n_pairs < dar_env.n_gpu:
        return [similarity_row(row, classifier=classifiers.pipelines[0]) for row in sent_pairs]
    # sim_pair = functools.partial(similarity_row, classifier=classifiers[0])
    # score_dataset = text_dataset.map(sim_pair, with_rank=True)  # pandas fragemented
    return similarity_ngpu(text_dataset, classifiers, n_pairs)


if __name__ == "__main__":
    sample_a = "Each computer program uses a region of memory called the stack to enable functions to work properly."
    sample_b = "From the outside, Les 4G, a Lyonnais bouchon (traditional restaurant), looked much like the nondescript cafe-cum-tobacco shops that can be found in most small French towns, but inside the decor was as warm and inviting as a country pub."
    sent_pairs = [" ".join([sample_a, "[SEP]", sample_b])]
    for i in range(100):
        sent_pairs.append(sent_pairs[0])
    results = similarity(
        sent_pairs=sent_pairs,
        classifiers=dar_env.mnli_classifiers["deberta"]
    )
    print(len(results))
    print(results)
