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
import text_preprocess
import functools


class AnyRef:
    def summarize_thread(self, dataset: Dataset, summarizer: dar_type.Pipeline, summaries: dar_type.TextList, min_len_expr: dar_type.SummaryLengthExpression, max_len_expr: dar_type.SummaryLengthExpression) -> None:
        min_length = min_len_expr(dataset)
        max_length = max_len_expr(dataset)
        summarizer_call = summarizer
        if min_length is not None:
            summarizer_call = functools.partial(summarizer_call, min_length=min_length)
        elif max_length is not None:
            summarizer_call = functools.partial(summarizer_call, max_length=max_length)
        for out in summarizer_call(KeyDataset(dataset, "text"), batch_size=8):
                summaries.append(out[0]["summary_text"])

    def summarize_ngpu(self, text_dataset: Dataset, summarizers: dar_type.PipelinesList, n_docs: int, min_len_expr: dar_type.SummaryLengthExpression, max_len_expr: dar_type.SummaryLengthExpression):
        # split dataset to n_gpu, batch 8 on every gpu at same time
        # ds[0]: [0, n_docs / n_gpu); ds[1]: [n_docs / n_gpu, 2 * (n_docs / n_gpu))
        ds_batch_sz = int(math.ceil(n_docs / env.n_gpu))
        thread_pool: typing.List[threading.Thread] = []
        results: typing.List[dar_type.TextList] = []
        for i_gpu in range(env.n_gpu):
            lower_bound = ds_batch_sz * i_gpu
            upper_bound = ds_batch_sz * (i_gpu + 1)
            if upper_bound > n_docs:
                upper_bound = n_docs
            if lower_bound >= upper_bound:
                break
            summaries_gpu: dar_type.TextList = []
            results.append(summaries_gpu)
            thread_pool.append(threading.Thread(target=self.summarize_thread, kwargs={
                "dataset": text_dataset.select(range(lower_bound, upper_bound, 1)),
                "summarizer": summarizers.pipelines[i_gpu],
                "summaries": summaries_gpu,
                "min_len_expr": min_len_expr,
                "max_len_expr": max_len_expr
            }))
        for thread in thread_pool:
            thread.start()
        for thread in thread_pool:
            thread.join()
        summaries = text_preprocess.flatten(results)
        if len(summaries) != n_docs:
            raise Exception("missing summary")
        return summaries

    def summarize(self, docs: dar_type.TextList, summarizers: dar_type.PipelinesList, min_len_expr: dar_type.SummaryLengthExpression, max_len_expr: dar_type.SummaryLengthExpression) -> dar_type.TextList:
        text_dataset = Dataset.from_dict({"text": docs})
        text_dataset = text_dataset.map(lambda row: {"len": len(row)})
        n_docs = len(docs)
        if n_docs < env.n_gpu:
            summaries = []
            self.summarize_thread(
                dataset=text_dataset,
                summarizer=summarizers.pipelines[0],
                summaries=summaries,
                min_len_expr=min_len_expr,
                max_len_expr=max_len_expr
            )
            return summaries
        return self.summarize_ngpu(
            text_dataset=text_dataset,
            summarizers=summarizers,
            n_docs=n_docs,
            min_len_expr=min_len_expr,
            max_len_expr=max_len_expr
        )

    def __call__(self, metric_compute_f: dar_type.MetricComputeFunc, predictions: dar_type.TextList, references: dar_type.TextList, summarizers: dar_type.PipelinesList, min_len_expr: dar_type.SummaryLengthExpression, max_len_expr: dar_type.SummaryLengthExpression) -> dar_type.MetricScoreDict:
        refs = self.summarize(
            docs=references,
            summarizers=summarizers,
            min_len_expr=min_len_expr,
            max_len_expr=max_len_expr
        )
        return metric_compute_f(predictions=predictions, references=refs)


anyref_compute = AnyRef()
