import sys
from os import path
# file_path = path.abspath(__file__)
# sys.path.append(path.dirname(path.dirname(file_path)))

from text_preprocess import list_segmentation
import dar_type
import math


class TopK:
    def extract_doc(self, ref_segments: dar_type.TextSegments, topk: int) -> str:
        topk_sents = ref_segments[0:topk]
        return " ".join(topk_sents)

    def extract(self, ref_segments_list: dar_type.TextListSegments, topk: int) -> dar_type.TextList:
        return [self.extract_doc(ref_segments, topk) for ref_segments in ref_segments_list]

    def __call__(self, metric_compute_f: dar_type.MetricComputeFunc, predictions: dar_type.TextList, references: dar_type.TextList, topk: int) -> dar_type.MetricScoreDict:
        refs = self.extract(ref_segments_list=list_segmentation(references), topk=topk)
        return metric_compute_f(predictions=predictions, references=refs)

topk_compute = TopK()


class TopP:
    def extract_doc(self, ref_segments: dar_type.TextSegments, topp: float) -> str:
        topk = int(math.ceil(len(ref_segments) * topp))
        topk_sents = ref_segments[0:topk]
        return " ".join(topk_sents)

    def extract(self, ref_segments_list: dar_type.TextListSegments, topp: float) -> dar_type.TextList:
        return [self.extract_doc(ref_segments, topp) for ref_segments in ref_segments_list]

    def __call__(self, metric_compute_f: dar_type.MetricComputeFunc, predictions: dar_type.TextList, references: dar_type.TextList, topp: int) -> dar_type.MetricScoreDict:
        refs = self.extract(ref_segments_list=list_segmentation(references), topp=topp)
        return metric_compute_f(predictions=predictions, references=refs)

topp_compute = TopP()
