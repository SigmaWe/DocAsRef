import typing
import numpy as np
import sentence_transformers
import transformers
from datasets.arrow_dataset import Dataset


MetricScoreList = typing.List[float]
MetricScoreDict = typing.Dict[str, MetricScoreList]
TextList = typing.List[str]
TextListSegments = typing.List[typing.List[str]]
TextSegments = typing.List[str]


class Embedder(sentence_transformers.SentenceTransformer):
    __name__: str


class SimilarityMatrixFunc(typing.Protocol):
    __name__: str

    def __call__(self, cand_segments: TextSegments, ref_segments: TextSegments) -> np.ndarray:
        ...


SimpleSimilarityMatrixFunc = typing.Callable[[TextSegments, TextSegments], np.ndarray]


Pipeline = transformers.Pipeline


class PipelinesList():
    __name__: str
    pipelines: typing.List[Pipeline]


PipelinesDict = typing.Dict[str, PipelinesList]


class MetricComputeFunc(typing.Protocol):
    def __call__(self, predictions: TextList, references: TextList) -> MetricScoreDict:
        ...


class DocWarning(Warning):
    """
    Warning raised when the document is malformed
    e.g., an empty document, which is meaningless to metrics
    """


class MNLICategory(typing.TypedDict):
    label: str
    score: float


MNLICategories = typing.List[MNLICategory]
MNLISimilarityExpression = typing.Callable[[MNLICategories], float]


SummaryLengthExpression = typing.Callable[[typing.Optional[Dataset]], typing.Optional[int]]

SentenceWeightFunction = typing.Callable[[np.ndarray], float]

IdfScoreFunction = typing.Callable[[TextSegments, np.ndarray], np.ndarray]
