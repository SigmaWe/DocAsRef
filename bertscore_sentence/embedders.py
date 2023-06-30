from multiprocessing import freeze_support
import sys
# from os import path
# file_path = path.abspath(__file__)
# sys.path.append(path.dirname(path.dirname(file_path)))

import dar_type
from sentence_transformers import SentenceTransformer, models
import typing
import functools


def init_sent_embedder(name: str, pooling_mode: typing.Optional[str] = None) -> dar_type.Embedder:
    if pooling_mode is None:
        model = SentenceTransformer(name)
    else:
        word_embedding_model = models.Transformer(name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # TODO: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
    # pool = model.start_multi_process_pool()
    # embedder: dar_type.Embedder = functools.partial(model.encode_multi_process, pool=pool)
    embedder: dar_type.Embedder = functools.partial(model.encode)
    embedder.__name__ = name
    return embedder


sent_embedders = {
    "mpnet": init_sent_embedder("all-mpnet-base-v2"),
    "roberta-large": init_sent_embedder("all-roberta-large-v1"),
    "deberta-large": init_sent_embedder("microsoft/deberta-large-mnli", pooling_mode="cls"),
    # "deberta-xlarge": init_sent_embedder("microsoft/deberta-xlarge-mnli", pooling_mode="cls"),
}
