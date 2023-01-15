import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import dar_type
import sentence_transformers


def init_sent_embedder(name: str) -> dar_type.Embedder:
    embedder: dar_type.Embedder = sentence_transformers.SentenceTransformer(name)
    embedder.__name__ = name


sent_embedders = {
    "mpnet": init_sent_embedder("all-mpnet-base-v2"),
    "roberta": init_sent_embedder("all-roberta-large-v1"),
}
