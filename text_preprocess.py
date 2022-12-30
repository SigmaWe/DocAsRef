import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from dar_env import nlp_spacy
import dar_type


def list_segmentation(raw_list: dar_type.TextList) -> dar_type.TextListSegments:
    return [text_segmentation(raw) for raw in raw_list]


def text_segmentation(piece: str) -> dar_type.TextSegments:
    doc = nlp_spacy(piece)
    doc_sents = [sent.text for sent in doc.sents]
    return doc_sents
