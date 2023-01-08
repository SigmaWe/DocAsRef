import spacy
import dar_type
import typing

nlp_spacy = spacy.load("en_core_web_lg")


def list_segmentation(raw_list: dar_type.TextList) -> dar_type.TextListSegments:
    return [text_segmentation(raw) for raw in raw_list]


def text_segmentation(piece: str) -> dar_type.TextSegments:
    doc = nlp_spacy(piece)
    doc_sents = [sent.text for sent in doc.sents]
    return doc_sents


def flatten(l: typing.List[typing.List]):
    return [item for sublist in l for item in sublist]
