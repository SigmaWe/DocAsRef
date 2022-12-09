import spacy
from transformers import pipeline
import sentence_transformers
import evaluate


nlp_spacy = spacy.load("en_core_web_lg")
mnli_classifier_roberta = pipeline("text-classification", model="roberta-large-mnli", top_k=None)
mnli_classifier_roberta.__name__ = "roberta-large-mnli"
mnli_classifier_bart = pipeline("text-classification", model="facebook/bart-large-mnli", top_k=None)
mnli_classifier_bart.__name__ = "bart-large-mnli"
sent_embedder_mpnet = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
sent_embedder_mpnet.__name__ = "all-mpnet-base-v2"
sent_embedder_roberta = sentence_transformers.SentenceTransformer("all-roberta-large-v1")
sent_embedder_roberta.__name__ = "all-roberta-large-v1"
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt", config_name="BLEURT-20", module_type="metric")
