import spacy
from transformers import pipeline
import sentence_transformers
import evaluate


nlp = spacy.load("en_core_web_lg")
mnli_classifier = pipeline("text-classification",
                           model="roberta-large-mnli", top_k=None)
sent_embedder = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt", config_name="BLEURT-20", module_type="metric")
