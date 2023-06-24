# Addon for EvalBase

| Metric      | Description                            |
| ----------- | -------------------------------------- |
| BLEU        | Sacrebleu & BLEU from HuggingFace      |
| METEOR      | HuggingFace                            |
| REUSE       | Follow Instructions on author's [github](https://github.com/AnanyaCoder/WMT22Submission_REUSE)       
| SDC*        | Already in this Repo                   |
| SMS         | Already in this Repo                   |
| Bart Score  | Already in this Repo                   |

## How to use
1. For BLEU METEOR, REUSE, SDC*, initialize a class first and call `compute` function see `additional_metrics.py`
2. SMS `from wmd_master import SMD_scorer`, and use `calculate_score` function. **Note: SMS is hard coded with GloVe embeddings and SMS Metric**

## Known problems for me
1. When loading METEOR from HF may display network errors warning messages. This is due to `nltk.download` not working. Probably because of bad internet
2. METEOR needs some addons such as wordnet. If `nltk.download('wordnet')` doesn't work. Go google nltk_data github and pull the repo to some local folders specified by the lookup error.
3. SMS (Sentence Mover Score) uses 2 embeddings, ELMo and GloVe. ELMo uses `allennlp.commands.elmo` which doesn't work for some (most) people. [Issue](https://github.com/eaclark07/sms/issues/3) by some other people. Therefore,  GloVe is hard coded.
4. You need `en_core_web_md` for SMS, this can be done with huggingface. This requires SpaCy to be >=3.5.0,<3.6.0 or else when loading `en_core_web_md`, it will display a warning message. The performance isn't affected (I tried with like 5 sentences) but it is annoying. Tried with 3.2.2 and with 3.5.0.
~~~
!pip install https://huggingface.co/spacy/en_core_web_md/resolve/main/en_core_web_md-any-py3-none-any.whl

# Using spacy.load().
import spacy
nlp = spacy.load("en_core_web_md")

# Importing as module.
import en_core_web_md
nlp = en_core_web_md.load()
~~~
5. SMS, `pip install wmd` will fail with you don't have a C++ compiler. It is suggested to run SMS on Linux.
