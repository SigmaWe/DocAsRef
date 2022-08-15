# DocAsRef

TL;DR: Using document as reference summary in summary evaluation

## Abstract

* Summarization is a task under the banner of Natural Language Generation (NLG). The input is called a __document__ and the output is called a __summary__. 
* To judge the quality of a summary generated by a summarizer (such a summary is called a __system summary__), one approach is to compare the system summary against a __reference summary__, which is usually written by human. This approach is known as the __reference-based approach__ (and thus __reference-based metrics__) including some prominent ones, such as [ROUGE]([url](https://en.wikipedia.org/wiki/ROUGE_(metric))), [BLEURT]([url](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html)), and [BERTScore](https://github.com/Tiiiger/bert_score). 
* But reference-based approaches are greatly limited by the cost of obtaining human-written reference summaries. (See Section I of [SueNes paper](https://openreview.net/pdf?id=rfGxaxhWr-5))
* Hence, we wanna tweak ref-based approaches by using the document in lieu of the reference summary. 

# Datasets
There are two kinds of datasets associated with summarization research, 
1. summarization datasets (pairs of documents and summaries, i.e., `(doc1, sum1), (doc2, sum2), ...`), such as 
   * [CNN/Dailymail](https://www.tensorflow.org/datasets/catalog/cnn_dailymail) (CNNDM)
   * [BigPatent](https://www.tensorflow.org/datasets/catalog/big_patent)
   * [Billsum](https://www.tensorflow.org/datasets/catalog/billsum)
   * [Newsroom](https://www.tensorflow.org/datasets/catalog/newsroom) (which is a not a good one as a summary is just one sentence)
   * [ScientificPapers](https://www.tensorflow.org/datasets/catalog/scientific_papers) (it has two subsets, arXiv and PubMed)
   
   and 
2. summarization evaluation datasets (tuples of one document and multiple summaries generated by different summarizers, and human ratings for each summary, i.e., `(doc1, sum1A, sum1B, ..., rate1A, rate1B, ...), (doc2, sum2A, sum2B, ..., rate2A, rate2B, ...), ...`), such as 
   * [TAC2010](https://tac.nist.gov//2010/) (I cannot give Non-ISU personnel access to. This is US gov data.)
   * [RealSumm](https://github.com/neulab/REALSumm)
   * [Newsroom](https://github.com/lil-lab/newsroom/) 

In summarization evaluation/quality studies, the second type of datasets always serve as **test sets** because human evaluation is the groundtruth on summary qualities. 

## Supervised approach is hard 
Because a summarization evaluation dataset (TAC2010, RealSumm, Newsroom) is usually very small, say 100 samples, it is prone to overfitting to train a model using human ratings as targets/labels on such a dataset. Instead, an unsupervised approach, like ROUGE, BLEU or BERTScore, or a weak/self/semi-surpervised approach, like SueNes or BLUERT, is preferred. 

# Pilot study: feeding document-summary pairs to reference-based metrics.

In the pilot study, we wanna see how well ROUGE, BLEU, BLEURT, and BERTScore work when being fed with (document, system summary) pairs instead of (reference, system summary) pairs which they were originally designed for. Use the four metrics to make predictions on system summaries in the test datasets, and then compute the alignment between their predictions and human ratings, which are also in the test sets. 

Do not reinvent the wheel:
1. For the four metrics, use Huggingface's API, e.g., [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore). 
2. For code to load document and system summaries in RealSumm and Newsroom, please refer to the [`human` folder of SueNes' Github repo](https://github.com/forrestbao/SueNes/tree/master/human). Just copy whatever is needed into this repo. 

# New approach 1: sentence-level, better similarity metrics, and better weighting methods
The pilot study results show that when reference summaries are of good qualities, vanilla use of BERTScore is not good. Hence, we will try the following changes: 

| | BERTScore | Our changes | 
|--|--|--|
|Comparison between |Token pairs| sentence pairs | 
| Similarity metrics| cosine | NLI-based, semantically tell whether two sentences are related, could be trained on our own tasks | 
| weighting scheme | IDF | semantic weighting  |


