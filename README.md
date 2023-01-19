# DocAsRef

TL;DR: Using document as reference summary in summary evaluation

Read the [**Background and terminology**](https://forrestbao.github.io/summarization_metrics.html) first.

# Pilot study: feeding document-summary pairs to reference-based metrics.

In the pilot study, we wanna see how well ROUGE, BLEU, BLEURT, and BERTScore work when being fed with (document, system summary) pairs instead of (reference, system summary) pairs which they were originally designed for. Use the four metrics to make predictions on system summaries in the test datasets, and then compute the alignment between their predictions and human ratings, which are also in the test sets. 

Do not reinvent the wheel:
1. For the four metrics, use Huggingface's API, e.g., [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore). 
2. For code to load document and system summaries in RealSumm and Newsroom, please refer to the [`human` folder of SueNes' Github repo](https://github.com/forrestbao/SueNes/tree/master/human). Just copy whatever is needed into this repo. 

# Approaches

For evaluation, set correct path to [EvalBase](https://github.com/SigmaWe/EvalBase), in [`eval.py`](eval.py)

For metrics, reference [`env_grp/`](env_grp/)

We have these groups of environments:

- Group 1: classic (bertscore, rouge, bleurt)
- Group 2: bertscore-sentence (cos, mnli)
- Group 3: AnyRef (bart, pegasus-xum) + Group 1
- Group 4: Top + Group 1
- Group 5: AnyRef (pegasus-newsroom, pegasus-cnndm) + Group 1
- Group 6: AnyRef (pegasus-large) + Group 1
- <del>Group 7 (Dropped): AnyRef (pegasus-x-large) + Group 1</del>
- Group 8: PageRank + Group 2
- Group 9: AnyRef (pegasus-newsroom) + Group 2
- Group 10: Top + Group 2
- Group 11: classic bertscore + deberta-large-mnli + Top
- Group 12: classic (moverscore)

## Approach 0: just replacing human summaries with documents

Metrics: BERTScore, ROUGE, BLEURT, MoverScore

Implemented in `/classic`

## Approach 1: sentence-level, better similarity metrics, and better weighting methods
The pilot study results show that when reference summaries are of good quality, vanilla use of BERTScore is not good. Hence, we will try the following changes: 

| | BERTScore | Our changes | 
|--|--|--|
|Comparison between |Token pairs| sentence pairs | 
| Similarity metrics| cosine | NLI-based, semantically tell whether two sentences are related, could be trained on our own tasks | 
| weighting scheme | IDF | semantic weighting  |

### Approach 1.1 Sentence embedding + cosine similarity, no weighting

The document is a list of $n$ sentences: $D=[D_1, D_2, ..., D_n]$, while the system/generated summary (to be evaluated) is a list of $m$ sentences: $S=[S_1, S_2, ..., S_m]$. And $m < < n$. 

A memory-saving pseudocode: 

```
for D in all_documents:
  [D1, D2, D3] = sentence_segmenter(D) # break D into sentences
  [E_1, E_2, ...] = sentence_embedder([D1, D2, ...]) # embed each sentence in D
  for S in summaries_of_D (not all summaries, only those of D):
    [S1, S2, ...] = sentence_segmenter(S) # break an S into sentences 
    [E'_1, E'_2, ...] = sentence_embedder([S1, S2, ...]) # embed each sentence in S
    
    score = summary_scorer(E1, E2, ..., E'1, E'2)
```

Implemented in `/bertscore_sentence`

### Approach 1.2

Use a bi-sentence model, instead of cosine similarity in Approach 1.1. In this new approach, we do not have to embed individual sentences. Instead, we embed a pair of sentences (one from documents and one from system summaries). We use models `roberta-large-mnli`, `facebook/bart-large-mnli`, `microsoft/deberta-xlarge-mnli`, and `microsoft/deberta-large-mnli` with different expressions for similarity.

Implemented in `/mnli`

### Approach 1.3 (Ruixuan)

Suppose a list of system summary sentences $L_S$ and a list of document sentences $L_D$, then we find the difference in weights generated by evaluating the attention of documents in the two lists respectively, instead of two sentences for one time in Approach 1.2.

### Approach 1.4 Sentence weighting


f1(D1, D2, ..., S1, S2, ...) 

f2( f3(D1, S1, S2, ..), f3(D2, S1, S2, ..), ...., f3(Dn, S1, S2, ...) ) 

entropy ( sim(S1, D1), sim(S1, D2), ... ) 
+ 
entropy ( sim(S2, D1), sim(S2, D2), ... )

Implemented in `/pagerank`

### Approach 1.5 Pseudo-reference by Top-K and Top-P

Implemented in `topk/`

### Approach 1.6 Pseudo-reference using decent enough summarizers

Instead of top-k and top-p in Approach 1.5, we use models `google/pegasus-xsum` and `facebook/bart-large-cnn` to generate pseudo-reference from documents.

Implemented in `anyref/`
