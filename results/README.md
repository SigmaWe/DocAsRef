# Report

Author: Ruixuan Tu (ruixuan.tu@wisc.edu), University of Wisconsin-Madison

Idea: Forrest Sheng Bao (forrest.bao@gmai.com), Iowa State University

## Analysis

### Expected Result

There should be no major difference between the results from the new approach with (document, system summary) pairs and the traditional approach with (reference, system summary) pairs. If so, we could rely less on human generated summary (i.e., reference) to enable the training be less supervised, as document and system summary are extracted and generated without human intervention.

### Actual Result

For ROUGE and BERTScore, we use the F-score (`fmeasure`, `f1`) of the medians, as it combines both precision and recall measures. For BLEURT, we use the median of all scores.

| | Newsroom (Trad) | Newsroom (New) | RealSumm (Trad) | RealSumm (New) |
| - | - | - | - | - |
| ROGUE-1 | 0.116266 | 0.151027 | 0.465393 | 0.057725 |
| ROGUE-2 | 0.032548 | 0.133317 | 0.225413 | 0.053366 |
| ROUGE-L | 0.098624 | 0.140724 | 0.329397 | 0.057289 |
| ROUGE-Lsum | 0.098678 | 0.140667 | 0.329510 | 0.057275 |
| BertScore | 0.824063 | 0.833208 | 0.889688 | 0.848760 |
| BLEU | 0.025807 | 1.52115e-07 | 0.2696018 | 4.773580e-12 |
| BLEURT | -0.895417 | -1.544569 | -0.100036 | -1.042327 |

### Observation

For ROGUE and BLEU, the differences in results from traditional and new evaluation approaches are significant. For BertScore, BLEURT, the differences are not significant. Therefore, it is better to replace with (document, system summary) pairs when we use the latter two metrics.

## Metrics

### ROUGE

https://huggingface.co/spaces/evaluate-metric/rouge

```json
{
    "rouge": {
        "trad": {
            "rouge1": AggregateScore(low=Score(precision, recall, fmeasure), mid=Score(...), high=Score(...)),
            "rouge2": AggregateScore(...),
            "rougeL": AggregateScore(...),
            "rougeLsum": AggregateScore(...)
        },
        "new": {
            ...
        }
    }
}
```

### BERTScore

https://huggingface.co/spaces/evaluate-metric/bertscore

```json
{
    "bertscore": {
        "trad": {
            "precision": precision ([0.0, 1.0]) for each sentence from the predictions + references lists,
            "recall": recall ([0.0, 1.0]) for each sentence from the predictions + references lists,
            "f1": F1 score ([0.0, 1.0]) for each sentence from the predictions + references lists,
            "hashcode": (hashcode of the library) "roberta-large_L17_no-idf_version=0.3.11(hug_trans=4.19.2)"
        },
        "new": {
            ...
        }
    }
}
```

### BLEU

https://huggingface.co/spaces/evaluate-metric/bleu

```json
{
    "bleu": {
        "trad": {
            "bleu": (float) blue score,
            "precisions": (List['float']) geometric mean of n-gram precisions,
            "brevity_penalty": (float),
            "length_ratio": (float),
            "translation_length": (int),
            "reference_length": (int)
        },
        "new": {
            ...
        }
    }
}
```

### BLEURT

https://huggingface.co/spaces/evaluate-metric/bleurt

```json
"bleurt": {
    "trad": {
        "scores": (list) a list of scores, one per prediction
    },
    "new": {
        "scores": ...
    }
}
```

## References

- Forrest Sheng Bao, Ruixuan Tu. https://github.com/SigmaWe/DocAsRef
- Forrest Sheng Bao, Ge Luo, et al. SueNes: A Weakly Supervised Approach to Evaluating Single-Document Summarization via Negative Sampling. NAACL 2022. https://github.com/forrestbao/SueNes
