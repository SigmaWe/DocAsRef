# https://github.com/facebookresearch/fairseq/tree/main/examples/roberta

import sim
import sim_expr
import env
import dar_type


def label_row(row: str, classifier: dar_type.Pipeline) -> str:
    classes = classifier(row)
    return classes[0][0]["label"]


if __name__ == "__main__":
    batch_of_pairs = [
        ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
        ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
        ['potatoes are awesome.', 'I like to run.'],
        ['Mars is very far from earth.', 'Mars is very close.'],
    ]
    sent_pairs = ["[SEP]".join([row[0], row[1]]) for row in batch_of_pairs]
    sim_results = sim.similarity(
        sent_pairs=sent_pairs,
        classifiers=env.mnli_classifiers["roberta"],
        expr=sim_expr.not_neutral
    )
    print(len(sim_results))
    print(sim_results)
    label_results = [label_row(row, classifier=env.mnli_classifiers["roberta"].pipelines[0]) for row in sent_pairs]
    print(len(label_results))
    print(label_results)
