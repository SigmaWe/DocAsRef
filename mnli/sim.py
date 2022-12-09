import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import typing
import transformers


def similarity(sent_pairs: typing.List[str], classifier: transformers.Pipeline):
    classes = classifier(sent_pairs)
    scores = []
    for c in classes:
        for category in c:
            if category["label"] == "NEUTRAL":
                scores.append(1 - category["score"])
                break
    return scores


if __name__ == "__main__":
    sample_a = "Each computer program uses a region of memory called the stack to enable functions to work properly."
    sample_b = "From the outside, Les 4G, a Lyonnais bouchon (traditional restaurant), looked much like the nondescript cafe-cum-tobacco shops that can be found in most small French towns, but inside the decor was as warm and inviting as a country pub."
    print(similarity([" ".join([sample_a, sample_b])]))
