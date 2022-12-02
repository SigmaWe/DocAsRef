import sys
from os import path

file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

from dar_env import mnli_classifier


def similarity(sentence_a: str, sentence_b: str):
    sequence = " ".join([sentence_a, sentence_b])
    classes = mnli_classifier(sequence)
    for c in classes[0]:
        if c["label"] == "NEUTRAL":
            return 1 - c["score"]
    raise Exception("Not found NEUTRAL class")


if __name__ == "__main__":
    print(similarity("Each computer program uses a region of memory called the stack to enable functions to work properly.",
          "From the outside, Les 4G, a Lyonnais bouchon (traditional restaurant), looked much like the nondescript cafe-cum-tobacco shops that can be found in most small French towns, but inside the decor was as warm and inviting as a country pub."))
