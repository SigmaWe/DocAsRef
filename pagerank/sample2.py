import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import pagerank.eval
import bertscore_sentence.eval as bertscore_sentence
import functools
import dar_type
import sentence_transformers
import numpy as np
import scipy.stats

sent_embedder_mpnet: dar_type.Embedder = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
sent_embedder_mpnet.__name__ = "all-mpnet-base-v2"
sim_mat_f = functools.partial(bertscore_sentence.cos_sim_mat_f, embedder=sent_embedder_mpnet)
# weight_f = np.sum
weight_f = scipy.stats.entropy

ref_segments = ["There is a zoo.", "This is a tiger.", "This is a monkey.", "This is a cat."]

pred_segments = ["There are many animals in a zoo.", "Computer can execute many programs."]

sim_mat = sim_mat_f(cand_segments=pred_segments, ref_segments=ref_segments)

idf = pagerank.eval.get_idf(ref_segments, sim_mat, sim_mat_f, weight_f)
idf_sum = np.sum(idf)

# print(idf)
print(idf / idf_sum)
