# Group 1: classic (bertscore, rouge, bleurt)

from env_root import *

### LIBRARY VARS ###

os.environ["MOVERSCORE_MODEL"] = "roberta-large"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

### METRICS ###

import classic.eval as classic

metrics = {
    "bertscore": classic.bertscore_compute,
    "rouge": classic.rouge_compute,
    "bleurt": classic.bleurt_compute,
    # "moverscore-1gram": classic.moverscore_compute_1gram,
    # "moverscore-2gram": classic.moverscore_compute_2gram
}
