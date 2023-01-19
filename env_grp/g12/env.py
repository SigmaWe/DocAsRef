# Group 12: classic (moverscore)

from env_root import *

### LIBRARY VARS ###

os.environ["MOVERSCORE_MODEL"] = "roberta-large"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

### METRICS ###

import classic.eval as classic

metrics = {
    "moverscore-1gram": classic.moverscore_compute_1gram,
    "moverscore-2gram": classic.moverscore_compute_2gram
}
