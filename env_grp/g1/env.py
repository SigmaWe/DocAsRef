# Group 1: classic (bertscore, rouge, bleurt)

from env_root import *

### METRICS ###

import classic.eval as classic

metrics = {
    "bertscore": classic.bertscore_compute,
    "rouge": classic.rouge_compute,
    "bleurt": classic.bleurt_compute,
}
