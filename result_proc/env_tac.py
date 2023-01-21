import sys
from os import path
file_path = path.abspath(__file__)
root = path.dirname(path.dirname(file_path))
sys.path.append(root)
sys.path.append(path.join(root, "env_grp/g0"))

import env

ds_list = ["tac2010"]
datasets = dict()
for ds_name in ds_list:
    datasets[ds_name] = env.evalbase.datasets[ds_name]["human_metrics"]

approach = "new"
result_path_bases = [
    "/home/turx/dar-archive/results-g1-230119-180618",
    "/home/turx/dar-archive/results-g11-230119-172344",
    "/home/turx/dar-archive/results-g11-230120-015934",
    "/home/turx/dar-archive/results-g11-230120-162136",
    "/home/turx/dar-archive/results-g11-230120-172646"
]
summary_dir = "/home/turx/dar-archive/results_tac_summary"
