import sys
from os import path
file_path = path.abspath(__file__)
root = path.dirname(path.dirname(file_path))
sys.path.append(root)

import dataset_config as config

ds_list = ["tac2010"]
datasets = dict()
for ds_name in ds_list:
    datasets[ds_name] = eval('config.' + ds_name + '_config["human_metrics"]')

approach = "new"
result_path_bases = [
    # "/home/turx/dar-archive/results-g1-230119-180618",
    # "/home/turx/dar-archive/results-g11-230119-172344",
    # "/home/turx/dar-archive/results-g11-230120-015934",
    # "/home/turx/dar-archive/results-g11-230120-162136",
    # "/home/turx/dar-archive/results-g11-230120-172646"
    "/home/turx/dar-archive/results-tac-rebuttal-230911-base",
    "/home/turx/dar-archive/results-tac-rebuttal-230911-deberta-base"
]
summary_dir = "/home/turx/dar-archive/results_tac_summary"
