import sys
from os import path
file_path = path.abspath(__file__)
root = path.dirname(path.dirname(file_path))
sys.path.append(root)

import config

ds_list = ["summeval", "newsroom", "realsumm_abs", "realsumm_ext"]
datasets = dict()
for ds_name in ds_list:
    datasets[ds_name] = eval('config.' + ds_name + '_config["human_metrics"]')

approach = "new"
result_path_bases = [
    # "/home/turx/dar-archive/results-1230",
    # "/home/turx/dar-archive/results-g3-230106-003117",
    # "/home/turx/dar-archive/results-g2-230107-190500",
    # "/home/turx/dar-archive/results-g3-230108-085258",
    # "/home/turx/dar-archive/results-g5-230110-063411",
    # "/home/turx/dar-archive/results-g8-230113-044120",
    # "/home/turx/dar-archive/results-g10-230114-020536",
    # "/home/turx/dar-archive/results-g10-230114-230347",
    # "/home/turx/dar-archive/results-g8-230115-191636",
    # "/home/turx/dar-archive/results-g9-230116-102957",
    # "/home/turx/dar-archive/results-g11-230116-131304",
    # "/home/turx/dar-archive/results-g12-230118-220300",
    # "/home/turx/dar-archive/results-g11-230119-020641",
    # "/home/turx/dar-archive/results-g8-230119-123900",
    # "/home/turx/dar-archive/results-g2-230119-223925",
    # "/home/turx/dar-archive/results-g11-230120-010031",
    # "/home/turx/dar-archive/results-g11-230120-162136"
    # "/home/turx/dar-archive/results-g13-230604",
    # "/home/turx/dar-archive/results-g13-230621",
    "/home/turx/dar-archive/results-new-classic-230623",
]
summary_dir = "/home/turx/dar-archive/results_snr_summary"
