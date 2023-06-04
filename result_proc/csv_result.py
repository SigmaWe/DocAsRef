# Print the JSON output into a CSV format for importing into Excel/Google Sheets
# modified from Forrest's EvalBase/print_result.py

import sys
from os import path
file_path = path.abspath(__file__)
sys.path.append(path.dirname(path.dirname(file_path)))

import os
import shutil
import pandas as pd
from typing import List

# Load results JSON using Pandas
def load_result_json_pandas(json_result: str, is_system: bool):
    print ("Loading result JSON using Pandas from ", json_result, " ...")
    if is_system:
        df = pd.read_json(json_result, orient='index').T
    else:
        df = pd.read_json(json_result)
    return df

# transform result dataframe

def transform_dataframe(df, reset_index: bool, is_system: bool) -> pd.Series:

    # Convert keys in JSON to multiindexes
    # This is probably due to that we stupidly set orient='index'
    # when dumping result DataFrame to JSON
    df.columns = pd.MultiIndex.from_tuples(
        [eval(column) for column in df.columns],
        # because loaded JSON has strings of tuple definitions
        names = ( # multiindex names
            "corr_metric",
            "aspect",
            "approach",
            "model_name",
            "scorer_name"
            )
        )

    # Transpose indexes and columns
    df = df.T

    # Leave only the average scores
    if not is_system:
        s = df["average"]
    else:
        s = df[0]
    # This resets to pandas.Series, with multiindex as index

    if reset_index:
        # This will insert repeating column names.
        # Default, False
        s.reset_index()

    return s

# the result JSON, once loaded, has the following structure:
# keys are tuples, (corr_metric, aspect, trad/new, metric_name, metric_)

# beautiful print

def parse(
    s: pd.Series,
    corr_metric: str,
    aspects: List[str],
    approach: str,
    model_list: List[str],
    header: bool,
    ):
    """
    aspect: List[str], dependending on the dataset
    corr_metric: str, a correlation metric,
        e.g., "spearmanr", "pearsonr", or "kendalltau"
    approach: List[str], e.g., ["trad", "new"]
    model_score_tuples: list of tuples of (model_name, score_name), e.g.,
       [("bertscore", "f1"), ("PreCalc", "supert")]
       When None, print all models and scores

    """

    # Step 1: build the table content
    print_table = [] # 2D list of strings/floats

    # narrow to the correlation metric
    s = s[
        (s.index.get_level_values('corr_metric') == corr_metric) &
        (s.index.get_level_values('model_name').isin(model_list)) &
        (s.index.get_level_values('approach') == approach)
    ]

    for aspect in aspects:
        s2 = s[
            (s.index.get_level_values('aspect') == aspect)
        ]
        column = s2.values.round(decimals=3).tolist()
        print_table.append(column)

    # Step 2: build the table header
    # first column is empty, for model x scorer
    header1 = [""]
    for aspect in aspects:
        header1 += [aspect]

    # Step 3: print the table
    model_names = s.index.get_level_values('model_name').to_list()
    scorer_names = s.index.get_level_values('scorer_name').to_list()
    model_scorer_tuples = zip(model_names, scorer_names)

    first_column = [
        f"{model}_{scorer}"
        for model, scorer in model_scorer_tuples
        ]

    print_table = [first_column] + print_table

    Rows = list(zip(*print_table))
    if header:
        Rows = [header1] + Rows

    # Code taken from https://stackoverflow.com/questions/13214809/pretty-print-2d-list
    s = [[str(e) for e in row] for row in Rows]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = ','.join('{{:{}}}'.format(0) for _ in lens)
    table = [fmt.format(*row) for row in s]
    return '\n'.join(table) + "\n"

corr_metrics = ["spearmanr", "pearsonr", "kendalltau"]
levels = ["summary", "system"]

from result_proc.env_snr import *

if __name__ == "__main__":
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)
    summary_path_tmpl = os.path.join(summary_dir, "{}_{}_{}.csv")
    for result_path_base in result_path_bases:
        for dataset_id, (dataset, aspects) in enumerate(datasets.items()):
            with open(os.path.join(result_path_base, "model_list.txt"), "r") as f:
                model_list = f.read().splitlines()

            if dataset_id == 0:
                if not os.path.exists(os.path.join(summary_dir, "model_list.txt")):
                    with open(os.path.join(summary_dir, "model_list.txt"), "w") as f:
                        f.write("\n".join(model_list) + "\n")
                else:
                    with open(os.path.join(summary_dir, "model_list.txt"), "a") as f:
                        f.write("\n".join(model_list) + "\n")

            for level in levels:
                pandas_json_path = os.path.join(result_path_base, "{}_{}.json".format(dataset, level))
                is_system = (level == "system")

                df = load_result_json_pandas(pandas_json_path, is_system=is_system)
                s  = transform_dataframe(df, reset_index=False, is_system=is_system)

                for corr_metric in corr_metrics:
                    summary_path = summary_path_tmpl.format(dataset, corr_metric, level)
                    if not os.path.exists(summary_path):
                        with open(summary_path, "w") as f:
                            f.write(parse(
                                s,
                                corr_metric=corr_metric,
                                aspects=aspects,
                                approach=approach,
                                model_list=model_list,
                                header=True,
                            ))
                    else:
                        with open(summary_path, "a") as f:
                            f.write(parse(
                                s,
                                corr_metric=corr_metric,
                                aspects=aspects,
                                approach=approach,
                                model_list=model_list,
                                header=False,
                            ))
