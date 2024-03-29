import pandas
from ast import literal_eval as make_tuple

datasets = ["summeval", "newsroom", "realsumm_abs", "realsumm_ext"]
path_tmpl = "/home/turx/dar-archive/results-1230/{}_summary.json"

for dataset in datasets:
    print("Dataset: {}".format(dataset))
    path = path_tmpl.format(dataset)
    df = pandas.read_json(path)
    df.columns = pandas.MultiIndex.from_tuples([make_tuple(col) for col in df.columns])
    with pandas.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.precision', 2,
                            'display.float_format', lambda x: '%.2f' % x
                            ):
        # TODO: turn multiindex to regular 2d table by unstack
        print(df.loc["average", ("pearsonr", slice(None), "new", slice(None), "P")].to_string())
