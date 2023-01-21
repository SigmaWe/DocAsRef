import os
import warnings
import pandas
import sys
import datetime

path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(path, "results")


def init():
    warnings.filterwarnings(
        action="ignore",
        message="You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset",
        category=UserWarning
    )
    warnings.simplefilter(
        action="ignore",
        category=pandas.errors.PerformanceWarning
    )


def create_result_path():
    if not os.path.exists(result_path):
        os.makedirs(result_path)


def eval():
    create_result_path()
    env.summeval.main()
    env.newsroom.main()
    env.realsumm.main("abs")
    env.realsumm.main("ext")
    env.tac2010.main()


if __name__ == '__main__':
    init()
    env_grp_i = 11
    print("[ENV] group " + str(env_grp_i))
    env_path = os.path.join(path, "env_grp", "g" + str(env_grp_i))
    sys.path.append(env_path)
    import env
    eval()
    if os.path.exists(result_path):
        os.rename(result_path, os.path.join(path, "results-g{}-{}".format(env_grp_i, datetime.datetime.now().strftime("%y%m%d-%H%M%S"))))
    sys.path.remove(env_path)
    if "env" in sys.modules:
        del sys.modules["env"]
    del env
