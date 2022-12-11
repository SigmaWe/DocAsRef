# for reference and type hint only, see EvalBase for full

import typing


class EvalPieces:
    raw_list: typing.List[str]
    segments_list: typing.List[typing.List[str]]

    def __init__(self):
        raise Exception("read only")
