import dar_type


def not_neutral(categories: dar_type.MNLICategories) -> float:
    for category in categories:
        if category["label"].lower() == "neutral":
            return 1 - category["score"]
    raise Exception("no neutral category")


def entail_only(categories: dar_type.MNLICategories) -> float:
    for category in categories:
        if category["label"].lower() == "entailment":
            return category["score"]
    raise Exception("no entailment category")


def entail_contradict(categories: dar_type.MNLICategories) -> float:
    entail_score = None
    contradict_score = None
    for category in categories:
        if category["label"].lower() == "entailment":
            if entail_score is not None:
                raise Exception("multiple entailment scores")
            entail_score = category["score"]
        elif category["label"].lower() == "contradiction":
            if contradict_score is not None:
                raise Exception("multiple contradiction scores")
            contradict_score = category["score"]
    if entail_score is None or contradict_score is None:
        raise Exception("no entailment or contradict category")
    return entail_score - contradict_score
