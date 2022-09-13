# Use pandas to redo it

# Copyleft 2022 Forrest Sheng Bao 
# with Iowa State University and Textea Inc. 
# forrest dot bao at gmail dot com 

from ast import dump
from tkinter import E
import pandas 
import typing 
import html
import difflib 
import json 
import tqdm


def clean_text(s:str):
    """Clean up the text in doc or summ in newsroom dataset
    including, removing HTML tags, unescap HTML control sequences 
    """
    s = s.replace("</p><p>", " ")
    s = html.unescape(s) 
    s = s.strip()
    return  s 

def append_reference_summaries_to_newsroom_human_evaluation(
    newsroom_human_evaluation_csv: str, 
    newsroom_test_jsonl: str, 
    dump_csv: str="") \
        -> \
    pandas.DataFrame :
    """Append reference summaries to the Newsroom dataset's human evaluation result. 

    The Newsroom dataset is awesome, specially its human evaluation part. However, the human evaluation result released does not have reference summaries, making it difficult to judge system summaries using reference-based approaches, such as ROUGE, BERTScore, MoverScore, or BLEURT. Hence, this script/function adds reference summaries to the human evaluation results. We do so by simply add a column to the original human evaluation CSV file. This function also returns a Pandas DataFrame as if it was loaded from the dumped CSV file. We do so by matching the title using difflib. 

    * newsroom_human_evaluation_csv: str, path to the Newsroom human evaluation CSV file, downloaded from https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv 
      Newsroom's human ratings are on 7 systems/summarizers, by 3 human raters, on 4 aspects ('coherence', 'informativeness', 'fluency', 'relevance'). Thus for each document, there are 21 (3x7) 1x4 ratings. There are 60 documents. So the CSV file has 1260 + 1 (the header) rows. Columns are: ['ArticleID', 'ArticleTitle', 'System', 'ArticleText', 'SystemSummary', 'CoherenceRating', 'FluencyRating', 'InformativenessRating', 'RelevanceRating']

    * newsroom_test_jsonl: str, the JSONL-format test split of Newsroom dataset from https://lil.nlp.cornell.edu/newsroom/download/index.html. All human evaluated data comes from the test split of Newsroom. Each line is as follows: 

    {
            "text": "...",
         "summary": "...",  # this is the reference summary 
           "title": "...",
         "archive": "http://...",
            "date": 20160302060024,
         "density": 1.25,
        "coverage": 0.75,
     "compression": 12.5,
    "compression_bin": "medium",
    "coverage_bin": "low",
     "density_bin": "abstractive"
    }

    * dump_csv: str, where to dump the reference-attached human evaluation result in CSV. If an empty string (default), then nothing dumps. 

    * df: a Pandas DataFrame of the following strucutre per row. Scores from 3 human evaluator are already pooled, via mean or median. A row of it is as follows: 
    In [5]: df.iloc[1]
    Out[5]: 
    ArticleID                                                              144
    ArticleTitle             '16 & Pregnant' Couple Arrested, Toddler Taken...
    System                                                           fragments
    ArticleText              Jacksonville, Ark., police arrested reality TV...
    SystemSummary            Jacksonville , Ark. , police arrested reality ...
    CoherenceRating                                                        4.0
    FluencyRating                                                     4.0
    InformativenessRating                                             4.0
    RelevanceRating                                                   4.0
    ReferenceSummary     We are SigmaWe NLP group from Iowa State University!

    This function is modified from the corresponding part from LS-Score (EMNLP 2019) and PerfScore (COLING 2022). 

    For any questions, please contact forrest dot bao at gmail dot com 

    """

    df = pandas.read_csv(newsroom_human_evaluation_csv) 
    # clean up the text 
    for column in ['SystemSummary', 'ArticleTitle', 'ArticleText']:
       df[column] = df[column].map(clean_text)

    # 1. extract all article IDs and titles. This avoids repeated string
    # matching later. It will save 3x7-1 times of the time. 
    
    id2title, id2ref = {}, {}
    for id in df["ArticleID"].unique():
        titles = df[df['ArticleID']==id] ['ArticleTitle'].unique()
        assert len(titles) == 1, "More than one articleTitle per articleID, wrong! "
        title= titles[0]
        id2title[id] = title
        id2ref[id] = ""

    # 2. Pair articleIDs to reference summaries in newsroom_raw_jsonl 
    with open(newsroom_test_jsonl, 'r') as f:
        for line in tqdm.tqdm(f):
            sample = json.loads(line)
            title_from_newsroom_jsonl = sample["title"]
            ref_from_newsroom_jsonl = sample["summary"]
            for articleID, title_from_human_evaluation_csv in id2title.items():
                ref = id2ref[articleID]
                if ref == "": # only compare those not matched 
                    if difflib.SequenceMatcher(
                        a=title_from_newsroom_jsonl, b=title_from_human_evaluation_csv).quick_ratio() > 0.9:
                        id2ref[articleID] = ref_from_newsroom_jsonl
                        print 

    # 3. insert a column called ref summary in input dataframe 
    df["ReferenceSummary"] = df["ArticleID"].map(id2ref)

    # 4. Dump the DF with Reference Summaries to a new CSV file 
    if dump_csv != "":
        df.to_csv(dump_csv, 
                index=False, 
                # FIXME: I am not sure about the three options below 
                # Need to use the same settings when loading dumpped CSV.
                quotechar="'", 
                doublequote=False, 
                escapechar= "\\" ) 

    return df


if __name__ == "__main__":
    df = append_reference_summaries_to_newsroom_human_evaluation(
        "../dataloader/newsroom-human-eval.csv", 
        "/media/forrest/12T_EasyStore1/data/NLP/resources/newsroom/test.jsonl", 
        dump_csv='merged.csv')

def pool_human_rating(
    df: pandas.DataFrame, 
    pool_method: str = "mean") \
        -> pandas.DataFrame:
    """Pool human ratings in Newsroom human evaluation results. 
    
    Each (document, system summary) pair is rated by 3 human raters. 
    The pooling shall reduce the number of rows from 1260 to 420. 

    * df: pandas.DataFrame, resulted from appending reference summaries to the human evaluation part of Newsroom dataset. Here, resulted from calling the append_reference_summaries_to_newsroom_human_evaluation function above. 

    * pool_method: str, "mean" or "median", how to obtain the score for each (document, system summary) pair from 3 human raters 


    Return data structure: 

    In [5]: df.iloc[1]
    Out[5]: 
    ArticleID                                                              144
    ArticleTitle             '16 & Pregnant' Couple Arrested, Toddler Taken...
    System                                                           fragments
    ArticleText              Jacksonville, Ark., police arrested reality TV...
    SystemSummary            Jacksonville , Ark. , police arrested reality ...
    CoherenceRating                                                        4.0
    FluencyRating                                                     4.333333
    InformativenessRating                                             4.333333
    RelevanceRating                                                   4.666667
    ReferenceSummary SigmaWe NLP group at Iowa State University!

    """

    # FIXME: Why cannot i use the method below to convert? 
    # exec(f'df_grouped = df_raw.groupby(by=["ArticleID", "System", "ArticleText", "SystemSummary"]).{consensus_method}().reset_index()')   

    if pool_method == "mean":
        df = df.groupby(by=["ArticleID", "ArticleTitle", "System", "ArticleText", "SystemSummary"]).mean().reset_index()  
    elif pool_method == "median":
        df = df.groupby(by=["ArticleID", "ArticleTitle", "System", "ArticleText", "SystemSummary"]).median().reset_index()  
    else: 
        print ("Wrong consensus method")
        exit()
    
    return df



import evaluate
# import sys
# sys.path.append("../")
# import bertscore_sentence.eval as bertscore_sentence

import functools
import env
import numpy, scipy


# TODO: Use dict rather than branch in eval 
# ref_{free,based}_metrics is a dict {str:function}
# ref_based_metrics = {
#     "bleurt": evaluate.load('bleurt', config_name='BLEURT-20', module_type='metric'), 
#     "rouge": functools.partial( evaluate.load("rouge"), use_aggregator=False)

#     }

# ref_free_metrics = {    
#     "bertscore-sentence": bertscore_sentence
# }
# all metrics shall return a dict {metric_name: List[float]}

def flatten_batch_result_to_DF(batch_result: dict) -> pandas.DataFrame:
    """Flatten a batch result to a Pandas DataFrame 

    The batch result is a multi-level dict  returned by eval.model_eval function in this lib:
    {
        "bleurt":{
            "trad":{
                "metric 1": List[float]
                "metric 2": List[float]
                ...
            }
            "new": {
                "metric 1": List[float]
                "metric 2": List[float]
                ...
            }
        }

    }

    """

    # Create a placeholder multiindex Dataframe
    index= pandas.MultiIndex.from_tuples([], names = ["approach", "model", "score_name"])
    df =pandas.DataFrame((), columns =index)


    for model_name in batch_result: # key in dict
        for approach in batch_result[model_name]: # key in dict
            for score_name, score_list in batch_result[model_name][approach].items(): 
                # detailed_score_name_tuple.append( (approach, model_name, score_name) )
                df[approach, model_name, score_name] = score_list

    return df 
                


def eval_summary_level(
    df:pandas.DataFrame, 
    document_column: str="ArticleText",
    system_summary_column: str="SystemSummary", 
    reference_summary_column: str="ReferenceSummary", 
    human_metrics: typing.List[str] = ["CoherenceRating", "FluencyRating", "InformativenessRating", "RelevanceRating"], 
    exp_models = env.models, 
    exp_approaches = env.approaches
):
    """Get summary-level scores for system summaries using various scoring methods. 

    Summary-level evaluation means that we compute corraltion for each document and then average across documents 

    """

    import eval 

    # batching based on articles. Also saves memory. 
    for articleID in df["ArticleID"].unique(): # summary-level, we so need to loop over articles 
        print (articleID)
        batch = df [ df["ArticleID"] == articleID] 

        # without .to_numpy(), will run into issues starting from 2nd iteration 
        docs   = batch[document_column].to_numpy()
        cands  = batch[system_summary_column].to_numpy()
        refs   = batch[reference_summary_column].to_numpy()

        human_scores = batch[human_metrics] # a DF

        batch_result = eval.model_eval(cands, refs, docs, exp_models, exp_approaches)
        result_df = flatten_batch_result_to_DF(batch_result)
        # result_df[approach, model, score_name] ===> a list for each pair in the batch 

        # correlation_types = ['pearsonr', 'kendalltau', 'spearmanr']
        for human_score_name, human_score in human_scores.iteritems(): 
            for (approach, model, score_name) in result_df.columns:
                metric_score = result_df[(approach, model, score_name)]
                cc = scipy.stats.spearmanr(human_score, metric_score)[0]

                print (articleID, human_score_name, approach, model, score_name,  cc)


def eval_system_level():
    """Get system-level scores for system summaries using various scoring methods. 

    System-level evaluation means that we compute corraltion for each system and then average across systems

    """

    pass

if __name__ == "__main__":
    # df = pandas.read_csv('merged.csv',              quotechar="'", doublequote=False,                escapechar= "\\" )
    eval_summary_level(df)





