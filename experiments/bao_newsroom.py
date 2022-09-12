# Use pandas to redo it

import pandas 
import typing 


def load_newsroom(newsroom_csv: str, consensus_method: str= "mean"):
    """Load newsroom human rating scores

    newsroom human ratings are on 7 systesms/summarizers, by 3 human raters, on 4 aspects ('coherence', 'informativeness', 'fluency', 'relevance'). Thus for each document, there are 21 (3x7) 1x4 ratings. There are 6 documents 


    newsroom_csv: str, path to the newsroom file, downloaded from  https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv 

    consensus_method: str, "mean" or "median", how to obtain the score for each (document, system summary) pair from 3 human raters 
    """

    # pd = pandas.DataFrame(columns=['docID', 'system', 'document', 'summary', 
    # 'coherence', 'informativeness', 'fluency', 'relevance']) 


    df_raw = pandas.read_csv(newsroom_csv) 

    # convert scores based on consensus method 

    # FIXME: Why cannot i use the method below to convert? 
    # exec(f'df_grouped = df_raw.groupby(by=["ArticleID", "System", "ArticleText", "SystemSummary"]).{consensus_method}().reset_index()')   

    if consensus_method == "mean":
        df_grouped = df_raw.groupby(by=["ArticleID", "System", "ArticleText", "SystemSummary"]).mean().reset_index()  
    elif consensus_method == "median":
        df_grouped = df_raw.groupby(by=["ArticleID", "System", "ArticleText", "SystemSummary"]).median().reset_index()  
    else: 
        print ("Wrong consensus method")
        exit()
    
    return df_grouped 

def get_scores(documents: typing.List[str], ref_summaries: typing.List[str], system_summaries: typing.List[str]):
    pass 

def eval_newsroom(newsroom_csv: str = "../dataloader/newsroom-human-eval.csv", consensus_method:str = "mean"):
    df = load_newsroom(newsroom_csv=newsroom_csv, consensus_method=consensus_method)

    print (df)

    return  df 


if __name__ == "__main__":
    eval_newsroom()
