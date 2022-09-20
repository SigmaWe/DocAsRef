import pandas
import json 
import typing 

def clean_text(s:str):
    s = s.replace("\t", " ")
    s = s.strip()
    return s 

def pool_human_rating(
    human_ratings: typing.List[dict], 
    pool_method: str = "mean") \
        -> dict:

# input list:
# [{'coherence': 2, 'consistency': 1, 'fluency': 4, 'relevance': 2},
#  {'coherence': 1, 'consistency': 1, 'fluency': 2, 'relevance': 1},
#  {'coherence': 1, 'consistency': 1, 'fluency': 3, 'relevance': 2}]    

    df = pandas.DataFrame(human_ratings)

    if pool_method == "mean": 
        q = df.mean() 

    return q.to_dict()

    # ratings = {}
    # for human_metric in ['coherence', 'consistency', 'fluency', 'relevance']: 
    #     tmp = 0 
    #     for i in range(3): 
    #         tmp += human_ratings[i][human_metric]

    #     if pool_method == "mean": 
    #         ratings[human_metric] = tmp/3

    # return ratings 

def load_summeval(paired_jsonl="summeval_annotations.aligned.paired.scored.jsonl"):
    human_metrics = ['coherence', 'consistency', 'fluency', 'relevance']
    with open(paired_jsonl, 'r', encoding='utf-8') as fd:
        dataset = [json.loads(line) for line in fd]

        df = pandas.DataFrame(dataset)

        # return df 
    # df.columns ==> 
    # ['id', 'decoded', 'expert_annotations', 'turker_annotations',
    #    'references', 'model_id', 'filepath', 'metric_scores_1',
    #    'metric_scores_6', 'metric_scores_11', 'text']


        # process nested precalcualted metrics 
        tdf = df['metric_scores_1'].to_list()
        for row in tdf: 
            row.update(row['rouge'])
            row['supert'] = row['supert'][0]
            del row['rouge']
        df = pandas.concat([df, pandas.DataFrame(tdf)], axis=1)

        for refId in range(11):
            df[f"ReferenceSummary_{refId}"] = df["text"] # place holder 
        for human_metric in human_metrics:
            df[human_metric] = df["id"] # place holder 

        # clean up 
        df = df.rename(columns={'decoded':'SystemSummary', 'text':'ArticleText','model_id':'system'})

        for index, row in df.iterrows():
            for refId in range(11):
                df.at[index, f"ReferenceSummary_{refId}"] = clean_text(row["references"][refId])

            pooled_human_ratings = pool_human_rating(row['expert_annotations'])
            for human_metric in human_metrics:
                df.at[index, human_metric] = pooled_human_ratings[human_metric]

            for column in ['ArticleText', 'SystemSummary']:
                df.at[index, column] = clean_text(row[column])

        df = df.drop(columns=['filepath', 'metric_scores_1', 'metric_scores_6', 'metric_scores_11', 'expert_annotations', 'turker_annotations', 'references'])

    return df 

    # In [7]: df.iloc[1]["expert_annotations"]
    # Out[7]: 
    # [{'coherence': 3, 'consistency': 5, 'fluency': 5, 'relevance': 2},
    #  {'coherence': 2, 'consistency': 5, 'fluency': 5, 'relevance': 3},
    #  {'coherence': 2, 'consistency': 5, 'fluency': 5, 'relevance': 3}]

if __name__ == "__main__":

    dataset_df = load_summeval()

    precalc_metrics = [ # keys from original SummEval json file 
        'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score', 
        'rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score',
        'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f_score',
        'rouge_we_1_p', 'rouge_we_1_r', 'rouge_we_1_f', 
        'rouge_we_2_p', 'rouge_we_2_r', 'rouge_we_2_f', 
        'meteor', 'cider', 's3_pyr', 's3_resp', 
        'mover_score', 'sentence_movers_glove_sms', 'bleu', 
        'bert_score_precision', 'bert_score_recall', 'bert_score_f1',         
        'blanc', 'summaqa_avg_prob', 'summaqa_avg_fscore', 'supert']

    import eval 

    corr_df = eval.eval_summary_level(
        dataset_df, 
        pre_calculated_metrics=precalc_metrics, 
        debug=False)
    with pandas.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       'display.float_format', lambda x: '%.3f' % x
                       ):
        with open("results/result_summeval.md", 'w') as f:
            f.write(corr_df['average'].to_string())
        print(corr_df['average'])

    with open(f"results/result_summeval.json", 'w') as f:
        json_ugly = corr_df.to_json(orient="index")
        json_parsed = json.loads(json_ugly)
        f.write(json.dumps(json_parsed, indent=2))