import pandas
import pickle 
import json 

def clean_text(s:str):
    """Clean up the text in doc or summ in RealSumm dataset
    including, removing HTML tags, unescap HTML control sequences 
    """
    s = s.replace("<t>", "")
    s = s.replace("</t>", "")
    s = s.replace("\t", " ")
    s = s.strip()
    return  s 

def load_realsumm(pickfile:str):
    """

    In [4]: sd = pickle.load(open('abs.pkl','rb'))

    In [5]: sd.keys()
    Out[5]: dict_keys([52, 2, 23, 62, 6, 92, 73, 47, 86, 32, 67, 42, 36, 3, 50, 35, 17, 45, 16, 48, 95, 91, 89, 85, 74, 28, 49, 58, 12, 33, 14, 9, 8, 29, 43, 39, 38, 84, 57, 31, 21, 71, 15, 24, 78, 5, 90, 87, 51, 40, 1, 46, 66, 56, 7, 88, 72, 77, 34, 68, 26, 64, 18, 76, 30, 80, 61, 99, 79, 41, 27, 94, 22, 4, 82, 0, 10, 37, 25, 60, 70, 13, 19, 75, 69, 54, 65, 81, 98, 11, 55, 44, 59, 20, 83, 96, 97, 93, 53, 63])

    In [6]: ! ls -lht
    total 4.2M
    -rw-rw-r-- 1 forrest forrest 956K Sep 12 15:15 ext_ours.pkl
    -rw-rw-r-- 1 forrest forrest 1.1M Sep 12 15:15 ext.pkl
    -rw-rw-r-- 1 forrest forrest 1.1M Sep 12 15:15 abs_ours.pkl
    -rw-rw-r-- 1 forrest forrest 1.2M Sep 12 15:15 abs.pkl

    In [7]: sdp[52].keys()
    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)
    Input In [7], in <cell line: 1>()
    ----> 1 sdp[52].keys()

    NameError: name 'sdp' is not defined

    In [8]: sd[52].keys()
    Out[8]: dict_keys(['doc_id', 'ref_summ', 'system_summaries'])

    In [9]: sd[52]['doc_id']
    Out[9]: 52

    In [10]: sd[52]['ref_summ']
    Out[10]: "<t> manchester united take on manchester city on sunday . </t>  <t> match will begin at 4pm local time at united 's old trafford home . </t>  <t> police have no objections to kick-off being so late in the afternoon . </t>  <t> last late afternoon weekend kick-off in the manchester derby saw 34 fans arrested at wembley in 2011 fa cup semi-final . </t>"

    In [11]: sd[52]['system_summaries'].keys()
    Out[11]: dict_keys(['presumm_out_trans_abs.txt', 'two_stage_rl_out.txt', 'unilm_out_v2.txt', 't5_out_large.txt', 'presumm_out_ext_abs.txt', 'ptr_generator_out_pointer_gen_cov.txt', 'bart_out.txt', 'fast_abs_rl_out_rerank.txt', 't5_out_11B.txt', 'presumm_out_abs.txt', 'bottom_up_out.txt', 'unilm_out_v1.txt', 't5_out_base.txt', 'semsim_out.txt'])

    In [12]: 


    """
    sd = pickle.load(open(pickfile, 'rb'))
    # 1. Get documents 

    # code from https://github.com/forrestbao/SueNes/blob/newsroom/human/realsumm/generate_test.py
    used_test_id = [1017, 10586, 11343, 1521, 2736, 3789, 5025, 5272, 5576, 6564, 7174, 7770, 8334, 9325, 9781, 10231, 10595, 11351, 1573, 2748, 3906, 5075, 5334, 5626, 6714, 7397, 7823, 8565, 9393, 9825, 10325, 10680, 11355, 1890, 307, 4043, 5099, 5357, 5635, 6731, 7535, 7910, 8613, 9502, 10368, 10721, 1153, 19, 3152, 4303, 5231, 5420, 5912, 6774, 7547, 8001, 8815, 9555, 10537, 10824, 1173, 1944, 3172, 4315, 5243, 5476, 6048, 6784, 7584, 8054, 8997, 9590, 10542, 11049, 1273, 2065, 3583, 4637, 5244, 5524, 6094, 6976, 7626, 8306, 9086, 9605, 10563, 11264, 1492, 2292, 3621, 4725, 5257, 5558, 6329, 7058, 7670, 8312, 9221, 9709]
    cnndm_test_articles = []
    with open("src.txt", "r", encoding="utf-8") as f:
        cnndm_test_articles = list(f)
    
    used_articles = [cnndm_test_articles[i] for i in used_test_id]

    dataset_df = pandas.DataFrame([], )
    
    for sample in  sd.values():
        doc_id = sample['doc_id']
        doc = used_articles[doc_id]
        ref_summ = sample['ref_summ']
        for sys_name in sample['system_summaries']:
            sys_summ = sample['system_summaries'][sys_name]['system_summary']
            score_dict  = sample['system_summaries'][sys_name]['scores']
            row = {
                "doc_id": [doc_id],
                "ArticleText": [clean_text(doc)],
                "ReferenceSummary": [clean_text(ref_summ)],
                "system": [sys_name[:-4]],
                "SystemSummary": [clean_text(sys_summ)],
                }
            for score_name, score in score_dict.items():
                row[score_name] = [score]
            # human rating is in row['litepyramid_recall']

            tmp_dataset_df = pandas.DataFrame.from_dict(row)

            dataset_df = pandas.concat([
                dataset_df, tmp_dataset_df])

    return dataset_df
        
if __name__ == "__main__":

    import eval # DocAsRef's

    system_type = "ext" # or "abs"

    dataset_df = load_realsumm(f'{system_type}.pkl')

    corr_df = eval.eval_summary_level(dataset_df, pre_calculated_metrics=['rouge_1_f_score',
       'rouge_2_recall', 'rouge_l_recall', 'rouge_2_precision',
       'rouge_2_f_score', 'rouge_1_precision', 'rouge_1_recall',
       'rouge_l_precision', 'rouge_l_f_score', 'js-2', 'mover_score',
       'bert_recall_score', 'bert_precision_score', 'bert_f_score'], debug=False)
    with pandas.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
        with open(f"results/result_realsumm_{system_type}.md", 'w') as f:
            f.write("```")
            f.write(corr_df['average'].to_string())
            f.write("```")
        print(corr_df['average'])

    with open(f"results/result_realsumm_{system_type}.json", 'w') as f:
        json_ugly = corr_df.to_json(orient="index")
        json_parsed = json.loads(json_ugly)
        f.write(json.dumps(json_parsed, indent=2))